"""
Gate Effect Analysis for Dissipative Cat Qubits
================================================
Investigates how applying single-qubit gate perturbations to the cat-qubit
Hamiltonian affects the control parameters (eps_d, g_2) and coherence
times T_X and T_Z.

Gate models
-----------
1. Z-rotation (detuning):  H -> H + delta * a†a
   Rotates the phase of coherent states differentially for |+alpha> vs |-alpha>.

2. X-gate (parity kick):  H -> H + Omega_x * (a†)^n_cat + h.c.  (via parity op)
   In practice modeled as a brief displacement pulse that swaps |alpha> <-> |-alpha>.
   Here we model it as adding a scaled parity operator to the Hamiltonian.

3. Phase rotation of g_2:  g_2 -> g_2 * exp(i*phi)
   Rotates the stabilization axis in phase space — changes the cat-state orientation.

For each gate scenario, the script:
  (a) Scans the gate strength and measures T_X, T_Z at the notebook baseline.
  (b) Runs a fast CMA-ES optimization of (eps_d, g_2) to find best lifetimes
      under the perturbation.
  (c) Produces comparison plots.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from concurrent.futures import ThreadPoolExecutor

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── Fixed system parameters (same as TzTx_optimization.py) ─────────────────
na       = 15
nb       = 5
kappa_b  = 10.0   # MHz
kappa_a  = 1.0    # MHz

NOTEBOOK_EPS_D = 4.0
NOTEBOOK_G2    = 1.0
TARGET_RATIO   = 320.0

# Fast optimizer settings
BATCH_SIZE = 6
N_EPOCHS   = 20
SIGMA0     = 0.5

BOUNDS = np.array([
    [0.5, 8.0],   # eps_d
    [0.2, 4.0],   # g_2
])

# ── Global operators ────────────────────────────────────────────────────────
a_s  = dq.destroy(na)
a_op = dq.tensor(a_s, dq.eye(nb))
b_op = dq.tensor(dq.eye(na), dq.destroy(nb))

parity_s  = (1j * jnp.pi * a_s.dag() @ a_s).expm()
parity_op = dq.tensor(parity_s, dq.eye(nb))

a2_op = dq.powm(a_op, 2)

# Number operator in full space (for detuning)
num_op = a_op.dag() @ a_op


# ══════════════════════════════════════════════════════════════════════════════
# 1. ANALYTIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def analytic_alpha(eps_d: float, g2: float) -> float:
    kappa_2 = 4.0 * g2**2 / kappa_b
    if kappa_2 < 1e-12:
        return 0.0
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    inner = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
    return float(np.sqrt(max(inner, 0.0)))


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM BUILDER WITH GATE PERTURBATION
# ══════════════════════════════════════════════════════════════════════════════

def build_system_with_gate(eps_d: float, g2: float,
                           gate_type: str = "none",
                           gate_strength: float = 0.0):
    """
    Build H (with optional gate perturbation), jump operators, and basis kets.

    gate_type : str
        "none"     — no perturbation (baseline stabilization)
        "Z"        — Z-rotation via detuning: H += delta * a†a
        "X"        — X-gate via parity term: H += Omega * parity_op
        "phase_g2" — rotate g_2 phase: g_2 -> g_2 * exp(i * phi)

    gate_strength : float
        "Z"        — delta  (MHz, detuning frequency)
        "X"        — Omega  (MHz, parity-kick coupling)
        "phase_g2" — phi    (radians, phase angle)
    """
    # Apply phase rotation to g_2 if requested
    g2_eff = g2
    if gate_type == "phase_g2":
        g2_eff = g2 * np.exp(1j * gate_strength)

    alpha = analytic_alpha(eps_d, float(np.abs(g2_eff)))

    # Base Hamiltonian (two-photon exchange + buffer drive)
    H = (jnp.conj(g2_eff) * a_op.dag() @ a_op.dag() @ b_op
         + g2_eff * a_op @ a_op @ b_op.dag()
         - eps_d * b_op.dag()
         - jnp.conj(eps_d) * b_op)

    # Add gate perturbation
    if gate_type == "Z":
        H = H + gate_strength * num_op
    elif gate_type == "X":
        H = H + gate_strength * parity_op

    L_b = jnp.sqrt(kappa_b) * b_op
    L_a = jnp.sqrt(kappa_a) * a_op

    ket_p = dq.coherent(na, alpha)
    ket_m = dq.coherent(na, -alpha)
    cat_x = (ket_p + ket_m).unit()
    psi0_x = dq.tensor(cat_x, dq.fock(nb, 0))

    return H, [L_b, L_a], alpha, psi0_x, ket_p, ket_m


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPONENTIAL FIT
# ══════════════════════════════════════════════════════════════════════════════

def _exp_model(p, t):
    A, tau, C = p
    return A * jnp.exp(-t / tau) + C


def _fit_lifetime(ts, signal):
    A0   = float(signal[0] - signal[-1])
    C0   = float(signal[-1])
    tau0 = float(ts[-1] - ts[0])
    def residuals(p):
        return _exp_model(p, ts) - signal
    result = least_squares(
        residuals, [A0, tau0, C0],
        bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
        loss="soft_l1", f_scale=0.05,
    )
    return float(result.x[1])


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIFETIME MEASUREMENTS (gate-aware)
# ══════════════════════════════════════════════════════════════════════════════

def measure_Tx_gate(eps_d, g2, gate_type="none", gate_strength=0.0,
                    tfinal=None, n_pts=25):
    H, jumps, alpha, psi0_x, _, _ = build_system_with_gate(
        eps_d, g2, gate_type, gate_strength)
    if tfinal is None:
        tfinal = float(np.clip(3.0 / max(kappa_a * alpha**2, 1e-6), 0.3, 15.0))
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    result = dq.mesolve(H, jumps, psi0_x, tsave,
                        exp_ops=[parity_op],
                        options=dq.Options(progress_meter=False))
    sxt = np.array(result.expects[0, :].real)
    ts  = np.array(tsave)
    if sxt[-1] > 0.9 * sxt[0]:
        Tx = 5.0 * tfinal
    else:
        try:
            Tx = _fit_lifetime(ts, sxt)
        except Exception:
            Tx = 0.0
    return {"T_X": Tx, "sxt": sxt, "ts": ts, "alpha": alpha}


def measure_Tz_gate(eps_d, g2, gate_type="none", gate_strength=0.0,
                    tfinal=None, n_pts=25):
    H, jumps, alpha, _, ket_p, ket_m = build_system_with_gate(
        eps_d, g2, gate_type, gate_strength)
    psi0_z = dq.tensor(ket_p, dq.fock(nb, 0))
    if tfinal is None:
        tz_est = float(np.exp(2.0 * alpha**2)) / kappa_a
        tfinal = float(np.clip(3.0 * tz_est, 50.0, 300.0))
    tsave = jnp.linspace(0.0, tfinal, n_pts)
    res = dq.mesolve(H, jumps, psi0_z, tsave,
                     options=dq.Options(progress_meter=False),
                     exp_ops=[a2_op, a_op, a_op.dag(), a_op.dag() @ a_op])
    a2_exp   = res.expects[0, :]
    a_exp    = res.expects[1, :]
    adag_exp = res.expects[2, :]
    num_exp  = jnp.maximum(res.expects[3, :].real, 1e-12)
    phi  = jnp.angle(a2_exp) / 2
    Xphi = 0.5 * (jnp.exp(1j*phi)*adag_exp + jnp.exp(-1j*phi)*a_exp) / jnp.sqrt(num_exp)
    szt = np.array(jnp.real(Xphi))
    ts  = np.array(res.tsave)
    if szt[-1] > 0.9 * szt[0]:
        Tz = 5.0 * tfinal
    else:
        try:
            Tz = _fit_lifetime(ts, szt)
        except Exception:
            Tz = 0.0
    return {"T_Z": Tz, "szt": szt, "ts": ts, "alpha": alpha}


def measure_joint_gate(eps_d, g2, gate_type="none", gate_strength=0.0):
    """Measure T_X and T_Z concurrently under a gate perturbation."""
    H, jumps, alpha, psi0_x, ket_p, ket_m = build_system_with_gate(
        eps_d, g2, gate_type, gate_strength)

    tx_est = 1.0 / max(kappa_a * alpha**2, 1e-6)
    tz_est = float(np.exp(2.0 * alpha**2)) / kappa_a
    tx_window = float(np.clip(3.0 * tx_est, 0.3, 15.0))
    tz_window = float(np.clip(3.0 * tz_est, 50.0, 300.0))

    # T_X simulation
    tsave_x = jnp.linspace(0.0, tx_window, 20)
    res_x = dq.mesolve(H, jumps, psi0_x, tsave_x,
                       exp_ops=[parity_op],
                       options=dq.Options(progress_meter=False))
    sxt = np.array(res_x.expects[0, :].real)
    ts_x = np.array(tsave_x)

    # T_Z simulation
    psi0_z = dq.tensor(ket_p, dq.fock(nb, 0))
    tsave_z = jnp.linspace(0.0, tz_window, 25)
    res_z = dq.mesolve(H, jumps, psi0_z, tsave_z,
                       options=dq.Options(progress_meter=False),
                       exp_ops=[a2_op, a_op, a_op.dag(), a_op.dag() @ a_op])
    a2_exp   = res_z.expects[0, :]
    a_exp    = res_z.expects[1, :]
    adag_exp = res_z.expects[2, :]
    num_exp  = jnp.maximum(res_z.expects[3, :].real, 1e-12)
    phi_z = jnp.angle(a2_exp) / 2
    Xphi  = 0.5 * (jnp.exp(1j*phi_z)*adag_exp + jnp.exp(-1j*phi_z)*a_exp) / jnp.sqrt(num_exp)
    szt = np.array(jnp.real(Xphi))
    ts_z = np.array(res_z.tsave)

    # Fit lifetimes
    try:
        Tx = _fit_lifetime(ts_x, sxt) if sxt[-1] <= 0.9 * sxt[0] else 5.0 * tx_window
    except Exception:
        Tx = 0.0
    try:
        Tz = _fit_lifetime(ts_z, szt) if szt[-1] <= 0.9 * szt[0] else 5.0 * tz_window
    except Exception:
        Tz = 0.0

    ratio = Tz / max(Tx, 1e-9)
    return {"T_X": Tx, "T_Z": Tz, "ratio": ratio, "alpha": alpha,
            "sxt": sxt, "ts_x": ts_x, "szt": szt, "ts_z": ts_z}


# ══════════════════════════════════════════════════════════════════════════════
# 5. PARAMETER SCAN — sweep gate strength at fixed (eps_d, g_2)
# ══════════════════════════════════════════════════════════════════════════════

def scan_gate_strength(gate_type: str, strengths: np.ndarray,
                       eps_d: float = NOTEBOOK_EPS_D,
                       g2: float = NOTEBOOK_G2,
                       verbose: bool = True):
    """
    Sweep gate_strength at fixed control parameters; return T_X, T_Z arrays.
    """
    Tx_arr = np.zeros(len(strengths))
    Tz_arr = np.zeros(len(strengths))
    alpha_arr = np.zeros(len(strengths))

    for i, s in enumerate(strengths):
        out = measure_joint_gate(eps_d, g2, gate_type, float(s))
        Tx_arr[i]    = out["T_X"]
        Tz_arr[i]    = out["T_Z"]
        alpha_arr[i] = out["alpha"]
        if verbose:
            print(f"  [{gate_type}] strength={s:+.4f}  "
                  f"T_X={out['T_X']:.4f}  T_Z={out['T_Z']:.1f}  "
                  f"ratio={out['ratio']:.1f}  alpha={out['alpha']:.3f}",
                  flush=True)

    return {"strengths": strengths, "T_X": Tx_arr, "T_Z": Tz_arr,
            "alpha": alpha_arr, "gate_type": gate_type}


# ══════════════════════════════════════════════════════════════════════════════
# 6. FAST CMA-ES OPTIMIZER (gate-aware)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_with_gate(gate_type: str, gate_strength: float,
                       eps_d_init: float = NOTEBOOK_EPS_D,
                       g2_init: float = NOTEBOOK_G2,
                       n_epochs: int = N_EPOCHS,
                       batch_size: int = BATCH_SIZE,
                       verbose: bool = True):
    """
    Fast CMA-ES optimization of (eps_d, g_2) under a gate perturbation.
    Returns (best_theta, best_Tx, best_Tz, history).
    """
    theta0 = np.array([eps_d_init, g2_init])

    if verbose:
        print(f"\n  Optimizing under gate={gate_type}, strength={gate_strength:.4f}")
        print(f"  Warm-start: eps_d={eps_d_init}, g_2={g2_init}")

    optimizer = SepCMA(mean=theta0, sigma=SIGMA0, bounds=BOUNDS,
                       population_size=batch_size)

    best_obj   = -np.inf
    best_theta = theta0.copy()
    best_Tx    = 0.0
    best_Tz    = 0.0
    history    = {"T_X": [], "T_Z": [], "ratio": [], "eps_d": [], "g2": []}

    for epoch in range(n_epochs):
        candidates = [optimizer.ask() for _ in range(batch_size)]
        solutions  = []
        best_ep_obj = -np.inf
        best_ep_idx = 0

        for i, x in enumerate(candidates):
            x = np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])
            try:
                out   = measure_joint_gate(float(x[0]), float(x[1]),
                                           gate_type, gate_strength)
                Tx    = out["T_X"]
                Tz    = out["T_Z"]
                ratio = out["ratio"]
                ratio_err = (ratio - TARGET_RATIO) / TARGET_RATIO
                loss  = (-(Tx + Tz / TARGET_RATIO) / 2.0
                         + 8.0 * ratio_err**2)
                obj   = Tx + Tz / TARGET_RATIO
            except Exception:
                Tx, Tz, ratio, loss, obj = 0.0, 0.0, 0.0, 1e6, 0.0
            solutions.append((x, loss))
            if obj > best_ep_obj:
                best_ep_obj = obj
                best_ep_idx = i

        optimizer.tell(solutions)

        bx = solutions[best_ep_idx][0]
        bm = measure_joint_gate(float(bx[0]), float(bx[1]),
                                gate_type, gate_strength)

        if best_ep_obj > best_obj:
            best_obj   = best_ep_obj
            best_theta = bx.copy()
            best_Tx    = bm["T_X"]
            best_Tz    = bm["T_Z"]

        history["T_X"].append(bm["T_X"])
        history["T_Z"].append(bm["T_Z"])
        history["ratio"].append(bm["ratio"])
        history["eps_d"].append(float(bx[0]))
        history["g2"].append(float(bx[1]))

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch:3d}/{n_epochs}: T_X={bm['T_X']:.4f}  "
                  f"T_Z={bm['T_Z']:.1f}  ratio={bm['ratio']:.1f}  "
                  f"eps_d={float(bx[0]):.3f}  g_2={float(bx[1]):.3f}",
                  flush=True)

    return best_theta, best_Tx, best_Tz, history


# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_gate_scan(scan_results: list, save_path: str = "gate_scan.png"):
    """
    Plot T_X, T_Z, and ratio vs gate strength for multiple gate types.
    scan_results: list of dicts from scan_gate_strength().
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Effect of Single-Qubit Gates on Cat-Qubit Coherence Times",
                 fontsize=13)

    colors = {"Z": "steelblue", "X": "crimson", "phase_g2": "seagreen"}
    labels = {"Z": r"Z-rot (detuning $\delta$)",
              "X": r"X-gate (parity $\Omega$)",
              "phase_g2": r"$g_2$ phase $\phi$"}

    for sr in scan_results:
        gt = sr["gate_type"]
        c  = colors.get(gt, "gray")
        lb = labels.get(gt, gt)

        axes[0].plot(sr["strengths"], sr["T_X"], "o-", ms=3, color=c, label=lb)
        axes[1].plot(sr["strengths"], sr["T_Z"], "o-", ms=3, color=c, label=lb)
        ratio = sr["T_Z"] / np.maximum(sr["T_X"], 1e-9)
        axes[2].plot(sr["strengths"], ratio, "o-", ms=3, color=c, label=lb)

    axes[0].set(title=r"Phase-flip lifetime $T_X$",
                xlabel="Gate strength", ylabel=r"$T_X$ (µs)")
    axes[0].legend(fontsize=8)

    axes[1].set(title=r"Bit-flip lifetime $T_Z$",
                xlabel="Gate strength", ylabel=r"$T_Z$ (µs)")
    axes[1].legend(fontsize=8)

    axes[2].set(title=r"Noise bias $T_Z / T_X$",
                xlabel="Gate strength", ylabel=r"$T_Z / T_X$")
    axes[2].axhline(TARGET_RATIO, color="red", linestyle="--", alpha=0.5,
                    label=f"target = {TARGET_RATIO:.0f}")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Gate scan figure saved to {save_path}")
    plt.show()


def plot_optimization_comparison(opt_results: dict,
                                  save_path: str = "gate_opt_comparison.png"):
    """
    Bar chart comparing best T_X, T_Z, and optimal (eps_d, g_2) across gate
    scenarios (including 'none' baseline).
    """
    names     = list(opt_results.keys())
    Tx_vals   = [opt_results[n]["T_X"] for n in names]
    Tz_vals   = [opt_results[n]["T_Z"] for n in names]
    epsd_vals = [opt_results[n]["eps_d"] for n in names]
    g2_vals   = [opt_results[n]["g_2"] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Optimized Parameters Under Different Gate Perturbations",
                 fontsize=13)

    x = np.arange(len(names))
    w = 0.5

    ax = axes[0, 0]
    bars = ax.bar(x, Tx_vals, w, color="steelblue", edgecolor="navy")
    ax.set(title=r"Optimized $T_X$", ylabel=r"$T_X$ (µs)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    for bar, v in zip(bars, Tx_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    bars = ax.bar(x, Tz_vals, w, color="darkorange", edgecolor="sienna")
    ax.set(title=r"Optimized $T_Z$", ylabel=r"$T_Z$ (µs)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    for bar, v in zip(bars, Tz_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    bars = ax.bar(x, epsd_vals, w, color="seagreen", edgecolor="darkgreen")
    ax.set(title=r"Optimal $\epsilon_d$", ylabel=r"$\epsilon_d$ (MHz)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    for bar, v in zip(bars, epsd_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 1]
    bars = ax.bar(x, g2_vals, w, color="purple", edgecolor="indigo")
    ax.set(title=r"Optimal $g_2$", ylabel=r"$g_2$ (MHz)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    for bar, v in zip(bars, g2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Optimization comparison figure saved to {save_path}")
    plt.show()


def plot_decay_under_gates(gate_decays: dict,
                           save_path: str = "gate_decay_curves.png"):
    """
    Plot T_X and T_Z decay curves for each gate scenario side-by-side.
    gate_decays: {name: {"tx_sim": {...}, "tz_sim": {...}, ...}}
    """
    names = list(gate_decays.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Decay Curves Under Gate Perturbations", fontsize=13, y=1.01)

    colors = ["steelblue", "crimson", "seagreen", "darkorange", "purple"]

    for i, name in enumerate(names):
        gd  = gate_decays[name]
        col = colors[i % len(colors)]

        # T_X panel
        ax = axes[i, 0]
        tx_sim = gd["tx_sim"]
        ax.plot(tx_sim["ts"], tx_sim["sxt"], "o", ms=3, color=col, alpha=0.7)
        t_fit = np.linspace(tx_sim["ts"][0], tx_sim["ts"][-1], 200)
        A = float(tx_sim["sxt"][0] - tx_sim["sxt"][-1])
        C = float(tx_sim["sxt"][-1])
        ax.plot(t_fit, A * np.exp(-t_fit / tx_sim["T_X"]) + C,
                color=col, linewidth=1.5,
                label=f"$T_X$ = {tx_sim['T_X']:.4f} µs")
        ax.set(title=f"{name} — $\\langle X \\rangle$ decay",
               xlabel="Time (µs)", ylabel=r"$\langle X \rangle$")
        ax.legend(fontsize=8)

        # T_Z panel
        ax = axes[i, 1]
        tz_sim = gd["tz_sim"]
        ax.plot(tz_sim["ts"], tz_sim["szt"], "s", ms=3, color=col, alpha=0.7)
        t_fit = np.linspace(tz_sim["ts"][0], tz_sim["ts"][-1], 200)
        A = float(tz_sim["szt"][0] - tz_sim["szt"][-1])
        C = float(tz_sim["szt"][-1])
        ax.plot(t_fit, A * np.exp(-t_fit / tz_sim["T_Z"]) + C,
                color=col, linewidth=1.5,
                label=f"$T_Z$ = {tz_sim['T_Z']:.1f} µs")
        ax.set(title=f"{name} — $\\langle Z \\rangle$ decay",
               xlabel="Time (µs)", ylabel=r"$\langle Z \rangle$")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Decay curves figure saved to {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN — full analysis pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Gate Effect Analysis for Dissipative Cat Qubits")
    print(f"  na={na}, nb={nb}, kappa_b={kappa_b}, kappa_a={kappa_a}")
    print("=" * 65)

    # ── A. Baseline (no gate) ─────────────────────────────────────────────────
    print("\n[1/4] Measuring baseline (no gate) ...", flush=True)
    baseline = measure_joint_gate(NOTEBOOK_EPS_D, NOTEBOOK_G2, "none", 0.0)
    print(f"  Baseline: T_X={baseline['T_X']:.4f}  T_Z={baseline['T_Z']:.1f}  "
          f"ratio={baseline['ratio']:.1f}  alpha={baseline['alpha']:.3f}")

    # ── B. Gate strength scans ────────────────────────────────────────────────
    print("\n[2/4] Scanning gate strengths at notebook baseline ...", flush=True)

    # Z-rotation (detuning): scan delta from 0 to 1 MHz
    print("\n  --- Z-rotation (detuning) ---")
    scan_Z = scan_gate_strength("Z", np.linspace(0.0, 1.0, 8))

    # X-gate (parity coupling): scan Omega from 0 to 0.5 MHz
    print("\n  --- X-gate (parity kick) ---")
    scan_X = scan_gate_strength("X", np.linspace(0.0, 0.5, 8))

    # Phase rotation of g_2: scan phi from 0 to pi
    print("\n  --- g_2 phase rotation ---")
    scan_phase = scan_gate_strength("phase_g2", np.linspace(0.0, np.pi, 8))

    # Plot scans
    plot_gate_scan([scan_Z, scan_X, scan_phase], save_path="gate_scan.png")

    # ── C. Optimize (eps_d, g_2) under each gate scenario ─────────────────────
    print("\n[3/4] Running CMA-ES optimization under each gate ...", flush=True)

    gate_scenarios = {
        "No gate":           ("none",     0.0),
        "Z (δ=0.3)":         ("Z",        0.3),
        "Z (δ=0.6)":         ("Z",        0.6),
        "X (Ω=0.15)":        ("X",        0.15),
        "X (Ω=0.3)":         ("X",        0.3),
        "Phase (φ=π/4)":     ("phase_g2", np.pi / 4),
        "Phase (φ=π/2)":     ("phase_g2", np.pi / 2),
    }

    opt_results = {}
    gate_decays = {}

    for name, (gtype, gstr) in gate_scenarios.items():
        print(f"\n{'─'*50}")
        print(f"  Scenario: {name}  (gate={gtype}, strength={gstr:.4f})")

        theta, Tx, Tz, hist = optimize_with_gate(
            gtype, gstr, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
        epsd_opt, g2_opt = theta

        opt_results[name] = {
            "T_X": Tx, "T_Z": Tz,
            "eps_d": float(epsd_opt), "g_2": float(g2_opt),
            "ratio": Tz / max(Tx, 1e-9),
            "history": hist,
        }

        # Measure decay curves at optimal parameters for this gate
        tx_sim = measure_Tx_gate(epsd_opt, g2_opt, gtype, gstr, n_pts=50)
        tz_sim = measure_Tz_gate(epsd_opt, g2_opt, gtype, gstr, n_pts=50)
        gate_decays[name] = {"tx_sim": tx_sim, "tz_sim": tz_sim}

        print(f"  => Optimal: eps_d={epsd_opt:.3f}  g_2={g2_opt:.3f}  "
              f"T_X={Tx:.4f}  T_Z={Tz:.1f}  ratio={Tz/max(Tx,1e-9):.1f}")

    # ── D. Final comparison plots ─────────────────────────────────────────────
    print("\n[4/4] Generating comparison plots ...", flush=True)

    plot_optimization_comparison(opt_results, save_path="gate_opt_comparison.png")
    plot_decay_under_gates(gate_decays, save_path="gate_decay_curves.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Scenario':<20s} {'eps_d':>7s} {'g_2':>7s} {'T_X (µs)':>10s} "
          f"{'T_Z (µs)':>10s} {'Ratio':>8s}")
    print("-" * 80)
    for name, r in opt_results.items():
        print(f"{name:<20s} {r['eps_d']:7.3f} {r['g_2']:7.3f} "
              f"{r['T_X']:10.4f} {r['T_Z']:10.1f} {r['ratio']:8.1f}")
    print("=" * 80)

    print("\nDone. All figures saved.")
