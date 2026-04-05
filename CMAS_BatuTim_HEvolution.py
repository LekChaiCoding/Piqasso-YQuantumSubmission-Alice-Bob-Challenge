"""
Joint T_X / T_Z Optimizer - Pure Hamiltonian Evolution (no dissipation)
=======================================================================
Same structure and figures as CMAS_BatuTim.py, but replaces Lindblad
(mesolve) with pure Schrodinger evolution (sesolve).

Purpose
-------
Investigate how eps_d and g2 affect "lifetime" observables <X> and <Z>
under coherent Hamiltonian dynamics alone -- no photon loss, no jump
operators.  This isolates the role of the two-photon exchange interaction
from dissipative stabilization.

Physical context
----------------
Under pure H evolution, cat states form transiently via coherent Rabi-like
oscillations but are NOT stabilized.  The parity <X> and coherent-state
projection <Z> will typically oscillate rather than decay exponentially.

Nonetheless, we fit the same A*exp(-t/tau)+C model as CMAS_BatuTim.py.
When there is no monotone decay, the fit captures an effective oscillation
envelope timescale, or the fallback returns tau -> 5*t_final (= no decay).

Analyses / Outputs (mirror CMAS_BatuTim exactly)
-------------------------------------------------
1. optimizer_progress      -- 2x3 panel: T_X, T_Z, ratio, params, alpha, objective
2. Tx_decay_comparison     -- 2x2 panel: notebook vs optimized decays
3. initial_Tx_Tz_combined  -- single panel, log-time overlay of <X> and <Z>
4. optimized_Tx_Tz_combined -- same, at optimized parameters

References
----------
[1] Mirrahimi et al., New J. Phys. 16, 045014 (2014)
[2] Leghtas et al., Science 347, 853 (2015)
[3] Grimm et al., Nature 584, 205 (2020)
[4] Gautier et al., PRX Quantum 3, 020339 (2022)
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

# -- Fixed system parameters -------------------------------------------------
na      = 15
nb      = 5
kappa_b = 10.0
kappa_a = 1.0

# -- Notebook baseline -------------------------------------------------------
NOTEBOOK_EPS_D    = 4.0
NOTEBOOK_G2       = 1.0
NOTEBOOK_TX_FINAL = 1.0
NOTEBOOK_TZ_FINAL = 200.0
TARGET_RATIO      = 320.0

# -- CMA-ES hyperparameters --------------------------------------------------
BATCH_SIZE    = 10
N_EPOCHS      = 50
SIGMA0        = 0.5
RATIO_PENALTY = 8.0

BOUNDS = np.array([
    [0.5, 8.0],
    [0.2, 4.0],
])

# -- Global operators --------------------------------------------------------
a_s  = dq.destroy(na)
a_op = dq.tensor(a_s, dq.eye(nb))
b_op = dq.tensor(dq.eye(na), dq.destroy(nb))

parity_s  = (1j * jnp.pi * a_s.dag() @ a_s).expm()
parity_op = dq.tensor(parity_s, dq.eye(nb))

a2_op = dq.powm(a_op, 2)


# =============================================================================
# 1.  ANALYTIC ESTIMATES
# =============================================================================

def analytic_alpha(eps_d, g2):
    kappa_2 = 4.0 * g2**2 / kappa_b
    if kappa_2 < 1e-12:
        return 0.0
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    inner = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
    return float(np.sqrt(max(inner, 0.0)))


def analytic_Tx(eps_d, g2):
    alpha = analytic_alpha(eps_d, g2)
    if alpha < 1e-6:
        return 1e-3
    return 1.0 / (kappa_a * alpha**2)


def analytic_Tz(eps_d, g2):
    alpha = analytic_alpha(eps_d, g2)
    return float(np.exp(2.0 * alpha**2)) / kappa_a


# =============================================================================
# 2.  SYSTEM BUILDER  (no jump operators)
# =============================================================================

def build_system(eps_d, g2):
    alpha = analytic_alpha(eps_d, g2)

    H = (jnp.conj(g2) * a_op.dag() @ a_op.dag() @ b_op
         + g2 * a_op @ a_op @ b_op.dag()
         - eps_d * b_op.dag()
         - jnp.conj(eps_d) * b_op) + 0.5 * a_op.dag() @ a_op

    ket_p = dq.coherent(na, alpha)
    ket_m = dq.coherent(na, -alpha)

    cat_x  = (ket_p + ket_m).unit()
    psi0_x = dq.tensor(cat_x, dq.fock(nb, 0))

    return H, alpha, psi0_x, ket_p, ket_m


# =============================================================================
# 3.  EXPONENTIAL FIT
# =============================================================================

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
        residuals,
        [A0, tau0, C0],
        bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
        loss="soft_l1",
        f_scale=0.05,
    )
    return float(result.x[1])


# =============================================================================
# 4.  LIFETIME MEASUREMENTS  (sesolve)
# =============================================================================

def _measure_Tx_inner(H, psi0_x, alpha, tfinal, n_pts):
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    result = dq.sesolve(
        H, psi0_x, tsave,
        exp_ops=[parity_op],
        options=dq.Options(progress_meter=False),
    )
    sxt = np.array(result.expects[0, :].real)
    ts  = np.array(tsave)
    if sxt[-1] > 0.9 * sxt[0]:
        Tx = 5.0 * tfinal
    else:
        try:
            Tx = _fit_lifetime(ts, sxt)
        except Exception:
            Tx = 0.0
    return {"T_X": Tx, "sxt": sxt, "ts": ts, "alpha": alpha, "tfinal": tfinal}


def _measure_Tz_inner(H, ket_p, ket_m, alpha, tfinal, n_pts):
    psi0_z = dq.tensor(ket_p, dq.fock(nb, 0))
    tsave  = jnp.linspace(0.0, tfinal, n_pts)

    res = dq.sesolve(
        H, psi0_z, tsave,
        options=dq.Options(progress_meter=False),
        exp_ops=[a2_op, a_op, a_op.dag(), a_op.dag() @ a_op],
    )
    a2_exp   = res.expects[0, :]
    a_exp    = res.expects[1, :]
    adag_exp = res.expects[2, :]
    num_exp  = jnp.maximum(res.expects[3, :].real, 1e-12)
    phi  = jnp.angle(a2_exp) / 2
    Xphi = 0.5 * (jnp.exp(1j * phi) * adag_exp + jnp.exp(-1j * phi) * a_exp) / jnp.sqrt(num_exp)
    szt = np.array(jnp.real(Xphi))
    ts  = np.array(tsave)
    if szt[-1] > 0.9 * szt[0]:
        Tz = 5.0 * tfinal
    else:
        try:
            Tz = _fit_lifetime(ts, szt)
        except Exception:
            Tz = 0.0
    return {"T_Z": Tz, "szt": szt, "ts": ts, "alpha": alpha, "tfinal": tfinal}


def measure_Tx(eps_d, g2, tfinal=None, n_pts=30):
    H, alpha, psi0_x, _, _ = build_system(eps_d, g2)
    if tfinal is None:
        tfinal = float(np.clip(3.0 / max(kappa_a * alpha**2, 1e-6), 0.3, 15.0))
    return _measure_Tx_inner(H, psi0_x, alpha, tfinal, n_pts)


def measure_Tz(eps_d, g2, tfinal=None, n_pts=25):
    H, alpha, _, ket_p, ket_m = build_system(eps_d, g2)
    if tfinal is None:
        tz_est = float(np.exp(2.0 * alpha**2)) / kappa_a
        tfinal = float(np.clip(3.0 * tz_est, 50.0, 1000.0))
    return _measure_Tz_inner(H, ket_p, ket_m, alpha, tfinal, n_pts)


def measure_joint(eps_d, g2):
    H, alpha, psi0_x, ket_p, ket_m = build_system(eps_d, g2)

    tx_est = 1.0 / max(kappa_a * alpha**2, 1e-6)
    tz_est = float(np.exp(2.0 * alpha**2)) / kappa_a
    tx_window = float(np.clip(3.0 * tx_est, 0.3, 15.0))
    tz_window = float(np.clip(3.0 * tz_est, 50.0, 500.0))

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_x = ex.submit(_measure_Tx_inner, H, psi0_x, alpha, tx_window, 30)
        fut_z = ex.submit(_measure_Tz_inner, H, ket_p, ket_m, alpha, tz_window, 50)
        out_x = fut_x.result()
        out_z = fut_z.result()

    Tx    = out_x["T_X"]
    Tz    = out_z["T_Z"]
    ratio = Tz / max(Tx, 1e-9)

    return {
        "T_X": Tx, "T_Z": Tz, "ratio": ratio,
        "sxt": out_x["sxt"], "ts_x": out_x["ts"],
        "szt": out_z["szt"], "ts_z": out_z["ts"],
        "alpha": alpha,
    }


# =============================================================================
# 5.  CMA-ES OPTIMIZATION
# =============================================================================

def clip_bounds(x):
    return np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])


def _eval_candidate(x):
    x = clip_bounds(x)
    try:
        out       = measure_joint(float(x[0]), float(x[1]))
        Tx        = out["T_X"]
        Tz        = out["T_Z"]
        ratio     = out["ratio"]
        ratio_err = (ratio - TARGET_RATIO) / TARGET_RATIO
        val       = (-(Tx + Tz / TARGET_RATIO) / 2.0
                     + RATIO_PENALTY * ratio_err**2)
    except Exception:
        Tx, Tz, ratio, val = 0.0, 0.0, 0.0, 1e6
        out = {"alpha": analytic_alpha(float(x[0]), float(x[1]))}
    return {"T_X": Tx, "T_Z": Tz, "ratio": ratio, "x": x, "loss": val,
            "alpha": out["alpha"]}


def optimize_joint(
    eps_d_init=NOTEBOOK_EPS_D,
    g2_init=NOTEBOOK_G2,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    sigma0=SIGMA0,
    verbose=True,
):
    theta0 = np.array([eps_d_init, g2_init])
    alpha0 = analytic_alpha(*theta0)

    if verbose:
        print(f"Warm-start t0 = (eps_d={eps_d_init}, g_2={g2_init})")
        print(f"  Analytic alpha = {alpha0:.3f}")
        print(f"  Analytic T_X   = {analytic_Tx(*theta0):.3f} us")
        print(f"  Analytic T_Z   = {analytic_Tz(*theta0):.1f} us")
        print(f"  Target ratio   = {TARGET_RATIO:.0f}")
        print("-" * 65)

    optimizer  = SepCMA(mean=theta0, sigma=sigma0, bounds=BOUNDS,
                        population_size=batch_size)
    max_workers = min(batch_size, os.cpu_count() or 4)

    history = {"T_X": [], "T_Z": [], "ratio": [],
               "alpha": [], "eps_d": [], "g2": []}
    best_obj   = -np.inf
    best_theta = theta0.copy()
    best_Tx    = 0.0
    best_Tz    = 0.0

    for epoch in range(n_epochs):
        candidates = [optimizer.ask() for _ in range(batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            batch_metrics = list(pool.map(_eval_candidate, candidates))

        batch_solutions = [(bm["x"], bm["loss"]) for bm in batch_metrics]
        optimizer.tell(batch_solutions)

        best_idx = min(range(batch_size), key=lambda i: batch_metrics[i]["loss"])
        bm = batch_metrics[best_idx]
        bx = bm["x"]

        obj_ep = bm["T_X"] + bm["T_Z"] / TARGET_RATIO
        if obj_ep > best_obj:
            best_obj   = obj_ep
            best_theta = bx.copy()
            best_Tx    = bm["T_X"]
            best_Tz    = bm["T_Z"]

        history["T_X"].append(bm["T_X"])
        history["T_Z"].append(bm["T_Z"])
        history["ratio"].append(bm["ratio"])
        history["alpha"].append(bm["alpha"])
        history["eps_d"].append(float(bx[0]))
        history["g2"].append(float(bx[1]))

        if verbose:
            print(f"Epoch {epoch:3d}/{n_epochs}:  T_X={bm['T_X']:7.3f} us  "
                  f"T_Z={bm['T_Z']:8.1f} us  "
                  f"ratio={bm['ratio']:6.1f}  "
                  f"eps_d={float(bx[0]):.3f}  g_2={float(bx[1]):.3f}",
                  flush=True)

    return best_theta, best_Tx, best_Tz, history


# =============================================================================
# 6.  PLOTTING
# =============================================================================

def plot_optimizer_progress(history, save_path="HEvo_optimizer_progress.png"):
    epochs = np.arange(len(history["T_X"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Joint T_X / T_Z Optimizer (H evolution) -- target ratio = {TARGET_RATIO:.0f}",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(epochs, history["T_X"], color="steelblue", linewidth=1.2)
    ax.axhline(max(history["T_X"]), color="steelblue", linestyle="--", alpha=0.6,
               label=f"max = {max(history['T_X']):.3f} us")
    ax.set(title="Phase-flip lifetime $T_X$", xlabel="Epoch", ylabel="$T_X$ (us)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(epochs, history["T_Z"], color="darkorange", linewidth=1.2)
    ax.axhline(max(history["T_Z"]), color="darkorange", linestyle="--", alpha=0.6,
               label=f"max = {max(history['T_Z']):.1f} us")
    ax.set(title="Bit-flip lifetime $T_Z$", xlabel="Epoch", ylabel="$T_Z$ (us)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(epochs, history["ratio"], color="purple", linewidth=1.2)
    ax.axhline(TARGET_RATIO, color="red", linestyle="--", alpha=0.7,
               label=f"target = {TARGET_RATIO:.0f}")
    ax.set(title="Noise bias $T_Z / T_X$", xlabel="Epoch", ylabel="$T_Z / T_X$")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(epochs, history["eps_d"], label=r"$\epsilon_d$ (MHz)", color="steelblue")
    ax.plot(epochs, history["g2"],    label=r"$g_2$ (MHz)",        color="crimson")
    ax.set(title="Control parameters", xlabel="Epoch", ylabel="Value (MHz)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(epochs, history["alpha"], color="seagreen", linewidth=1.2)
    ax.set(title=r"Cat size $\alpha$ (analytic)", xlabel="Epoch", ylabel=r"$\alpha$")

    objective = [tx + tz / TARGET_RATIO
                 for tx, tz in zip(history["T_X"], history["T_Z"])]
    ax = axes[1, 2]
    ax.plot(epochs, objective, color="indigo", linewidth=1.2)
    ax.axhline(max(objective), color="indigo", linestyle="--", alpha=0.6,
               label=f"max = {max(objective):.3f} us")
    ax.set(title=r"Objective $T_X + T_Z/320$", xlabel="Epoch", ylabel="(us)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Optimizer progress figure saved to {save_path}")
    plt.show()


def plot_decay_comparison(
    nb_sim_x, nb_sim_z, opt_sim_x, opt_sim_z,
    nb_eps_d, nb_g2, opt_eps_d, opt_g2,
    save_path="HEvo_Tx_decay_comparison.png",
):
    NB_COLOR  = "#e05c2a"
    OPT_COLOR = "steelblue"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Cat-Qubit Lifetimes (H evolution): Notebook Baseline vs Optimized",
        fontsize=13,
    )

    def _fit(signal, lifetime, t_grid):
        A = float(signal[0] - signal[-1])
        C = float(signal[-1])
        return A * np.exp(-t_grid / lifetime) + C

    # (0,0): T_X comparison
    ax = axes[0, 0]
    ts_nb = nb_sim_x["ts"]; sxt_nb = nb_sim_x["sxt"]; Tx_nb = nb_sim_x["T_X"]
    t_fit = np.linspace(ts_nb[0], ts_nb[-1], 400)
    ax.plot(ts_nb, sxt_nb, "o", ms=3, color=NB_COLOR, alpha=0.8,
            label=fr"Notebook ($\epsilon_d$={nb_eps_d}, $g_2$={nb_g2})")
    ax.plot(t_fit, _fit(sxt_nb, Tx_nb, t_fit), color=NB_COLOR, linestyle="--",
            label=f"Fit  $T_X$ = {Tx_nb:.3f} us")

    ts_op = opt_sim_x["ts"]; sxt_op = opt_sim_x["sxt"]; Tx_op = opt_sim_x["T_X"]
    t_fit = np.linspace(ts_op[0], ts_op[-1], 400)
    ax.plot(ts_op, sxt_op, "o", ms=3, color=OPT_COLOR, alpha=0.8,
            label=fr"Optimized ($\epsilon_d$={opt_eps_d:.2f}, $g_2$={opt_g2:.2f})")
    ax.plot(t_fit, _fit(sxt_op, Tx_op, t_fit), color=OPT_COLOR, linestyle="--",
            label=f"Fit  $T_X$ = {Tx_op:.3f} us")

    ax.set(title=r"$\langle X \rangle$ -- notebook window (H evol.)",
           xlabel="Time (us)", ylabel=r"$\langle X \rangle$", ylim=(-0.05, 1.15))
    ax.legend(fontsize=8)
    improvement_x = Tx_op / max(Tx_nb, 1e-9)
    ax.text(0.03, 0.08, f"Improvement: {improvement_x:.1f}x",
            transform=ax.transAxes, fontsize=9, color=OPT_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # (0,1): T_Z comparison
    ax = axes[0, 1]
    ts_nb = nb_sim_z["ts"]; szt_nb = nb_sim_z["szt"]; Tz_nb = nb_sim_z["T_Z"]
    t_fit = np.linspace(ts_nb[0], ts_nb[-1], 400)
    ax.plot(ts_nb, szt_nb, "o", ms=3, color=NB_COLOR, alpha=0.8,
            label=fr"Notebook ($\epsilon_d$={nb_eps_d}, $g_2$={nb_g2})")
    ax.plot(t_fit, _fit(szt_nb, Tz_nb, t_fit), color=NB_COLOR, linestyle="--",
            label=f"Fit  $T_Z$ = {Tz_nb:.1f} us")

    ts_op = opt_sim_z["ts"]; szt_op = opt_sim_z["szt"]; Tz_op = opt_sim_z["T_Z"]
    t_fit = np.linspace(ts_op[0], ts_op[-1], 400)
    ax.plot(ts_op, szt_op, "o", ms=3, color=OPT_COLOR, alpha=0.8,
            label=fr"Optimized ($\epsilon_d$={opt_eps_d:.2f}, $g_2$={opt_g2:.2f})")
    ax.plot(t_fit, _fit(szt_op, Tz_op, t_fit), color=OPT_COLOR, linestyle="--",
            label=f"Fit  $T_Z$ = {Tz_op:.1f} us")

    ax.set(title=r"$\langle Z \rangle$ -- notebook window (H evol.)",
           xlabel="Time (us)", ylabel=r"$\langle Z \rangle$", ylim=(-0.05, 1.15))
    ax.legend(fontsize=8)
    improvement_z = Tz_op / max(Tz_nb, 1e-9)
    ax.text(0.03, 0.08, f"Improvement: {improvement_z:.1f}x",
            transform=ax.transAxes, fontsize=9, color=OPT_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # (1,0): Initial combined
    _plot_combined_panel(
        axes[1, 0], sim_x=nb_sim_x, sim_z=nb_sim_z,
        color_x=NB_COLOR, color_z=NB_COLOR, linestyle_z="--",
        title=(f"Initial: $T_X$ and $T_Z$ together\n"
               fr"($\epsilon_d$={nb_eps_d}, $g_2$={nb_g2})"),
        Tx_label=f"$\\langle X\\rangle$  $T_X$ = {nb_sim_x['T_X']:.3f} us",
        Tz_label=f"$\\langle Z\\rangle$  $T_Z$ = {nb_sim_z['T_Z']:.1f} us",
    )

    # (1,1): Optimized combined
    _plot_combined_panel(
        axes[1, 1], sim_x=opt_sim_x, sim_z=opt_sim_z,
        color_x=OPT_COLOR, color_z="darkorange", linestyle_z="--",
        title=(f"Optimized: $T_X$ and $T_Z$ together\n"
               fr"($\epsilon_d$={opt_eps_d:.2f}, $g_2$={opt_g2:.2f})"),
        Tx_label=f"$\\langle X\\rangle$  $T_X$ = {opt_sim_x['T_X']:.3f} us",
        Tz_label=f"$\\langle Z\\rangle$  $T_Z$ = {opt_sim_z['T_Z']:.1f} us",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved to {save_path}")
    plt.show()


def _plot_combined_panel(ax, sim_x, sim_z,
                         color_x, color_z, linestyle_z,
                         title, Tx_label, Tz_label):
    ts_x = sim_x["ts"]; sxt = sim_x["sxt"]; Tx = sim_x["T_X"]
    ts_z = sim_z["ts"]; szt = sim_z["szt"]; Tz = sim_z["T_Z"]

    sxt0 = sxt[0] if abs(sxt[0]) > 1e-9 else 1.0
    szt0 = szt[0] if abs(szt[0]) > 1e-9 else 1.0
    sxt_n = sxt / sxt0
    szt_n = szt / szt0

    t_min = max(min(ts_x[1] if len(ts_x) > 1 else 1e-4,
                    ts_z[1] if len(ts_z) > 1 else 1e-4), 1e-4)
    t_max = max(ts_x[-1], ts_z[-1])
    t_log = np.logspace(np.log10(t_min), np.log10(t_max), 400)

    mask_x = ts_x > 0
    mask_z = ts_z > 0

    ax.plot(ts_x[mask_x], sxt_n[mask_x], "o", ms=3,
            color=color_x, alpha=0.7, label=Tx_label)
    ax.plot(t_log, np.exp(-t_log / Tx), color=color_x, linewidth=1.5)

    ax.plot(ts_z[mask_z], szt_n[mask_z], "s", ms=3,
            color=color_z, alpha=0.7, linestyle="none", label=Tz_label)
    ax.plot(t_log, np.exp(-t_log / Tz),
            color=color_z, linewidth=1.5, linestyle=linestyle_z)

    ax.axvline(Tx, color=color_x, linestyle=":", alpha=0.5, label="$t = T_X$")
    ax.axvline(Tz, color=color_z, linestyle=":", alpha=0.5, label="$t = T_Z$")
    ax.axhline(np.exp(-1), color="gray", linestyle="--", alpha=0.4, label=r"$e^{-1}$")

    ax.set_xscale("log")
    ax.set(title=title,
           xlabel="Time (us)  [log scale]",
           ylabel="Normalised expectation value",
           ylim=(-0.05, 1.15))
    ax.legend(fontsize=7, loc="upper right")

    ratio = Tz / max(Tx, 1e-9)
    ax.text(0.03, 0.12, f"$T_Z / T_X$ = {ratio:.1f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# =============================================================================
# 7.  MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Joint T_X / T_Z Optimizer -- Pure Hamiltonian Evolution")
    print(f"  na={na}, nb={nb}, kappa_b={kappa_b} MHz, kappa_a={kappa_a} MHz")
    print(f"  Target T_Z/T_X ratio = {TARGET_RATIO:.0f}")
    print("  NOTE: sesolve (no dissipation) -- expect oscillations, not decay")
    print("=" * 65)

    print(f"\nBaseline: eps_d={NOTEBOOK_EPS_D}, g_2={NOTEBOOK_G2}", flush=True)
    print(f"  Analytic alpha = {analytic_alpha(NOTEBOOK_EPS_D, NOTEBOOK_G2):.3f}", flush=True)
    print(f"  Analytic T_X   = {analytic_Tx(NOTEBOOK_EPS_D, NOTEBOOK_G2):.3f} us", flush=True)
    print(f"  Analytic T_Z   = {analytic_Tz(NOTEBOOK_EPS_D, NOTEBOOK_G2):.1f} us", flush=True)

    print(f"\nWarming up JAX ...", flush=True)
    print(f"Simulating notebook baseline T_X (window={NOTEBOOK_TX_FINAL} us) ...", flush=True)
    nb_sim_x = measure_Tx(NOTEBOOK_EPS_D, NOTEBOOK_G2,
                          tfinal=NOTEBOOK_TX_FINAL, n_pts=30)
    print(f"  Notebook T_X = {nb_sim_x['T_X']:.4f} us", flush=True)

    print(f"Simulating notebook baseline T_Z (window={NOTEBOOK_TZ_FINAL} us) ...", flush=True)
    nb_sim_z = measure_Tz(NOTEBOOK_EPS_D, NOTEBOOK_G2,
                          tfinal=NOTEBOOK_TZ_FINAL, n_pts=50)
    print(f"  Notebook T_Z = {nb_sim_z['T_Z']:.2f} us", flush=True)
    print(f"  Notebook T_Z/T_X = {nb_sim_z['T_Z'] / max(nb_sim_x['T_X'], 1e-9):.1f}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("Starting joint CMA-ES optimization (H evolution) ...", flush=True)
    print("=" * 65 + "\n", flush=True)

    best_theta, best_Tx, best_Tz, history = optimize_joint(
        eps_d_init=NOTEBOOK_EPS_D, g2_init=NOTEBOOK_G2,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, sigma0=SIGMA0,
    )

    eps_d_opt, g2_opt = best_theta
    alpha_opt = analytic_alpha(eps_d_opt, g2_opt)

    print("\n" + "=" * 65)
    print("Optimal parameters (H evolution):")
    print(f"  eps_d  = {eps_d_opt:+.4f} MHz")
    print(f"  g_2    = {g2_opt:+.4f} MHz")
    print(f"  Analytic alpha = {alpha_opt:.4f}")
    print(f"  Best T_X       = {best_Tx:.4f} us")
    print(f"  Best T_Z       = {best_Tz:.2f} us")
    print(f"  Best T_Z/T_X   = {best_Tz / max(best_Tx, 1e-9):.1f}  (target: {TARGET_RATIO:.0f})")
    print(f"  T_X improvement: {best_Tx / max(nb_sim_x['T_X'], 1e-9):.2f}x")
    print(f"  T_Z improvement: {best_Tz / max(nb_sim_z['T_Z'], 1e-9):.2f}x")

    print(f"\nFinal simulations at optimal params ...")

    print(f"  T_X on notebook window ({NOTEBOOK_TX_FINAL} us) ...")
    opt_sim_x = measure_Tx(eps_d_opt, g2_opt, tfinal=NOTEBOOK_TX_FINAL, n_pts=100)

    print(f"  T_Z on notebook window ({NOTEBOOK_TZ_FINAL} us) ...")
    opt_sim_z = measure_Tz(eps_d_opt, g2_opt, tfinal=NOTEBOOK_TZ_FINAL, n_pts=100)

    tx_long_window = float(np.clip(5.0 * analytic_Tx(eps_d_opt, g2_opt), 0.5, 20.0))
    tz_long_window = float(np.clip(4.0 * analytic_Tz(eps_d_opt, g2_opt), 100.0, 2000.0))

    print(f"  T_X long-window ({tx_long_window:.1f} us) for combined plot ...")
    opt_sim_x_long = measure_Tx(eps_d_opt, g2_opt, tfinal=tx_long_window, n_pts=100)
    nb_sim_x_long  = measure_Tx(NOTEBOOK_EPS_D, NOTEBOOK_G2, tfinal=tx_long_window, n_pts=100)

    print(f"  T_Z long-window ({tz_long_window:.0f} us) for combined plot ...")
    opt_sim_z_long = measure_Tz(eps_d_opt, g2_opt, tfinal=tz_long_window, n_pts=100)
    nb_sim_z_long  = measure_Tz(NOTEBOOK_EPS_D, NOTEBOOK_G2, tfinal=tz_long_window, n_pts=100)

    print(f"\n  Optimized T_X (long window) = {opt_sim_x_long['T_X']:.4f} us")
    print(f"  Optimized T_Z (long window) = {opt_sim_z_long['T_Z']:.2f} us")
    print(f"  Optimized T_Z/T_X           = "
          f"{opt_sim_z_long['T_Z'] / max(opt_sim_x_long['T_X'], 1e-9):.1f}")

    print("\nGenerating plots ...")

    plot_optimizer_progress(history, save_path="HEvo_optimizer_progress.png")

    plot_decay_comparison(
        nb_sim_x=nb_sim_x, nb_sim_z=nb_sim_z,
        opt_sim_x=opt_sim_x, opt_sim_z=opt_sim_z,
        nb_eps_d=NOTEBOOK_EPS_D, nb_g2=NOTEBOOK_G2,
        opt_eps_d=eps_d_opt, opt_g2=g2_opt,
        save_path="HEvo_Tx_decay_comparison.png",
    )

    # Separate figure: initial T_X and T_Z
    fig_init, ax_init = plt.subplots(figsize=(7, 5))
    fig_init.suptitle(
        fr"Initial parameters (H evol.): $\epsilon_d$={NOTEBOOK_EPS_D}, $g_2$={NOTEBOOK_G2}",
        fontsize=12,
    )
    _plot_combined_panel(
        ax_init, sim_x=nb_sim_x_long, sim_z=nb_sim_z_long,
        color_x="#e05c2a", color_z="#e05c2a", linestyle_z="--",
        title=(f"Initial $T_X$ and $T_Z$ -- log time axis\n"
               f"$T_X$ = {nb_sim_x_long['T_X']:.3f} us,  "
               f"$T_Z$ = {nb_sim_z_long['T_Z']:.1f} us"),
        Tx_label=f"$\\langle X\\rangle$  (phase-flip)  $T_X$ = {nb_sim_x_long['T_X']:.3f} us",
        Tz_label=f"$\\langle Z\\rangle$  (bit-flip)    $T_Z$ = {nb_sim_z_long['T_Z']:.1f} us",
    )
    plt.tight_layout()
    plt.savefig("HEvo_initial_Tx_Tz_combined.png", dpi=150, bbox_inches="tight")
    print("Figure saved to HEvo_initial_Tx_Tz_combined.png")
    plt.show()

    # Separate figure: optimized T_X and T_Z
    fig_opt, ax_opt = plt.subplots(figsize=(7, 5))
    fig_opt.suptitle(
        fr"Optimized parameters (H evol.): $\epsilon_d$={eps_d_opt:.2f}, $g_2$={g2_opt:.2f}",
        fontsize=12,
    )
    _plot_combined_panel(
        ax_opt, sim_x=opt_sim_x_long, sim_z=opt_sim_z_long,
        color_x="steelblue", color_z="darkorange", linestyle_z="--",
        title=(f"Optimized $T_X$ and $T_Z$ -- log time axis\n"
               f"$T_X$ = {opt_sim_x_long['T_X']:.3f} us,  "
               f"$T_Z$ = {opt_sim_z_long['T_Z']:.1f} us"),
        Tx_label=f"$\\langle X\\rangle$  (phase-flip)  $T_X$ = {opt_sim_x_long['T_X']:.3f} us",
        Tz_label=f"$\\langle Z\\rangle$  (bit-flip)    $T_Z$ = {opt_sim_z_long['T_Z']:.1f} us",
    )
    plt.tight_layout()
    plt.savefig("HEvo_optimized_Tx_Tz_combined.png", dpi=150, bbox_inches="tight")
    print("Figure saved to HEvo_optimized_Tx_Tz_combined.png")
    plt.show()

    print("\nDone.")
    print("=" * 65)
