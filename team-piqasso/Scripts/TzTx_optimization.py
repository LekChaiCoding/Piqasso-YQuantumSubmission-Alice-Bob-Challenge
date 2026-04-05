"""
Joint T_X / T_Z Optimizer for Dissipative Cat Qubits
=====================================================
Simultaneously maximizes both the phase-flip lifetime T_X and the bit-flip
lifetime T_Z while keeping their ratio T_Z / T_X = TARGET_RATIO = 320.

Physical context
----------------
In the cat-qubit encoding  |0>_L = |+alpha>,  |1>_L = |-alpha>:

  T_X  — lifetime of <X> = parity  (phase-flip, initialized in even-cat |+x>)
         Decays as single-photon loss jumps flip even <-> odd cat parity.
         Analytic estimate:  T_X ~ 1 / (kappa_a * alpha^2)   [polynomially short]

  T_Z  — lifetime of <Z> = coherent-state projection
         (bit-flip, initialized in |+z> = |+alpha>)
         Tunneling between |+alpha> and |-alpha> is exponentially suppressed.
         Analytic estimate:  T_Z ~ exp(2*alpha^2) / kappa_a   [exponentially long]

  Noise bias  =  T_Z / T_X  ~  alpha^2 * exp(2*alpha^2)  (grows with cat size)

The challenge-notebook baseline (eps_d=4, g_2=1, kappa_b=10, kappa_a=1)
gives a bias of approximately 320 (Cell 42 of 1-challenge.ipynb).

Optimization goal
-----------------
  Maximize  (T_X + T_Z / 320) / 2   subject to   T_Z / T_X = 320.
  Uses CMA-ES with an additive ratio-constraint penalty.

Control parameters
------------------
  theta = (eps_d, g_2)   [MHz, real]

From adiabatic elimination (notebook formula):
  eps_2   = 2 * g_2 * eps_d / kappa_b
  kappa_2 = 4 * g_2^2 / kappa_b
  alpha   = sqrt(2 * (eps_2 - kappa_a/4) / kappa_2)
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

# ── Fixed system parameters ────────────────────────────────────────────────────
na      = 15    # storage Fock-space truncation (reduced from 15 for speed)
nb      = 5    # buffer Fock-space truncation  (reduced from 5 for speed)
kappa_b = 10.0  # MHz, buffer decay rate (fast, fixed)
kappa_a = 1.0   # MHz, storage single-photon loss (fixed hardware noise)

# ── Notebook baseline (1-challenge.ipynb, Cells 38–42) ────────────────────────
NOTEBOOK_EPS_D    = 4.0    # MHz
NOTEBOOK_G2       = 1.0    # MHz
NOTEBOOK_TX_FINAL = 1.0    # µs  (Cell 41: measure_lifetime("+x", 1.0))
NOTEBOOK_TZ_FINAL = 200.0  # µs  (Cell 40: measure_lifetime("+z", 200))
TARGET_RATIO      = 320.0  # T_Z / T_X noise bias (Cell 42)

# ── CMA-ES hyperparameters ─────────────────────────────────────────────────────
BATCH_SIZE    = 10
N_EPOCHS      = 50
SIGMA0        = 0.5
RATIO_PENALTY = 8.0   # relative penalty weight for T_Z/T_X ratio constraint

# Control bounds: [eps_d, g_2]  (MHz)
BOUNDS = np.array([
    [0.5, 8.0],   # eps_d — pump drive amplitude
    [0.2, 4.0],   # g_2   — two-photon coupling
])

# ── Global operators ───────────────────────────────────────────────────────────
a_s  = dq.destroy(na)
a_op = dq.tensor(a_s, dq.eye(nb))
b_op = dq.tensor(dq.eye(na), dq.destroy(nb))

# Parity (logical-X) operator:  (-1)^n  in storage space
parity_s  = (1j * jnp.pi * a_s.dag() @ a_s).expm()
parity_op = dq.tensor(parity_s, dq.eye(nb))

# Pre-computed a^2 operator (reused in every T_Z simulation)
a2_op = dq.powm(a_op, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ANALYTIC ESTIMATES
# ══════════════════════════════════════════════════════════════════════════════

def analytic_alpha(eps_d: float, g2: float) -> float:
    """
    Cat size from adiabatic elimination (notebook Cell 38 formula):
      alpha^2 = 2 * (eps_2 - kappa_a/4) / kappa_2
    """
    kappa_2 = 4.0 * g2**2 / kappa_b
    if kappa_2 < 1e-12:
        return 0.0
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    inner = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
    return float(np.sqrt(max(inner, 0.0)))


def analytic_Tx(eps_d: float, g2: float) -> float:
    """Phase-flip lifetime estimate: T_X ~ 1 / (kappa_a * alpha^2)."""
    alpha = analytic_alpha(eps_d, g2)
    if alpha < 1e-6:
        return 1e-3
    return 1.0 / (kappa_a * alpha**2)


def analytic_Tz(eps_d: float, g2: float) -> float:
    """Bit-flip lifetime estimate: T_Z ~ exp(2*alpha^2) / kappa_a."""
    alpha = analytic_alpha(eps_d, g2)
    return float(np.exp(2.0 * alpha**2)) / kappa_a


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYSTEM BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_system(eps_d: float, g2: float):
    """
    Build H, jump operators, cat-size estimate, and basis kets.

    Returns
    -------
    H       : Hamiltonian (full na*nb space)
    jumps   : [L_b, L_a]
    alpha   : analytic cat-size estimate
    psi0_x  : |+x> ⊗ |0>_b  (even cat — for T_X measurement)
    ket_p   : |+alpha>  (coherent ket — for T_Z measurement & Z operator)
    ket_m   : |-alpha>
    """
    alpha = analytic_alpha(eps_d, g2)

    # Two-photon exchange Hamiltonian (matches notebook Cell 38)
    H = (jnp.conj(g2) * a_op.dag() @ a_op.dag() @ b_op
         + g2 * a_op @ a_op @ b_op.dag()
         - eps_d * b_op.dag()
         - jnp.conj(eps_d) * b_op) + 0.5 * a_op.dag() @ a_op

    L_b = jnp.sqrt(kappa_b) * b_op
    L_a = jnp.sqrt(kappa_a) * a_op

    ket_p = dq.coherent(na, alpha)
    ket_m = dq.coherent(na, -alpha)

    cat_x  = (ket_p + ket_m).unit()          # |+x> = even cat
    psi0_x = dq.tensor(cat_x, dq.fock(nb, 0))

    return H, [L_b, L_a], alpha, psi0_x, ket_p, ket_m


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EXPONENTIAL FIT
# ══════════════════════════════════════════════════════════════════════════════

def _exp_model(p, t):
    A, tau, C = p
    return A * jnp.exp(-t / tau) + C


def _fit_lifetime(ts: np.ndarray, signal: np.ndarray) -> float:
    """Least-squares fit of A*exp(-t/tau)+C; return tau."""
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


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LIFETIME MEASUREMENTS
# ══════════════════════════════════════════════════════════════════════════════

def _measure_Tx_inner(H, jumps, psi0_x, alpha: float,
                      tfinal: float, n_pts: int) -> dict:
    """Run the T_X simulation given a pre-built system."""
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    result = dq.mesolve(
        H, jumps, psi0_x, tsave,
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


def _measure_Tz_inner(H, jumps, ket_p, ket_m, alpha: float,
                      tfinal: float, n_pts: int) -> dict:
    """Run the T_Z simulation given a pre-built system."""
    psi0_z = dq.tensor(ket_p, dq.fock(nb, 0))
    tsave  = jnp.linspace(0.0, tfinal, n_pts)

    res = dq.mesolve(
        H,
        jumps,
        psi0_z,
        tsave,
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
    ts  = np.array(res.tsave)
    if szt[-1] > 0.9 * szt[0]:
        Tz = 5.0 * tfinal
    else:
        try:
            Tz = _fit_lifetime(ts, szt)
        except Exception:
            Tz = 0.0
    return {"T_Z": Tz, "szt": szt, "ts": ts, "alpha": alpha, "tfinal": tfinal}


def measure_Tx(eps_d: float, g2: float,
               tfinal: float | None = None,
               n_pts: int = 30) -> dict:
    """
    Simulate <X> = parity decay starting from |+x> and extract T_X.

    Default window: clip(3 * analytic_Tx, 0.3, 15) µs
    """
    H, jumps, alpha, psi0_x, _, _ = build_system(eps_d, g2)
    if tfinal is None:
        tfinal = float(np.clip(3.0 / max(kappa_a * alpha**2, 1e-6), 0.3, 15.0))
    return _measure_Tx_inner(H, jumps, psi0_x, alpha, tfinal, n_pts)


def measure_Tz(eps_d: float, g2: float,
               tfinal: float | None = None,
               n_pts: int = 25) -> dict:
    """
    Simulate <Z> decay starting from |+z> = |+alpha> and extract T_Z.

    The logical-Z operator is:  sz = |+alpha><+alpha| - |-alpha><-alpha|
    (constructed from alpha at the given parameters, matching notebook Cell 38).

    Default window: clip(3 * analytic_Tz, 50, 1000) µs
    """
    H, jumps, alpha, _, ket_p, ket_m = build_system(eps_d, g2)
    if tfinal is None:
        tz_est = float(np.exp(2.0 * alpha**2)) / kappa_a
        tfinal = float(np.clip(3.0 * tz_est, 50.0, 1000.0))
    return _measure_Tz_inner(H, jumps, ket_p, ket_m, alpha, tfinal, n_pts)


def measure_joint(eps_d: float, g2: float) -> dict:
    """
    Measure T_X and T_Z for given parameters, building the system only once.

    T_Z window is set from the analytic estimate to keep it stable across
    the CMA-ES search (capped at 500 µs for optimizer speed).

    Returns dict with T_X, T_Z, ratio, and raw time-traces.
    """
    # Single build_system call — operators, kets, and alpha shared by both sims
    H, jumps, alpha, psi0_x, ket_p, ket_m = build_system(eps_d, g2)

    # Derive time windows directly from alpha (avoids two extra analytic_alpha calls)
    tx_est = 1.0 / max(kappa_a * alpha**2, 1e-6)
    tz_est = float(np.exp(2.0 * alpha**2)) / kappa_a
    tx_window = float(np.clip(3.0 * tx_est, 0.3, 15.0))
    tz_window = float(np.clip(3.0 * tz_est, 50.0, 500.0))

    # Run T_X and T_Z simulations concurrently — they are fully independent
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_x = ex.submit(_measure_Tx_inner, H, jumps, psi0_x, alpha, tx_window, 30)
        fut_z = ex.submit(_measure_Tz_inner, H, jumps, ket_p, ket_m, alpha, tz_window, 50)
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


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CMA-ES OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def clip_bounds(x: np.ndarray) -> np.ndarray:
    return np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])


def _eval_candidate(x: np.ndarray) -> dict:
    """Evaluate a single CMA-ES candidate; returns metrics dict."""
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
    eps_d_init: float = NOTEBOOK_EPS_D,
    g2_init:    float = NOTEBOOK_G2,
    n_epochs:   int   = N_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    sigma0:     float = SIGMA0,
    verbose:    bool  = True,
) -> tuple:
    """
    Maximize (T_X + T_Z/TARGET_RATIO)/2 subject to T_Z/T_X = TARGET_RATIO.

    Loss = -(T_X + T_Z/TARGET_RATIO)/2
           + RATIO_PENALTY * ((T_Z/T_X - TARGET_RATIO)/TARGET_RATIO)^2

    Candidates within each CMA-ES batch are evaluated in parallel via
    ThreadPoolExecutor (JAX releases the GIL during XLA dispatch).

    Returns (best_theta, best_Tx, best_Tz, history)
    """
    theta0 = np.array([eps_d_init, g2_init])
    alpha0 = analytic_alpha(*theta0)

    if verbose:
        print(f"Warm-start θ₀ = (eps_d={eps_d_init}, g_2={g2_init})")
        print(f"  Analytic α   = {alpha0:.3f}")
        print(f"  Analytic T_X = {analytic_Tx(*theta0):.3f} µs")
        print(f"  Analytic T_Z = {analytic_Tz(*theta0):.1f} µs")
        print(f"  Target T_Z/T_X ratio = {TARGET_RATIO:.0f}")
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

        # Evaluate all candidates in parallel — each call is independent
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            batch_metrics = list(pool.map(_eval_candidate, candidates))

        batch_solutions = [(bm["x"], bm["loss"]) for bm in batch_metrics]
        optimizer.tell(batch_solutions)

        # Best candidate this epoch by combined objective
        best_idx = min(range(batch_size), key=lambda i: batch_metrics[i]["loss"])
        bm = batch_metrics[best_idx]
        bx = bm["x"]

        obj_ep = bm["T_X"] + bm["T_Z"] / TARGET_RATIO
        if obj_ep > best_obj:
            best_obj   = obj_ep
            best_theta = bx.copy()
            best_Tx    = bm["T_X"]
            best_Tz    = bm["T_Z"]

        # alpha already computed inside measure_joint; retrieve from metrics
        history["T_X"].append(bm["T_X"])
        history["T_Z"].append(bm["T_Z"])
        history["ratio"].append(bm["ratio"])
        history["alpha"].append(bm["alpha"])
        history["eps_d"].append(float(bx[0]))
        history["g2"].append(float(bx[1]))

        if verbose:
            print(f"Epoch {epoch:3d}/{n_epochs}:  T_X={bm['T_X']:7.3f} µs  "
                  f"T_Z={bm['T_Z']:8.1f} µs  "
                  f"ratio={bm['ratio']:6.1f}  "
                  f"eps_d={float(bx[0]):.3f}  g_2={float(bx[1]):.3f}",
                  flush=True)

    return best_theta, best_Tx, best_Tz, history


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimizer_progress(history: dict,
                             save_path: str = "optimizer_progress.png"):
    """
    Six-panel figure showing CMA-ES optimization history:
      row 0: T_X, T_Z, ratio vs epoch
      row 1: control params, alpha, combined objective
    """
    epochs = np.arange(len(history["T_X"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Joint T_X / T_Z Optimizer — target ratio = {TARGET_RATIO:.0f}",
        fontsize=13,
    )

    # (0,0) T_X vs epoch
    ax = axes[0, 0]
    ax.plot(epochs, history["T_X"], color="steelblue", linewidth=1.2)
    ax.axhline(max(history["T_X"]), color="steelblue", linestyle="--", alpha=0.6,
               label=f"max = {max(history['T_X']):.3f} µs")
    ax.set(title="Phase-flip lifetime $T_X$", xlabel="Epoch", ylabel="$T_X$ (µs)")
    ax.legend(fontsize=8)

    # (0,1) T_Z vs epoch
    ax = axes[0, 1]
    ax.plot(epochs, history["T_Z"], color="darkorange", linewidth=1.2)
    ax.axhline(max(history["T_Z"]), color="darkorange", linestyle="--", alpha=0.6,
               label=f"max = {max(history['T_Z']):.1f} µs")
    ax.set(title="Bit-flip lifetime $T_Z$", xlabel="Epoch", ylabel="$T_Z$ (µs)")
    ax.legend(fontsize=8)

    # (0,2) T_Z/T_X ratio vs epoch
    ax = axes[0, 2]
    ax.plot(epochs, history["ratio"], color="purple", linewidth=1.2)
    ax.axhline(TARGET_RATIO, color="red", linestyle="--", alpha=0.7,
               label=f"target = {TARGET_RATIO:.0f}")
    ax.set(title="Noise bias $T_Z / T_X$", xlabel="Epoch", ylabel="$T_Z / T_X$")
    ax.legend(fontsize=8)

    # (1,0) Control parameters
    ax = axes[1, 0]
    ax.plot(epochs, history["eps_d"], label=r"$\epsilon_d$ (MHz)", color="steelblue")
    ax.plot(epochs, history["g2"],    label=r"$g_2$ (MHz)",        color="crimson")
    ax.set(title="Control parameters", xlabel="Epoch", ylabel="Value (MHz)")
    ax.legend(fontsize=8)

    # (1,1) Analytic cat size alpha
    ax = axes[1, 1]
    ax.plot(epochs, history["alpha"], color="seagreen", linewidth=1.2)
    ax.set(title=r"Cat size $\alpha$ (analytic)", xlabel="Epoch", ylabel=r"$\alpha$")

    # (1,2) Combined objective  T_X + T_Z/TARGET_RATIO
    objective = [tx + tz / TARGET_RATIO
                 for tx, tz in zip(history["T_X"], history["T_Z"])]
    ax = axes[1, 2]
    ax.plot(epochs, objective, color="indigo", linewidth=1.2)
    ax.axhline(max(objective), color="indigo", linestyle="--", alpha=0.6,
               label=f"max = {max(objective):.3f} µs")
    ax.set(title=r"Objective $T_X + T_Z/320$", xlabel="Epoch", ylabel="(µs)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Optimizer progress figure saved to {save_path}")
    plt.show()


def plot_decay_comparison(
    nb_sim_x:  dict,   # notebook T_X sim (NOTEBOOK_TX_FINAL window)
    nb_sim_z:  dict,   # notebook T_Z sim (NOTEBOOK_TZ_FINAL window)
    opt_sim_x: dict,   # optimized T_X sim (NOTEBOOK_TX_FINAL window)
    opt_sim_z: dict,   # optimized T_Z sim (NOTEBOOK_TZ_FINAL window)
    nb_eps_d: float, nb_g2: float,
    opt_eps_d: float, opt_g2: float,
    save_path: str = "Tx_decay_comparison.png",
):
    """
    2×2 comparison figure:
      (0,0) T_X decays — notebook vs optimized (short window)
      (0,1) T_Z decays — notebook vs optimized (long window)
      (1,0) Initial T_X and T_Z on the same graph (log time axis)
      (1,1) Optimized T_X and T_Z on the same graph (log time axis)
    """
    NB_COLOR  = "#e05c2a"   # orange-red: notebook / initial
    OPT_COLOR = "steelblue" # blue: optimized

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Cat-Qubit Lifetimes: Notebook Baseline vs Optimized",
        fontsize=13,
    )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _fit(signal, lifetime, t_grid):
        A = float(signal[0] - signal[-1])
        C = float(signal[-1])
        return A * np.exp(-t_grid / lifetime) + C

    # ─────────────────────────────────────────────────────────────────────────
    # (0,0): T_X  comparison
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]

    ts_nb  = nb_sim_x["ts"]
    sxt_nb = nb_sim_x["sxt"]
    Tx_nb  = nb_sim_x["T_X"]
    t_fit  = np.linspace(ts_nb[0], ts_nb[-1], 400)
    ax.plot(ts_nb, sxt_nb, "o", ms=3, color=NB_COLOR, alpha=0.8,
            label=fr"Notebook ($\epsilon_d$={nb_eps_d}, $g_2$={nb_g2})")
    ax.plot(t_fit, _fit(sxt_nb, Tx_nb, t_fit),
            color=NB_COLOR, linestyle="--",
            label=f"Fit  $T_X$ = {Tx_nb:.3f} µs")

    ts_op  = opt_sim_x["ts"]
    sxt_op = opt_sim_x["sxt"]
    Tx_op  = opt_sim_x["T_X"]
    t_fit  = np.linspace(ts_op[0], ts_op[-1], 400)
    ax.plot(ts_op, sxt_op, "o", ms=3, color=OPT_COLOR, alpha=0.8,
            label=fr"Optimized ($\epsilon_d$={opt_eps_d:.2f}, $g_2$={opt_g2:.2f})")
    ax.plot(t_fit, _fit(sxt_op, Tx_op, t_fit),
            color=OPT_COLOR, linestyle="--",
            label=f"Fit  $T_X$ = {Tx_op:.3f} µs")

    ax.set(title=r"$\langle X \rangle$ decay — notebook window",
           xlabel="Time (µs)", ylabel=r"$\langle X \rangle$", ylim=(-0.05, 1.15))
    ax.legend(fontsize=8)

    improvement_x = Tx_op / max(Tx_nb, 1e-9)
    ax.text(0.03, 0.08,
            f"Improvement: {improvement_x:.1f}×",
            transform=ax.transAxes, fontsize=9, color=OPT_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ─────────────────────────────────────────────────────────────────────────
    # (0,1): T_Z comparison
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]

    ts_nb  = nb_sim_z["ts"]
    szt_nb = nb_sim_z["szt"]
    Tz_nb  = nb_sim_z["T_Z"]
    t_fit  = np.linspace(ts_nb[0], ts_nb[-1], 400)
    ax.plot(ts_nb, szt_nb, "o", ms=3, color=NB_COLOR, alpha=0.8,
            label=fr"Notebook ($\epsilon_d$={nb_eps_d}, $g_2$={nb_g2})")
    ax.plot(t_fit, _fit(szt_nb, Tz_nb, t_fit),
            color=NB_COLOR, linestyle="--",
            label=f"Fit  $T_Z$ = {Tz_nb:.1f} µs")

    ts_op  = opt_sim_z["ts"]
    szt_op = opt_sim_z["szt"]
    Tz_op  = opt_sim_z["T_Z"]
    t_fit  = np.linspace(ts_op[0], ts_op[-1], 400)
    ax.plot(ts_op, szt_op, "o", ms=3, color=OPT_COLOR, alpha=0.8,
            label=fr"Optimized ($\epsilon_d$={opt_eps_d:.2f}, $g_2$={opt_g2:.2f})")
    ax.plot(t_fit, _fit(szt_op, Tz_op, t_fit),
            color=OPT_COLOR, linestyle="--",
            label=f"Fit  $T_Z$ = {Tz_op:.1f} µs")

    ax.set(title=r"$\langle Z \rangle$ decay — notebook window",
           xlabel="Time (µs)", ylabel=r"$\langle Z \rangle$", ylim=(-0.05, 1.15))
    ax.legend(fontsize=8)

    improvement_z = Tz_op / max(Tz_nb, 1e-9)
    ax.text(0.03, 0.08,
            f"Improvement: {improvement_z:.1f}×",
            transform=ax.transAxes, fontsize=9, color=OPT_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ─────────────────────────────────────────────────────────────────────────
    # (1,0): INITIAL  T_X and T_Z on the same graph (log time axis)
    # ─────────────────────────────────────────────────────────────────────────
    _plot_combined_panel(
        axes[1, 0],
        sim_x=nb_sim_x, sim_z=nb_sim_z,
        color_x=NB_COLOR, color_z=NB_COLOR,
        linestyle_z="--",
        title=(f"Initial: $T_X$ and $T_Z$ together\n"
               fr"($\epsilon_d$={nb_eps_d}, $g_2$={nb_g2})"),
        Tx_label=f"$\\langle X\\rangle$  $T_X$ = {nb_sim_x['T_X']:.3f} µs",
        Tz_label=f"$\\langle Z\\rangle$  $T_Z$ = {nb_sim_z['T_Z']:.1f} µs",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # (1,1): OPTIMIZED T_X and T_Z on the same graph (log time axis)
    # ─────────────────────────────────────────────────────────────────────────
    _plot_combined_panel(
        axes[1, 1],
        sim_x=opt_sim_x, sim_z=opt_sim_z,
        color_x=OPT_COLOR, color_z="darkorange",
        linestyle_z="--",
        title=(f"Optimized: $T_X$ and $T_Z$ together\n"
               fr"($\epsilon_d$={opt_eps_d:.2f}, $g_2$={opt_g2:.2f})"),
        Tx_label=f"$\\langle X\\rangle$  $T_X$ = {opt_sim_x['T_X']:.3f} µs",
        Tz_label=f"$\\langle Z\\rangle$  $T_Z$ = {opt_sim_z['T_Z']:.1f} µs",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved to {save_path}")
    plt.show()


def _plot_combined_panel(ax, sim_x: dict, sim_z: dict,
                         color_x: str, color_z: str, linestyle_z: str,
                         title: str, Tx_label: str, Tz_label: str):
    """
    Plot <X> and <Z> decay curves on the same axes with a log time axis.
    Both are normalized to their t=0 value so they share the y-axis [0, 1].
    The log x-axis reveals both the fast T_X decay and the slow T_Z decay.
    """
    ts_x  = sim_x["ts"]
    sxt   = sim_x["sxt"]
    Tx    = sim_x["T_X"]
    ts_z  = sim_z["ts"]
    szt   = sim_z["szt"]
    Tz    = sim_z["T_Z"]

    # Normalize to initial value
    sxt0 = sxt[0] if abs(sxt[0]) > 1e-9 else 1.0
    szt0 = szt[0] if abs(szt[0]) > 1e-9 else 1.0
    sxt_n = sxt / sxt0
    szt_n = szt / szt0

    # Dense grid for fitted exponentials
    t_min = max(min(ts_x[1], ts_z[1]), 1e-4)
    t_max = max(ts_x[-1], ts_z[-1])
    t_log = np.logspace(np.log10(t_min), np.log10(t_max), 400)

    # Skip t=0 for log scale
    mask_x = ts_x > 0
    mask_z = ts_z > 0

    # <X> parity
    ax.plot(ts_x[mask_x], sxt_n[mask_x], "o", ms=3,
            color=color_x, alpha=0.7, label=Tx_label)
    ax.plot(t_log, np.exp(-t_log / Tx),
            color=color_x, linewidth=1.5)

    # <Z> coherent-state projection
    ax.plot(ts_z[mask_z], szt_n[mask_z], "s", ms=3,
            color=color_z, alpha=0.7, linestyle="none", label=Tz_label)
    ax.plot(t_log, np.exp(-t_log / Tz),
            color=color_z, linewidth=1.5, linestyle=linestyle_z)

    # Mark T_X and T_Z with vertical dashed lines
    ax.axvline(Tx, color=color_x, linestyle=":", alpha=0.5,
               label=f"$t = T_X$")
    ax.axvline(Tz, color=color_z, linestyle=":", alpha=0.5,
               label=f"$t = T_Z$")
    ax.axhline(np.exp(-1), color="gray", linestyle="--", alpha=0.4,
               label=r"$e^{-1}$")

    ax.set_xscale("log")
    ax.set(title=title,
           xlabel="Time (µs)  [log scale]",
           ylabel="Normalised expectation value",
           ylim=(-0.05, 1.15))
    ax.legend(fontsize=7, loc="upper right")

    # Annotate ratio
    ratio = Tz / max(Tx, 1e-9)
    ax.text(0.03, 0.12,
            f"$T_Z / T_X$ = {ratio:.1f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Joint T_X / T_Z Optimizer for Dissipative Cat Qubits")
    print(f"  na={na}, nb={nb}, kappa_b={kappa_b} MHz, kappa_a={kappa_a} MHz")
    print(f"  Target T_Z/T_X ratio = {TARGET_RATIO:.0f}")
    print("=" * 65)

    # ── Notebook baseline simulations ─────────────────────────────────────────
    print(f"\nBaseline: eps_d={NOTEBOOK_EPS_D}, g_2={NOTEBOOK_G2}", flush=True)
    print(f"  Analytic α   = {analytic_alpha(NOTEBOOK_EPS_D, NOTEBOOK_G2):.3f}", flush=True)
    print(f"  Analytic T_X = {analytic_Tx(NOTEBOOK_EPS_D, NOTEBOOK_G2):.3f} µs", flush=True)
    print(f"  Analytic T_Z = {analytic_Tz(NOTEBOOK_EPS_D, NOTEBOOK_G2):.1f} µs", flush=True)

    print(f"\nWarming up JAX (first compile may take several minutes) ...", flush=True)
    print(f"Simulating notebook baseline T_X (window={NOTEBOOK_TX_FINAL} µs) ...", flush=True)
    # n_pts=30 matches the optimizer's inner loop — warms up JAX JIT for epoch 0
    nb_sim_x = measure_Tx(NOTEBOOK_EPS_D, NOTEBOOK_G2,
                          tfinal=NOTEBOOK_TX_FINAL, n_pts=30)
    print(f"  Notebook T_X = {nb_sim_x['T_X']:.4f} µs", flush=True)

    print(f"Simulating notebook baseline T_Z (window={NOTEBOOK_TZ_FINAL} µs) ...", flush=True)
    # n_pts=50 matches measure_joint's T_Z call — warms up JAX JIT for epoch 0
    nb_sim_z = measure_Tz(NOTEBOOK_EPS_D, NOTEBOOK_G2,
                          tfinal=NOTEBOOK_TZ_FINAL, n_pts=50)
    print(f"  Notebook T_Z = {nb_sim_z['T_Z']:.2f} µs", flush=True)
    print(f"  Notebook T_Z/T_X = {nb_sim_z['T_Z'] / max(nb_sim_x['T_X'], 1e-9):.1f}", flush=True)

    # ── CMA-ES joint optimization ──────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print("Starting joint CMA-ES optimization ...", flush=True)
    print("=" * 65 + "\n", flush=True)

    best_theta, best_Tx, best_Tz, history = optimize_joint(
        eps_d_init = NOTEBOOK_EPS_D,
        g2_init    = NOTEBOOK_G2,
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
        sigma0     = SIGMA0,
    )

    eps_d_opt, g2_opt = best_theta
    alpha_opt = analytic_alpha(eps_d_opt, g2_opt)

    print("\n" + "=" * 65)
    print("Optimal parameters θ*:")
    print(f"  eps_d  = {eps_d_opt:+.4f} MHz")
    print(f"  g_2    = {g2_opt:+.4f} MHz")
    print(f"  Analytic α   = {alpha_opt:.4f}")
    print(f"  Best T_X     = {best_Tx:.4f} µs")
    print(f"  Best T_Z     = {best_Tz:.2f} µs")
    print(f"  Best T_Z/T_X = {best_Tz / max(best_Tx, 1e-9):.1f}  (target: {TARGET_RATIO:.0f})")
    print(f"  T_X improvement: {best_Tx / max(nb_sim_x['T_X'], 1e-9):.2f}×")
    print(f"  T_Z improvement: {best_Tz / max(nb_sim_z['T_Z'], 1e-9):.2f}×")

    # ── Final simulations at optimal parameters ────────────────────────────────
    print(f"\nFinal simulations at optimal θ*...")

    # Use the notebook windows so comparisons are apple-to-apple
    print(f"  T_X on notebook window ({NOTEBOOK_TX_FINAL} µs) ...")
    opt_sim_x = measure_Tx(eps_d_opt, g2_opt,
                            tfinal=NOTEBOOK_TX_FINAL, n_pts=100)

    print(f"  T_Z on notebook window ({NOTEBOOK_TZ_FINAL} µs) ...")
    opt_sim_z = measure_Tz(eps_d_opt, g2_opt,
                            tfinal=NOTEBOOK_TZ_FINAL, n_pts=100)

    # Accurate long-window sims for the combined "same graph" panels
    tx_long_window = float(np.clip(5.0 * analytic_Tx(eps_d_opt, g2_opt), 0.5, 20.0))
    tz_long_window = float(np.clip(4.0 * analytic_Tz(eps_d_opt, g2_opt), 100.0, 2000.0))

    print(f"  T_X long-window ({tx_long_window:.1f} µs) for combined plot ...")
    opt_sim_x_long = measure_Tx(eps_d_opt, g2_opt,
                                 tfinal=tx_long_window, n_pts=100)
    nb_sim_x_long  = measure_Tx(NOTEBOOK_EPS_D, NOTEBOOK_G2,
                                 tfinal=tx_long_window, n_pts=100)

    print(f"  T_Z long-window ({tz_long_window:.0f} µs) for combined plot ...")
    opt_sim_z_long = measure_Tz(eps_d_opt, g2_opt,
                                 tfinal=tz_long_window, n_pts=100)
    nb_sim_z_long  = measure_Tz(NOTEBOOK_EPS_D, NOTEBOOK_G2,
                                 tfinal=tz_long_window, n_pts=100)

    print(f"\n  Optimized T_X (long window) = {opt_sim_x_long['T_X']:.4f} µs")
    print(f"  Optimized T_Z (long window) = {opt_sim_z_long['T_Z']:.2f} µs")
    print(f"  Optimized T_Z/T_X           = "
          f"{opt_sim_z_long['T_Z'] / max(opt_sim_x_long['T_X'], 1e-9):.1f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    plot_optimizer_progress(history, save_path="optimizer_progress.png")

    plot_decay_comparison(
        nb_sim_x  = nb_sim_x,
        nb_sim_z  = nb_sim_z,
        opt_sim_x = opt_sim_x,
        opt_sim_z = opt_sim_z,
        nb_eps_d  = NOTEBOOK_EPS_D,
        nb_g2     = NOTEBOOK_G2,
        opt_eps_d = eps_d_opt,
        opt_g2    = g2_opt,
        save_path = "Tx_decay_comparison.png",
    )

    # Separate figure: initial T_X and T_Z on the same graph
    fig_init, ax_init = plt.subplots(figsize=(7, 5))
    fig_init.suptitle(
        fr"Initial parameters: $\epsilon_d$={NOTEBOOK_EPS_D}, $g_2$={NOTEBOOK_G2}",
        fontsize=12,
    )
    _plot_combined_panel(
        ax_init,
        sim_x=nb_sim_x_long, sim_z=nb_sim_z_long,
        color_x="#e05c2a", color_z="#e05c2a",
        linestyle_z="--",
        title=(f"Initial $T_X$ and $T_Z$ — log time axis\n"
               f"$T_X$ = {nb_sim_x_long['T_X']:.3f} µs,  "
               f"$T_Z$ = {nb_sim_z_long['T_Z']:.1f} µs"),
        Tx_label=f"$\\langle X\\rangle$  (phase-flip)  $T_X$ = {nb_sim_x_long['T_X']:.3f} µs",
        Tz_label=f"$\\langle Z\\rangle$  (bit-flip)    $T_Z$ = {nb_sim_z_long['T_Z']:.1f} µs",
    )
    plt.tight_layout()
    plt.savefig("initial_Tx_Tz_combined.png", dpi=150, bbox_inches="tight")
    print("Figure saved to initial_Tx_Tz_combined.png")
    plt.show()

    # Separate figure: optimized T_X and T_Z on the same graph
    fig_opt, ax_opt = plt.subplots(figsize=(7, 5))
    fig_opt.suptitle(
        fr"Optimized parameters: $\epsilon_d$={eps_d_opt:.2f}, $g_2$={g2_opt:.2f}",
        fontsize=12,
    )
    _plot_combined_panel(
        ax_opt,
        sim_x=opt_sim_x_long, sim_z=opt_sim_z_long,
        color_x="steelblue", color_z="darkorange",
        linestyle_z="--",
        title=(f"Optimized $T_X$ and $T_Z$ — log time axis\n"
               f"$T_X$ = {opt_sim_x_long['T_X']:.3f} µs,  "
               f"$T_Z$ = {opt_sim_z_long['T_Z']:.1f} µs"),
        Tx_label=f"$\\langle X\\rangle$  (phase-flip)  $T_X$ = {opt_sim_x_long['T_X']:.3f} µs",
        Tz_label=f"$\\langle Z\\rangle$  (bit-flip)    $T_Z$ = {opt_sim_z_long['T_Z']:.1f} µs",
    )
    plt.tight_layout()
    plt.savefig("optimized_Tx_Tz_combined.png", dpi=150, bbox_inches="tight")
    print("Figure saved to optimized_Tx_Tz_combined.png")
    plt.show()

    print("\nDone.")