"""
leon.py – Accelerated Cat Qubit Optimizer
==========================================
Speed-ups vs optimize_cat_colab.py:

  1. Log-linear τ fitting   – replaces iterative scipy.least_squares with
     a weighted log-linear regression (numpy.polyfit). ~100× faster per fit.
  2. Adaptive integration   – short mesolve probe first (t≈20 for Tz); only
     extends to t=100 if the signal has barely decayed.  Saves the full ODE
     integration for the majority of candidates whose Tz < 100.
  3. Strategic time-points  – 5 geometrically-spaced save-points instead of
     30 uniform.  Points past ~2τ (where the signal has converged) are
     skipped; the asymptote is *inferred* from the fit, not computed.
  4. Single-pass CMA-ES     – one reward function per run (default
     log_quadratic), not three sequential passes.
  5. Multi-fidelity finish  – fast τ during search; the final best candidate
     is re-evaluated with 40 points + scipy for accurate reporting.
  6. Early rejection        – if Tz is terrible (< 0.5), Tx simulation is
     skipped entirely.

Gaussian noise model (control + measurement) carried over from
optimize_cat_colab.py for sim-to-real robustness.

Target wall-clock: 5–10 min on Mac CPU.

Usage:
  python leon.py                        # defaults
  LEON_EPOCHS=25 python leon.py         # more epochs
  CAT_NOISE=0 python leon.py            # disable noise
  LEON_REWARD=mixed_log python leon.py  # different reward
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from cmaes import SepCMA
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

try:
    import dynamiqs as dq
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dynamiqs is not installed. Run:\n"
        "  pip install 'dynamiqs>=0.3.0' cmaes scipy\n"
        "  pip install -U 'jax[cuda12]>=0.4.36,<0.7'  # (GPU, optional)\n"
    ) from exc

jax.config.update("jax_enable_x64", False)

# ── Physics ──────────────────────────────────────────────────
NA, NB = 15, 5
KAPPA_B = 10.0
KAPPA_A = 1.0
ETA_TARGET = 320.0
TX_MAX = 5.0
TZ_MAX = 2000.0

# ── Gaussian noise ───────────────────────────────────────────
NOISE_ENABLED = os.getenv("CAT_NOISE", "1") == "1"
NOISE_SIGMA_CTRL = float(os.getenv("CAT_NOISE_CTRL", "0.01"))
NOISE_SIGMA_MEAS = float(os.getenv("CAT_NOISE_MEAS", "0.01"))
_RNG = np.random.default_rng(int(os.getenv("CAT_NOISE_SEED", "42")))

# ── Cache (auto-disabled when noise is on) ───────────────────
_CACHE: dict[tuple, tuple[float, float]] = {}

# ── CMA-ES search bounds ────────────────────────────────────
BOUNDS = np.array([[0.05, 1.5], [2.0, 6.0]], dtype=float)

# ── Probe time arrays ───────────────────────────────────────
# Geometric-ish spacing.  Short arrays cover the common τ range;
# long arrays extend only when the short probe shows < 15 % decay.
# Points beyond ~2τ carry almost no new information ("2-sigma cutoff")
# – the converged value is inferred by the log-linear fit instead.
_TZ_SHORT = np.array([0.0, 0.8, 3.0, 8.0, 20.0])
_TZ_LONG = np.array([0.0, 3.0, 12.0, 40.0, 100.0])
_TX_SHORT = np.array([0.0, 0.02, 0.07, 0.20, 0.50])
_TX_LONG = np.array([0.0, 0.05, 0.20, 0.50, 1.20])

# ── Pre-computed operators ───────────────────────────────────
_A = dq.tensor(dq.destroy(NA), dq.eye(NB))
_B = dq.tensor(dq.eye(NA), dq.destroy(NB))
_LOSS_B = jnp.sqrt(KAPPA_B) * _B
_LOSS_A = jnp.sqrt(KAPPA_A) * _A
_FOCK_B0 = dq.fock(NB, 0)
_PARITY = dq.tensor(
    jnp.diag(jnp.array([(-1.0) ** n for n in range(NA)], dtype=jnp.float32)),
    jnp.eye(NB),
)
_AA_BD = _A @ _A @ _B.dag()
_ADAD_B = _A.dag() @ _A.dag() @ _B
_B_DRIVE = _B.dag() + _B


# ── Core helpers ─────────────────────────────────────────────

def compute_alpha(g2: float, eps_d: float) -> float:
    kappa2 = 4.0 * g2 * g2 / KAPPA_B
    eps2 = 2.0 * g2 * eps_d / KAPPA_B
    if kappa2 < 1e-12:
        return 0.5
    return float(np.sqrt(max(2.0 / kappa2 * (eps2 - KAPPA_A / 4.0), 0.01)))


def log_linear_tau(t, y, min_signal: float = 0.05) -> float:
    """Extract τ from y ≈ A·exp(−t/τ) via weighted log-linear regression.

    Points with |y| < *min_signal* are below the "2-sigma" convergence
    threshold and are discarded – the asymptotic value is *inferred*, not
    computed.  Remaining points are weighted by |y| so noisy low-amplitude
    samples contribute less.
    """
    t = np.asarray(t, dtype=float)
    y_abs = np.abs(np.asarray(y, dtype=float))
    mask = y_abs > min_signal
    if mask.sum() < 2:
        return float(np.ptp(t)) / 2.0  # fallback
    tf, logy, w = t[mask], np.log(y_abs[mask]), y_abs[mask]
    slope = np.polyfit(tf, logy, 1, w=w)[0]
    if slope >= -1e-12:
        return float(np.ptp(t)) * 3.0  # barely decaying
    return min(-1.0 / slope, TZ_MAX)


def precise_tau(t, y) -> float:
    """Full scipy exponential fit (used only for the final best candidate)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A0 = max(float(np.ptp(y)), 1e-6)
    tau0 = max(float(np.ptp(t)) / 3.0, 1e-6)
    C0 = float(y[-1])
    try:
        res = least_squares(
            lambda p, tx, ty: p[0] * np.exp(-tx / p[1]) + p[2] - ty,
            [A0, tau0, C0],
            args=(t, y),
            bounds=([0.0, 1e-10, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1",
            f_scale=0.1,
        )
        return max(float(res.x[1]), 1e-10)
    except Exception:
        return tau0


# ── Simulation ───────────────────────────────────────────────

def simulate(g2, eps_d, init_state, tsave, add_noise: bool = True):
    """Run mesolve at the requested save-points.

    Returns (t, <X_L>, <Z_L>).
    """
    alpha = compute_alpha(g2, eps_d)
    H = g2 * (_AA_BD + _ADAD_B) - eps_d * _B_DRIVE

    cat_p = dq.coherent(NA, alpha)
    cat_m = dq.coherent(NA, -alpha)
    if init_state == "+z":
        psi0_a = cat_p
    elif init_state == "+x":
        psi0_a = (cat_p + cat_m) / jnp.sqrt(2.0)
    else:
        psi0_a = cat_m

    Z_L = dq.tensor(cat_p @ cat_p.dag() - cat_m @ cat_m.dag(), dq.eye(NB))
    psi0 = dq.tensor(psi0_a, _FOCK_B0)

    result = dq.mesolve(
        H,
        [_LOSS_B, _LOSS_A],
        psi0,
        jnp.asarray(tsave, dtype=jnp.float32),
        exp_ops=[_PARITY, Z_L],
        options=dq.Options(progress_meter=False),
    )
    t_out = np.asarray(result.tsave)
    x_out = np.asarray(result.expects[0].real)
    z_out = np.asarray(result.expects[1].real)

    if add_noise and NOISE_ENABLED and NOISE_SIGMA_MEAS > 0:
        x_out = x_out + _RNG.normal(0.0, NOISE_SIGMA_MEAS, size=x_out.shape)
        z_out = z_out + _RNG.normal(0.0, NOISE_SIGMA_MEAS, size=z_out.shape)

    return t_out, x_out, z_out


def _noisy_controls(g2, eps_d):
    if not NOISE_ENABLED or NOISE_SIGMA_CTRL <= 0:
        return g2, eps_d
    return (
        max(g2 * (1.0 + _RNG.normal(0.0, NOISE_SIGMA_CTRL)), 1e-6),
        max(eps_d * (1.0 + _RNG.normal(0.0, NOISE_SIGMA_CTRL)), 1e-6),
    )


# ── Adaptive τ probing ──────────────────────────────────────

_DECAY_THRESHOLD = 0.85  # extend if last/first > this (< 15 % decay)

def _probe_tau(g2, eps_d, init_state, component, short_ts, long_ts, tau_max):
    """Two-phase adaptive τ extraction.

    Phase 1 – short integration with 5 geometric points.
    Phase 2 – extend only if the signal barely decayed (ratio > 0.85).
    """
    _, sx, sz = simulate(g2, eps_d, init_state, short_ts)
    sig = sz if component == "z" else sx

    y0 = abs(float(sig[0]))
    if y0 < 0.05:
        return 1e-6  # cat state barely formed → bad candidate

    ratio = abs(float(sig[-1])) / y0
    if ratio < _DECAY_THRESHOLD:
        return min(max(log_linear_tau(short_ts, sig), 1e-10), tau_max)

    # Signal barely decayed → need longer integration
    _, sx2, sz2 = simulate(g2, eps_d, init_state, long_ts)
    sig2 = sz2 if component == "z" else sx2
    return min(max(log_linear_tau(long_ts, sig2), 1e-10), tau_max)


def measure_Tx_Tz(g2, eps_d):
    """Adaptive measurement with caching + noise."""
    if not NOISE_ENABLED:
        key = (round(float(g2), 4), round(float(eps_d), 4))
        if key in _CACHE:
            return _CACHE[key]

    g2_n, eps_d_n = _noisy_controls(g2, eps_d)

    Tz = _probe_tau(g2_n, eps_d_n, "+z", "z", _TZ_SHORT, _TZ_LONG, TZ_MAX)

    # Early rejection: terrible Tz → skip Tx entirely
    if Tz < 0.5:
        metrics = (1e-6, max(Tz, 1e-6))
    else:
        Tx = _probe_tau(g2_n, eps_d_n, "+x", "x", _TX_SHORT, _TX_LONG, TX_MAX)
        metrics = (max(Tx, 1e-6), max(Tz, 1e-6))

    if not NOISE_ENABLED:
        _CACHE[key] = metrics
    return metrics


def precise_measure(g2, eps_d, n_pts: int = 40):
    """High-fidelity noiseless measurement for the final best candidate."""
    tsave_tz = np.linspace(0.0, 100.0, n_pts)
    tsave_tx = np.linspace(0.0, 1.0, n_pts)

    _, _, sz = simulate(g2, eps_d, "+z", tsave_tz, add_noise=False)
    Tz = min(precise_tau(tsave_tz, sz), TZ_MAX)

    _, sx, _ = simulate(g2, eps_d, "+x", tsave_tx, add_noise=False)
    Tx = min(precise_tau(tsave_tx, sx), TX_MAX)

    return max(Tx, 1e-6), max(Tz, 1e-6)


# ── Reward functions ─────────────────────────────────────────

def reward_log_quadratic(Tx, Tz, a=1.0, b=1.0, c=2.0):
    eta = Tz / Tx
    return -(a * np.log(Tx) + b * np.log(Tz)) + c * (np.log(eta / ETA_TARGET)) ** 2


def reward_log_exp(Tx, Tz, a=1.0, b=1.0, c=1.0, k=3.0):
    eta = Tz / Tx
    return -(a * np.log(Tx) + b * np.log(Tz)) + c * np.exp(k * abs(eta / ETA_TARGET - 1.0))


def reward_mixed_log(Tx, Tz, a=1.0, b=1.0, c=3.0):
    eta = Tz / Tx
    return -(a * np.log(Tx) + b * np.log(Tz)) + c * abs(np.log(eta / ETA_TARGET))


REWARD_FUNCTIONS = {
    "log_quadratic": reward_log_quadratic,
    "log_exp": reward_log_exp,
    "mixed_log": reward_mixed_log,
}


# ── CMA-ES loop ─────────────────────────────────────────────

def evaluate_candidate(x, reward_fn):
    g2 = float(np.clip(x[0], BOUNDS[0, 0], BOUNDS[0, 1]))
    eps_d = float(np.clip(x[1], BOUNDS[1, 0], BOUNDS[1, 1]))
    try:
        Tx, Tz = measure_Tx_Tz(g2, eps_d)
        return float(reward_fn(Tx, Tz)), Tx, Tz, Tz / Tx
    except Exception as exc:
        print(f"  [!] g2={g2:.4f} eps_d={eps_d:.4f}: {exc}")
        return 1e6, 1e-6, 1e-6, 1.0


def optimize(reward_name="log_quadratic", batch_size=14, n_epochs=20, seed=0):
    reward_fn = REWARD_FUNCTIONS[reward_name]
    x0 = np.array([0.2, 4.0], dtype=float)

    opt = SepCMA(
        mean=x0,
        sigma=0.15,
        bounds=BOUNDS,
        population_size=batch_size,
        seed=seed,
    )

    history = {"loss": [], "Tx": [], "Tz": [], "eta": [], "params": [], "t": []}

    for ep in range(n_epochs):
        t0 = time.time()
        xs = np.array([opt.ask() for _ in range(opt.population_size)], dtype=float)
        results = [evaluate_candidate(row, reward_fn) for row in xs]

        losses = np.array([r[0] for r in results], dtype=float)
        txs = np.array([r[1] for r in results], dtype=float)
        tzs = np.array([r[2] for r in results], dtype=float)
        etas = np.array([r[3] for r in results], dtype=float)

        opt.tell([(xs[j], losses[j]) for j in range(len(xs))])

        history["loss"].append(float(np.mean(losses)))
        history["Tx"].append(float(np.mean(txs)))
        history["Tz"].append(float(np.mean(tzs)))
        history["eta"].append(float(np.mean(etas)))
        history["params"].append(opt.mean.copy())
        history["t"].append(time.time() - t0)

        if ep % 3 == 0:
            print(
                f"  Epoch {ep:3d}/{n_epochs} | loss={np.mean(losses):.3f} "
                f"| Tx={np.mean(txs):.3f} Tz={np.mean(tzs):.1f} "
                f"| \u03b7={np.mean(etas):.0f} "
                f"| g2={opt.mean[0]:.4f} eps_d={opt.mean[1]:.4f} "
                f"| {history['t'][-1]:.1f}s"
            )

    # ── Multi-fidelity: precise final evaluation ─────────────
    best = opt.mean.copy()
    print("\nPrecise re-evaluation of best candidate …")
    Tx, Tz = precise_measure(float(best[0]), float(best[1]))
    eta = Tz / Tx

    print("\n" + "=" * 50)
    print(f"  RESULT [{reward_name}]")
    print("=" * 50)
    print(f"  g2    = {best[0]:.4f}")
    print(f"  eps_d = {best[1]:.4f}")
    print(f"  Tx    = {Tx:.4f} \u00b5s")
    print(f"  Tz    = {Tz:.1f} \u00b5s")
    print(f"  \u03b7     = {eta:.1f}  (target {ETA_TARGET})")
    print("=" * 50)

    return best, history, (Tx, Tz, eta)


# ── Plotting ─────────────────────────────────────────────────

def plot_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ep = np.arange(len(history["loss"]))

    axes[0, 0].plot(ep, history["loss"])
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Loss convergence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, history["eta"])
    axes[0, 1].axhline(ETA_TARGET, color="r", linestyle="--", alpha=0.5, label="target")
    axes[0, 1].set(xlabel="Epoch", ylabel="\u03b7 = Tz / Tx", title="Bias ratio")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ep, history["Tx"], label="Tx")
    axes[1, 0].plot(ep, history["Tz"], label="Tz", linestyle="--")
    axes[1, 0].set(xlabel="Epoch", ylabel="Lifetime (\u00b5s)", title="Lifetimes")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    params = np.array(history["params"], dtype=float)
    axes[1, 1].plot(ep, params[:, 0], label="g2")
    axes[1, 1].plot(ep, params[:, 1], label="eps_d", linestyle="--")
    axes[1, 1].set(xlabel="Epoch", ylabel="Value", title="Parameters")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    devs = jax.devices()
    in_colab = "google.colab" in sys.modules
    print(
        f"Environment={'COLAB' if in_colab else 'LOCAL'} | "
        f"devices={[d.platform + ':' + str(d.id) for d in devs]}"
    )
    noise_str = (
        f"ON  ctrl_\u03c3={NOISE_SIGMA_CTRL}  meas_\u03c3={NOISE_SIGMA_MEAS}"
        if NOISE_ENABLED
        else "OFF"
    )
    print(f"Noise: {noise_str}")

    # Warm-up (compiles mesolve graph once)
    print("Warming up JAX + dynamiqs …")
    t_warm = time.time()
    _ = simulate(0.2, 4.0, "+z", np.array([0.0, 1.0, 5.0]), add_noise=False)
    jax.block_until_ready(jnp.zeros(1))
    print(f"Warm-up done in {time.time() - t_warm:.1f}s\n")

    reward_name = os.getenv("LEON_REWARD", "log_quadratic")
    batch = int(os.getenv("LEON_BATCH", "14"))
    epochs = int(os.getenv("LEON_EPOCHS", "20"))
    print(f"CMA-ES: reward={reward_name}  batch={batch}  epochs={epochs}\n")

    best, hist, metrics = optimize(
        reward_name=reward_name,
        batch_size=batch,
        n_epochs=epochs,
    )

    total = sum(hist["t"])
    print(f"\nTotal optimisation time: {total:.0f}s ({total / 60:.1f} min)")

    plot_results(hist)
