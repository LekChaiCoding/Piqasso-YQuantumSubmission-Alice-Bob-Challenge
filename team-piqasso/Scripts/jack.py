"""
Colab-ready Cat Qubit Optimizer (FAST mode by default).

This version mirrors the 2-parameter FAST workflow from optimize_cat.py and
is designed to run reliably on Google Colab.

Colab setup (run in a notebook cell before executing this file):
  !pip install -q "dynamiqs>=0.3.0" cmaes scipy
  !pip install -q -U "jax[cuda12]>=0.4.36,<0.7"

After installing, restart the runtime, then run:
  %run optimize_cat_colab.py

Why this avoids your reshape error:
  - No manual reshape assumptions are used for batched mesolve outputs.
  - Default evaluation is sequential (stable on Colab GPU).
  - Optional threaded population evaluation has guarded fallback.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
        "dynamiqs is not installed. In Colab run:\n"
        "  !pip install -q \"dynamiqs>=0.3.0\" cmaes scipy\n"
        "  !pip install -q -U \"jax[cuda12]>=0.4.36,<0.7\"\n"
        "Then restart the runtime and run this script again."
    ) from exc


jax.config.update("jax_enable_x64", False)


# Physics constants
NA, NB = 15, 5
KAPPA_B = 10.0
KAPPA_A = 1.0

# Objective
ETA_TARGET = 320.0

# Runtime mode
FAST_MODE = False
#os.getenv("CAT_FAST_MODE", "1") == "1"

# Lifetime clamps
TX_MAX = 5.0
TZ_MAX = 2000.0

# Simulation tuning
N_POINTS = int(os.getenv("CAT_N_POINTS", "30" if FAST_MODE else "50"))
TZ_TFINAL = float(os.getenv("CAT_TZ_TFINAL", "100.0" if FAST_MODE else "200.0"))
TX_TFINAL = float(os.getenv("CAT_TX_TFINAL", "1.0"))

# Evaluation mode. "sequential" is safest on Colab.
EVAL_MODE = os.getenv("CAT_EVAL_MODE", "sequential").strip().lower()
MAX_WORKERS = max(1, int(os.getenv("CAT_MAX_WORKERS", "2")))

# Cache to avoid re-simulating near-identical points explored by CMA-ES.
CACHE_DECIMALS = int(os.getenv("CAT_CACHE_DECIMALS", "4"))
_MEASURE_CACHE = {}


# Precomputed operators
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


def fit_decay(t, y):
    """Fit y = A * exp(-t/tau) + C, then return tau."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A0 = max(float(np.ptp(y)), 1e-6)
    tau0 = max(float(np.ptp(t)) / 3.0, 1e-6)
    C0 = float(y[-1])
    try:
        result = least_squares(
            lambda p, tx, ty: p[0] * np.exp(-tx / p[1]) + p[2] - ty,
            [A0, tau0, C0],
            args=(t, y),
            bounds=([0.0, 1e-10, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1",
            f_scale=0.1,
        )
        return max(float(result.x[1]), 1e-10)
    except Exception:
        return tau0


def compute_alpha(g2, eps_d):
    """Estimate cat amplitude alpha from real-valued g2 and eps_d."""
    kappa2 = 4.0 * g2 * g2 / KAPPA_B
    eps2 = 2.0 * g2 * eps_d / KAPPA_B
    if kappa2 < 1e-12:
        return 0.5
    val = 2.0 / kappa2 * (eps2 - KAPPA_A / 4.0)
    return float(np.sqrt(max(val, 0.01)))


def simulate(g2, eps_d, init_state, tfinal, n_points):
    """Run one mesolve and return (t, <X_L>, <Z_L>)."""
    alpha = compute_alpha(g2, eps_d)

    # Real-parameter Hamiltonian:
    # H = g2*(a^2 b^dagger + (a^dagger)^2 b) - eps_d*(b^dagger + b)
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
    tsave = jnp.linspace(0.0, float(tfinal), int(n_points))

    result = dq.mesolve(
        H,
        [_LOSS_B, _LOSS_A],
        psi0,
        tsave,
        exp_ops=[_PARITY, Z_L],
        options=dq.Options(progress_meter=False),
    )
    return (
        np.asarray(result.tsave),
        np.asarray(result.expects[0].real),
        np.asarray(result.expects[1].real),
    )


def _measure_cache_key(g2, eps_d, n_points, tz_tfinal, tx_tfinal):
    return (
        round(float(g2), CACHE_DECIMALS),
        round(float(eps_d), CACHE_DECIMALS),
        int(n_points),
        round(float(tz_tfinal), 4),
        round(float(tx_tfinal), 4),
    )


def measure_Tx_Tz(g2, eps_d, n_points=N_POINTS, tz_tfinal=TZ_TFINAL, tx_tfinal=TX_TFINAL):
    """Run two simulations and return physically clamped (Tx, Tz)."""
    key = _measure_cache_key(g2, eps_d, n_points, tz_tfinal, tx_tfinal)
    if key in _MEASURE_CACHE:
        return _MEASURE_CACHE[key]

    tz_t, _, sz = simulate(g2, eps_d, "+z", tz_tfinal, n_points=n_points)
    Tz = min(fit_decay(tz_t, sz), TZ_MAX)

    tx_t, sx, _ = simulate(g2, eps_d, "+x", tx_tfinal, n_points=n_points)
    Tx = min(fit_decay(tx_t, sx), TX_MAX)

    metrics = (max(Tx, 1e-6), max(Tz, 1e-6))
    _MEASURE_CACHE[key] = metrics
    return metrics


def reward_log_quadratic(Tx, Tz, a=1.0, b=1.0, c=2.0):
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * (np.log(eta / ETA_TARGET)) ** 2
    return lifetime_term + eta_penalty


def reward_log_exp(Tx, Tz, a=1.0, b=1.0, c=1.0, k=3.0):
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * np.exp(k * abs(eta / ETA_TARGET - 1.0))
    return lifetime_term + eta_penalty


def reward_mixed_log(Tx, Tz, a=1.0, b=1.0, c=3.0):
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * abs(np.log(eta / ETA_TARGET))
    return lifetime_term + eta_penalty


REWARD_FUNCTIONS = {
    "log_quadratic": reward_log_quadratic,
    "log_exp": reward_log_exp,
    "mixed_log": reward_mixed_log,
}


BOUNDS = np.array(
    [
        [0.05, 1.5],
        [2.0, 6.0],
    ],
    dtype=float,
)


def evaluate_candidate(x, reward_fn, n_points, tz_tfinal):
    g2 = float(np.clip(x[0], BOUNDS[0, 0], BOUNDS[0, 1]))
    eps_d = float(np.clip(x[1], BOUNDS[1, 0], BOUNDS[1, 1]))
    try:
        Tx, Tz = measure_Tx_Tz(g2, eps_d, n_points=n_points, tz_tfinal=tz_tfinal)
        eta = Tz / Tx
        loss = float(reward_fn(Tx, Tz))
        return loss, Tx, Tz, eta
    except Exception as exc:
        print(f"  [!] sim failed for g2={g2:.4f}, eps_d={eps_d:.4f}: {exc}")
        return 1e6, 1e-6, 1e-6, 1.0


def evaluate_population(xs, reward_fn, n_points, tz_tfinal, eval_mode=EVAL_MODE, max_workers=MAX_WORKERS):
    """
    Evaluate one CMA-ES population robustly.

    - sequential: safest for Colab and default.
    - threaded: optional CPU threading around candidate-level calls.
    """
    xs = np.asarray(xs, dtype=float)
    if eval_mode == "threaded" and len(xs) > 1:
        try:
            workers = min(max_workers, len(xs))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(
                    pool.map(
                        lambda row: evaluate_candidate(row, reward_fn, n_points, tz_tfinal),
                        xs,
                    )
                )
        except Exception as exc:
            print(f"  [!] batch failed ({exc}), falling back to sequential")

    return [evaluate_candidate(row, reward_fn, n_points, tz_tfinal) for row in xs]


def optimize(reward_name="log_quadratic", batch_size=8, n_epochs=20, seed=0, eval_mode=EVAL_MODE):
    reward_fn = REWARD_FUNCTIONS[reward_name]
    x0 = np.array([0.2, 4.0], dtype=float)

    optimizer = SepCMA(
        mean=x0,
        sigma=0.1,
        bounds=BOUNDS,
        population_size=batch_size,
        seed=seed,
    )

    history = {"loss": [], "Tx": [], "Tz": [], "eta": [], "params": [], "epoch_s": []}
    log_every = 5 if FAST_MODE else 10

    for epoch in range(n_epochs):
        t0 = time.time()
        xs = np.array([optimizer.ask() for _ in range(optimizer.population_size)], dtype=float)

        results = evaluate_population(
            xs,
            reward_fn,
            n_points=N_POINTS,
            tz_tfinal=TZ_TFINAL,
            eval_mode=eval_mode,
            max_workers=MAX_WORKERS,
        )

        losses = np.array([r[0] for r in results], dtype=float)
        txs = np.array([r[1] for r in results], dtype=float)
        tzs = np.array([r[2] for r in results], dtype=float)
        etas = np.array([r[3] for r in results], dtype=float)

        optimizer.tell([(xs[j], losses[j]) for j in range(len(xs))])

        history["loss"].append(float(np.mean(losses)))
        history["Tx"].append(float(np.mean(txs)))
        history["Tz"].append(float(np.mean(tzs)))
        history["eta"].append(float(np.mean(etas)))
        history["params"].append(optimizer.mean.copy())
        history["epoch_s"].append(float(time.time() - t0))

        if epoch % log_every == 0:
            print(
                f"[{reward_name}] Epoch {epoch:3d} | loss={np.mean(losses):.3f} "
                f"| Tx={np.mean(txs):.3f} Tz={np.mean(tzs):.1f} "
                f"| eta={np.mean(etas):.0f} "
                f"| g2={optimizer.mean[0]:.3f} eps_d={optimizer.mean[1]:.3f} "
                f"| {history['epoch_s'][-1]:.2f}s"
            )

    best = optimizer.mean.copy()
    Tx, Tz = measure_Tx_Tz(float(best[0]), float(best[1]))
    eta = Tz / Tx

    print("\n" + "=" * 50)
    print(f"RESULT [{reward_name}]")
    print("=" * 50)
    print(f"g_2   = {best[0]:.4f}")
    print(f"eps_d = {best[1]:.4f}")
    print(f"T_x   = {Tx:.4f} us")
    print(f"T_z   = {Tz:.1f} us")
    print(f"eta   = {eta:.1f} (target: {ETA_TARGET})")

    return best, history, (Tx, Tz, eta)


def plot_results(all_results):
    """Plot convergence for all tested reward functions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name, (_, hist, _) in all_results.items():
        epochs = np.arange(len(hist["loss"]))
        axes[0, 0].plot(epochs, hist["loss"], label=name)
        axes[0, 1].plot(epochs, hist["eta"], label=name)
        axes[1, 0].plot(epochs, hist["Tx"], label=f"{name} Tx")
        axes[1, 0].plot(epochs, hist["Tz"], label=f"{name} Tz", linestyle="--")
        params = np.array(hist["params"], dtype=float)
        axes[1, 1].plot(epochs, params[:, 0], label=f"{name} g2")
        axes[1, 1].plot(epochs, params[:, 1], label=f"{name} eps_d", linestyle="--")

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("eta = T_z/T_x")
    axes[0, 1].set_title("Bias Ratio")
    axes[0, 1].axhline(ETA_TARGET, color="r", linestyle="--", alpha=0.5, label="target")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Lifetime (us)")
    axes[1, 0].set_title("Lifetimes")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Parameter Value")
    axes[1, 1].set_title("Parameters")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_runtime_banner():
    in_colab = "google.colab" in sys.modules
    devices = jax.devices()
    gpu_count = sum(1 for d in devices if d.platform == "gpu")
    print(
        f"Environment={'COLAB' if in_colab else 'LOCAL'} | "
        f"devices={[d.platform + ':' + str(d.id) for d in devices]}"
    )
    if in_colab and gpu_count == 0:
        print("[!] No GPU detected in Colab. Runtime -> Change runtime type -> GPU.")


if __name__ == "__main__":
    print_runtime_banner()

    print("Warming up JAX + dynamiqs...")
    warm_t, warm_x, _ = simulate(0.2, 4.0, "+z", tfinal=1.0, n_points=min(10, N_POINTS))
    _ = warm_t
    jax.block_until_ready(jnp.asarray(warm_x))
    print("Ready.\n")

    default_rewards = [os.getenv("CAT_REWARD", "log_quadratic")] if FAST_MODE else list(REWARD_FUNCTIONS.keys())
    default_epochs = int(os.getenv("CAT_EPOCHS", "100"))
    default_batch = int(os.getenv("CAT_BATCH_SIZE", "8" if FAST_MODE else "12"))
    default_eval_mode = os.getenv("CAT_EVAL_MODE", EVAL_MODE).strip().lower()

    print(
        f"Mode={'FAST' if FAST_MODE else 'FULL'} | rewards={default_rewards} "
        f"| epochs={default_epochs} | batch={default_batch} "
        f"| points={N_POINTS} | Tz_final={TZ_TFINAL} | eval={default_eval_mode}"
    )

    all_results = {}
    for reward_name in default_rewards:
        print("\n" + "=" * 60)
        print(f"Optimizing with reward: {reward_name}")
        print("=" * 60)
        best, history, metrics = optimize(
            reward_name=reward_name,
            batch_size=default_batch,
            n_epochs=default_epochs,
            eval_mode=default_eval_mode,
        )
        all_results[reward_name] = (best, history, metrics)

    plot_results(all_results)