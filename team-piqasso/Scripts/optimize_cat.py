"""
Cat Qubit Optimizer

Tunes real-valued g_2 and eps_d (2 knobs) to:
  - Maximize T_x and T_z
  - Keep bias eta = T_z / T_x in the hundreds (~200-999)

Three reward functions available:
  1. log_quadratic: -(a*log(Tx) + b*log(Tz)) + c*(log(eta/target))^2
  2. log_exp:       -(a*log(Tx) + b*log(Tz)) + c*exp(k*|eta/target - 1|)
  3. mixed_log:     -(a*log(Tx) + b*log(Tz)) + c*|log(eta/target)|
"""

import numpy as np
import jax.numpy as jnp
import dynamiqs as dq
import os
import time
from scipy.optimize import least_squares
from cmaes import SepCMA
from matplotlib import pyplot as plt

# ── Physics constants ────────────────────────────────────────
NA, NB = 15, 5
KAPPA_B = 10.0   # buffer decay [MHz]
KAPPA_A = 1.0    # single-photon loss [MHz]

# ── Objective ────────────────────────────────────────────────
ETA_TARGET = 500.0  # target bias T_z/T_x (want ~200-999)

# ── Runtime mode ─────────────────────────────────────────────
# Fast mode is on by default to keep turnaround short.
# Set CAT_FAST_MODE=0 for full-quality runs.
FAST_MODE = os.getenv("CAT_FAST_MODE", "1") == "1"

# ── Lifetime clamps ──────────────────────────────────────────
TX_MAX = 5.0     # us
TZ_MAX = 2000.0  # us

# ── Simulation tuning ────────────────────────────────────────
N_POINTS = int(os.getenv("CAT_N_POINTS", "30" if FAST_MODE else "50"))
TZ_TFINAL = float(os.getenv("CAT_TZ_TFINAL", "100.0" if FAST_MODE else "200.0"))  # us
TX_TFINAL = 1.0    # us

# ── Precomputed operators ────────────────────────────────────
_a = dq.tensor(dq.destroy(NA), dq.eye(NB))
_b = dq.tensor(dq.eye(NA), dq.destroy(NB))
_loss_b = jnp.sqrt(KAPPA_B) * _b
_loss_a = jnp.sqrt(KAPPA_A) * _a
_fock_b0 = dq.fock(NB, 0)
_parity = dq.tensor(
    jnp.diag(jnp.array([(-1.0) ** n for n in range(NA)])),
    jnp.eye(NB),
)


# ── Exponential fit ──────────────────────────────────────────
def fit_decay(t, y):
    """Fit y = A * exp(-t/tau) + C, return tau."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    A0 = max(float(np.ptp(y)), 1e-6)
    tau0 = max(float(np.ptp(t)) / 3, 1e-6)
    C0 = float(y[-1])
    try:
        res = least_squares(
            lambda p, t, y: p[0] * np.exp(-t / p[1]) + p[2] - y,
            [A0, tau0, C0], args=(t, y),
            bounds=([0, 1e-10, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1", f_scale=0.1,
        )
        return max(float(res.x[1]), 1e-10)
    except Exception:
        return tau0


# ── Simulation ───────────────────────────────────────────────
def compute_alpha(g2, eps_d):
    """Estimate cat size from real g2 and eps_d."""
    kappa2 = 4.0 * g2 ** 2 / KAPPA_B
    eps2 = 2.0 * g2 * eps_d / KAPPA_B
    if kappa2 < 1e-12:
        return 0.5
    val = 2.0 / kappa2 * (eps2 - KAPPA_A / 4.0)
    return float(np.sqrt(max(val, 0.01)))


def simulate(g2, eps_d, init_state, tfinal, n_points=N_POINTS):
    """Run mesolve for real g2, eps_d. Returns (t, <X_L>, <Z_L>)."""
    alpha = compute_alpha(g2, eps_d)

    # H = g2*(a^2 b† + (a†)^2 b) - eps_d*(b† + b)  [real g2, eps_d]
    H = (g2 * (_a @ _a @ _b.dag() + _a.dag() @ _a.dag() @ _b)
         - eps_d * (_b.dag() + _b))

    cat_p = dq.coherent(NA, alpha)
    cat_m = dq.coherent(NA, -alpha)
    if init_state == "+z":
        psi0_a = cat_p
    elif init_state == "+x":
        psi0_a = (cat_p + cat_m) / jnp.sqrt(2)
    else:
        psi0_a = cat_m

    Z_L = dq.tensor(cat_p @ cat_p.dag() - cat_m @ cat_m.dag(), dq.eye(NB))
    psi0 = dq.tensor(psi0_a, _fock_b0)
    tsave = jnp.linspace(0, tfinal, n_points)

    res = dq.mesolve(
        H, [_loss_b, _loss_a], psi0, tsave,
        exp_ops=[_parity, Z_L],
        options=dq.Options(progress_meter=False),
    )
    return np.array(res.tsave), np.array(res.expects[0].real), np.array(res.expects[1].real)


def measure_Tx_Tz(g2, eps_d):
    """Two sims → (T_x, T_z), clamped to physical bounds."""
    tz_t, _, sz = simulate(g2, eps_d, "+z", TZ_TFINAL)
    Tz = min(fit_decay(tz_t, sz), TZ_MAX)

    tx_t, sx, _ = simulate(g2, eps_d, "+x", TX_TFINAL)
    Tx = min(fit_decay(tx_t, sx), TX_MAX)

    return max(Tx, 1e-6), max(Tz, 1e-6)


# ── Reward functions (all return loss to MINIMIZE) ───────────
def reward_log_quadratic(Tx, Tz, a=1.0, b=1.0, c=2.0):
    """
    -(a*log(Tx) + b*log(Tz)) + c*(log(eta/target))^2
    Smooth, symmetric in multiplicative deviations of eta.
    """
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * (np.log(eta / ETA_TARGET)) ** 2
    return lifetime_term + eta_penalty


def reward_log_exp(Tx, Tz, a=1.0, b=1.0, c=1.0, k=3.0):
    """
    -(a*log(Tx) + b*log(Tz)) + c*exp(k*|eta/target - 1|)
    Exponential penalty grows fast for large eta deviations.
    """
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * np.exp(k * abs(eta / ETA_TARGET - 1))
    return lifetime_term + eta_penalty


def reward_mixed_log(Tx, Tz, a=1.0, b=1.0, c=3.0):
    """
    -(a*log(Tx) + b*log(Tz)) + c*|log(eta/target)|
    Softer than quadratic, sharper than nothing.
    """
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * abs(np.log(eta / ETA_TARGET))
    return lifetime_term + eta_penalty


REWARD_FUNCTIONS = {
    "log_quadratic": reward_log_quadratic,
    "log_exp": reward_log_exp,
    "mixed_log": reward_mixed_log,
}


# ── CMA-ES Optimization ─────────────────────────────────────
def optimize(reward_name="log_quadratic", batch_size=12, n_epochs=60, seed=0):
    reward_fn = REWARD_FUNCTIONS[reward_name]

    # g2 in [0.05, 0.5], eps_d in [2.0, 6.0]
    bounds = np.array([
        [0.05, 0.5],
        [2.0, 6.0],
    ])
    x0 = np.array([0.2, 4.0])

    optimizer = SepCMA(
        mean=x0,
        sigma=0.1,
        bounds=bounds,
        population_size=batch_size,
        seed=seed,
    )

    history = {"loss": [], "Tx": [], "Tz": [], "eta": [], "params": []}

    for epoch in range(n_epochs):
        t0 = time.time()
        xs = np.array([optimizer.ask() for _ in range(optimizer.population_size)])

        losses, txs, tzs, etas = [], [], [], []
        for x in xs:
            g2, eps_d = float(x[0]), float(x[1])
            try:
                Tx, Tz = measure_Tx_Tz(g2, eps_d)
                eta = Tz / Tx
                loss = reward_fn(Tx, Tz)
            except Exception as e:
                print(f"  [!] sim failed: {e}")
                Tx, Tz, eta, loss = 0.0, 0.0, 0.0, 1e6
            losses.append(loss)
            txs.append(Tx)
            tzs.append(Tz)
            etas.append(eta)

        losses = np.array(losses)
        optimizer.tell([(xs[j], losses[j]) for j in range(len(xs))])

        history["loss"].append(float(np.mean(losses)))
        history["Tx"].append(float(np.mean(txs)))
        history["Tz"].append(float(np.mean(tzs)))
        history["eta"].append(float(np.mean(etas)))
        history["params"].append(optimizer.mean.copy())
        epoch_s = time.time() - t0

        if epoch % 10 == 0:
            print(
                f"[{reward_name}] Epoch {epoch:3d} | loss={np.mean(losses):.3f} "
                f"| Tx={np.mean(txs):.3f} Tz={np.mean(tzs):.1f} "
                f"| eta={np.mean(etas):.0f} "
                f"| g2={optimizer.mean[0]:.3f} eps_d={optimizer.mean[1]:.3f} "
                f"| {epoch_s:.2f}s"
            )

    # Final eval on optimizer mean
    best = optimizer.mean
    Tx, Tz = measure_Tx_Tz(float(best[0]), float(best[1]))
    eta = Tz / Tx

    print(f"\n{'='*50}")
    print(f"RESULT [{reward_name}]")
    print(f"{'='*50}")
    print(f"g_2   = {best[0]:.4f}")
    print(f"eps_d = {best[1]:.4f}")
    print(f"T_x   = {Tx:.4f} us")
    print(f"T_z   = {Tz:.1f} us")
    print(f"eta   = {eta:.1f}  (target: {ETA_TARGET})")

    return best, history, (Tx, Tz, eta)


# ── Plotting ─────────────────────────────────────────────────
def plot_results(all_results):
    """Plot convergence for all reward functions side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name, (_, hist, _) in all_results.items():
        epochs = np.arange(len(hist["loss"]))
        axes[0, 0].plot(epochs, hist["loss"], label=name)
        axes[0, 1].plot(epochs, hist["eta"], label=name)
        axes[1, 0].plot(epochs, hist["Tx"], label=f"{name} Tx")
        axes[1, 0].plot(epochs, hist["Tz"], label=f"{name} Tz", linestyle="--")
        params = np.array(hist["params"])
        axes[1, 1].plot(epochs, params[:, 0], label=f"{name} g₂")
        axes[1, 1].plot(epochs, params[:, 1], label=f"{name} ε_d", linestyle="--")

    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Convergence"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("η = T_z/T_x")
    axes[0, 1].set_title("Bias Ratio")
    axes[0, 1].axhline(ETA_TARGET, color="r", linestyle="--", alpha=0.5, label="target")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Lifetime (µs)")
    axes[1, 0].set_title("Lifetimes"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Parameter Value")
    axes[1, 1].set_title("Parameters"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Warming up JAX + dynamiqs...")
    _ = simulate(0.2, 4.0, "+z", 1.0, 10)
    print("Ready.\n")

    default_rewards = [os.getenv("CAT_REWARD", "log_quadratic")] if FAST_MODE else list(REWARD_FUNCTIONS.keys())
    default_epochs = int(os.getenv("CAT_EPOCHS", "20" if FAST_MODE else "60"))
    default_batch = int(os.getenv("CAT_BATCH_SIZE", "8" if FAST_MODE else "12"))

    print(
        f"Mode={'FAST' if FAST_MODE else 'FULL'} | rewards={default_rewards} "
        f"| epochs={default_epochs} | batch={default_batch} "
        f"| points={N_POINTS} | Tz_final={TZ_TFINAL}"
    )

    all_results = {}
    for name in default_rewards:
        print(f"\n{'='*60}")
        print(f"Optimizing with reward: {name}")
        print(f"{'='*60}")
        best, hist, metrics = optimize(
            reward_name=name,
            batch_size=default_batch,
            n_epochs=default_epochs,
        )
        all_results[name] = (best, hist, metrics)

    plot_results(all_results)
