"""
Cat Qubit Optimizer — Fast Batched Version

Tunes real-valued g_2 and eps_d (2 knobs) to maximize T_x, T_z
while keeping bias eta = T_z/T_x in the hundreds (~200-999).

Performance vs optimize_cat.py:
  - Batched mesolve: 2 calls/epoch instead of 24
  - Fast log-linear decay estimator (no scipy)
  - Reduced grid: 30 points, Tz window 100µs
  - Precomputed dense operators for batching
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import dynamiqs as dq
from cmaes import SepCMA
from matplotlib import pyplot as plt

# ── Physics constants ────────────────────────────────────────
NA, NB = 15, 5
DIM = NA * NB  # total Hilbert space dim = 75
KAPPA_B = 10.0
KAPPA_A = 1.0

# ── Objective ────────────────────────────────────────────────
ETA_TARGET = 500.0

# ── Lifetime clamps ──────────────────────────────────────────
TX_MAX = 5.0
TZ_MAX = 2000.0

# ── Simulation (reduced for speed) ───────────────────────────
N_POINTS = 30
TZ_TFINAL = 100.0
TX_TFINAL = 1.0

# ── Precomputed sparse operators (for single-sample simulate)
_a = dq.tensor(dq.destroy(NA), dq.eye(NB))
_b = dq.tensor(dq.eye(NA), dq.destroy(NB))
_loss_b = jnp.sqrt(KAPPA_B) * _b
_loss_a = jnp.sqrt(KAPPA_A) * _a
_fock_b0 = dq.fock(NB, 0)
_parity_op = dq.tensor(
    jnp.diag(jnp.array([(-1.0) ** n for n in range(NA)])),
    jnp.eye(NB),
)

# ── Precomputed DENSE operators (for batching) ───────────────
_aa_bd_d = jnp.array(_a @ _a @ _b.dag())        # a^2 b†
_adad_b_d = jnp.array(_a.dag() @ _a.dag() @ _b)  # (a†)^2 b
_bd_d = jnp.array(_b.dag())                       # b†
_b_dense_d = jnp.array(_b)                        # b

# Combined: H = g2 * _interact_d - eps_d * _drive_d
_interact_d = _aa_bd_d + _adad_b_d   # a^2 b† + (a†)^2 b
_drive_d = _bd_d + _b_dense_d         # b† + b

_loss_b_d = jnp.sqrt(KAPPA_B) * jnp.array(_b)
_loss_a_d = jnp.sqrt(KAPPA_A) * jnp.array(_a)
_parity_d = jnp.array(_parity_op)


# ── Fast decay estimator ─────────────────────────────────────
def fit_decay_fast(t, y):
    """Fast tau estimator via log-linear regression on y = A*exp(-t/tau) + C.

    Estimates C from last value, then log-linear fit on (y - C).
    Falls back to heuristic if log fails.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    C_est = float(y[-1])
    y_shifted = y - C_est
    y_shifted = np.clip(y_shifted, 1e-10, None)

    if y_shifted[0] < 1e-6:
        return max(float(np.ptp(t)) / 3, 1e-6)

    try:
        mask = y_shifted > 1e-8
        if np.sum(mask) < 3:
            return max(float(np.ptp(t)) / 3, 1e-6)
        log_y = np.log(y_shifted[mask])
        slope, _ = np.polyfit(t[mask], log_y, 1)
        if slope >= 0:
            return max(float(np.ptp(t)) / 3, 1e-6)
        tau = -1.0 / slope
        tau = np.clip(tau, 1e-6, float(t[-1]) * 10)
        return float(tau)
    except Exception:
        return max(float(np.ptp(t)) / 3, 1e-6)


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
        exp_ops=[_parity_op, Z_L],
        options=dq.Options(progress_meter=False),
    )
    return np.array(res.tsave), np.array(res.expects[0].real), np.array(res.expects[1].real)


# ── Batched simulation ───────────────────────────────────────
def simulate_batched(g2_arr, epsd_arr, init_state, tfinal, n_points=N_POINTS):
    """Batched mesolve for B samples. 1 call replaces B sequential calls.

    Returns (tsave, X_expects[B, T], Z_expects[B, T]).
    """
    B = len(g2_arr)
    g2_arr = np.asarray(g2_arr, dtype=float)
    epsd_arr = np.asarray(epsd_arr, dtype=float)

    g2_batch = jnp.array(g2_arr)[:, None, None]
    epsd_batch = jnp.array(epsd_arr)[:, None, None]

    # H_batch shape: (B, DIM, DIM)
    H_batch = g2_batch * _interact_d - epsd_batch * _drive_d

    psi0_list = []
    Z_L_list = []
    for i in range(B):
        alpha = compute_alpha(float(g2_arr[i]), float(epsd_arr[i]))
        cat_p = jnp.array(dq.coherent(NA, alpha))
        cat_m = jnp.array(dq.coherent(NA, -alpha))

        if init_state == "+z":
            psi0_a = cat_p
        elif init_state == "+x":
            psi0_a = (cat_p + cat_m) / jnp.sqrt(2)
        else:
            psi0_a = cat_m

        fock_b0 = jnp.array(dq.fock(NB, 0))
        psi0_list.append(jnp.kron(psi0_a, fock_b0))

        Z_a = cat_p @ cat_p.conj().T - cat_m @ cat_m.conj().T
        Z_L_list.append(jnp.kron(Z_a, jnp.eye(NB)))

    psi0_batch = jnp.stack(psi0_list)   # (B, DIM, 1)
    Z_L_batch = jnp.stack(Z_L_list)     # (B, DIM, DIM)

    tsave = jnp.linspace(0, tfinal, n_points)

    res = dq.mesolve(
        H_batch, [_loss_b_d, _loss_a_d], psi0_batch, tsave,
        exp_ops=[_parity_d, Z_L_batch],
        options=dq.Options(progress_meter=False),
    )

    # res.expects shape: (n_exp_ops, B, T)
    X_exp = np.array(res.expects[0].real)  # (B, T)
    Z_exp = np.array(res.expects[1].real)  # (B, T)
    return np.array(tsave), X_exp, Z_exp


# ── Measurement ──────────────────────────────────────────────
def measure_Tx_Tz(g2, eps_d):
    """Single-sample lifetime measurement."""
    tz_t, _, sz = simulate(g2, eps_d, "+z", TZ_TFINAL)
    Tz = min(fit_decay_fast(tz_t, sz), TZ_MAX)

    tx_t, sx, _ = simulate(g2, eps_d, "+x", TX_TFINAL)
    Tx = min(fit_decay_fast(tx_t, sx), TX_MAX)

    return max(Tx, 1e-6), max(Tz, 1e-6)


def batch_measure_Tx_Tz(g2_arr, epsd_arr):
    """Batched: 2 mesolve calls for B samples → (Tx_arr, Tz_arr)."""
    B = len(g2_arr)

    # +z init → fit Tz from Z_L decay
    tsave_z, _, Z_exp = simulate_batched(g2_arr, epsd_arr, "+z", TZ_TFINAL)
    Tz_arr = np.array([
        max(min(fit_decay_fast(tsave_z, Z_exp[i]), TZ_MAX), 1e-6)
        for i in range(B)
    ])

    # +x init → fit Tx from parity (X_L) decay
    tsave_x, X_exp, _ = simulate_batched(g2_arr, epsd_arr, "+x", TX_TFINAL)
    Tx_arr = np.array([
        max(min(fit_decay_fast(tsave_x, X_exp[i]), TX_MAX), 1e-6)
        for i in range(B)
    ])

    return Tx_arr, Tz_arr


# ── Reward functions (all return loss to MINIMIZE) ───────────
def reward_log_quadratic(Tx, Tz, a=1.0, b=1.0, c=2.0):
    """-(a*log(Tx) + b*log(Tz)) + c*(log(eta/target))^2"""
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * (np.log(eta / ETA_TARGET)) ** 2
    return lifetime_term + eta_penalty


def reward_log_exp(Tx, Tz, a=1.0, b=1.0, c=1.0, k=3.0):
    """-(a*log(Tx) + b*log(Tz)) + c*exp(k*|eta/target - 1|)"""
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * np.exp(k * abs(eta / ETA_TARGET - 1))
    return lifetime_term + eta_penalty


def reward_mixed_log(Tx, Tz, a=1.0, b=1.0, c=3.0):
    """-(a*log(Tx) + b*log(Tz)) + c*|log(eta/target)|"""
    eta = Tz / Tx
    lifetime_term = -(a * np.log(Tx) + b * np.log(Tz))
    eta_penalty = c * abs(np.log(eta / ETA_TARGET))
    return lifetime_term + eta_penalty


REWARD_FUNCTIONS = {
    "log_quadratic": reward_log_quadratic,
    "log_exp": reward_log_exp,
    "mixed_log": reward_mixed_log,
}


# ── CMA-ES Optimization (batched) ───────────────────────────
def optimize(reward_name="log_quadratic", batch_size=12, n_epochs=60, seed=0):
    reward_fn = REWARD_FUNCTIONS[reward_name]

    optimizer = SepCMA(
        mean=np.array([0.2, 4.0]),
        sigma=0.1,
        bounds=np.array([[0.05, 0.5], [2.0, 6.0]]),
        population_size=batch_size,
        seed=seed,
    )

    history = {"loss": [], "Tx": [], "Tz": [], "eta": [], "params": [], "epoch_time": []}

    for epoch in range(n_epochs):
        t0 = time.time()

        xs = np.array([optimizer.ask() for _ in range(optimizer.population_size)])
        g2_arr = xs[:, 0]
        epsd_arr = xs[:, 1]

        try:
            Tx_arr, Tz_arr = batch_measure_Tx_Tz(g2_arr, epsd_arr)
        except Exception as e:
            print(f"  [!] batch failed ({e}), falling back to sequential")
            Tx_arr = np.zeros(len(xs))
            Tz_arr = np.zeros(len(xs))
            for i in range(len(xs)):
                Tx_arr[i], Tz_arr[i] = measure_Tx_Tz(float(g2_arr[i]), float(epsd_arr[i]))

        losses = np.array([reward_fn(Tx_arr[i], Tz_arr[i]) for i in range(len(xs))])
        etas = Tz_arr / Tx_arr

        optimizer.tell([(xs[j], losses[j]) for j in range(len(xs))])

        epoch_time = time.time() - t0

        history["loss"].append(float(np.mean(losses)))
        history["Tx"].append(float(np.mean(Tx_arr)))
        history["Tz"].append(float(np.mean(Tz_arr)))
        history["eta"].append(float(np.mean(etas)))
        history["params"].append(optimizer.mean.copy())
        history["epoch_time"].append(epoch_time)

        if epoch % 10 == 0:
            print(
                f"[{reward_name}] Epoch {epoch:3d} | loss={np.mean(losses):.3f} "
                f"| Tx={np.mean(Tx_arr):.3f} Tz={np.mean(Tz_arr):.1f} "
                f"| eta={np.mean(etas):.0f} "
                f"| g2={optimizer.mean[0]:.3f} eps_d={optimizer.mean[1]:.3f} "
                f"| {epoch_time:.2f}s"
            )

    # Final eval on optimizer mean
    best = optimizer.mean
    Tx, Tz = measure_Tx_Tz(float(best[0]), float(best[1]))
    eta = Tz / Tx

    total_time = sum(history["epoch_time"])
    print(f"\n{'='*50}")
    print(f"RESULT [{reward_name}]  (total: {total_time:.1f}s)")
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

    first_hist = list(all_results.values())[0][1]
    total = sum(first_hist["epoch_time"])

    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"Loss Convergence ({total:.0f}s total)")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("η = T_z/T_x")
    axes[0, 1].set_title("Bias Ratio")
    axes[0, 1].axhline(ETA_TARGET, color="r", linestyle="--", alpha=0.5, label="target")
    axes[0, 1].axhline(200, color="gray", linestyle=":", alpha=0.3)
    axes[0, 1].axhline(999, color="gray", linestyle=":", alpha=0.3)
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Lifetime (µs)")
    axes[1, 0].set_title("Lifetimes"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Parameter Value")
    axes[1, 1].set_title("Parameters"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Cat Qubit Optimizer — Fast Batched Version")
    print("=" * 60)
    print(f"Hilbert space: {NA}×{NB} = {DIM}")
    print(f"Grid: {N_POINTS} points, Tz window: {TZ_TFINAL}µs, Tx window: {TX_TFINAL}µs")
    print()

    print("Warming up JAX + dynamiqs...")
    t0 = time.time()
    _ = simulate(0.2, 4.0, "+z", 1.0, 10)
    print(f"Warmup done in {time.time() - t0:.1f}s\n")

    all_results = {}
    for name in REWARD_FUNCTIONS:
        print(f"\n{'='*60}")
        print(f"Optimizing with reward: {name}")
        print(f"{'='*60}")
        best, hist, metrics = optimize(reward_name=name, n_epochs=60)
        all_results[name] = (best, hist, metrics)

    plot_results(all_results)
