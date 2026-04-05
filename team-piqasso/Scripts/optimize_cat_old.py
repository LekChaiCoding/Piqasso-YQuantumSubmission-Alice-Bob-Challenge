"""
Simple Cat-Qubit Lifetime Optimizer

Tunes complex g_2 and eps_d (4 real knobs) to:
  - Maximize T_x and T_z
  - Keep bias eta = T_z / T_x ≈ 320

Loss: -(a * log(T_x) + b * log(T_z)) + c * exp(k * |eta/320 - 1|)
"""

import numpy as np
import jax.numpy as jnp
import dynamiqs as dq
from scipy.optimize import least_squares
from cmaes import SepCMA
from matplotlib import pyplot as plt

# ── Physics constants ────────────────────────────────────────
NA, NB = 15, 5
KAPPA_B = 10.0   # buffer decay [MHz]
KAPPA_A = 1.0    # single-photon loss [MHz]

# ── Objective ────────────────────────────────────────────────
ETA_TARGET = 320.0
A_COEFF = 4.0   # weight on log(T_x)  — upweights bit-flip since log(Tx)≈−1 vs log(Tz)≈5
B_COEFF = 1.0   # weight on log(T_z)
C_COEFF = 2.0   # penalty prefactor   — exp penalty dominates when η drifts >30% from target
K_ETA   = 5.0   # exponential steepness on η deviation

# ── Lifetime clamps (suppress fit artifacts) ─────────────────
TX_MAX = 5.0     # us — physical cap for bit-flip time
TZ_MAX = 2000.0  # us — physical cap for phase-flip time

# ── Simulation tuning ────────────────────────────────────────
N_POINTS = 50       # time steps per sim (was 100 — 50 is enough for exp fit)
TZ_TFINAL = 100.0   # us — phase-flip window (was 200 — shorter is fine)
TX_TFINAL = 1.0     # us — bit-flip window

# ── Precomputed static operators (built once, reused) ────────
_a_op = dq.tensor(dq.destroy(NA), dq.eye(NB))
_b_op = dq.tensor(dq.eye(NA), dq.destroy(NB))
_parity = dq.tensor(
    jnp.diag(jnp.array([(-1.0) ** n for n in range(NA)])),
    jnp.eye(NB),
)
_loss_b = jnp.sqrt(KAPPA_B) * _b_op
_loss_a = jnp.sqrt(KAPPA_A) * _a_op
_fock_b0 = dq.fock(NB, 0)

# Precompute operator products (avoid recomputing every call)
_aa = _a_op @ _a_op                    # a^2
_adad = _a_op.dag() @ _a_op.dag()      # (a†)^2
_bd = _b_op.dag()                       # b†
_aa_bd = _aa @ _bd                      # a^2 b†
_adad_b = _adad @ _b_op                 # (a†)^2 b

# Dense JAX arrays for batched broadcasting (SparseDIAQArray can't be jnp.stacked)
_aa_bd_d = jnp.array(_aa_bd)
_adad_b_d = jnp.array(_adad_b)
_bd_d = jnp.array(_bd)
_b_d = jnp.array(_b_op)
_parity_d = jnp.array(_parity)
_loss_b_d = jnp.array(_loss_b)
_loss_a_d = jnp.array(_loss_a)


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
def _compute_alpha(g2_complex, eps_d_complex):
    """Estimate cat size from parameters."""
    kappa2 = 4 * abs(g2_complex) ** 2 / KAPPA_B
    eps2 = 2 * g2_complex * eps_d_complex / KAPPA_B
    if kappa2 > 1e-12:
        return float(np.sqrt(max(2 / kappa2 * (abs(eps2) - KAPPA_A / 4), 0.01)))
    return 0.5


def simulate(g2_complex, eps_d_complex, init_state, tfinal, n_points=N_POINTS):
    """Run mesolve and return (t, <X_L>, <Z_L>). Uses precomputed operators."""
    alpha = _compute_alpha(g2_complex, eps_d_complex)

    H = (np.conj(g2_complex) * _aa_bd
         + g2_complex * _adad_b
         - eps_d_complex * _bd
         - np.conj(eps_d_complex) * _b_op)

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


def simulate_batched(g2_arr, epsd_arr, init_state, tfinal, n_points=N_POINTS):
    """
    Batched mesolve: run one solver call for all (g2, eps_d) pairs.
    g2_arr, epsd_arr: 1-D arrays of complex values, length B.
    Returns (tsave, X_expects[B, T], Z_expects[B, T]).
    """
    B = len(g2_arr)
    g2 = jnp.array(g2_arr, dtype=jnp.complex64)
    epsd = jnp.array(epsd_arr, dtype=jnp.complex64)

    # Build batched Hamiltonian via broadcasting: (B, 1, 1) * (dim, dim) → (B, dim, dim)
    H_batch = (jnp.conj(g2)[:, None, None] * _aa_bd_d
               + g2[:, None, None] * _adad_b_d
               - epsd[:, None, None] * _bd_d
               - jnp.conj(epsd)[:, None, None] * _b_d)

    # Per-sample alpha, cat states, Z_L, psi0 (alpha varies per sample)
    psi0_list = []
    Z_L_list = []
    for i in range(B):
        alpha = _compute_alpha(complex(g2_arr[i]), complex(epsd_arr[i]))
        cat_p = jnp.array(dq.coherent(NA, alpha))    # (NA, 1)
        cat_m = jnp.array(dq.coherent(NA, -alpha))
        if init_state == "+z":
            psi0_a = cat_p
        elif init_state == "+x":
            psi0_a = (cat_p + cat_m) / jnp.sqrt(2)
        else:
            psi0_a = cat_m

        fock_b0 = jnp.array(dq.fock(NB, 0))
        psi0_list.append(jnp.kron(psi0_a, fock_b0))
        Z_a = cat_p @ cat_p.conj().T - cat_m @ cat_m.conj().T  # (NA, NA)
        Z_L_list.append(jnp.kron(Z_a, jnp.eye(NB)))

    psi0_batch = jnp.stack(psi0_list)    # (B, dim, 1)
    Z_L_batch = jnp.stack(Z_L_list)      # (B, dim, dim)

    tsave = jnp.linspace(0, tfinal, n_points)

    res = dq.mesolve(
        H_batch, [_loss_b_d, _loss_a_d], psi0_batch, tsave,
        exp_ops=[_parity_d, Z_L_batch],
        options=dq.Options(progress_meter=False),
    )
    # res.expects shape: (n_exp_ops, B, T)
    X_exp = np.array(res.expects[0].real)
    Z_exp = np.array(res.expects[1].real)
    return np.array(tsave), X_exp, Z_exp


def measure_Tx_Tz(g2_complex, eps_d_complex):
    """Two simulations → (T_x, T_z). Single-sample convenience wrapper."""
    tz_t, _, sz = simulate(g2_complex, eps_d_complex, "+z", tfinal=TZ_TFINAL)
    Tz = fit_decay(tz_t, sz)
    tx_t, sx, _ = simulate(g2_complex, eps_d_complex, "+x", tfinal=TX_TFINAL)
    Tx = fit_decay(tx_t, sx)
    return Tx, Tz


def batch_measure_Tx_Tz(g2_arr, epsd_arr):
    """
    Batched lifetime measurement: 2 mesolve calls for all B samples.
    Returns (Tx_arr, Tz_arr) each of length B.
    """
    # Batch Tz sims (+z init, long window)
    tz_t, _, Z_exp = simulate_batched(g2_arr, epsd_arr, "+z", TZ_TFINAL)
    Tz_arr = np.array([fit_decay(tz_t, Z_exp[i]) for i in range(len(g2_arr))])

    # Batch Tx sims (+x init, short window)
    tx_t, X_exp, _ = simulate_batched(g2_arr, epsd_arr, "+x", TX_TFINAL)
    Tx_arr = np.array([fit_decay(tx_t, X_exp[i]) for i in range(len(g2_arr))])

    return Tx_arr, Tz_arr


# ── Loss function ────────────────────────────────────────────
def loss_fn(params):
    """Single-sample loss for debugging."""
    g2 = complex(params[0], params[1])
    eps_d = complex(params[2], params[3])
    try:
        Tx, Tz = measure_Tx_Tz(g2, eps_d)
        Tx = float(np.clip(Tx, 1e-6, TX_MAX))
        Tz = float(np.clip(Tz, 1e-6, TZ_MAX))
        eta = Tz / Tx
        reward = A_COEFF * np.log(Tx) + B_COEFF * np.log(Tz)
        penalty = C_COEFF * np.exp(K_ETA * abs(eta / ETA_TARGET - 1.0))
        return float(-reward + penalty), Tx, Tz, eta
    except Exception as e:
        print(f"  [!] sim failed: {e}")
        return 1e6, 0.0, 0.0, 0.0


def batch_loss_from_params(xs):
    """
    Batched loss: 2 mesolve calls for the entire population.
    xs: (B, 4) array — [Re(g2), Im(g2), Re(eps_d), Im(eps_d)] per row.
    Returns list of (loss, Tx, Tz, eta) tuples.
    """
    xs = np.asarray(xs, float)
    g2_arr = xs[:, 0] + 1j * xs[:, 1]
    epsd_arr = xs[:, 2] + 1j * xs[:, 3]

    try:
        Tx_arr, Tz_arr = batch_measure_Tx_Tz(g2_arr, epsd_arr)
    except Exception as e:
        print(f"  [!] batched sim failed: {e}")
        return [(1e6, 0.0, 0.0, 0.0)] * len(xs)

    results = []
    for i in range(len(xs)):
        Tx = float(np.clip(Tx_arr[i], 1e-6, TX_MAX))
        Tz = float(np.clip(Tz_arr[i], 1e-6, TZ_MAX))
        eta = Tz / Tx
        reward = A_COEFF * np.log(Tx) + B_COEFF * np.log(Tz)
        penalty = C_COEFF * np.exp(K_ETA * abs(eta / ETA_TARGET - 1.0))
        results.append((float(-reward + penalty), Tx, Tz, eta))
    return results


# ── CMA-ES Optimization ─────────────────────────────────────
def optimize(batch_size=12, n_epochs=60, seed=0):
    bounds = np.array([
        [0.1, 5.0],    # Re(g2)
        [-2.0, 2.0],   # Im(g2)
        [1.0, 20.0],   # Re(eps_d)
        [-5.0, 5.0],   # Im(eps_d)
    ])
    x0 = np.array([1.0, 0.0, 4.0, 0.0])

    optimizer = SepCMA(
        mean=x0,
        sigma=0.3,
        bounds=bounds,
        population_size=batch_size,
        seed=seed,
    )

    loss_history = []
    mean_history = []
    tx_history = []
    tz_history = []
    eta_history = []

    for epoch in range(n_epochs):
        # Sample and evaluate — 2 batched mesolve calls for entire population
        xs = np.array([optimizer.ask() for _ in range(optimizer.population_size)])
        results = batch_loss_from_params(xs)
        losses = np.array([r[0] for r in results])
        txs = np.array([r[1] for r in results])
        tzs = np.array([r[2] for r in results])
        etas = np.array([r[3] for r in results])

        # Tell optimizer
        optimizer.tell([(xs[j], losses[j]) for j in range(len(xs))])

        avg_loss = float(np.mean(losses))
        loss_history.append(avg_loss)
        mean_history.append(optimizer.mean.copy())
        tx_history.append(float(np.mean(txs)))
        tz_history.append(float(np.mean(tzs)))
        eta_history.append(float(np.mean(etas)))

        if epoch % 5 == 0:
            best_idx = np.argmin(losses)
            print(
                f"Epoch {epoch:3d} | loss={avg_loss:+.3f} "
                f"| best={losses[best_idx]:+.3f} "
                f"| Tx={np.mean(txs):.3f} Tz={np.mean(tzs):.1f} "
                f"| eta={np.mean(etas):.0f} "
                f"| mean={np.round(optimizer.mean, 3)}"
            )

    # Final evaluation
    best_params = optimizer.mean
    g2_best = complex(best_params[0], best_params[1])
    eps_d_best = complex(best_params[2], best_params[3])
    Tx, Tz = measure_Tx_Tz(g2_best, eps_d_best)
    eta = Tz / Tx

    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULT")
    print("=" * 50)
    print(f"g_2   = {g2_best:.4f}")
    print(f"eps_d = {eps_d_best:.4f}")
    print(f"T_x   = {Tx:.4f} us")
    print(f"T_z   = {Tz:.4f} us")
    print(f"eta   = {eta:.1f}  (target: {ETA_TARGET})")

    # ── Plots ────────────────────────────────────────────────
    epochs = np.arange(len(loss_history))
    mean_history = np.array(mean_history)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, loss_history)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss vs Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    for i, label in enumerate(["Re(g₂)", "Im(g₂)", "Re(ε_d)", "Im(ε_d)"]):
        axes[0, 1].plot(epochs, mean_history[:, i], label=label)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Parameter value")
    axes[0, 1].set_title("Parameter Convergence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, tx_history, label="T_x (µs)")
    axes[1, 0].plot(epochs, tz_history, label="T_z (µs)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Lifetime (µs)")
    axes[1, 0].set_title("Lifetimes vs Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, eta_history, label="η = T_z / T_x")
    axes[1, 1].axhline(ETA_TARGET, color="r", linestyle="--", label=f"Target η = {ETA_TARGET}")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("η")
    axes[1, 1].set_title("Bias Ratio vs Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return best_params, (Tx, Tz, eta)


if __name__ == "__main__":
    optimize()
