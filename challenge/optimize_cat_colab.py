# %%
# ============================================================
#  Cell 1: Install dependencies & configure JAX for GPU
# ============================================================
# Install dynamiqs + deps, then reinstall JAX with matching CUDA 12 support
!pip install -q "dynamiqs>=0.3.0" cmaes scipy
!pip install -q -U "jax[cuda12]>=0.4.36,<0.7"

import jax
try:
    # Verify GPU is usable
    _ = jax.devices("gpu")
    print("JAX GPU backend:", jax.devices())
except RuntimeError:
    print("WARNING: GPU not available, falling back to CPU")
    print("JAX CPU backend:", jax.devices())

# %%
# ============================================================
#  Cell 2: Imports & constants
# ============================================================
"""
Simple Cat-Qubit Lifetime Optimizer (Colab GPU version)

Tunes complex g_2 and eps_d (4 real knobs) to:
  - Maximize T_x and T_z
  - Keep bias eta = T_z / T_x ≈ 320

Loss: -(a * log(T_x) + b * log(T_z)) + c * (eta/320 - 1)^2
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
A_COEFF = 1.0           # weight on log(T_x)
B_COEFF = 1.0           # weight on log(T_z)
C_COEFF = 5.0           # penalty weight on (eta/eta_target - 1)^2

# ── Lifetime clamps (suppress fit artifacts) ─────────────────
TX_MAX = 5.0     # us — physical cap for bit-flip time
TZ_MAX = 2000.0  # us — physical cap for phase-flip time

# %%
# ============================================================
#  Cell 3: Exponential fit helper
# ============================================================
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

# %%
# ============================================================
#  Cell 4: Simulation (runs on GPU via JAX/dynamiqs)
# ============================================================
def simulate(g2_complex, eps_d_complex, init_state, tfinal, n_points=100):
    """Run mesolve and return (t, <X_L>, <Z_L>)."""
    a = dq.tensor(dq.destroy(NA), dq.eye(NB))
    b = dq.tensor(dq.eye(NA), dq.destroy(NB))

    kappa2 = 4 * abs(g2_complex) ** 2 / KAPPA_B
    eps2 = 2 * g2_complex * eps_d_complex / KAPPA_B

    if kappa2 > 1e-12:
        alpha = float(np.sqrt(max(2 / kappa2 * (abs(eps2) - KAPPA_A / 4), 0.01)))
    else:
        alpha = 0.5

    # Hamiltonian
    H = (np.conj(g2_complex) * a @ a @ b.dag()
         + g2_complex * a.dag() @ a.dag() @ b
         - eps_d_complex * b.dag()
         - np.conj(eps_d_complex) * b)

    # Cat basis states
    cat_p = dq.coherent(NA, alpha)
    cat_m = dq.coherent(NA, -alpha)
    basis = {
        "+z": cat_p,
        "-z": cat_m,
        "+x": (cat_p + cat_m) / jnp.sqrt(2),
    }

    # Logical operators
    X_L = dq.tensor(
        jnp.diag(jnp.array([(-1.0) ** n for n in range(NA)])),
        jnp.eye(NB),
    )
    Z_L = dq.tensor(
        cat_p @ cat_p.dag() - cat_m @ cat_m.dag(),
        dq.eye(NB),
    )

    psi0 = dq.tensor(basis[init_state], dq.fock(NB, 0))
    tsave = jnp.linspace(0, tfinal, n_points)

    res = dq.mesolve(
        H,
        [jnp.sqrt(KAPPA_B) * b, jnp.sqrt(KAPPA_A) * a],
        psi0,
        tsave,
        exp_ops=[X_L, Z_L],
        options=dq.Options(progress_meter=False),
    )
    return np.array(res.tsave), np.array(res.expects[0].real), np.array(res.expects[1].real)


def measure_Tx_Tz(g2_complex, eps_d_complex):
    """Two simulations → (T_x, T_z)."""
    tz_t, _, sz = simulate(g2_complex, eps_d_complex, "+z", tfinal=200.0)
    Tz = fit_decay(tz_t, sz)

    tx_t, sx, _ = simulate(g2_complex, eps_d_complex, "+x", tfinal=1.0)
    Tx = fit_decay(tx_t, sx)

    return Tx, Tz

# %%
# ============================================================
#  Cell 5: Loss function
# ============================================================
def loss_fn(params):
    """
    params: [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]
    Loss = -(a*log(Tx) + b*log(Tz)) + c*(eta/eta_target - 1)^2
    """
    g2 = complex(params[0], params[1])
    eps_d = complex(params[2], params[3])
    try:
        Tx, Tz = measure_Tx_Tz(g2, eps_d)
        Tx = float(np.clip(Tx, 1e-6, TX_MAX))
        Tz = float(np.clip(Tz, 1e-6, TZ_MAX))
        eta = Tz / Tx
        reward = A_COEFF * np.log(Tx) + B_COEFF * np.log(Tz)
        penalty = C_COEFF * (eta / ETA_TARGET - 1.0) ** 2
        return float(-reward + penalty), Tx, Tz, eta
    except Exception as e:
        print(f"  [!] sim failed: {e}")
        return 1e6, 0.0, 0.0, 0.0

# %%
# ============================================================
#  Cell 6: Warm up JAX (first mesolve compiles XLA kernels)
# ============================================================
print("Warming up JAX + dynamiqs on GPU (first call compiles)...")
_ = simulate(1.0+0j, 4.0+0j, "+z", tfinal=1.0, n_points=10)
print("Done! Subsequent calls will be fast.")

# %%
# ============================================================
#  Cell 7: Run optimization
# ============================================================
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
        xs = np.array([optimizer.ask() for _ in range(optimizer.population_size)])
        results = [loss_fn(x) for x in xs]
        losses = np.array([r[0] for r in results])
        txs = np.array([r[1] for r in results])
        tzs = np.array([r[2] for r in results])
        etas = np.array([r[3] for r in results])

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

best_params, (Tx, Tz, eta) = optimize()
