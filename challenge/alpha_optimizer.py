"""
Alpha Maximizer for Dissipative Cat Qubits
==========================================
Strategy (following the outlined framework):

PRIMARY REWARD  (a² estimator)
  m2(θ)        = Tr(ρ_θ a²)
  α̂(θ)        = sqrt(|m2|)
  χ²(θ)        = Tr(ρ_θ (a²−m2)†(a²−m2)) = ⟨a†²a²⟩ − |m2|²   [eigenspace quality]
  L(θ)         = ⟨n_b⟩                                          [buffer leakage]
  R(θ)         = α̂(θ) − λ·χ²(θ) − μ·L(θ)

VALIDATION  (cat-state overlap)
  |C±(β)⟩     = N±(β) (|β⟩ ± |−β⟩)
  S_cat(β; θ) = ⟨C+(β)|ρ_θ|C+(β)⟩ + ⟨C−(β)|ρ_θ|C−(β)⟩
  α̂_cat(θ)   = argmax_β S_cat(β; θ)

OPTIMIZER: CMA-ES (warm-started from analytic adiabatic-elimination model)

Controls: θ = (Re g₂, Im g₂, Re ε_d, Im ε_d)  [MHz]
"""

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA
import matplotlib.pyplot as plt

# ── JAX configuration ──────────────────────────────────────────────────────────
jax.config.update("jax_enable_x64", True)

# ── System parameters (fixed, not tunable online) ──────────────────────────────
na = 15          # storage Fock space truncation
nb = 5           # buffer Fock space truncation
kappa_b = 10.0   # MHz, buffer single-photon decay rate
kappa_a = 1.0    # MHz, storage single-photon loss rate

# ── Penalty weights ────────────────────────────────────────────────────────────
LAMBDA_CHI2 = 0.3   # eigenspace-quality penalty  (χ²)
MU_LEAK     = 0.1   # buffer-leakage penalty       (⟨n_b⟩)

# ── CMA-ES hyperparameters ─────────────────────────────────────────────────────
BATCH_SIZE = 12
N_EPOCHS   = 100
SIGMA0     = 0.5    # initial CMA step size (MHz)

# Control bounds: [Re g₂, Im g₂, Re ε_d, Im ε_d]  (MHz)
BOUNDS = np.array([
    [-5.0,  5.0],   # Re g₂
    [-5.0,  5.0],   # Im g₂
    [-10.0, 10.0],  # Re ε_d
    [-10.0, 10.0],  # Im ε_d
])

# ── Operators (full na×nb space) ───────────────────────────────────────────────
a_s   = dq.destroy(na)                           # storage annihilation (subspace)
a_op  = dq.tensor(a_s, dq.eye(nb))               # full-space storage a
b_op  = dq.tensor(dq.eye(na), dq.destroy(nb))    # full-space buffer b

a2_full  = a_op @ a_op                           # a²  (full space)
a4_full  = dq.dag(a2_full) @ a2_full             # a†²a²  (full space)
nb_full  = dq.dag(b_op) @ b_op                   # buffer number operator

# Storage-subspace a² (for cat-overlap validation)
a2_s = a_s @ a_s


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ANALYTIC WARM START
# ══════════════════════════════════════════════════════════════════════════════

def analytic_alpha(theta: np.ndarray) -> float:
    """
    Cat size from adiabatic elimination (real-parameter approximation):
      ε₂  = 2 g₂ ε_d / κ_b
      κ₂  = 4|g₂|² / κ_b
      α   = sqrt(|ε₂|/κ₂ − κ_a/(4κ₂))
    """
    g2_re, g2_im, eps_d_re, eps_d_im = theta
    g2    = complex(g2_re, g2_im)
    eps_d = complex(eps_d_re, eps_d_im)
    kappa_2 = 4.0 * abs(g2)**2 / kappa_b
    if kappa_2 < 1e-12:
        return 0.0
    eps_2   = 2.0 * g2 * eps_d / kappa_b
    inner   = abs(eps_2) / kappa_2 - kappa_a / (4.0 * kappa_2)
    return float(np.sqrt(max(inner, 0.0)))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STEADY-STATE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def get_steady_state_expects(theta: np.ndarray,
                              t_steady: float = 3.0,
                              n_pts: int = 5) -> dict:
    """
    Evolve the Lindblad master equation to near-steady state and return
    the expectation values needed for the reward.

    Returns dict with keys: 'm2', 'a4', 'nb', 'rho'
      m2  = ⟨a²⟩         (complex)
      a4  = ⟨a†²a²⟩      (real)
      nb  = ⟨n_b⟩         (real)
      rho = final density matrix (full space)
    """
    g2_re, g2_im, eps_d_re, eps_d_im = theta
    g2    = complex(g2_re, g2_im)
    eps_d = complex(eps_d_re, eps_d_im)

    # Hamiltonian: H = conj(g₂) a†²b + g₂ a²b† − ε_d b† − conj(ε_d) b
    H = (jnp.conj(g2) * dq.dag(a_op) @ dq.dag(a_op) @ b_op
         + g2 * a2_full @ dq.dag(b_op)
         - eps_d * dq.dag(b_op)
         - jnp.conj(eps_d) * b_op)

    # Jump operators
    L_b = jnp.sqrt(kappa_b) * b_op
    L_a = jnp.sqrt(kappa_a) * a_op

    # Initial state: vacuum ⊗ vacuum
    psi0 = dq.tensor(dq.fock(na, 0), dq.fock(nb, 0))
    rho0 = dq.todm(psi0)

    # Time grid (only need last point, but solvers need at least 2)
    tsave = jnp.linspace(0.0, t_steady, n_pts)

    result = dq.mesolve(
        H, [L_b, L_a], rho0, tsave,
        exp_ops=[a2_full, a4_full, nb_full],
    )

    # result.expects shape: [n_ops, n_times]
    m2 = complex(result.expects[0, -1])
    a4 = float(jnp.real(result.expects[1, -1]))
    nb = float(jnp.real(result.expects[2, -1]))
    rho_final = result.states[-1]

    return {'m2': m2, 'a4': a4, 'nb': nb, 'rho': rho_final}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  REWARD  R_a²(θ)
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(theta: np.ndarray,
                   lambda_chi2: float = LAMBDA_CHI2,
                   mu_leak: float = MU_LEAK,
                   t_steady: float = 3.0) -> tuple:
    """
    R(θ) = α̂ − λ·χ² − μ·⟨n_b⟩

    where
      α̂  = sqrt(|⟨a²⟩|)
      χ²  = ⟨a†²a²⟩ − |⟨a²⟩|²     (eigenspace variance of a²)
      ⟨n_b⟩                         (buffer leakage)

    Returns (reward, alpha_hat, chi2, n_b_mean)
    """
    ex = get_steady_state_expects(theta, t_steady=t_steady)
    m2   = ex['m2']
    a4   = ex['a4']
    nb   = ex['nb']

    alpha_hat = float(np.sqrt(abs(m2)))                  # α̂ = sqrt(|⟨a²⟩|)
    chi2      = float(max(a4 - abs(m2)**2, 0.0))         # χ² ≥ 0 by construction
    reward    = alpha_hat - lambda_chi2 * chi2 - mu_leak * nb

    return reward, alpha_hat, chi2, nb


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CAT-STATE OVERLAP VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def cat_ket(n: int, beta: float, sign: int = 1):
    """
    Normalized even (sign=+1) or odd (sign=-1) cat state:
      |C±(β)⟩ = N±(β)(|β⟩ ± |−β⟩),  N±(β) = 1/sqrt(2(1 ± exp(−2|β|²)))
    """
    ket_p = dq.coherent(n, beta)
    ket_m = dq.coherent(n, -beta)
    unnorm = ket_p + sign * ket_m
    norm_sq = 2.0 * (1.0 + sign * float(np.exp(-2.0 * beta**2)))
    return unnorm / jnp.sqrt(norm_sq)


def cat_overlap_score(rho_a, beta: float) -> float:
    """
    S_cat(β) = ⟨C+(β)|ρ_a|C+(β)⟩ + ⟨C−(β)|ρ_a|C−(β)⟩

    This is the fraction of ρ_a that lives in the cat manifold of size β.
    """
    cp = cat_ket(na, beta, sign=+1)
    cm = cat_ket(na, beta, sign=-1)
    # ⟨ψ|ρ|ψ⟩ = Tr(|ψ⟩⟨ψ| ρ) = dq.expect(projector, rho)
    F_plus  = float(jnp.real(dq.expect(dq.todm(cp), rho_a)))
    F_minus = float(jnp.real(dq.expect(dq.todm(cm), rho_a)))
    return F_plus + F_minus


def validate_cat_size(theta: np.ndarray,
                       beta_grid: np.ndarray | None = None,
                       t_steady: float = 3.0) -> tuple:
    """
    Compute S_cat(β; θ) on a grid and return the argmax as α̂_cat.

    Returns (alpha_cat, beta_grid, scores)
    """
    if beta_grid is None:
        beta_grid = np.linspace(0.1, 5.0, 50)

    ex    = get_steady_state_expects(theta, t_steady=t_steady)
    rho   = ex['rho']

    # Partial trace over buffer (keep subsystem 0 = storage)
    rho_a = dq.ptrace(rho, keep=[0], dims=(na, nb))

    scores = np.array([cat_overlap_score(rho_a, float(b)) for b in beta_grid])
    alpha_cat = float(beta_grid[np.argmax(scores)])
    return alpha_cat, beta_grid, scores


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CMA-ES OPTIMIZATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def clip_bounds(x: np.ndarray) -> np.ndarray:
    return np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])


def optimize_alpha(
    g2_init:    float = 1.0,
    eps_d_init: float = 4.0,
    n_epochs:   int   = N_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    sigma0:     float = SIGMA0,
    t_steady:   float = 3.0,
    verbose:    bool  = True,
) -> tuple:
    """
    Maximize R_a²(θ) over θ = (Re g₂, Im g₂, Re ε_d, Im ε_d) using CMA-ES.

    Warm start: θ₀ = (g2_init, 0, eps_d_init, 0) from analytic model.

    Returns (best_theta, best_reward, history)
    """
    theta0 = np.array([g2_init, 0.0, eps_d_init, 0.0])
    alpha_warm = analytic_alpha(theta0)
    if verbose:
        print(f"Warm-start: θ₀ = {theta0}")
        print(f"Analytic α estimate: {alpha_warm:.4f}")
        print(f"Penalty weights: λ={LAMBDA_CHI2}, μ={MU_LEAK}")
        print("-" * 60)

    optimizer = SepCMA(mean=theta0, sigma=sigma0, bounds=BOUNDS)

    history = {
        'reward':    [],
        'alpha_hat': [],
        'chi2':      [],
        'n_b':       [],
        'theta':     [],
    }
    best_reward = -np.inf
    best_theta  = theta0.copy()

    for epoch in range(n_epochs):
        batch_solutions = []
        batch_rewards   = []

        for _ in range(batch_size):
            x = clip_bounds(optimizer.ask())
            try:
                reward, alpha_hat, chi2, nb = compute_reward(x, t_steady=t_steady)
            except Exception:
                reward, alpha_hat, chi2, nb = -1e6, 0.0, 0.0, 0.0

            batch_solutions.append((x, -reward))   # CMA-ES minimises
            batch_rewards.append(reward)

        optimizer.tell(batch_solutions)

        # Track best in epoch
        best_idx = int(np.argmax(batch_rewards))
        best_x_epoch = batch_solutions[best_idx][0]
        best_r_epoch = batch_rewards[best_idx]

        if best_r_epoch > best_reward:
            best_reward = best_r_epoch
            best_theta  = best_x_epoch.copy()

        # Re-evaluate best point for logging
        r, a_hat, c2, nb_val = compute_reward(best_x_epoch, t_steady=t_steady)
        history['reward'].append(r)
        history['alpha_hat'].append(a_hat)
        history['chi2'].append(c2)
        history['n_b'].append(nb_val)
        history['theta'].append(best_x_epoch.copy())

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: R={r:+.4f}  α̂={a_hat:.3f}  "
                  f"χ²={c2:.4f}  ⟨n_b⟩={nb_val:.4f}  "
                  f"θ=({best_x_epoch[0]:.2f},{best_x_epoch[1]:.2f},"
                  f"{best_x_epoch[2]:.2f},{best_x_epoch[3]:.2f})")

    return best_theta, best_reward, history


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(history: dict,
                 best_theta: np.ndarray,
                 beta_grid: np.ndarray,
                 scores: np.ndarray,
                 alpha_cat: float,
                 save_path: str = "alpha_optimizer_results.png"):

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Cat-Qubit α Maximizer Results", fontsize=14)

    epochs = np.arange(len(history['reward']))

    # ── (0,0) Reward trace ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history['reward'], color='steelblue')
    ax.set(title='Reward  $R_{a^2}(\\theta)$', xlabel='Epoch', ylabel='R')
    ax.axhline(max(history['reward']), color='r', linestyle='--', alpha=0.5,
               label=f"max = {max(history['reward']):.3f}")
    ax.legend(fontsize=9)

    # ── (0,1) α̂ trace ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history['alpha_hat'], color='steelblue', label='$\\hat{\\alpha}$ (a² estimator)')
    ax.axhline(alpha_cat, color='crimson', linestyle='--',
               label=f'$\\hat{{\\alpha}}_{{cat}}$ (overlap) = {alpha_cat:.3f}')
    ax.set(title='Cat size estimate', xlabel='Epoch', ylabel='α')
    ax.legend(fontsize=9)

    # ── (1,0) Penalty terms ───────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, history['chi2'], label='$\\chi^2$ (eigenspace penalty)', color='darkorange')
    ax.plot(epochs, history['n_b'],  label='$\\langle n_b \\rangle$ (leakage)', color='purple')
    ax.set(title='Penalty terms', xlabel='Epoch', ylabel='Value')
    ax.legend(fontsize=9)

    # ── (1,1) Cat-overlap validation ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(beta_grid, scores, color='steelblue')
    ax.axvline(alpha_cat, color='crimson', linestyle='--',
               label=f'$\\hat{{\\alpha}}_{{cat}}$ = {alpha_cat:.3f}')
    ax.set(title='Cat-state overlap  $S_{cat}(\\beta; \\theta^*)$',
           xlabel='β', ylabel='$S_{cat}$')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  Cat-Qubit Alpha Maximizer")
    print(f"  na={na}, nb={nb}, κ_b={kappa_b} MHz, κ_a={kappa_a} MHz")
    print("=" * 60)

    # ── Optimization ──────────────────────────────────────────────────────────
    best_theta, best_reward, history = optimize_alpha(
        g2_init    = 1.0,    # MHz — analytic warm start
        eps_d_init = 4.0,    # MHz — analytic warm start
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
        sigma0     = SIGMA0,
        t_steady   = 3.0,    # µs — integration time to near steady state
    )

    print("\n" + "=" * 60)
    print("Optimal controls θ*:")
    labels = ['Re g₂', 'Im g₂', 'Re ε_d', 'Im ε_d']
    for lbl, val in zip(labels, best_theta):
        print(f"  {lbl:10s} = {val:+.4f} MHz")
    print(f"Analytic α (warm model): {analytic_alpha(best_theta):.4f}")
    print(f"Best reward R_a²:        {best_reward:.4f}")

    # ── Validation with cat-state overlaps ────────────────────────────────────
    print("\nValidating via cat-state overlaps ...")
    beta_grid = np.linspace(0.1, 5.0, 50)
    alpha_cat, beta_grid, scores = validate_cat_size(best_theta, beta_grid, t_steady=3.0)

    # Re-evaluate reward at best theta to get final α̂
    _, alpha_hat_final, chi2_final, nb_final = compute_reward(best_theta)
    print(f"a²-estimator  α̂  = {alpha_hat_final:.4f}")
    print(f"Cat-overlap   α̂  = {alpha_cat:.4f}")
    print(f"χ² (eigenspace)  = {chi2_final:.4f}")
    print(f"⟨n_b⟩ (leakage)  = {nb_final:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(history, best_theta, beta_grid, scores, alpha_cat)
