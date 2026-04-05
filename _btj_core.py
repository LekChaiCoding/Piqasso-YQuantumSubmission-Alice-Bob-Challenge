"""
betterthanJacks.py — Core Optimizer Section
Part 1: Imports, constants, batched reward evaluation, CMA-ES optimization loop.
"""

# ── Section A: Imports and Constants ────────────────────────────────────────

import os, sys, time
import numpy as np
import jax.numpy as jnp
import dynamiqs as dq
from scipy.optimize import brentq, least_squares
from cmaes import SepCMA
from matplotlib import pyplot as plt

# Path setup — catqubit.py is in team-piqasso/
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_DIR = os.path.join(REPO_DIR, "team-piqasso")
sys.path.insert(0, TEAM_DIR)
sys.path.insert(0, REPO_DIR)

from catqubit import (
    NA, NB, KAPPA_A, KAPPA_B, N_KNOBS,
    KNOB_BOUNDS, DEFAULT_KNOBS,
    estimate_alpha, simulate_lifetimes, robust_exp_fit,
    compute_full_reward, apply_drift,
)

# ── Override target to 320 (catqubit.py uses 100 as its default) ──
ETA_TARGET = 320.0


# ── Section B: Kalman interface placeholder ──────────────────────────────────

class _NullDriftEstimator:
    """No-op drift estimator — used when Kalman is disabled."""
    def __init__(self): self.d = np.zeros(4)
    def predict(self): pass
    def get_correction(self): return np.zeros(4)
    def update(self, commanded, reward_observed, reward_fn): pass
    @property
    def drift_history(self): return np.zeros((0, 4)), np.zeros((0, 4))


# ── Section C: Batched Reward Evaluator ─────────────────────────────────────

# Precompute two-mode ladder operators as plain JAX arrays once at import.
_a    = dq.tensor(dq.destroy(NA), dq.eye(NB))
_b    = dq.tensor(dq.eye(NA),    dq.destroy(NB))
_adag = _a.dag()
_bdag = _b.dag()
_fock_b0 = dq.fock(NB, 0)

_loss_b = (jnp.sqrt(KAPPA_B) * _b).to_jax()
_loss_a = (jnp.sqrt(KAPPA_A) * _a).to_jax()
_parity = dq.tensor(
    jnp.diag(jnp.array([(-1.0)**n for n in range(NA)])), jnp.eye(NB)
).to_jax()

# 4 base matrices for complex-knob Hamiltonian
# H = conj(g2)*a²b† + g2*(a†)²b − ε_d*b† − conj(ε_d)*b
# Decomposed as:
#   g2r * (a²b† + a†²b)  +  g2i * i*(-a²b† + a†²b)
#   − εr * (b† + b)      −  εi  * i*(b† - b)
_H_g2r  = (_a@_a@_bdag + _adag@_adag@_b).to_jax()           # Re(g₂) term
_H_g2i  = (1j*(-_a@_a@_bdag + _adag@_adag@_b)).to_jax()     # Im(g₂) term
_H_epsr = (_bdag + _b).to_jax()                               # Re(ε_d) term
_H_epsi = (1j*(_bdag - _b)).to_jax()                         # Im(ε_d) term

TZ_TFINAL = 200.0
TX_TFINAL = 1.0
N_POINTS  = 50
TZ_MAX    = 2000.0
TX_MAX    = 5.0


def simulate_batch_complex(knobs_batch, init_state, tfinal, n_points=N_POINTS):
    """
    Batched mesolve for B = len(knobs_batch) 4-knob vectors.
    knobs_batch: (B, 4) array of [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]
    Returns: tsave (T,), parity_exp (B,T), ZL_exp (B,T)
    """
    B = len(knobs_batch)
    knobs_np = np.asarray(knobs_batch)

    g2r = jnp.array(knobs_np[:, 0])[:, None, None]
    g2i = jnp.array(knobs_np[:, 1])[:, None, None]
    er  = jnp.array(knobs_np[:, 2])[:, None, None]
    ei  = jnp.array(knobs_np[:, 3])[:, None, None]

    H_batch = (g2r * _H_g2r[None] + g2i * _H_g2i[None]
               - er * _H_epsr[None] - ei * _H_epsi[None])

    psi0_list, ZL_list = [], []
    for k in knobs_np:
        alpha = estimate_alpha(k)
        alpha_c = complex(alpha)
        cat_p = dq.coherent(NA, alpha_c)
        cat_m = dq.coherent(NA, -alpha_c)
        if init_state == "+z":
            psi0_a = cat_p
        elif init_state == "+x":
            psi0_a = (cat_p + cat_m) / jnp.sqrt(2)
        else:
            psi0_a = cat_m
        psi0_list.append(dq.tensor(psi0_a, _fock_b0).to_jax())
        ZL_list.append(
            dq.tensor(cat_p@cat_p.dag() - cat_m@cat_m.dag(), dq.eye(NB)).to_jax()
        )

    psi0_batch = jnp.stack(psi0_list)   # (B, N, 1)
    ZL_batch   = jnp.stack(ZL_list)     # (B, N, N)
    tsave = jnp.linspace(0, tfinal, n_points)

    res = dq.mesolve(
        H_batch, [_loss_b, _loss_a], psi0_batch, tsave,
        options=dq.Options(progress_meter=False, cartesian_batching=False, save_states=True),
    )

    states     = jnp.array(res.states)
    parity_exp = jnp.einsum("ij,Btji->Bt", _parity,   states).real  # (B, T)
    ZL_exp     = jnp.einsum("Bij,Btji->Bt", ZL_batch, states).real  # (B, T)

    return np.array(tsave), np.array(parity_exp), np.array(ZL_exp)


def _fit_decay(t, y, t_max):
    """Fit A·exp(-t/τ)+C using robust least_squares (soft_l1), return clamped τ."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    A0   = max(float(np.ptp(y)), 1e-6)
    tau0 = max(float(np.ptp(t)) / 3, 1e-6)
    C0   = float(y[-1])

    try:
        res = least_squares(
            lambda p, t_, y_: p[0] * np.exp(-t_ / np.maximum(p[1], 1e-12)) + p[2] - y_,
            [A0, tau0, C0],
            args=(t, y),
            bounds=([0.0, 1e-10, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1",
            f_scale=0.1,
        )
        tau = float(res.x[1])
        if tau <= 0 or not np.isfinite(tau):
            raise ValueError("Non-positive lifetime from fit.")
    except Exception:
        tau = tau0

    return min(max(tau, 1e-10), t_max)


def batch_lifetimes(knobs_batch):
    """
    Return (Tx_arr, Tz_arr) for each row in knobs_batch.
    Two batched mesolve calls instead of 2*B sequential.
    """
    knobs_batch = np.asarray(knobs_batch)

    # Z run: initial state |-α⟩, observe Z_L
    tz_t, _parity_z, sz_b = simulate_batch_complex(knobs_batch, "+z", TZ_TFINAL)
    # X run: initial state (|+α⟩+|-α⟩)/√2, observe parity (X_L)
    tx_t, sx_b, _ZL_x = simulate_batch_complex(knobs_batch, "+x", TX_TFINAL)

    Txs, Tzs = [], []
    for i in range(len(knobs_batch)):
        Tz = _fit_decay(tz_t, sz_b[i], TZ_MAX)
        Tx = _fit_decay(tx_t, sx_b[i], TX_MAX)
        Tzs.append(max(Tz, 1e-6))
        Txs.append(max(Tx, 1e-6))

    return np.array(Txs), np.array(Tzs)


def fast_reward_320(knobs):
    """
    Proxy reward with target_bias=320 (2-point probe, fast).
    Replicates catqubit.proxy_reward but with ETA_TARGET=320.
    """
    from catqubit import build_hamiltonian, build_measurement_ops, estimate_alpha

    ops = build_measurement_ops(knobs)
    sx, sz, alpha = ops['sx'], ops['sz'], ops['alpha']
    ham = build_hamiltonian(knobs)
    H, loss_ops = ham['H'], ham['loss_ops']

    g_state = dq.coherent(NA, complex(alpha))
    e_state = dq.coherent(NA, complex(-alpha))
    b_vac   = dq.fock(NB, 0)

    # Z probe — start in |-α⟩, probe at t=50
    psi0_z = dq.tensor(e_state, b_vac)
    tsave_z = jnp.array([0.0, 50.0])
    res_z = dq.mesolve(H, loss_ops, psi0_z, tsave_z,
                       exp_ops=[sx, sz],
                       options=dq.Options(progress_meter=False))
    sz_probe = float(np.array(res_z.expects[1, -1]).real)
    sz_probe = np.clip(abs(sz_probe), 1e-12, 1 - 1e-12)
    T_Z = -50.0 / np.log(sz_probe)

    # X probe — start in superposition, probe at t=0.3
    psi0_x = dq.tensor((g_state + e_state) / jnp.sqrt(2.0), b_vac)
    tsave_x = jnp.array([0.0, 0.3])
    res_x = dq.mesolve(H, loss_ops, psi0_x, tsave_x,
                       exp_ops=[sx, sz],
                       options=dq.Options(progress_meter=False))
    sx_probe = float(np.array(res_x.expects[0, -1]).real)
    sx_probe = np.clip(abs(sx_probe), 1e-12, 1 - 1e-12)
    T_X = -0.3 / np.log(sx_probe)

    T_Z = max(T_Z, 1e-9)
    T_X = max(T_X, 1e-9)
    eta = T_Z / T_X

    reward = (0.3 * np.log10(T_Z)
              + 0.3 * np.log10(T_X)
              - 0.4 * abs(np.log10(eta / ETA_TARGET)))
    return float(reward)


# ── Section D: CMA-ES with Kalman integration ────────────────────────────────

def optimize_with_kalman(drift_fn=None, kalman=None, n_epochs=80, batch_size=16, seed=0):
    """
    CMA-ES optimizer for the cat qubit system with optional Kalman drift correction.

    Key improvements over Jack_fast.py:
    1. Warm start from DEFAULT_KNOBS [1.0, 0.0, 4.0, 0.0]
    2. 4 knobs (all of KNOB_BOUNDS from catqubit.py, not just 2)
    3. sigma=0.15 (tighter since we start near optimum)
    4. Correct reward (fast_reward_320, target_bias=320)
    5. Kalman integration: if kalman is not None, apply correction and update each epoch

    Parameters
    ----------
    drift_fn : callable or None
        Function (epoch -> drift array of shape (4,)). If None, no drift.
    kalman : KalmanDriftEstimator or None
        Kalman drift estimator. If None, uses _NullDriftEstimator (no-op).
    n_epochs : int
        Number of CMA-ES epochs.
    batch_size : int
        CMA-ES population size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        name               : str, identifier for this run
        reward_history     : list of float, mean reward per epoch
        reward_std_history : list of float, std of reward per epoch
        mean_history       : list of np.ndarray(4,), CMA mean per epoch
        commanded_history  : list of np.ndarray(4,), commanded mean (mean + correction)
        actual_history     : list of np.ndarray(4,), actual knobs after drift
        best_reward        : float
        best_knobs         : np.ndarray(4,)
    """
    if kalman is None:
        kalman = _NullDriftEstimator()

    bounds_lo = np.array([b[0] for b in KNOB_BOUNDS])
    bounds_hi = np.array([b[1] for b in KNOB_BOUNDS])
    bounds_arr = np.array(KNOB_BOUNDS, dtype=float)

    x0 = np.array(DEFAULT_KNOBS, dtype=float)

    optimizer = SepCMA(
        mean=x0,
        sigma=0.15,
        bounds=bounds_arr,
        population_size=batch_size,
        seed=seed,
    )

    reward_history     = []
    reward_std_history = []
    mean_history       = []
    commanded_history  = []
    actual_history     = []
    best_reward        = -np.inf
    best_knobs         = x0.copy()

    for epoch in range(n_epochs):
        drift      = drift_fn(epoch) if drift_fn is not None else np.zeros(4)
        correction = kalman.get_correction()
        kalman.predict()

        solutions, rewards = [], []

        for _ in range(optimizer.population_size):
            x_cma    = optimizer.ask()                                        # CMA internal coords
            x_cmd    = x_cma + correction                                     # commanded to hardware
            x_actual = np.clip(x_cmd + drift, bounds_lo, bounds_hi)          # true physical knobs

            try:
                r = fast_reward_320(x_actual)
            except Exception as exc:
                print(f"  [!] fast_reward_320 failed at epoch {epoch}: {exc}")
                r = -1e6

            solutions.append((x_cma, -r))   # CMA minimises, so negate reward
            rewards.append(r)

        optimizer.tell(solutions)

        rewards_arr = np.array(rewards)
        mean_reward = float(np.mean(rewards_arr))
        std_reward  = float(np.std(rewards_arr))

        # Kalman update using the mean candidate
        mean_cmd    = optimizer.mean + correction
        mean_actual = np.clip(mean_cmd + drift, bounds_lo, bounds_hi)
        try:
            r_mean = fast_reward_320(mean_actual)
        except Exception:
            r_mean = mean_reward

        kalman.update(mean_cmd, r_mean, fast_reward_320)

        # Track best
        best_idx = int(np.argmax(rewards_arr))
        if rewards_arr[best_idx] > best_reward:
            best_reward = float(rewards_arr[best_idx])
            x_best_cma  = solutions[best_idx][0]
            best_knobs  = np.clip(x_best_cma + correction + drift, bounds_lo, bounds_hi)

        reward_history.append(mean_reward)
        reward_std_history.append(std_reward)
        mean_history.append(optimizer.mean.copy())
        commanded_history.append(mean_cmd.copy())
        actual_history.append(mean_actual.copy())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            mean = optimizer.mean
            print(
                f"[BTJ] Epoch {epoch:3d}/{n_epochs} | "
                f"reward={mean_reward:.4f}±{std_reward:.4f} | "
                f"best={best_reward:.4f} | "
                f"g2=({mean[0]:.3f},{mean[1]:.3f}) "
                f"eps_d=({mean[2]:.3f},{mean[3]:.3f})"
            )

    return {
        "name":               "optimize_with_kalman",
        "reward_history":     reward_history,
        "reward_std_history": reward_std_history,
        "mean_history":       mean_history,
        "commanded_history":  commanded_history,
        "actual_history":     actual_history,
        "best_reward":        best_reward,
        "best_knobs":         best_knobs,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("This is the core module. Run betterthanJacks.py for the full benchmark.")
