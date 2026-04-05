"""
catqubit.py — Foundational shared module for cat qubit optimizer comparison.

Physics: Two-mode (storage a, buffer b) cat qubit stabilized by the Hamiltonian
    H = g2* a^2 b† + g2 (a†)^2 b − ε_d b† − ε_d* b

with single-photon loss on both modes.  Four real knobs parameterize the system:
    knobs = [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]

All other agents import from this module.  Only numpy arrays are returned; JAX
arrays from dynamiqs are converted to numpy before being handed back.
"""

import warnings
import numpy as np
import jax.numpy as jnp
import dynamiqs as dq
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

NA = 15          # storage Hilbert-space dimension
NB = 5           # buffer Hilbert-space dimension
KAPPA_B = 10.0   # buffer decay rate  (MHz)
KAPPA_A = 1.0    # single-photon loss rate (MHz)
TARGET_BIAS = 100.0  # target η = T_Z / T_X
N_KNOBS = 4

KNOB_BOUNDS = [
    [0.2, 3.0],    # Re(g2)
    [-1.0, 1.0],   # Im(g2)
    [1.0, 8.0],    # Re(eps_d)
    [-2.0, 2.0],   # Im(eps_d)
]

DEFAULT_KNOBS = [1.0, 0.0, 4.0, 0.0]  # nominal operating point

# Soft upper-limit on |alpha| before Hilbert-space truncation becomes risky
_ALPHA_MAX = 3.5

__all__ = [
    "NA", "NB", "KAPPA_B", "KAPPA_A", "TARGET_BIAS", "N_KNOBS",
    "KNOB_BOUNDS", "DEFAULT_KNOBS",
    "estimate_alpha",
    "build_hamiltonian",
    "build_measurement_ops",
    "simulate_lifetimes",
    "robust_exp_fit",
    "compute_full_reward",
    "proxy_reward",
    "apply_drift",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_knobs(knobs):
    """Return (g2, eps_d) as complex numbers from a 4-vector of reals."""
    knobs = np.asarray(knobs, dtype=float)
    g2 = complex(knobs[0], knobs[1])
    eps_d = complex(knobs[2], knobs[3])
    return g2, eps_d


def _make_ops():
    """Return two-mode ladder operators a and b in the full NA*NB space."""
    a = dq.tensor(dq.destroy(NA), dq.eye(NB))
    b = dq.tensor(dq.eye(NA), dq.destroy(NB))
    return a, b


# ---------------------------------------------------------------------------
# 1. estimate_alpha
# ---------------------------------------------------------------------------

def estimate_alpha(knobs):
    """Estimate the cat-qubit coherent amplitude α from the operating knobs.

    Uses the adiabatic (leading-order) formula:
        α² = ε_d / conj(g₂)

    which is the steady-state solution obtained by eliminating the buffer mode
    in the limit κ_b → ∞ with finite ε₂ and κ₂.

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]

    Returns
    -------
    alpha : complex
        The cat coherent amplitude α (principal square root of ε_d / conj(g₂)).

    Warns
    -----
    UserWarning
        If |α| > 3.5, there is a risk that the NA=15 Hilbert space is too
        small, and the result is clamped to |α| = 3.5 in magnitude.
    """
    g2, eps_d = _unpack_knobs(knobs)
    if abs(g2) < 1e-12:
        warnings.warn("g2 is near zero; cannot compute alpha reliably. Returning 0.")
        return complex(0.0)

    alpha_sq = eps_d / np.conj(g2)
    # Take the principal square root: choose the branch with Re(alpha) >= 0
    alpha = np.sqrt(complex(alpha_sq))
    if alpha.real < 0:
        alpha = -alpha

    if abs(alpha) > _ALPHA_MAX:
        warnings.warn(
            f"|alpha| = {abs(alpha):.3f} > {_ALPHA_MAX} — Hilbert-space truncation "
            f"risk.  Clamping to {_ALPHA_MAX}.",
            UserWarning,
            stacklevel=2,
        )
        alpha = alpha * (_ALPHA_MAX / abs(alpha))

    return alpha


# ---------------------------------------------------------------------------
# 2. build_hamiltonian
# ---------------------------------------------------------------------------

def build_hamiltonian(knobs):
    """Build the two-mode cat-qubit Hamiltonian and loss operators.

    The Hamiltonian is:
        H = conj(g₂) · a² b† + g₂ · (a†)² b − ε_d · b† − conj(ε_d) · b

    Loss operators:
        L_b = sqrt(κ_b) · b        (buffer decay)
        L_a = sqrt(κ_a) · a        (storage single-photon loss)

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]

    Returns
    -------
    dict with keys:
        'H'         : dq-compatible Hamiltonian matrix (NA*NB × NA*NB)
        'loss_ops'  : list [L_b, L_a]
    """
    g2, eps_d = _unpack_knobs(knobs)
    a, b = _make_ops()

    g2_j = complex(g2)
    eps_d_j = complex(eps_d)

    H = (
        np.conj(g2_j) * a @ a @ b.dag()
        + g2_j * a.dag() @ a.dag() @ b
        - eps_d_j * b.dag()
        - np.conj(eps_d_j) * b
    )

    loss_b = jnp.sqrt(KAPPA_B) * b
    loss_a = jnp.sqrt(KAPPA_A) * a

    return {"H": H, "loss_ops": [loss_b, loss_a]}


# ---------------------------------------------------------------------------
# 3. build_measurement_ops
# ---------------------------------------------------------------------------

def build_measurement_ops(knobs):
    """Build the logical X and Z measurement operators for the current knobs.

    Logical X (photon-number parity) does NOT depend on α:
        X_L = exp(iπ a†a)

    Logical Z is constructed from the coherent-state projectors and DOES
    depend on the current α:
        Z_L = |+α⟩⟨+α| ⊗ I_b  −  |−α⟩⟨−α| ⊗ I_b

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]

    Returns
    -------
    dict with keys:
        'sx'    : logical X operator (NA*NB × NA*NB)
        'sz'    : logical Z operator (NA*NB × NA*NB)
        'alpha' : complex, the estimated cat amplitude
    """
    alpha = estimate_alpha(knobs)
    a, _ = _make_ops()

    # Parity operator — alpha-independent
    sx = (1j * jnp.pi * a.dag() @ a).expm()

    # Coherent-state projectors
    g_state = dq.coherent(NA, complex(alpha))       # |+α⟩
    e_state = dq.coherent(NA, complex(-alpha))      # |−α⟩

    sz_storage = g_state @ g_state.dag() - e_state @ e_state.dag()
    sz = dq.tensor(sz_storage, dq.eye(NB))

    return {"sx": sx, "sz": sz, "alpha": alpha}


# ---------------------------------------------------------------------------
# 4. simulate_lifetimes
# ---------------------------------------------------------------------------

def simulate_lifetimes(knobs, t_max_z=200.0, t_max_x=1.0, n_points=50):
    """Run two full mesolve simulations to obtain the T_Z and T_X decay curves.

    T_Z simulation (phase-flip lifetime):
        Initial state: |−α⟩ ⊗ |0⟩_b  (logical Z eigenstate)
        Observable:    ⟨Z_L⟩(t)
        Time range:    [0, t_max_z]

    T_X simulation (bit-flip lifetime):
        Initial state: (|+α⟩ + |−α⟩)/√2 ⊗ |0⟩_b  (logical X eigenstate)
        Observable:    ⟨X_L⟩(t)
        Time range:    [0, t_max_x]

    Both initial states and measurement operators are built using the
    adaptively estimated α from `estimate_alpha`.

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]
    t_max_z : float
        Total simulation time for the Z (phase-flip) run in μs.
    t_max_x : float
        Total simulation time for the X (bit-flip) run in μs.
    n_points : int
        Number of time points saved for each run.

    Returns
    -------
    dict with keys:
        'sz_t'    : numpy array, ⟨Z_L⟩ at each time step
        'sx_t'    : numpy array, ⟨X_L⟩ at each time step
        'tsave_z' : numpy array, time points for Z run (μs)
        'tsave_x' : numpy array, time points for X run (μs)
        'alpha'   : float, |α| used for initial states
    """
    ops = build_measurement_ops(knobs)
    sx = ops["sx"]
    sz = ops["sz"]
    alpha = ops["alpha"]

    ham_dict = build_hamiltonian(knobs)
    H = ham_dict["H"]
    loss_ops = ham_dict["loss_ops"]

    g_state = dq.coherent(NA, complex(alpha))
    e_state = dq.coherent(NA, complex(-alpha))
    b_vacuum = dq.fock(NB, 0)

    # ---- T_Z run: start in |-alpha> ----
    psi0_z = dq.tensor(e_state, b_vacuum)
    tsave_z = jnp.linspace(0.0, t_max_z, n_points)
    res_z = dq.mesolve(
        H, loss_ops, psi0_z, tsave_z,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )
    sz_t = np.array(res_z.expects[1, :].real)
    tsave_z_np = np.array(tsave_z)

    # ---- T_X run: start in (|+alpha> + |-alpha>)/sqrt(2) ----
    psi0_x = dq.tensor((g_state + e_state) / jnp.sqrt(2.0), b_vacuum)
    tsave_x = jnp.linspace(0.0, t_max_x, n_points)
    res_x = dq.mesolve(
        H, loss_ops, psi0_x, tsave_x,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )
    sx_t = np.array(res_x.expects[0, :].real)
    tsave_x_np = np.array(tsave_x)

    return {
        "sz_t": sz_t,
        "sx_t": sx_t,
        "tsave_z": tsave_z_np,
        "tsave_x": tsave_x_np,
        "alpha": float(abs(alpha)),
    }


# ---------------------------------------------------------------------------
# 5. robust_exp_fit
# ---------------------------------------------------------------------------

def robust_exp_fit(t, y):
    """Fit an exponential decay y = A·exp(−t/T) + C using robust least squares.

    Uses scipy.optimize.least_squares with soft_l1 loss (Cauchy-like, robust
    to outliers) and smart initialisation from the data range.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time points.
    y : array-like, shape (N,)
        Observed values.

    Returns
    -------
    T : float
        Fitted lifetime (time constant).  Returns 1e-6 on failure.
    A : float
        Fitted amplitude.
    C : float
        Fitted offset.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def _model(p, t_):
        A, tau, C = p
        return A * np.exp(-t_ / np.maximum(tau, 1e-12)) + C

    def _residuals(p, t_, y_):
        return _model(p, t_) - y_

    # Smart initialisation
    A0 = float(y.max() - y.min())
    C0 = float(y.min())
    tau0 = float(t.max() - t.min())
    p0 = [A0 if A0 > 0 else 0.1, max(tau0, 1e-6), C0]

    try:
        res = least_squares(
            _residuals,
            p0,
            args=(t, y),
            bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1",
            f_scale=0.1,
        )
        A, T, C = float(res.x[0]), float(res.x[1]), float(res.x[2])
        if T <= 0 or not np.isfinite(T):
            raise ValueError("Non-positive lifetime from fit.")
        return T, A, C
    except Exception as exc:
        warnings.warn(f"robust_exp_fit failed ({exc}); returning T=1e-6.", UserWarning, stacklevel=2)
        return 1e-6, float(y[0]) if len(y) > 0 else 0.0, 0.0


# ---------------------------------------------------------------------------
# 6. compute_full_reward
# ---------------------------------------------------------------------------

def compute_full_reward(knobs, target_bias=TARGET_BIAS):
    """Compute the full reward by running two complete mesolve simulations.

    Reward formula:
        reward = 0.3·log10(T_Z) + 0.3·log10(T_X) − 0.4·|log10(η / target_bias)|

    where η = T_Z / T_X.  A higher reward is better.  When passing to a
    minimising optimiser, negate this value.

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]
    target_bias : float
        Desired T_Z / T_X ratio.  Defaults to TARGET_BIAS = 100.

    Returns
    -------
    reward : float
        Scalar reward (higher = better).
    """
    sim = simulate_lifetimes(knobs)

    T_Z, _, _ = robust_exp_fit(sim["tsave_z"], sim["sz_t"])
    T_X, _, _ = robust_exp_fit(sim["tsave_x"], sim["sx_t"])

    T_Z = max(T_Z, 1e-9)
    T_X = max(T_X, 1e-9)
    eta = T_Z / T_X

    reward = (
        0.3 * np.log10(T_Z)
        + 0.3 * np.log10(T_X)
        - 0.4 * abs(np.log10(eta / target_bias))
    )
    return float(reward)


# ---------------------------------------------------------------------------
# 7. proxy_reward
# ---------------------------------------------------------------------------

def proxy_reward(knobs, t_probe_z=50.0, t_probe_x=0.3):
    """Fast single-point proxy for the full reward (use this inside optimisers).

    Instead of fitting an exponential to many time points, we run mesolve to
    exactly two time points [0, t_probe] and estimate the lifetime as:
        T_Z ≈ −t_probe_z / log(|⟨Z⟩(t_probe_z)|)
        T_X ≈ −t_probe_x / log(|⟨X⟩(t_probe_x)|)

    Then the same reward formula is applied:
        reward = 0.3·log10(T_Z) + 0.3·log10(T_X) − 0.4·|log10(η / target_bias)|

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]
    t_probe_z : float
        Probe time for Z decay in μs.
    t_probe_x : float
        Probe time for X decay in μs.

    Returns
    -------
    reward : float
        Proxy reward (higher = better).  Negate when passing to minimisers.
    """
    ops = build_measurement_ops(knobs)
    sx = ops["sx"]
    sz = ops["sz"]
    alpha = ops["alpha"]

    ham_dict = build_hamiltonian(knobs)
    H = ham_dict["H"]
    loss_ops = ham_dict["loss_ops"]

    g_state = dq.coherent(NA, complex(alpha))
    e_state = dq.coherent(NA, complex(-alpha))
    b_vacuum = dq.fock(NB, 0)

    # ---- Z probe ----
    psi0_z = dq.tensor(e_state, b_vacuum)
    tsave_z = jnp.array([0.0, float(t_probe_z)])
    res_z = dq.mesolve(
        H, loss_ops, psi0_z, tsave_z,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )
    sz_probe = float(np.array(res_z.expects[1, -1]).real)
    sz_probe = np.clip(abs(sz_probe), 1e-12, 1.0 - 1e-12)
    T_Z = -t_probe_z / np.log(sz_probe)

    # ---- X probe ----
    psi0_x = dq.tensor((g_state + e_state) / jnp.sqrt(2.0), b_vacuum)
    tsave_x = jnp.array([0.0, float(t_probe_x)])
    res_x = dq.mesolve(
        H, loss_ops, psi0_x, tsave_x,
        exp_ops=[sx, sz],
        options=dq.Options(progress_meter=False),
    )
    sx_probe = float(np.array(res_x.expects[0, -1]).real)
    sx_probe = np.clip(abs(sx_probe), 1e-12, 1.0 - 1e-12)
    T_X = -t_probe_x / np.log(sx_probe)

    T_Z = max(T_Z, 1e-9)
    T_X = max(T_X, 1e-9)
    eta = T_Z / T_X

    reward = (
        0.3 * np.log10(T_Z)
        + 0.3 * np.log10(T_X)
        - 0.4 * abs(np.log10(eta / TARGET_BIAS))
    )
    return float(reward)


# ---------------------------------------------------------------------------
# 8. apply_drift
# ---------------------------------------------------------------------------

def apply_drift(knobs, drift):
    """Apply a parameter drift to the knob vector (element-wise addition).

    Parameters
    ----------
    knobs : array-like, shape (4,)
        Current knob values [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)].
    drift : array-like, shape (4,)
        Drift vector [Δg₂_re, Δg₂_im, Δε_d_re, Δε_d_im].

    Returns
    -------
    new_knobs : numpy.ndarray, shape (4,)
        Updated knob values after adding the drift.
    """
    knobs = np.asarray(knobs, dtype=float)
    drift = np.asarray(drift, dtype=float)
    return knobs + drift


# ---------------------------------------------------------------------------
# Validation / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("catqubit.py — self-test at DEFAULT_KNOBS")
    print("=" * 60)

    knobs = DEFAULT_KNOBS

    # 1. estimate_alpha
    alpha = estimate_alpha(knobs)
    print(f"\n[1] estimate_alpha({knobs}) = {alpha:.4f}  (|alpha| = {abs(alpha):.4f})")

    # 2. proxy_reward (fast)
    print("\n[2] Running proxy_reward (fast single-point estimate)...")
    pr = proxy_reward(knobs)
    print(f"    proxy_reward = {pr:.4f}")

    # 3. build_measurement_ops
    mops = build_measurement_ops(knobs)
    print(f"\n[3] build_measurement_ops: alpha = {mops['alpha']:.4f}")
    print(f"    sx shape = {np.array(mops['sx']).shape}")
    print(f"    sz shape = {np.array(mops['sz']).shape}")

    # 4. robust_exp_fit smoke test
    t_test = np.linspace(0, 10, 50)
    y_test = 0.95 * np.exp(-t_test / 5.0) + 0.02 + 0.01 * np.random.randn(50)
    T_fit, A_fit, C_fit = robust_exp_fit(t_test, y_test)
    print(f"\n[4] robust_exp_fit smoke test (true T=5.0):")
    print(f"    T = {T_fit:.4f}, A = {A_fit:.4f}, C = {C_fit:.4f}")

    # 5. apply_drift
    drift = np.array([0.05, 0.01, -0.1, 0.0])
    new_knobs = apply_drift(knobs, drift)
    print(f"\n[5] apply_drift: {knobs} + {drift} = {new_knobs}")

    print("\n[OK] All checks passed.")
