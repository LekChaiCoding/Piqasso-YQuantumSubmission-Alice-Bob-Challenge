"""
optimizers_a.py — CMA-ES and SPSA optimizers for cat qubit knob tuning.

Both optimizers expose a common results-dict interface:
    {
        'reward_history'     : np.ndarray  shape (n_epochs,)   — mean reward per epoch
        'reward_std_history' : np.ndarray  shape (n_epochs,)   — std of rewards per epoch
        'mean_history'       : np.ndarray  shape (n_epochs, 4) — mean knob vector per epoch
        'best_knobs'         : np.ndarray  shape (4,)          — knobs at best reward seen
        'best_reward'        : float                           — best reward seen
        'name'               : str                             — 'CMA-ES' or 'SPSA'
    }

Dependencies
------------
    pip install cmaes   # provides SepCMA
"""

import numpy as np
from cmaes import SepCMA

from catqubit import KNOB_BOUNDS, DEFAULT_KNOBS, N_KNOBS

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BOUNDS_LOW  = np.array([b[0] for b in KNOB_BOUNDS], dtype=float)
_BOUNDS_HIGH = np.array([b[1] for b in KNOB_BOUNDS], dtype=float)


def _clip_knobs(knobs: np.ndarray) -> np.ndarray:
    """Clip a knob vector into the feasible box defined by KNOB_BOUNDS."""
    return np.clip(knobs, _BOUNDS_LOW, _BOUNDS_HIGH)


def _default_knobs() -> np.ndarray:
    return np.array(DEFAULT_KNOBS, dtype=float)


# ---------------------------------------------------------------------------
# 1. CMA-ES  (using SepCMA from the `cmaes` package)
# ---------------------------------------------------------------------------

def run_cmaes(
    reward_fn,
    n_epochs: int = 150,
    batch_size: int = 12,
    drift_fn=None,
    seed: int = 0,
    sigma0: float = 0.3,
) -> dict:
    """Optimize cat-qubit knobs with a separable CMA-ES.

    Parameters
    ----------
    reward_fn : callable
        Either ``reward_fn(knobs)`` -> float  or
        ``reward_fn(knobs, drift)`` -> float  when *drift_fn* is provided.
    n_epochs : int
        Number of CMA-ES generations (each processes *batch_size* candidates).
    batch_size : int
        Population size per generation.
    drift_fn : callable or None
        If provided, called as ``drift_fn(epoch)`` -> drift_vector (shape 4).
        The drift is forwarded as the second argument to *reward_fn*.
    seed : int
        Random seed for reproducibility.
    sigma0 : float
        Initial step-size for CMA-ES.

    Returns
    -------
    dict
        Keys: ``reward_history``, ``reward_std_history``, ``mean_history``,
        ``best_knobs``, ``best_reward``, ``name``.
    """
    rng = np.random.default_rng(seed)

    # SepCMA expects: mean (initial), sigma0, bounds, population_size, seed
    bounds = np.column_stack([_BOUNDS_LOW, _BOUNDS_HIGH])  # shape (4, 2)

    optimizer = SepCMA(
        mean=_default_knobs(),
        sigma=sigma0,
        bounds=bounds,
        population_size=batch_size,
        seed=int(rng.integers(0, 2**31)),
    )

    reward_history      = np.zeros(n_epochs, dtype=float)
    reward_std_history  = np.zeros(n_epochs, dtype=float)
    mean_history        = np.zeros((n_epochs, N_KNOBS), dtype=float)

    best_reward = -np.inf
    best_knobs  = _default_knobs()

    for epoch in range(n_epochs):
        # Compute drift for this epoch (if applicable)
        drift = drift_fn(epoch) if drift_fn is not None else None

        # Sample a batch of candidates
        solutions = []
        rewards   = []

        for _ in range(batch_size):
            candidate = optimizer.ask()
            candidate = _clip_knobs(candidate)

            if drift is not None:
                r = float(reward_fn(candidate, drift))
            else:
                r = float(reward_fn(candidate))

            solutions.append((candidate, -r))   # CMA-ES minimises -> negate
            rewards.append(r)

        # Tell the optimizer the (negated) results
        optimizer.tell(solutions)

        rewards_arr = np.array(rewards, dtype=float)
        epoch_mean_reward = float(rewards_arr.mean())
        epoch_std_reward  = float(rewards_arr.std())

        reward_history[epoch]     = epoch_mean_reward
        reward_std_history[epoch] = epoch_std_reward
        mean_history[epoch]       = optimizer.mean.copy()

        # Track global best
        best_idx = int(np.argmax(rewards_arr))
        if rewards_arr[best_idx] > best_reward:
            best_reward = float(rewards_arr[best_idx])
            best_knobs  = solutions[best_idx][0].copy()

        if epoch % 10 == 0:
            print(
                f"[CMA-ES] epoch {epoch:4d}/{n_epochs}  "
                f"mean_reward={epoch_mean_reward:+.4f}  "
                f"std={epoch_std_reward:.4f}  "
                f"best_so_far={best_reward:+.4f}"
            )

    return {
        "reward_history":     reward_history,
        "reward_std_history": reward_std_history,
        "mean_history":       mean_history,
        "best_knobs":         best_knobs,
        "best_reward":        best_reward,
        "name":               "CMA-ES",
    }


# ---------------------------------------------------------------------------
# 2. SPSA — Simultaneous Perturbation Stochastic Approximation
# ---------------------------------------------------------------------------

def run_spsa(
    reward_fn,
    n_epochs: int = 150,
    drift_fn=None,
    seed: int = 0,
    a: float = 0.15,
    c: float = 0.05,
    alpha_exp: float = 0.602,
    gamma_exp: float = 0.101,
) -> dict:
    """Optimize cat-qubit knobs with SPSA.

    At each epoch the gradient is estimated with two reward evaluations using
    a random +-1 Bernoulli perturbation vector, then a noisy gradient step is
    taken.  The learning rate and perturbation magnitude follow the standard
    SPSA schedule:

        a_k = a / (epoch + 1)^alpha_exp
        c_k = c / (epoch + 1)^gamma_exp

    Parameters
    ----------
    reward_fn : callable
        Either ``reward_fn(knobs)`` -> float  or
        ``reward_fn(knobs, drift)`` -> float  when *drift_fn* is provided.
    n_epochs : int
        Total number of gradient steps.
    drift_fn : callable or None
        If provided, called as ``drift_fn(epoch)`` -> drift_vector (shape 4).
        The drift is forwarded as the second argument to *reward_fn*.
    seed : int
        Random seed for the Bernoulli perturbation draws.
    a : float
        Baseline learning-rate coefficient.
    c : float
        Baseline perturbation magnitude.
    alpha_exp : float
        Decay exponent for the learning rate (SPSA theory: ~0.602).
    gamma_exp : float
        Decay exponent for the perturbation size (SPSA theory: ~0.101).

    Returns
    -------
    dict
        Keys: ``reward_history``, ``reward_std_history``, ``mean_history``,
        ``best_knobs``, ``best_reward``, ``name``.

    Notes
    -----
    * ``reward_std_history`` is all-zeros for SPSA because the algorithm
      evaluates a single point per epoch (after the update step).  The field
      is retained so both optimizers share the same results-dict schema.
    * ``mean_history[epoch]`` records the knob vector *after* the update.
    """
    rng = np.random.default_rng(seed)

    knobs = _default_knobs()

    reward_history      = np.zeros(n_epochs, dtype=float)
    reward_std_history  = np.zeros(n_epochs, dtype=float)   # always 0 for SPSA
    mean_history        = np.zeros((n_epochs, N_KNOBS), dtype=float)

    best_reward = -np.inf
    best_knobs  = knobs.copy()

    for epoch in range(n_epochs):
        # Compute schedules
        a_k = a / float(epoch + 1) ** alpha_exp
        c_k = c / float(epoch + 1) ** gamma_exp

        # Random +-1 Bernoulli perturbation
        delta = rng.choice([-1.0, 1.0], size=N_KNOBS).astype(float)

        knobs_plus  = _clip_knobs(knobs + c_k * delta)
        knobs_minus = _clip_knobs(knobs - c_k * delta)

        # Compute drift (if applicable)
        drift = drift_fn(epoch) if drift_fn is not None else None

        if drift is not None:
            r_plus  = float(reward_fn(knobs_plus,  drift))
            r_minus = float(reward_fn(knobs_minus, drift))
        else:
            r_plus  = float(reward_fn(knobs_plus))
            r_minus = float(reward_fn(knobs_minus))

        # Gradient estimate (we are MAXIMISING reward, so + direction)
        g_hat = (r_plus - r_minus) / (2.0 * c_k * delta)

        # Update knobs (ascent step) and clip
        knobs = _clip_knobs(knobs + a_k * g_hat)

        # Evaluate reward at updated knobs for logging
        if drift is not None:
            r_current = float(reward_fn(knobs, drift))
        else:
            r_current = float(reward_fn(knobs))

        reward_history[epoch] = r_current
        mean_history[epoch]   = knobs.copy()

        if r_current > best_reward:
            best_reward = r_current
            best_knobs  = knobs.copy()

        if epoch % 10 == 0:
            print(
                f"[SPSA]   epoch {epoch:4d}/{n_epochs}  "
                f"reward={r_current:+.4f}  "
                f"a_k={a_k:.5f}  c_k={c_k:.5f}  "
                f"best_so_far={best_reward:+.4f}"
            )

    return {
        "reward_history":     reward_history,
        "reward_std_history": reward_std_history,
        "mean_history":       mean_history,
        "best_knobs":         best_knobs,
        "best_reward":        best_reward,
        "name":               "SPSA",
    }


# ---------------------------------------------------------------------------
# Validation / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("optimizers_a.py — smoke test (5 epochs, mock reward)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Mock reward: a simple bowl centred at [1.5, 0.0, 4.5, 0.0]
    # (well within KNOB_BOUNDS).  Returns a float in a plausible range.
    # ------------------------------------------------------------------
    _TARGET = np.array([1.5, 0.0, 4.5, 0.0], dtype=float)

    def mock_reward(knobs, drift=None):
        """Quadratic bowl: reward = -||knobs - target||^2  (max = 0)."""
        k = np.asarray(knobs, dtype=float)
        if drift is not None:
            k = k + np.asarray(drift, dtype=float)
        return float(-np.sum((k - _TARGET) ** 2))

    def mock_drift(epoch):
        """Small sinusoidal drift on the first knob dimension."""
        return np.array([0.01 * np.sin(epoch * 0.3), 0.0, 0.0, 0.0])

    N_TEST = 5

    # ------------------------------------------------------------------
    # Test 1: CMA-ES without drift
    # ------------------------------------------------------------------
    print("\n--- CMA-ES (no drift, 5 epochs) ---")
    res_cmaes = run_cmaes(mock_reward, n_epochs=N_TEST, batch_size=6, seed=42, sigma0=0.3)

    assert res_cmaes["name"] == "CMA-ES",                              "name mismatch"
    assert res_cmaes["reward_history"].shape      == (N_TEST,),        "reward_history shape"
    assert res_cmaes["reward_std_history"].shape  == (N_TEST,),        "reward_std_history shape"
    assert res_cmaes["mean_history"].shape        == (N_TEST, 4),      "mean_history shape"
    assert res_cmaes["best_knobs"].shape          == (4,),             "best_knobs shape"
    assert isinstance(res_cmaes["best_reward"], float),                "best_reward type"
    print(f"    best_reward = {res_cmaes['best_reward']:.4f}")
    print(f"    best_knobs  = {res_cmaes['best_knobs']}")
    print("    [PASS] CMA-ES without drift")

    # ------------------------------------------------------------------
    # Test 2: CMA-ES with drift
    # ------------------------------------------------------------------
    print("\n--- CMA-ES (with drift, 5 epochs) ---")
    res_cmaes_d = run_cmaes(
        mock_reward, n_epochs=N_TEST, batch_size=6,
        drift_fn=mock_drift, seed=7, sigma0=0.3,
    )
    assert res_cmaes_d["name"] == "CMA-ES"
    assert res_cmaes_d["reward_history"].shape == (N_TEST,)
    print(f"    best_reward = {res_cmaes_d['best_reward']:.4f}")
    print("    [PASS] CMA-ES with drift")

    # ------------------------------------------------------------------
    # Test 3: SPSA without drift
    # ------------------------------------------------------------------
    print("\n--- SPSA (no drift, 5 epochs) ---")
    res_spsa = run_spsa(mock_reward, n_epochs=N_TEST, seed=42)

    assert res_spsa["name"] == "SPSA",                                 "name mismatch"
    assert res_spsa["reward_history"].shape      == (N_TEST,),         "reward_history shape"
    assert res_spsa["reward_std_history"].shape  == (N_TEST,),         "reward_std_history shape"
    assert res_spsa["mean_history"].shape        == (N_TEST, 4),       "mean_history shape"
    assert res_spsa["best_knobs"].shape          == (4,),              "best_knobs shape"
    assert isinstance(res_spsa["best_reward"], float),                 "best_reward type"
    print(f"    best_reward = {res_spsa['best_reward']:.4f}")
    print(f"    best_knobs  = {res_spsa['best_knobs']}")
    print("    [PASS] SPSA without drift")

    # ------------------------------------------------------------------
    # Test 4: SPSA with drift
    # ------------------------------------------------------------------
    print("\n--- SPSA (with drift, 5 epochs) ---")
    res_spsa_d = run_spsa(
        mock_reward, n_epochs=N_TEST,
        drift_fn=mock_drift, seed=13,
    )
    assert res_spsa_d["name"] == "SPSA"
    assert res_spsa_d["reward_history"].shape == (N_TEST,)
    print(f"    best_reward = {res_spsa_d['best_reward']:.4f}")
    print("    [PASS] SPSA with drift")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("All smoke tests passed.")
    print("=" * 65)
    sys.exit(0)
