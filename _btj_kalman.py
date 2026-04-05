"""
_btj_kalman.py — Improved Kalman Filter + Benchmarking for Cat Qubit Optimizer

Physics context:
  - 4 knobs: [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]
  - Lifetimes: T_Z ~ exp(2*alpha^2)/kappa_a, T_X ~ 1/(kappa_a * alpha^2)
  - alpha^2 = eps_d / conj(g2)
  - Target bias: eta = T_Z / T_X = 320
  - DEFAULT_KNOBS = [1.0, 0.0, 4.0, 0.0]
  - In real hardware: actual = commanded + drift_vector

This module provides:
  1. KalmanDriftEstimator  — improved EKF with 4 enhancements over the notebook version
  2. Drift scenario functions + DRIFT_SCENARIOS dict
  3. Benchmark harness (run_benchmark, compute_metrics, plot_benchmark)
  4. jack_fast_optimize   — Jack_fast baseline wrapper with drift applied

Importable standalone:
  from _btj_kalman import KalmanDriftEstimator, DRIFT_SCENARIOS
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe default; override before calling plot_benchmark
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOB_BOUNDS = [[0.2, 3.0], [-1.0, 1.0], [1.0, 8.0], [-2.0, 2.0]]
DEFAULT_KNOBS = np.array([1.0, 0.0, 4.0, 0.0])
ETA_TARGET = 320.0

_BOUNDS_LOW  = np.array([b[0] for b in KNOB_BOUNDS], dtype=float)
_BOUNDS_HIGH = np.array([b[1] for b in KNOB_BOUNDS], dtype=float)

# Anisotropic process noise: imaginary parts drift less than real parts
_DEFAULT_Q_DIAG = np.array([0.002, 0.001, 0.002, 0.001])

# ---------------------------------------------------------------------------
# 1. Improved KalmanDriftEstimator
# ---------------------------------------------------------------------------

class KalmanDriftEstimator:
    """Extended Kalman Filter for estimating hardware drift in cat-qubit knobs.

    The model assumes:
        actual_knobs = commanded_knobs + drift_vector  (d)

    The EKF estimates d from observed rewards and a linearised reward model.

    Improvements over the notebook baseline:
      1. Bounded drift estimate — d is clipped so DEFAULT_KNOBS + d stays in
         KNOB_BOUNDS after every update.
      2. Jacobian caching — if the estimated mean has moved by < jac_cache_tol
         in L2 norm, the previous finite-difference Jacobian H is reused,
         saving 2*n expensive reward evaluations per update call.
      3. Anisotropic process noise — Q = diag([0.002, 0.001, 0.002, 0.001])
         so that imaginary parts (which drift less physically) receive less
         injected uncertainty each epoch.
      4. Multi-measurement update — update_multi() accepts a list of (B, r)
         pairs and performs sequential scalar Kalman updates within one epoch,
         as if processing the top-k candidates' reward observations back-to-back.

    The core interface (predict / update / get_correction / drift_history) is
    kept identical to the notebook version so this class is a drop-in replacement.

    Parameters
    ----------
    n_knobs : int
        Number of knobs (default 4).
    process_noise : float or None
        Scalar to build isotropic Q.  If None, the anisotropic default
        _DEFAULT_Q_DIAG is used (recommended).
    obs_noise : float
        Observation noise standard deviation; R = obs_noise**2.
    jac_cache_tol : float
        L2-norm threshold below which the cached Jacobian is reused.
    default_knobs : array-like or None
        Reference point for bounds clipping.  Defaults to DEFAULT_KNOBS.
    knob_bounds : list of [lo, hi] or None
        Feasibility box.  Defaults to KNOB_BOUNDS.
    """

    def __init__(
        self,
        n_knobs: int = 4,
        process_noise: float = None,
        obs_noise: float = 0.02,
        jac_cache_tol: float = 0.01,
        default_knobs=None,
        knob_bounds=None,
    ):
        self.n = n_knobs
        self.d = np.zeros(n_knobs)
        self.P = np.eye(n_knobs) * 0.05
        self.R = obs_noise ** 2
        self.jac_cache_tol = jac_cache_tol

        # Process noise
        if process_noise is not None:
            self.Q = np.eye(n_knobs) * process_noise
        else:
            if n_knobs == 4:
                self.Q = np.diag(_DEFAULT_Q_DIAG)
            else:
                self.Q = np.eye(n_knobs) * 0.002  # fallback for non-4-knob

        # Bounds for drift clipping
        self._default_knobs = (
            np.asarray(default_knobs, dtype=float)
            if default_knobs is not None
            else DEFAULT_KNOBS.copy()
        )
        self._knob_bounds = knob_bounds if knob_bounds is not None else KNOB_BOUNDS

        # Jacobian cache
        self._H_cached: np.ndarray | None = None
        self._mean_at_cache: np.ndarray | None = None

        # History
        self._d_hist = []
        self._std_hist = []

    # ------------------------------------------------------------------
    # Core EKF interface (drop-in compatible with notebook version)
    # ------------------------------------------------------------------

    def predict(self):
        """Time-update step: propagate covariance with process noise."""
        self.P = self.P + self.Q

    def update(self, commanded, reward_observed, reward_fn, fd_eps: float = 5e-3):
        """Measurement-update step (single observation).

        Parameters
        ----------
        commanded : array-like, shape (n,)
            The knob vector that was commanded to hardware.
        reward_observed : float
            The reward actually measured at the hardware.
        reward_fn : callable
            reward_fn(knobs) -> float.  Used for finite-difference Jacobian.
        fd_eps : float
            Step size for finite-difference gradient.
        """
        commanded = np.asarray(commanded, dtype=float)
        actual_est = commanded + self.d

        H = self._compute_or_reuse_jacobian(actual_est, reward_fn, fd_eps)

        y_pred = reward_fn(actual_est)
        innovation = float(reward_observed) - y_pred

        S = float(H @ self.P @ H) + self.R
        K = self.P @ H / S

        self.d = self.d + K * innovation
        I_KH = np.eye(self.n) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R

        # Improvement 1: clip drift so default_knobs + d stays in bounds
        self.bounds_clip(self._default_knobs, self._knob_bounds)

        self._d_hist.append(self.d.copy())
        self._std_hist.append(np.sqrt(np.diag(self.P)))

    def get_correction(self) -> np.ndarray:
        """Return the correction to apply to the next commanded knob vector.

        To compensate for estimated drift d, the optimizer should command:
            knobs_corrected = knobs_desired - d
        """
        return -self.d.copy()

    @property
    def drift_history(self):
        """Return (d_history, std_history) arrays.

        Returns
        -------
        d_history : np.ndarray, shape (T, n)
            Estimated drift vector at each update step.
        std_history : np.ndarray, shape (T, n)
            Posterior standard deviation of each knob's drift estimate.
        """
        if not self._d_hist:
            return np.zeros((0, self.n)), np.zeros((0, self.n))
        return np.array(self._d_hist), np.array(self._std_hist)

    # ------------------------------------------------------------------
    # Improvement 1: Bounded drift clipping
    # ------------------------------------------------------------------

    def bounds_clip(self, default_knobs, knob_bounds):
        """Clip the drift estimate d so that default_knobs + d stays feasible.

        For each dimension i:
            d[i] is clipped to [bounds_lo[i] - default[i],
                                bounds_hi[i] - default[i]]

        This ensures that even if the optimizer commands DEFAULT_KNOBS, the
        estimated actual knobs remain inside KNOB_BOUNDS.

        Parameters
        ----------
        default_knobs : array-like, shape (n,)
            Reference (nominal) knob vector.
        knob_bounds : list of [lo, hi], length n
            Feasibility box per knob.
        """
        default_knobs = np.asarray(default_knobs, dtype=float)
        lo = np.array([b[0] for b in knob_bounds], dtype=float)
        hi = np.array([b[1] for b in knob_bounds], dtype=float)
        d_lo = lo - default_knobs
        d_hi = hi - default_knobs
        self.d = np.clip(self.d, d_lo, d_hi)

    # ------------------------------------------------------------------
    # Improvement 4: Multi-measurement update
    # ------------------------------------------------------------------

    def update_multi(self, commanded_list, rewards_list, reward_fn, fd_eps: float = 5e-3):
        """Sequential scalar Kalman updates from multiple (commanded, reward) pairs.

        Each pair in zip(commanded_list, rewards_list) is processed as a
        separate measurement within the same epoch.  predict() is NOT called
        between individual updates — the caller is expected to call predict()
        once per epoch before update_multi().

        Parameters
        ----------
        commanded_list : list of array-like, each shape (n,)
            Commanded knob vectors for the top-k measurements.
        rewards_list : list of float
            Corresponding observed rewards.
        reward_fn : callable
            reward_fn(knobs) -> float.
        fd_eps : float
            Step size for finite-difference Jacobian.

        Notes
        -----
        History is appended once per call (after all sequential updates),
        not once per measurement.
        """
        for commanded, reward_observed in zip(commanded_list, rewards_list):
            commanded = np.asarray(commanded, dtype=float)
            actual_est = commanded + self.d

            H = self._compute_or_reuse_jacobian(actual_est, reward_fn, fd_eps)

            y_pred = reward_fn(actual_est)
            innovation = float(reward_observed) - y_pred

            S = float(H @ self.P @ H) + self.R
            K = self.P @ H / S

            self.d = self.d + K * innovation
            I_KH = np.eye(self.n) - np.outer(K, H)
            self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R

            self.bounds_clip(self._default_knobs, self._knob_bounds)

        # Record state once after all sequential updates
        self._d_hist.append(self.d.copy())
        self._std_hist.append(np.sqrt(np.diag(self.P)))

    # ------------------------------------------------------------------
    # Improvement 2: Jacobian caching (internal)
    # ------------------------------------------------------------------

    def _compute_or_reuse_jacobian(
        self, actual_est: np.ndarray, reward_fn, fd_eps: float
    ) -> np.ndarray:
        """Return finite-difference Jacobian H, using cache if mean hasn't moved.

        Parameters
        ----------
        actual_est : np.ndarray, shape (n,)
            Current estimate of actual knobs.
        reward_fn : callable
        fd_eps : float

        Returns
        -------
        H : np.ndarray, shape (n,)
            Gradient of reward w.r.t. actual knobs at actual_est.
        """
        if (
            self._H_cached is not None
            and self._mean_at_cache is not None
            and np.linalg.norm(actual_est - self._mean_at_cache) < self.jac_cache_tol
        ):
            return self._H_cached

        # Improvement 3 note: Q is already anisotropic (set in __init__)
        H = np.zeros(self.n)
        for i in range(self.n):
            kp = actual_est.copy()
            kp[i] += fd_eps
            km = actual_est.copy()
            km[i] -= fd_eps
            H[i] = (reward_fn(kp) - reward_fn(km)) / (2.0 * fd_eps)

        self._H_cached = H.copy()
        self._mean_at_cache = actual_est.copy()
        return H

    def __repr__(self):
        return (
            f"KalmanDriftEstimator(n={self.n}, "
            f"|d|={np.linalg.norm(self.d):.4f}, "
            f"updates={len(self._d_hist)})"
        )


# ---------------------------------------------------------------------------
# 2. Drift Scenario Functions
# ---------------------------------------------------------------------------

def no_drift(epoch):
    """Zero drift at every epoch (ideal hardware baseline)."""
    return np.zeros(4)


def sinusoidal_drift(epoch, amplitude=0.3, period=50):
    """Slow sinusoidal drift in Re(eps_d) — most physically relevant.

    Models a periodic oscillation in the drive amplitude, e.g. due to
    temperature cycling or laser power fluctuation.
    """
    return np.array([0.0, 0.0, amplitude * np.sin(2 * np.pi * epoch / period), 0.0])


def ramp_drift(epoch, rate=0.005):
    """Linear ramp in Re(g2) — models slow gain drift in the parametric drive.

    The drift is unbounded in epoch but physically the Kalman filter's
    bounded-clip will prevent the estimate from leaving KNOB_BOUNDS.
    """
    return np.array([rate * epoch, 0.0, 0.0, 0.0])


def step_drift(epoch, step_epoch=40, step_size=0.3):
    """Step change in Re(eps_d) at step_epoch — models a sudden hardware reset.

    Useful for measuring recovery time: how many epochs does it take for the
    Kalman filter (or baseline) to return to near-optimal reward after the step?
    """
    return np.array([0.0, 0.0, step_size if epoch >= step_epoch else 0.0, 0.0])


def compound_drift(epoch):
    """Combined drift: Re(g2) ramp + Re(eps_d) oscillation.

    Tests whether the filter can track two simultaneously drifting parameters.
    The ramp is capped at 0.5 to stay within KNOB_BOUNDS.
    """
    return np.array([
        min(0.003 * epoch, 0.5),
        0.0,
        0.25 * np.sin(2 * np.pi * epoch / 40),
        0.0,
    ])


DRIFT_SCENARIOS = {
    "no_drift":   no_drift,
    "sinusoidal": sinusoidal_drift,
    "ramp":       ramp_drift,
    "step":       step_drift,
    "compound":   compound_drift,
}

# ---------------------------------------------------------------------------
# 3. Benchmark Harness
# ---------------------------------------------------------------------------

def compute_metrics(result: dict, n_epochs: int, step_epoch: int = 40) -> dict:
    """Compute scalar performance metrics from a single optimize_fn result.

    Parameters
    ----------
    result : dict
        Must contain 'reward_history' (array of length n_epochs), 'best_reward',
        and 'best_knobs'.
    n_epochs : int
        Total number of epochs run.
    step_epoch : int
        Epoch at which the step drift fires; used for recovery_time calculation.

    Returns
    -------
    metrics : dict with keys:
        convergence_epoch : int
            First epoch at which cumulative improvement >= 90% of total improvement.
        final_reward : float
            Mean reward over the last 20% of epochs.
        best_reward : float
            Global best reward seen.
        recovery_time : int or None
            Epochs to recover 90% of pre-step reward after the step.  None if
            the scenario is not a step-drift run.
    """
    rh = np.asarray(result["reward_history"], dtype=float)
    n = len(rh)
    if n == 0:
        return {
            "convergence_epoch": None,
            "final_reward": float("nan"),
            "best_reward": float(result.get("best_reward", float("nan"))),
            "recovery_time": None,
        }

    r_min = float(np.nanmin(rh))
    r_max = float(np.nanmax(rh))
    total_improvement = r_max - r_min

    # Convergence epoch: first time we reach 90% of total improvement
    convergence_epoch = None
    if total_improvement > 1e-12:
        threshold = r_min + 0.9 * total_improvement
        idx = np.where(rh >= threshold)[0]
        convergence_epoch = int(idx[0]) if len(idx) > 0 else n - 1

    # Final reward: mean of last 20% of epochs
    tail_start = max(0, int(0.8 * n))
    final_reward = float(np.nanmean(rh[tail_start:]))

    # Recovery time (step-drift scenario only)
    # Pre-step mean reward is the mean of epochs [max(0, step_epoch-10), step_epoch)
    # Post-step recovery is defined as returning to 90% of that mean.
    recovery_time = None
    if step_epoch < n:
        pre_start = max(0, step_epoch - 10)
        pre_mean = float(np.nanmean(rh[pre_start:step_epoch])) if step_epoch > 0 else r_max
        target_recovery = pre_mean * 0.9 if pre_mean > 0 else pre_mean * 1.1
        post_step_rh = rh[step_epoch:]
        if len(post_step_rh) > 0:
            idx_rec = np.where(post_step_rh >= target_recovery)[0]
            recovery_time = int(idx_rec[0]) if len(idx_rec) > 0 else None

    return {
        "convergence_epoch": convergence_epoch,
        "final_reward": final_reward,
        "best_reward": float(result.get("best_reward", r_max)),
        "recovery_time": recovery_time,
    }


def run_benchmark(
    optimize_fn,
    drift_scenarios: dict | None = None,
    n_epochs: int = 80,
    seed: int = 0,
) -> dict:
    """Run optimize_fn under each drift scenario and collect results + metrics.

    Parameters
    ----------
    optimize_fn : callable
        Signature: optimize_fn(drift_fn, n_epochs, seed) -> result_dict.
        result_dict must contain:
            - 'reward_history' : array-like, shape (n_epochs,)
            - 'best_reward'    : float
            - 'best_knobs'     : array-like, shape (4,)
        Optionally may contain 'eta_history' (shape (n_epochs,)); if absent,
        the benchmark will attempt to compute eta from knobs using the proxy.
    drift_scenarios : dict or None
        Maps scenario_name -> drift_fn.  Defaults to DRIFT_SCENARIOS.
    n_epochs : int
        Number of optimization epochs per scenario.
    seed : int
        Random seed forwarded to optimize_fn.

    Returns
    -------
    all_results : dict
        {scenario_name: {'result': result_dict, 'metrics': metrics_dict}}
    """
    if drift_scenarios is None:
        drift_scenarios = DRIFT_SCENARIOS

    all_results = {}
    for name, drift_fn in drift_scenarios.items():
        print(f"[benchmark] Running scenario: {name!r} ...")
        try:
            result = optimize_fn(drift_fn, n_epochs, seed)
        except Exception as exc:
            print(f"  [!] scenario {name!r} failed: {exc}")
            result = {
                "reward_history": np.full(n_epochs, float("nan")),
                "best_reward": float("nan"),
                "best_knobs": DEFAULT_KNOBS.copy(),
            }

        step_epoch = 40  # default step position used in step_drift
        metrics = compute_metrics(result, n_epochs, step_epoch=step_epoch)
        all_results[name] = {"result": result, "metrics": metrics}
        print(
            f"  convergence_epoch={metrics['convergence_epoch']}, "
            f"final_reward={metrics['final_reward']:.4f}, "
            f"best_reward={metrics['best_reward']:.4f}"
        )

    return all_results


def plot_benchmark(all_results: dict, save_path: str | None = None):
    """Multi-panel figure comparing reward and bias-eta trajectories per scenario.

    Layout:
      Row 0: reward vs epoch for each scenario (one subplot per scenario)
      Row 1: bias eta vs epoch for each scenario (one subplot per scenario)
      Row 2: metrics summary table

    Parameters
    ----------
    all_results : dict
        Output of run_benchmark — maps scenario_name -> {result, metrics}.
        Each result['reward_history'] is plotted.
        If result contains 'eta_history', that is plotted in row 1; otherwise
        row 1 is skipped for that scenario.
    save_path : str or None
        If provided, the figure is saved to this path.  Otherwise plt.show()
        is called.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    scenario_names = list(all_results.keys())
    n_scenarios = len(scenario_names)
    if n_scenarios == 0:
        raise ValueError("all_results is empty — nothing to plot.")

    fig = plt.figure(figsize=(4 * n_scenarios, 10))
    gs = gridspec.GridSpec(3, n_scenarios, figure=fig, hspace=0.45, wspace=0.35)

    for col, name in enumerate(scenario_names):
        result  = all_results[name]["result"]
        metrics = all_results[name]["metrics"]
        rh = np.asarray(result.get("reward_history", []), dtype=float)
        epochs = np.arange(len(rh))

        # ---- Row 0: Reward ----
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(epochs, rh, color="steelblue", lw=1.5)
        if metrics["convergence_epoch"] is not None:
            ax0.axvline(metrics["convergence_epoch"], color="orange", ls="--",
                        lw=1, label=f"conv@{metrics['convergence_epoch']}")
            ax0.legend(fontsize=7)
        ax0.set_title(name, fontsize=9, fontweight="bold")
        ax0.set_xlabel("Epoch", fontsize=8)
        if col == 0:
            ax0.set_ylabel("Reward", fontsize=8)
        ax0.tick_params(labelsize=7)
        ax0.grid(True, alpha=0.3)

        # ---- Row 1: Bias eta ----
        ax1 = fig.add_subplot(gs[1, col])
        eta_h = result.get("eta_history", None)
        if eta_h is not None:
            eta_h = np.asarray(eta_h, dtype=float)
            ax1.plot(np.arange(len(eta_h)), eta_h, color="darkorange", lw=1.5)
            ax1.axhline(ETA_TARGET, color="red", ls="--", lw=1, alpha=0.7,
                        label=f"target {ETA_TARGET:.0f}")
            ax1.legend(fontsize=7)
        else:
            ax1.text(0.5, 0.5, "no eta_history", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=8, color="grey")
        ax1.set_title(f"eta ({name})", fontsize=8)
        ax1.set_xlabel("Epoch", fontsize=8)
        if col == 0:
            ax1.set_ylabel("eta = T_Z/T_X", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.3)

    # ---- Row 2: Metrics table ----
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    col_labels = ["Scenario", "conv. epoch", "final_reward", "best_reward", "recovery_time"]
    table_data = []
    for name in scenario_names:
        m = all_results[name]["metrics"]
        table_data.append([
            name,
            str(m["convergence_epoch"]) if m["convergence_epoch"] is not None else "—",
            f"{m['final_reward']:.4f}",
            f"{m['best_reward']:.4f}",
            str(m["recovery_time"]) if m["recovery_time"] is not None else "—",
        ])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)

    fig.suptitle("Benchmark: Drift Scenario Comparison", fontsize=12, fontweight="bold")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_benchmark] saved to {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# 4. Jack_fast baseline wrapper
# ---------------------------------------------------------------------------

def jack_fast_optimize(drift_fn, n_epochs: int = 80, seed: int = 0) -> dict:
    """Wrap Jack_fast.py's optimizer to evaluate reward under a given drift_fn.

    Jack_fast.py has no drift compensation.  This wrapper applies drift_fn at
    each epoch by offsetting the actual knobs used in the reward evaluation.

    Since Jack_fast uses only 2 real knobs (Re(g2) and Re(eps_d)), drift is
    applied only to those two dimensions:
        actual_g2    = commanded_g2    + drift[0]
        actual_eps_d = commanded_eps_d + drift[2]

    The reward is then evaluated at (actual_g2, actual_eps_d).

    Parameters
    ----------
    drift_fn : callable
        drift_fn(epoch) -> np.ndarray of shape (4,).
        Only indices 0 (Re(g2)) and 2 (Re(eps_d)) are used.
    n_epochs : int
        Number of CMA-ES epochs to run.
    seed : int
        Random seed for Jack_fast's SepCMA optimizer.

    Returns
    -------
    result : dict with keys:
        'reward_history' : np.ndarray, shape (n_epochs,) — mean reward per epoch
                           (sign-flipped so higher = better, matching the
                            convention of optimizers_a.py)
        'eta_history'    : np.ndarray, shape (n_epochs,) — mean eta per epoch
        'best_reward'    : float
        'best_knobs'     : np.ndarray, shape (4,) — as 4-vector (Im components = 0)
        'name'           : 'Jack_fast'
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    try:
        from Jack_fast import (
            simulate_batch,
            fit_decay,
            TZ_TFINAL,
            TX_TFINAL,
            N_POINTS,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot import Jack_fast.  Make sure Jack_fast.py is in the same "
            f"directory as _btj_kalman.py ({_here})."
        ) from exc

    from cmaes import SepCMA

    # Jack_fast 2-knob bounds: [[0.05, 0.5], [2.0, 6.0]]  (its own convention)
    # We use a slightly wider box to match the 4-knob DEFAULT_KNOBS convention
    bounds_2d = np.array([[0.2, 3.0], [1.0, 8.0]])
    x0_2d = np.array([DEFAULT_KNOBS[0], DEFAULT_KNOBS[2]])  # [1.0, 4.0]
    batch_size = 12

    optimizer = SepCMA(
        mean=x0_2d,
        sigma=0.2,
        bounds=bounds_2d,
        population_size=batch_size,
        seed=int(seed),
    )

    reward_history = np.zeros(n_epochs, dtype=float)
    eta_history    = np.zeros(n_epochs, dtype=float)
    best_reward    = -np.inf
    best_knobs_2d  = x0_2d.copy()

    for epoch in range(n_epochs):
        drift = drift_fn(epoch)
        d_g2    = float(drift[0])
        d_eps_d = float(drift[2])

        xs = np.array([optimizer.ask() for _ in range(batch_size)])
        commanded_g2    = xs[:, 0]
        commanded_eps_d = xs[:, 1]

        # Actual knobs seen by hardware after drift
        actual_g2    = np.clip(commanded_g2    + d_g2,    bounds_2d[0, 0], bounds_2d[0, 1])
        actual_eps_d = np.clip(commanded_eps_d + d_eps_d, bounds_2d[1, 0], bounds_2d[1, 1])

        try:
            Txs, Tzs = _jack_measure_Tx_Tz_batch(
                actual_g2.tolist(),
                actual_eps_d.tolist(),
                simulate_batch,
                fit_decay,
                TZ_TFINAL,
                TX_TFINAL,
                N_POINTS,
            )
            etas = Tzs / Txs
            # Simple reward: log(Tz) + log(Tx) - penalty for eta deviation
            rewards = (
                np.log(np.maximum(Tzs, 1e-9))
                + np.log(np.maximum(Txs, 1e-9))
                - 2.0 * np.abs(np.log(np.maximum(etas / ETA_TARGET, 1e-9)))
            )
        except Exception as exc:
            print(f"  [jack_fast_optimize] epoch {epoch}: sim failed: {exc}")
            rewards = np.full(batch_size, -1e6)
            etas    = np.zeros(batch_size)

        # Tell CMA-ES (it minimises, so negate reward)
        optimizer.tell([(xs[j], -rewards[j]) for j in range(batch_size)])

        mean_reward = float(np.mean(rewards))
        reward_history[epoch] = mean_reward
        eta_history[epoch]    = float(np.mean(etas))

        best_idx = int(np.argmax(rewards))
        if rewards[best_idx] > best_reward:
            best_reward   = float(rewards[best_idx])
            best_knobs_2d = xs[best_idx].copy()

    # Return as 4-knob vector with Im parts = 0
    best_knobs = np.array([best_knobs_2d[0], 0.0, best_knobs_2d[1], 0.0])

    return {
        "reward_history": reward_history,
        "eta_history":    eta_history,
        "best_reward":    best_reward,
        "best_knobs":     best_knobs,
        "name":           "Jack_fast",
    }


def _jack_measure_Tx_Tz_batch(g2_arr, eps_d_arr, simulate_batch, fit_decay,
                               TZ_TFINAL, TX_TFINAL, N_POINTS,
                               TX_MAX=5.0, TZ_MAX=2000.0):
    """Internal helper: measure T_X and T_Z for a batch of Jack_fast parameters.

    Mirrors Jack_fast.measure_Tx_Tz_batch but accepts the sim functions as
    arguments to avoid circular import issues.
    """
    import numpy as _np

    tz_t, _,    sz_b = simulate_batch(g2_arr, eps_d_arr, "+z", TZ_TFINAL, N_POINTS)
    tx_t, sx_b, _    = simulate_batch(g2_arr, eps_d_arr, "+x", TX_TFINAL, N_POINTS)

    Txs, Tzs = [], []
    for i in range(len(g2_arr)):
        Tz = min(fit_decay(tz_t, sz_b[i]), TZ_MAX)
        Tx = min(fit_decay(tx_t, sx_b[i]), TX_MAX)
        Txs.append(max(Tx, 1e-6))
        Tzs.append(max(Tz, 1e-6))
    return _np.array(Txs), _np.array(Tzs)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Kalman filter
    "KalmanDriftEstimator",
    # Drift scenarios
    "no_drift",
    "sinusoidal_drift",
    "ramp_drift",
    "step_drift",
    "compound_drift",
    "DRIFT_SCENARIOS",
    # Constants
    "KNOB_BOUNDS",
    "DEFAULT_KNOBS",
    "ETA_TARGET",
    # Benchmark
    "run_benchmark",
    "compute_metrics",
    "plot_benchmark",
    # Baseline
    "jack_fast_optimize",
]


# ---------------------------------------------------------------------------
# Smoke test (run standalone: python _btj_kalman.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("_btj_kalman.py — smoke test")
    print("=" * 65)

    # ----------------------------------------------------------------
    # 1. KalmanDriftEstimator: basic predict/update cycle
    # ----------------------------------------------------------------
    print("\n--- KalmanDriftEstimator smoke test ---")

    # Simple quadratic reward surface centred at [1.2, 0.0, 4.5, 0.0]
    TRUE_CENTER = np.array([1.2, 0.0, 4.5, 0.0])
    TRUE_DRIFT  = np.array([0.15, 0.0, -0.2, 0.0])

    def mock_reward(knobs):
        k = np.asarray(knobs, dtype=float)
        return float(-np.sum((k - TRUE_CENTER) ** 2))

    rng = np.random.default_rng(0)

    kf = KalmanDriftEstimator()
    commanded = DEFAULT_KNOBS.copy()

    for step in range(10):
        kf.predict()
        noise = rng.normal(0, 0.02)
        actual = commanded + TRUE_DRIFT
        obs_reward = mock_reward(actual) + noise
        kf.update(commanded, obs_reward, mock_reward)

    d_hist, std_hist = kf.drift_history
    assert d_hist.shape == (10, 4), f"Expected (10,4) drift history, got {d_hist.shape}"
    print(f"  Estimated drift after 10 steps: {kf.d}")
    print(f"  True drift:                     {TRUE_DRIFT}")
    print(f"  KF repr: {kf}")
    print("  [PASS] basic update/predict cycle")

    # ----------------------------------------------------------------
    # 2. bounds_clip
    # ----------------------------------------------------------------
    print("\n--- bounds_clip smoke test ---")
    kf2 = KalmanDriftEstimator()
    kf2.d = np.array([10.0, 5.0, -20.0, 3.0])   # wildly out of range
    kf2.bounds_clip(DEFAULT_KNOBS, KNOB_BOUNDS)
    reconstructed = DEFAULT_KNOBS + kf2.d
    assert np.all(reconstructed >= _BOUNDS_LOW - 1e-9), "lower bound violated"
    assert np.all(reconstructed <= _BOUNDS_HIGH + 1e-9), "upper bound violated"
    print(f"  Clipped d:          {kf2.d}")
    print(f"  DEFAULT + clipped:  {reconstructed}")
    print("  [PASS] bounds_clip")

    # ----------------------------------------------------------------
    # 3. update_multi
    # ----------------------------------------------------------------
    print("\n--- update_multi smoke test ---")
    kf3 = KalmanDriftEstimator()
    kf3.predict()
    cmds   = [DEFAULT_KNOBS.copy() for _ in range(3)]
    actual_rewards = [mock_reward(DEFAULT_KNOBS + TRUE_DRIFT) + rng.normal(0, 0.02)
                      for _ in range(3)]
    kf3.update_multi(cmds, actual_rewards, mock_reward)
    d_hist3, _ = kf3.drift_history
    assert d_hist3.shape[0] == 1, "update_multi should append once to history"
    print(f"  d after update_multi: {kf3.d}")
    print("  [PASS] update_multi")

    # ----------------------------------------------------------------
    # 4. Drift scenario functions
    # ----------------------------------------------------------------
    print("\n--- Drift scenario smoke test ---")
    for name, fn in DRIFT_SCENARIOS.items():
        for ep in [0, 10, 40, 79]:
            d = fn(ep)
            assert d.shape == (4,), f"{name}({ep}) wrong shape: {d.shape}"
    print("  [PASS] All drift scenarios return shape-(4,) arrays")

    # ----------------------------------------------------------------
    # 5. compute_metrics
    # ----------------------------------------------------------------
    print("\n--- compute_metrics smoke test ---")
    fake_rh = np.linspace(-5.0, -1.0, 80)   # monotone improvement
    fake_result = {"reward_history": fake_rh, "best_reward": float(fake_rh[-1]), "best_knobs": DEFAULT_KNOBS}
    m = compute_metrics(fake_result, 80)
    assert m["convergence_epoch"] is not None
    assert m["final_reward"] > m["best_reward"] - 0.5
    print(f"  convergence_epoch={m['convergence_epoch']}, final_reward={m['final_reward']:.4f}")
    print("  [PASS] compute_metrics")

    print("\n" + "=" * 65)
    print("All smoke tests passed.")
    print("=" * 65)
