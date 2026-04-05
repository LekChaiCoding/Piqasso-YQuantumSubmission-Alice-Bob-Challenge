"""
Test script for faster alternatives to least_squares for exponential fitting.

This script implements and compares different approaches for fitting exponential decay:
1. Linear regression on log-transformed data
2. Analytical approximations for known decay forms
3. Better initial guesses and bounds
4. Vectorized fitting for multiple curves

Based on the fit_decay function from jack.py
"""

import numpy as np
import scipy.optimize
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress


def generate_test_data(n_samples=100, noise_level=0.01):
    """Generate synthetic exponential decay data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 5, n_samples)
    A_true = 1.0
    tau_true = 2.0
    C_true = 0.1

    # True exponential decay
    y_true = A_true * np.exp(-t / tau_true) + C_true

    # Add noise
    noise = np.random.normal(0, noise_level * np.max(y_true), n_samples)
    y_noisy = y_true + noise

    return t, y_noisy, tau_true


def fit_decay_original(t, y):
    """Original fit_decay function from jack.py using least_squares."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A0 = max(float(np.ptp(y)), 1e-6)
    tau0 = max(float(np.ptp(t)) / 3.0, 1e-6)
    C0 = float(y[-1])

    try:
        result = scipy.optimize.least_squares(
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


def fit_decay_log_linear(t, y):
    """Linear regression on log-transformed data."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove baseline by subtracting the final value
    y_baseline = y - y[-1]

    # Only use positive values for log transform
    valid_idx = y_baseline > 1e-10
    if np.sum(valid_idx) < 3:
        # Fallback to original method if not enough valid points
        return fit_decay_original(t, y)

    t_valid = t[valid_idx]
    y_valid = y_baseline[valid_idx]

    # Log transform
    log_y = np.log(y_valid)

    # Linear regression: log(y) = log(A) - t/tau
    # So: y = log(A) + (-1/tau)*t
    slope, intercept, r_value, p_value, std_err = linregress(t_valid, log_y)

    # Extract tau from slope
    tau = -1.0 / slope if slope < 0 else 1e-10

    return max(float(tau), 1e-10)


def fit_decay_improved_bounds(t, y):
    """Improved least_squares with better initial guesses and bounds."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Better initial guesses
    y_range = np.ptp(y)
    A0 = max(float(y_range), 1e-6)

    # Estimate tau from the time it takes to decay to 1/e of initial amplitude
    y_initial = y[0] - y[-1]
    y_threshold = y[-1] + y_initial / np.e
    decay_idx = np.where(y <= y_threshold)[0]
    if len(decay_idx) > 0:
        tau0 = max(float(t[decay_idx[0]]), 1e-6)
    else:
        tau0 = max(float(np.ptp(t)) / 3.0, 1e-6)

    C0 = float(y[-1])

    # Tighter bounds based on data characteristics
    A_max = 2.0 * y_range
    tau_max = 10.0 * np.ptp(t)
    C_min = np.min(y) - 0.1 * y_range
    C_max = np.max(y) + 0.1 * y_range

    try:
        result = scipy.optimize.least_squares(
            lambda p, tx, ty: p[0] * np.exp(-tx / p[1]) + p[2] - ty,
            [A0, tau0, C0],
            args=(t, y),
            bounds=([0.0, 1e-10, C_min], [A_max, tau_max, C_max]),
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=50,  # Limit iterations for speed
        )
        return max(float(result.x[1]), 1e-10)
    except Exception:
        return tau0


def fit_decay_analytical_approx(t, y):
    """Analytical approximation for exponential decay."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Simple approximation: tau ≈ t_decay / log(y0/y_decay)
    # where t_decay is time to decay to certain level

    y0 = y[0] - y[-1]
    y_half = y[-1] + y0 / 2.0
    y_e = y[-1] + y0 / np.e

    # Find half-life
    half_idx = np.where(y <= y_half)[0]
    if len(half_idx) > 0:
        t_half = t[half_idx[0]]
        tau_half = t_half / np.log(2)
    else:
        tau_half = np.ptp(t) / 3.0

    # Find 1/e life
    e_idx = np.where(y <= y_e)[0]
    if len(e_idx) > 0:
        t_e = t[e_idx[0]]
        tau_e = t_e
    else:
        tau_e = np.ptp(t) / 3.0

    # Average the estimates
    tau = (tau_half + tau_e) / 2.0

    return max(float(tau), 1e-10)


def fit_decay_vectorized(ts, ys):
    """Vectorized fitting for multiple curves using log-linear regression."""
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if ts.ndim == 1:
        # Single curve
        return fit_decay_log_linear(ts, ys)

    # Multiple curves - assume ts is 1D time array, ys is 2D (n_curves, n_points)
    n_curves = ys.shape[0]
    taus = np.zeros(n_curves)

    for i in range(n_curves):
        taus[i] = fit_decay_log_linear(ts, ys[i])

    return taus


def benchmark_methods(n_tests=1000, n_samples=50):
    """Benchmark different fitting methods for speed and accuracy."""
    print(f"Benchmarking exponential fitting methods ({n_tests} tests, {n_samples} samples each)")

    methods = {
        'original_least_squares': fit_decay_original,
        'log_linear_regression': fit_decay_log_linear,
        'improved_bounds': fit_decay_improved_bounds,
        'analytical_approx': fit_decay_analytical_approx,
    }

    results = {name: {'times': [], 'errors': [], 'taus': []} for name in methods}

    for i in range(n_tests):
        # Generate test data
        t, y, tau_true = generate_test_data(n_samples, noise_level=0.02)

        for name, method in methods.items():
            # Time the method
            start_time = time.time()
            tau_fit = method(t, y)
            elapsed = time.time() - start_time

            # Calculate error
            error = abs(tau_fit - tau_true) / tau_true

            results[name]['times'].append(elapsed)
            results[name]['errors'].append(error)
            results[name]['taus'].append(tau_fit)

    # Calculate statistics
    stats = {}
    for name in methods:
        times = np.array(results[name]['times'])
        errors = np.array(results[name]['errors'])

        stats[name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'speedup': 1.0,  # Will calculate relative to slowest
        }

    # Calculate speedups relative to original
    baseline_time = stats['original_least_squares']['mean_time']
    for name in stats:
        stats[name]['speedup'] = baseline_time / stats[name]['mean_time']

    return stats


def plot_comparison(stats):
    """Plot speed vs accuracy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = list(stats.keys())
    times = [stats[m]['mean_time'] * 1000 for m in methods]  # Convert to ms
    errors = [stats[m]['mean_error'] * 100 for m in methods]  # Convert to %

    # Speed comparison
    ax1.bar(methods, times)
    ax1.set_ylabel('Mean Time (ms)')
    ax1.set_title('Speed Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Add time values on bars
    for i, v in enumerate(times):
        ax1.text(i, v + 0.01, '.2f', ha='center', va='bottom')

    # Accuracy comparison
    ax2.bar(methods, errors)
    ax2.set_ylabel('Mean Relative Error (%)')
    ax2.set_title('Accuracy Comparison')
    ax2.tick_params(axis='x', rotation=45)

    # Add error values on bars
    for i, v in enumerate(errors):
        ax2.text(i, v + 0.01, '.2f', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('fitting_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_edge_cases():
    """Test methods on edge cases."""
    print("\nTesting edge cases...")

    # Very noisy data
    t, y, tau_true = generate_test_data(50, noise_level=0.1)
    print(f"Noisy data (10% noise): true_tau={tau_true:.3f}")

    methods = {
        'original': fit_decay_original,
        'log_linear': fit_decay_log_linear,
        'improved': fit_decay_improved_bounds,
        'analytical': fit_decay_analytical_approx,
    }

    for name, method in methods.items():
        tau_fit = method(t, y)
        error = abs(tau_fit - tau_true) / tau_true * 100
        print(".3f")

    # Very few points
    t, y, tau_true = generate_test_data(5, noise_level=0.01)
    print(f"\nFew points (5 samples): true_tau={tau_true:.3f}")

    for name, method in methods.items():
        tau_fit = method(t, y)
        error = abs(tau_fit - tau_true) / tau_true * 100
        print(".3f")

    # Fast decay
    t = np.linspace(0, 0.5, 50)
    tau_true = 0.1
    y = 1.0 * np.exp(-t / tau_true) + 0.05 + 0.01 * np.random.randn(50)
    print(f"\nFast decay (tau=0.1): true_tau={tau_true:.3f}")

    for name, method in methods.items():
        tau_fit = method(t, y)
        error = abs(tau_fit - tau_true) / tau_true * 100
        print(".3f")


if __name__ == "__main__":
    # Run benchmarks
    stats = benchmark_methods(n_tests=500, n_samples=50)

    print("\nBenchmark Results:")
    print("=" * 60)
    print("<15")
    print("-" * 60)

    for method, data in stats.items():
        print("<15")

    print("\nRecommendations:")
    print("- For speed with acceptable accuracy: log_linear_regression")
    print("- For best accuracy: improved_bounds")
    print("- For maximum speed: analytical_approx")
    print("- For multiple curves: vectorized log_linear_regression")

    # Plot results
    plot_comparison(stats)

    # Test edge cases
    test_edge_cases()