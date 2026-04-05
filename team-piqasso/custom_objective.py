"""
custom_objective.py — Custom objective functions for the Piqasso cat qubit model.

Implements the target objective:
    A*T_X + B*T_Z - exp(|eta - 320|)
with A = B/eta and eta = T_Z / T_X.

Two implementations are provided:
- `objective_with_ratio`: computes A = B / eta explicitly.
- `objective_expanded`: uses the equivalent expanded expression without
  assuming the shorthand formula in the final expression.
"""

import numpy as np
import matplotlib.pyplot as plt

from landscape_plot import proxy_lifetimes

__all__ = [
    "objective_with_ratio",
    "objective_expanded",
    "default_knobs",
    "compute_objective_landscape",
    "plot_objective_landscape",
    "print_top_objective_table",
]


def default_knobs():
    """Return the default real operating knobs [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]."""
    return [1.0, 0.0, 4.0, 0.0]


def objective_with_ratio(knobs, B=1.0, target_eta=320.0, t_probe_z=50.0, t_probe_x=0.3):
    """Compute the objective using A = B / eta explicitly.

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]
    B : float
        Overall scaling factor for the lifetime terms.
    target_eta : float
        Penalty center for the eta mismatch term.
    t_probe_z : float
        Probe time for the Z lifetime estimate (μs).
    t_probe_x : float
        Probe time for the X lifetime estimate (μs).

    Returns
    -------
    objective : float
        A*T_X + B*T_Z - exp(|eta - target_eta|), where A = B / eta.
    """
    T_Z, T_X = proxy_lifetimes(knobs, t_probe_z=t_probe_z, t_probe_x=t_probe_x)
    eta = T_Z / T_X
    A = B / eta
    return float(A * T_X + B * T_Z - np.exp(abs(eta - target_eta)))


def objective_expanded(knobs, B=1.0, target_eta=320.0, t_probe_z=50.0, t_probe_x=0.3):
    """Compute the objective using the expanded formula.

    This version does not explicitly use A = B / eta in the final
    expression. It is equivalent to the result of that substitution.

    objective = B * (T_X**2 / T_Z + T_Z) - exp(|eta - target_eta|)
    """
    T_Z, T_X = proxy_lifetimes(knobs, t_probe_z=t_probe_z, t_probe_x=t_probe_x)
    eta = T_Z / T_X
    return float(B * (T_X**2 / T_Z + T_Z) - np.exp(abs(eta - target_eta)))


def compute_objective_landscape(
    g2_vals=None,
    eps_d_vals=None,
    B=1.0,
    target_eta=320.0,
    t_probe_z=50.0,
    t_probe_x=0.3,
):
    """Compute the objective landscape over real Re(g2) and Re(eps_d)."""
    if g2_vals is None:
        g2_vals = np.linspace(0.2, 3.0, 25)
    if eps_d_vals is None:
        eps_d_vals = np.linspace(1.0, 8.0, 25)

    objective_grid = np.zeros((len(g2_vals), len(eps_d_vals)))
    T_Z_grid = np.zeros_like(objective_grid)
    T_X_grid = np.zeros_like(objective_grid)
    eta_grid = np.zeros_like(objective_grid)

    for j, g2_val in enumerate(g2_vals):
        for i, eps_d_val in enumerate(eps_d_vals):
            knobs = [g2_val, 0.0, eps_d_val, 0.0]
            T_Z, T_X = proxy_lifetimes(knobs, t_probe_z=t_probe_z, t_probe_x=t_probe_x)
            eta = T_Z / T_X
            objective = (B / eta) * T_X + B * T_Z - np.exp(abs(eta - target_eta))

            T_Z_grid[j, i] = T_Z
            T_X_grid[j, i] = T_X
            eta_grid[j, i] = eta
            objective_grid[j, i] = objective

    return {
        "g2_vals": g2_vals,
        "eps_d_vals": eps_d_vals,
        "T_Z": T_Z_grid,
        "T_X": T_X_grid,
        "eta": eta_grid,
        "objective": objective_grid,
    }


def plot_objective_landscape(data, save_path="custom_objective_landscape.png"):
    """Plot the custom objective landscape and save it to a PNG file."""
    g2_vals = data["g2_vals"]
    eps_d_vals = data["eps_d_vals"]
    objective = data["objective"]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    pcm = ax.pcolormesh(eps_d_vals, g2_vals, objective, cmap="viridis", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Objective", fontsize=12)

    ax.set_title("Custom objective landscape: A*T_X + B*T_Z - exp(|eta - 320|)")
    ax.set_xlabel(r"Re($\varepsilon_d$)")
    ax.set_ylabel(r"Re($g_2$)")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def print_top_objective_table(data, top_n=5):
    """Print a summary table of the top objective values."""
    g2_vals = data["g2_vals"]
    eps_d_vals = data["eps_d_vals"]
    T_Z = data["T_Z"]
    T_X = data["T_X"]
    eta = data["eta"]
    objective = data["objective"]

    flat_idx = np.argsort(objective.ravel())[::-1][:top_n]
    print("Top objective values:")
    print(
        "{:<3} {:<3} {:<8} {:<9} {:<10} {:<10} {:<10} {:<12}".format(
            "r", "c", "g2", "eps_d", "T_Z", "T_X", "eta", "objective"
        )
    )
    for idx in flat_idx:
        j, i = np.unravel_index(idx, objective.shape)
        print(
            "{:<3} {:<3} {:<8.3f} {:<9.3f} {:<10.3f} {:<10.5f} {:<10.3f} {:<12.6f}".format(
                j,
                i,
                g2_vals[j],
                eps_d_vals[i],
                T_Z[j, i],
                T_X[j, i],
                eta[j, i],
                objective[j, i],
            )
        )


if __name__ == "__main__":
    knobs = default_knobs()
    val_ratio = objective_with_ratio(knobs)
    val_expanded = objective_expanded(knobs)
    print("default knobs:", knobs)
    print("objective_with_ratio:", val_ratio)
    print("objective_expanded:", val_expanded)
    print("equal:", np.allclose(val_ratio, val_expanded, rtol=1e-9, atol=0.0))

    print("\nComputing objective landscape...")
    data = compute_objective_landscape()
    plot_path = plot_objective_landscape(data)
    print(f"Saved objective landscape plot to: {plot_path}")
    print_top_objective_table(data, top_n=5)
