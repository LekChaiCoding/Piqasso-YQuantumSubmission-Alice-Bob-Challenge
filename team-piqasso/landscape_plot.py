"""
landscape_plot.py — Compute and (optionally) plot the T_Z / T_X / eta / reward
landscape over the (eps_d, g2) parameter plane for the two-mode cat qubit.

Usage
-----
    python landscape_plot.py            # load cached data if available, else compute
    python landscape_plot.py --recompute  # always recompute
"""

import sys
import os
import time

import numpy as np
import jax.numpy as jnp
import dynamiqs as dq

# ---------------------------------------------------------------------------
# Path setup — catqubit.py lives in the same directory as this file
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from catqubit import build_hamiltonian, build_measurement_ops, NA, NB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_BIAS = 320.0

# ---------------------------------------------------------------------------
# Core simulation helper
# ---------------------------------------------------------------------------

def proxy_lifetimes(knobs, t_probe_z=50.0, t_probe_x=0.3):
    """Estimate T_Z and T_X for the given knobs using short mesolve probes.

    Parameters
    ----------
    knobs : array-like, shape (4,)
        [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]
    t_probe_z : float
        Probe time used for estimating T_Z (the bit-flip lifetime).
    t_probe_x : float
        Probe time used for estimating T_X (the phase-flip lifetime).

    Returns
    -------
    (T_Z, T_X) : tuple of float
        Estimated lifetimes in the same units as the probe times.
        Both values are clamped to a minimum of 1e-9.
    """
    try:
        ops = build_measurement_ops(knobs)
        sx = ops['sx']
        sz = ops['sz']
        alpha = ops['alpha']

        hd = build_hamiltonian(knobs)
        H = hd['H']
        loss_ops = hd['loss_ops']

        g_state = dq.coherent(NA, complex(alpha))
        e_state = dq.coherent(NA, complex(-alpha))
        b_vacuum = dq.fock(NB, 0)

        # --- T_Z: start in |-alpha>, measure <sz> ---
        psi0_z = dq.tensor(e_state, b_vacuum)
        res_z = dq.mesolve(
            H,
            loss_ops,
            psi0_z,
            jnp.array([0.0, t_probe_z]),
            exp_ops=[sx, sz],
            options=dq.Options(progress_meter=False),
        )
        sz_probe = float(
            np.clip(abs(np.array(res_z.expects[1, -1]).real), 1e-12, 1 - 1e-12)
        )
        T_Z = -t_probe_z / np.log(sz_probe)
        T_Z = max(T_Z, 1e-9)

        # --- T_X: start in (|+alpha> + |-alpha>) / sqrt(2), measure <sx> ---
        psi0_x = dq.tensor((g_state + e_state) / jnp.sqrt(2.0), b_vacuum)
        res_x = dq.mesolve(
            H,
            loss_ops,
            psi0_x,
            jnp.array([0.0, t_probe_x]),
            exp_ops=[sx, sz],
            options=dq.Options(progress_meter=False),
        )
        sx_probe = float(
            np.clip(abs(np.array(res_x.expects[0, -1]).real), 1e-12, 1 - 1e-12)
        )
        T_X = -t_probe_x / np.log(sx_probe)
        T_X = max(T_X, 1e-9)

        return (T_Z, T_X)

    except Exception:
        return (1e-9, 1e-9)


# ---------------------------------------------------------------------------
# Landscape computation
# ---------------------------------------------------------------------------

def compute_landscape(n_eps=40, n_g2=40):
    """Sweep (eps_d, g2) and compute T_Z, T_X, eta, and reward at each point.

    Parameters
    ----------
    n_eps : int
        Number of eps_d samples (Re component; Im set to 0).
    n_g2 : int
        Number of g2 samples (Re component; Im set to 0).

    Returns
    -------
    dict with keys:
        eps_d_arr : ndarray, shape (n_eps,)
        g2_arr    : ndarray, shape (n_g2,)
        T_Z       : ndarray, shape (n_g2, n_eps)
        T_X       : ndarray, shape (n_g2, n_eps)
        eta       : ndarray, shape (n_g2, n_eps)   — T_Z / T_X
        reward    : ndarray, shape (n_g2, n_eps)
    """
    eps_d_arr = np.linspace(1.0, 8.0, n_eps)
    g2_arr = np.linspace(0.2, 3.0, n_g2)

    T_Z_grid = np.full((n_g2, n_eps), 1e-9)
    T_X_grid = np.full((n_g2, n_eps), 1e-9)

    t_start = time.time()
    first_call = True

    for j, g2_val in enumerate(g2_arr):
        row_t0 = time.time()

        for i, eps_d_val in enumerate(eps_d_arr):
            knobs = [g2_val, 0.0, eps_d_val, 0.0]

            if first_call:
                print("First call includes JAX JIT compilation, please wait...")
                first_call = False

            try:
                tz, tx = proxy_lifetimes(knobs)
            except Exception:
                tz, tx = 1e-9, 1e-9

            T_Z_grid[j, i] = tz
            T_X_grid[j, i] = tx

        elapsed = time.time() - t_start
        rows_done = j + 1
        rows_left = n_g2 - rows_done
        time_per_row = elapsed / rows_done
        eta_sec = time_per_row * rows_left

        print(
            f"[Row {rows_done}/{n_g2}] g2={g2_val:.2f} | "
            f"elapsed {elapsed:.0f}s | ETA ~{eta_sec:.0f}s"
        )

    eta_grid = T_Z_grid / np.where(T_X_grid > 0, T_X_grid, 1e-300)

    # reward = 0.3*log10(T_Z) + 0.3*log10(T_X) - 0.4*|log10(eta/TARGET_BIAS)|
    reward_grid = (
        0.3 * np.log10(np.maximum(T_Z_grid, 1e-300))
        + 0.3 * np.log10(np.maximum(T_X_grid, 1e-300))
        - 0.4 * np.abs(np.log10(np.maximum(eta_grid, 1e-300) / TARGET_BIAS))
    )

    return {
        "eps_d_arr": eps_d_arr,
        "g2_arr": g2_arr,
        "T_Z": T_Z_grid,
        "T_X": T_X_grid,
        "eta": eta_grid,
        "reward": reward_grid,
    }


# ---------------------------------------------------------------------------
# Placeholder plotter
# ---------------------------------------------------------------------------

def plot_landscape(data, save_path="landscape_plot.png"):
    import matplotlib.pyplot as plt

    eps_d_arr = data["eps_d_arr"]
    g2_arr    = data["g2_arr"]
    T_Z       = data["T_Z"]
    T_X       = data["T_X"]
    eta       = data["eta"]
    reward    = data["reward"]

    log_TZ     = np.log10(np.where(T_Z  > 0, T_Z,  np.nan))
    log_TX     = np.log10(np.where(T_X  > 0, T_X,  np.nan))
    log_eta    = np.log10(np.where(eta  > 0, eta,   np.nan))
    log_target = np.log10(TARGET_BIAS)

    max_dev  = np.nanmax(np.abs(log_eta - log_target))
    vmin_eta = log_target - max_dev
    vmax_eta = log_target + max_dev

    fig, axes = plt.subplots(2, 2, figsize=(22, 18), dpi=150)
    fig.suptitle(
        r"Cat Qubit Landscape: $T_Z$, $T_X$, $\eta$, Reward"
        f" over Re($\\varepsilon_d$) × Re($g_2$)   [η=320 contour shown]",
        fontsize=18,
    )

    panel_cfg = [
        (axes[0, 0], log_TZ,  "plasma",  None,     None,
         r"$\log_{10}(T_Z\ [\mu s])$",  "Phase-flip lifetime  log₁₀(T_Z)"),
        (axes[0, 1], log_TX,  "viridis", None,     None,
         r"$\log_{10}(T_X\ [\mu s])$",  "Bit-flip lifetime  log₁₀(T_X)"),
        (axes[1, 0], log_eta, "RdBu_r",  vmin_eta, vmax_eta,
         r"$\log_{10}(\eta)$",           r"Bias  log₁₀(η)   [η=320 contour]"),
        (axes[1, 1], reward,  "magma",   None,     None,
         "Reward R",                     r"Reward R   [η=320 contour]"),
    ]

    for ax, Z, cmap, vmin, vmax, cbar_label, title in panel_cfg:
        kwargs = dict(cmap=cmap, shading="auto")
        if vmin is not None:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax

        pcm  = ax.pcolormesh(eps_d_arr, g2_arr, Z, **kwargs)
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=12)

        # White dashed η=320 contour on every panel
        try:
            cs = ax.contour(
                eps_d_arr, g2_arr, eta,
                levels=[TARGET_BIAS],
                colors="white",
                linewidths=3,
                linestyles="--",
            )
            if cs.allsegs and any(len(s) > 0 for s in cs.allsegs[0]):
                ax.clabel(cs, fmt=r"η=320", fontsize=11, inline=True)
        except Exception:
            pass

        # Extra bold yellow contour on the bias panel only
        if ax is axes[1, 0]:
            try:
                cs2 = ax.contour(
                    eps_d_arr, g2_arr, eta,
                    levels=[TARGET_BIAS],
                    colors="yellow",
                    linewidths=5,
                    linestyles="--",
                )
                if cs2.allsegs and any(len(s) > 0 for s in cs2.allsegs[0]):
                    ax.clabel(cs2, fmt=r"η=320", fontsize=11, inline=True)
            except Exception:
                pass

        ax.set_xlabel(r"Re($\varepsilon_d$)", fontsize=14)
        ax.set_ylabel(r"Re($g_2$)",           fontsize=14)
        ax.set_title(title,                   fontsize=15)
        ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    npz_path = os.path.join(SCRIPT_DIR, "landscape_data.npz")

    if os.path.exists(npz_path) and "--recompute" not in sys.argv:
        print(f"Loading cached landscape data from {npz_path}")
        raw = np.load(npz_path)
        data = {k: raw[k] for k in raw.files}
    else:
        print("Computing landscape (this may take a while)...")
        data = compute_landscape()
        np.savez(
            npz_path,
            eps_d_arr=data["eps_d_arr"],
            g2_arr=data["g2_arr"],
            T_Z=data["T_Z"],
            T_X=data["T_X"],
            eta=data["eta"],
            reward=data["reward"],
        )
        print(f"Saved landscape data to {npz_path}")

    plot_landscape(data, save_path=os.path.join(SCRIPT_DIR, "landscape_plot.png"))


if __name__ == "__main__":
    main()
