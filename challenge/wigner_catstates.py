"""
Wigner Function Visualisation — Even & Odd Cat States
======================================================
Plots the Wigner quasi-probability distributions of even and odd cat states
(|+x⟩ and |−x⟩) and shows how they evolve under single-photon loss,
illustrating the parity-exchange mechanism between the two states.

Outputs:
  wigner_static.png       — static Wigner of even, odd, and a superposition
  wigner_evolution.png    — time-evolution snapshots under photon loss
  wigner_parity_decay.png — parity ⟨Π⟩ decay curves for both states
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── Parameters ──────────────────────────────────────────────────────────────
na    = 25       # Fock-space truncation (larger for clean Wigners)
alpha = 2.0      # cat-state amplitude
kappa = 1.0      # single-photon loss rate (MHz)

# ── Operators ───────────────────────────────────────────────────────────────
a      = dq.destroy(na)
n_op   = a.dag() @ a
parity = (1j * jnp.pi * n_op).expm()          # (-1)^n  parity operator

# ── Cat states ──────────────────────────────────────────────────────────────
ket_p    = dq.coherent(na, alpha)
ket_m    = dq.coherent(na, -alpha)
cat_even = dq.unit(ket_p + ket_m)              # |+x⟩  (even Fock numbers)
cat_odd  = dq.unit(ket_p - ket_m)              # |−x⟩  (odd  Fock numbers)

# Equal superposition of even and odd = coherent state |alpha⟩ (sanity check)
cat_sup  = dq.unit(cat_even + cat_odd)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Static Wigner functions
# ══════════════════════════════════════════════════════════════════════════════
def plot_static_wigners():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    dq.plot.wigner(cat_even, ax=axes[0])
    axes[0].set_title(r"Even cat  $|+x\rangle$", fontsize=13)

    dq.plot.wigner(cat_odd, ax=axes[1])
    axes[1].set_title(r"Odd cat  $|-x\rangle$", fontsize=13)

    dq.plot.wigner(cat_sup, ax=axes[2])
    axes[2].set_title(r"$|+x\rangle + |-x\rangle = |\alpha\rangle$", fontsize=13)

    fig.suptitle(rf"Cat states with $\alpha = {alpha}$", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig("wigner_static.png", dpi=150, bbox_inches="tight")
    print("Saved wigner_static.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Time evolution under single-photon loss
# ══════════════════════════════════════════════════════════════════════════════
def plot_time_evolution():
    """Evolve both cats under L = sqrt(kappa) * a  and snapshot Wigners."""
    L      = [jnp.sqrt(kappa) * a]
    H      = jnp.zeros((na, na), dtype=jnp.complex128)   # free decay (no drive)

    # Time points for snapshots
    snap_times = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 4.0])
    tsave      = jnp.array(snap_times)

    res_even = dq.mesolve(H, L, cat_even, tsave,
                          options=dq.Options(progress_meter=False))
    res_odd  = dq.mesolve(H, L, cat_odd, tsave,
                          options=dq.Options(progress_meter=False))

    n_snap = len(snap_times)
    fig, axes = plt.subplots(2, n_snap, figsize=(3.5 * n_snap, 7))

    for i in range(n_snap):
        dq.plot.wigner(res_even.states[i], ax=axes[0, i])
        axes[0, i].set_title(rf"$t = {snap_times[i]:.1f}\,\mu s$", fontsize=11)
        dq.plot.wigner(res_odd.states[i], ax=axes[1, i])

    axes[0, 0].set_ylabel("Even cat", fontsize=12)
    axes[1, 0].set_ylabel("Odd cat", fontsize=12)

    fig.suptitle(
        rf"Cat-state decay under single-photon loss ($\kappa = {kappa}$ MHz)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("wigner_evolution.png", dpi=150, bbox_inches="tight")
    print("Saved wigner_evolution.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Parity decay and exchange
# ══════════════════════════════════════════════════════════════════════════════
def plot_parity_decay():
    """
    Show ⟨Π⟩ vs time for even-cat and odd-cat initial states.
    Even cat starts at ⟨Π⟩ = +1, odd cat at ⟨Π⟩ = -1.
    Under loss they decay toward 0 (mixed parity).
    """
    L     = [jnp.sqrt(kappa) * a]
    H     = jnp.zeros((na, na), dtype=jnp.complex128)
    tsave = jnp.linspace(0.0, 5.0, 120)

    res_even = dq.mesolve(H, L, cat_even, tsave, exp_ops=[parity],
                          options=dq.Options(progress_meter=False))
    res_odd  = dq.mesolve(H, L, cat_odd, tsave, exp_ops=[parity],
                          options=dq.Options(progress_meter=False))

    parity_even = np.array(res_even.expects[0, :].real)
    parity_odd  = np.array(res_odd.expects[0, :].real)
    ts          = np.array(tsave)

    # Analytic estimate  T_X ~ 1 / (kappa * alpha^2)
    Tx_est = 1.0 / (kappa * alpha**2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, parity_even, "b-",  lw=2, label=r"Even cat $\langle\Pi\rangle$")
    ax.plot(ts, parity_odd,  "r-",  lw=2, label=r"Odd cat $\langle\Pi\rangle$")
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.axvline(Tx_est, color="green", ls=":", lw=1.5,
               label=rf"$T_X \approx 1/(\kappa\alpha^2) = {Tx_est:.2f}\,\mu s$")
    ax.set_xlabel(r"Time ($\mu$s)", fontsize=12)
    ax.set_ylabel(r"$\langle \Pi \rangle$  (parity)", fontsize=12)
    ax.set_title(
        rf"Parity decay of even / odd cat states ($\alpha={alpha}$, $\kappa={kappa}$ MHz)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(-1.15, 1.15)
    fig.tight_layout()
    fig.savefig("wigner_parity_decay.png", dpi=150, bbox_inches="tight")
    print("Saved wigner_parity_decay.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Cat-state Wigner analysis  (alpha={alpha}, na={na}, kappa={kappa})")
    print("=" * 60)

    plot_static_wigners()
    plot_time_evolution()
    plot_parity_decay()

    print("=" * 60)
    print("All figures saved.")
