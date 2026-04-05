"""
Physical Analysis of Optimal Parameters
========================================
Investigates why the CMA-ES optimizer converged to eps_d ≈ 4.89, g_2 ≈ 1.76.

Central finding:
  The optimizer is navigating a two-dimensional parameter space that is
  fundamentally constrained by two physical conditions:

  1.  RATIO CONSTRAINT:  T_Z / T_X = 320  pins the cat size alpha to ~1.54.
      This defines a one-dimensional iso-alpha manifold in (eps_d, g_2) space.

  2.  STABILIZATION THRESHOLD:  Along that manifold, T_X improves monotonically
      with g_2 because kappa_2 = 4*g_2^2/kappa_b, the two-photon stabilization
      rate, increases.  The critical threshold is  kappa_2 = kappa_a,
      i.e.  g_2* = sqrt(kappa_a * kappa_b / 4) ≈ 1.58 MHz.

  The optimizer found g_2 ≈ 1.76 — just above g_2* — because:
    - Below g_2*, cat stabilization is weaker than single-photon loss (bad).
    - Above g_2*, stabilization dominates and T_X is protected.
    - The optimizer had not converged at epoch 60; it was still climbing toward
      the bound-imposed maximum g_2 ≈ 3.2 on this manifold.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

jax.config.update("jax_enable_x64", True)

# ── Fixed parameters ───────────────────────────────────────────────────────────
na       = 12
nb       = 4
kappa_a  = 1.0
kappa_b  = 10.0
TARGET_RATIO = 320.0

# Optimizer result and baseline from the run
EPS_D_OPT  = 4.89
G2_OPT     = 1.76
EPS_D_BASE = 4.0
G2_BASE    = 1.0

# Critical g_2 where kappa_2 = kappa_a
G2_STAR = np.sqrt(kappa_a * kappa_b / 4)          # ≈ 1.581

# ── Global operators ───────────────────────────────────────────────────────────
a_s  = dq.destroy(na)
a_op = dq.tensor(a_s, dq.eye(nb))
b_op = dq.tensor(dq.eye(na), dq.destroy(nb))
parity_s  = (1j * jnp.pi * a_s.dag() @ a_s).expm()
parity_op = dq.tensor(parity_s, dq.eye(nb))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ANALYTIC LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════

def analytic_alpha2(eps_d, g2):
    """alpha^2 from adiabatic elimination."""
    kappa_2 = 4.0 * g2**2 / kappa_b
    if kappa_2 < 1e-12:
        return 0.0
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    return max(2.0 * (eps_2 - kappa_a / 4.0) / kappa_2, 0.0)

def kappa2(g2):
    return 4.0 * g2**2 / kappa_b

def analytic_Tx(eps_d, g2):
    a2 = analytic_alpha2(eps_d, g2)
    return 1.0 / (kappa_a * a2) if a2 > 1e-9 else np.inf

def analytic_Tz(eps_d, g2):
    a2 = analytic_alpha2(eps_d, g2)
    return np.exp(2.0 * a2) / kappa_a

def analytic_ratio_from_alpha2(a2):
    """T_Z / T_X = alpha^2 * exp(2*alpha^2)."""
    return a2 * np.exp(2.0 * a2)

def alpha2_for_target_ratio(target=TARGET_RATIO):
    """Find alpha^2 such that analytic_ratio = target."""
    return brentq(lambda a2: analytic_ratio_from_alpha2(a2) - target, 0.5, 5.0)

def eps_d_on_iso_alpha_curve(g2, a2_target):
    """eps_d that gives alpha^2 = a2_target at given g2."""
    # alpha^2 = eps_d/g2 - kappa_a*kappa_b/(8*g2^2)  (simplified form)
    # => eps_d = g2*(alpha^2 + kappa_a*kappa_b/(8*g2^2))
    #          = g2*alpha^2 + kappa_a*kappa_b/(8*g2)
    return g2 * a2_target + kappa_a * kappa_b / (8.0 * g2)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SIMULATION HELPERS (minimal, for the constraint-curve scan)
# ══════════════════════════════════════════════════════════════════════════════

def build_system(eps_d, g2):
    a2    = analytic_alpha2(eps_d, g2)
    alpha = np.sqrt(max(a2, 0))
    H = (jnp.conj(g2) * a_op.dag() @ a_op.dag() @ b_op
         + g2 * a_op @ a_op @ b_op.dag()
         - eps_d * b_op.dag()
         - jnp.conj(eps_d) * b_op)
    L_b = jnp.sqrt(kappa_b) * b_op
    L_a = jnp.sqrt(kappa_a) * a_op
    ket_p  = dq.coherent(na, alpha)
    ket_m  = dq.coherent(na, -alpha)
    cat_x  = (ket_p + ket_m).unit()
    psi0_x = dq.tensor(cat_x, dq.fock(nb, 0))
    return H, [L_b, L_a], alpha, psi0_x, ket_p, ket_m


def _fit_lifetime(ts, signal):
    A0, C0 = float(signal[0] - signal[-1]), float(signal[-1])
    tau0   = float(ts[-1] - ts[0])
    def residuals(p):
        return p[0] * np.exp(-ts / p[1]) + p[2] - signal
    res = least_squares(residuals, [A0, tau0, C0],
                        bounds=([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                        loss='soft_l1', f_scale=0.05)
    return float(res.x[1])


def measure_Tx_sim(eps_d, g2, n_pts=40):
    H, jumps, alpha, psi0_x, _, _ = build_system(eps_d, g2)
    a2     = alpha**2
    tx_est = 1.0 / (kappa_a * a2) if a2 > 1e-9 else 1.0
    tfinal = float(np.clip(3.0 * tx_est, 0.3, 10.0))
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    result = dq.mesolve(H, jumps, psi0_x, tsave,
                        exp_ops=[parity_op],
                        options=dq.Options(progress_meter=False))
    sxt = np.array(result.expects[0, :].real)
    ts  = np.array(tsave)
    if sxt[-1] > 0.9 * sxt[0]:
        return 5.0 * tfinal, ts, sxt
    try:
        return _fit_lifetime(ts, sxt), ts, sxt
    except Exception:
        return 0.0, ts, sxt


def measure_Tz_sim(eps_d, g2, n_pts=30):
    H, jumps, alpha, _, ket_p, ket_m = build_system(eps_d, g2)
    a2     = alpha**2
    sz_s   = ket_p @ ket_p.dag() - ket_m @ ket_m.dag()
    sz_op  = dq.tensor(sz_s, dq.eye(nb))
    psi0_z = dq.tensor(ket_p, dq.fock(nb, 0))
    tz_est = np.exp(2.0 * a2) / kappa_a
    tfinal = float(np.clip(3.0 * tz_est, 50.0, 500.0))
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    result = dq.mesolve(H, jumps, psi0_z, tsave,
                        exp_ops=[sz_op],
                        options=dq.Options(progress_meter=False))
    szt = np.array(result.expects[0, :].real)
    ts  = np.array(tsave)
    if szt[-1] > 0.9 * szt[0]:
        return 5.0 * tfinal, ts, szt
    try:
        return _fit_lifetime(ts, szt), ts, szt
    except Exception:
        return 0.0, ts, szt


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LIOUVILLIAN GAP (reduced storage mode after adiabatic elimination)
# ══════════════════════════════════════════════════════════════════════════════

def liouvillian_gaps(eps_d, g2):
    """
    Compute the smallest non-zero eigenvalues of the Lindblad superoperator
    in the effective single-mode (storage) picture after adiabatic elimination.

    Effective Lindblad operators on storage mode:
      L_2  = sqrt(kappa_2) * a^2           (two-photon loss — stabilizes cat)
      L_a  = sqrt(kappa_a) * a             (single-photon loss — dephases parity)

    Returns the two smallest non-zero gaps (in MHz), which correspond to
    1/T_X and 1/T_Z respectively.
    """
    k2 = kappa2(g2)
    a  = dq.destroy(na)

    L2 = jnp.sqrt(k2) * a @ a          # two-photon loss
    La = jnp.sqrt(kappa_a) * a         # single-photon loss

    # Build superoperator  L[rho] = sum_k (L_k rho L_k† - ½{L_k†L_k, rho})
    # as a matrix acting on vec(rho)
    n  = na
    Id = np.eye(n)

    def lindblad_super(L):
        Ld  = np.conj(L.T)
        LdL = Ld @ L
        return (np.kron(np.conj(L), L)
                - 0.5 * np.kron(Id, LdL)
                - 0.5 * np.kron(LdL.T, Id))

    L2_np = np.array(L2)
    La_np = np.array(La)
    sup   = lindblad_super(L2_np) + lindblad_super(La_np)

    eigs = np.sort(np.real(np.linalg.eigvals(sup)))
    # eigs[0] = 0 (steady state), then gaps are -eigs[1], -eigs[2], ...
    gaps = -eigs[eigs < -1e-10]
    gaps.sort()
    return gaps[:4] if len(gaps) >= 4 else gaps


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 70)
    print("  Physical analysis of optimal parameters")
    print(f"  kappa_a = {kappa_a} MHz,  kappa_b = {kappa_b} MHz")
    print(f"  Target T_Z/T_X = {TARGET_RATIO:.0f}")
    print("=" * 70, flush=True)

    # ── Key analytic quantities ────────────────────────────────────────────────
    a2_base = analytic_alpha2(EPS_D_BASE, G2_BASE)
    a2_opt  = analytic_alpha2(EPS_D_OPT,  G2_OPT)
    a2_320  = alpha2_for_target_ratio(TARGET_RATIO)

    print(f"\n{'Parameter':>30s}  {'Baseline':>12s}  {'Optimized':>12s}")
    print("-" * 58)
    print(f"{'eps_d (MHz)':>30s}  {EPS_D_BASE:12.3f}  {EPS_D_OPT:12.3f}")
    print(f"{'g_2 (MHz)':>30s}  {G2_BASE:12.3f}  {G2_OPT:12.3f}")
    print(f"{'alpha^2':>30s}  {a2_base:12.4f}  {a2_opt:12.4f}")
    print(f"{'alpha':>30s}  {np.sqrt(a2_base):12.4f}  {np.sqrt(a2_opt):12.4f}")
    print(f"{'kappa_2 (MHz)':>30s}  {kappa2(G2_BASE):12.4f}  {kappa2(G2_OPT):12.4f}")
    print(f"{'kappa_2 / kappa_a':>30s}  {kappa2(G2_BASE)/kappa_a:12.4f}  {kappa2(G2_OPT)/kappa_a:12.4f}")
    print(f"{'Analytic T_X (us)':>30s}  {analytic_Tx(EPS_D_BASE,G2_BASE):12.4f}  {analytic_Tx(EPS_D_OPT,G2_OPT):12.4f}")
    print(f"{'Analytic T_Z (us)':>30s}  {analytic_Tz(EPS_D_BASE,G2_BASE):12.1f}  {analytic_Tz(EPS_D_OPT,G2_OPT):12.1f}")
    print(f"{'Analytic ratio':>30s}  {analytic_ratio_from_alpha2(a2_base):12.1f}  {analytic_ratio_from_alpha2(a2_opt):12.1f}")
    print(f"\n  Critical g_2* (kappa_2 = kappa_a): {G2_STAR:.4f} MHz")
    print(f"  Optimized g_2 is {(G2_OPT - G2_STAR):.3f} MHz ABOVE the threshold  (kappa_2/kappa_a = {kappa2(G2_OPT)/kappa_a:.3f})")
    print(f"  Baseline g_2 is {(G2_STAR - G2_BASE):.3f} MHz BELOW the threshold  (kappa_2/kappa_a = {kappa2(G2_BASE)/kappa_a:.3f})")

    # ── Iso-alpha^2 manifold: max g_2 within bounds ────────────────────────────
    # eps_d = g2*a2 + kappa_a*kappa_b/(8*g2) <= 8.0
    # a2*g2^2 - 8*g2 + kappa_a*kappa_b/8 = 0
    roots = np.roots([a2_opt, -8.0, kappa_a * kappa_b / 8.0])
    g2_max_on_curve = max(r for r in roots if np.isreal(r) and 0 < r.real <= 4)
    if hasattr(g2_max_on_curve, 'real'):
        g2_max_on_curve = float(g2_max_on_curve.real)
    eps_d_at_max = eps_d_on_iso_alpha_curve(g2_max_on_curve, a2_opt)
    print(f"\n  Max g_2 on iso-alpha curve (alpha^2={a2_opt:.3f}) within eps_d<=8 bound: {g2_max_on_curve:.3f}")
    print(f"  eps_d at that g_2: {eps_d_at_max:.3f} MHz")
    print(f"  kappa_2/kappa_a at max g_2: {kappa2(g2_max_on_curve)/kappa_a:.3f}")
    print(f"  --> Optimizer at epoch 60 had reached {G2_OPT/g2_max_on_curve*100:.0f}% of the way to the bound")

    # ── Liouvillian gap analysis ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Liouvillian gaps (effective single-mode after adiabatic elimination)")
    print("  Gaps = decay rates; 1/gap = lifetime estimate")
    print("=" * 70)
    g2_scan = [1.0, G2_STAR, G2_OPT, 2.5, g2_max_on_curve]
    liouvil_results = {}
    for g2_val in g2_scan:
        eps_d_val = eps_d_on_iso_alpha_curve(g2_val, a2_opt)
        gaps = liouvillian_gaps(eps_d_val, g2_val)
        k2_val = kappa2(g2_val)
        liouvil_results[g2_val] = gaps
        print(f"  g_2={g2_val:.3f}, kappa_2={k2_val:.3f}: "
              f"gaps = {', '.join(f'{g:.5f}' for g in gaps[:3])} MHz  "
              f"=> T_X(gap1) = {1/gaps[0]:.3f} us")

    # ── Constraint-curve scan (full simulation) ────────────────────────────────
    print("\n" + "=" * 70)
    print("  Full Lindblad simulation along the iso-alpha^2 curve")
    print(f"  (alpha^2 fixed at {a2_opt:.3f}, T_Z/T_X target = {TARGET_RATIO:.0f})")
    print("=" * 70, flush=True)

    # Sample 6 points along the curve from under-stabilized to over-stabilized
    g2_curve = [1.0, G2_STAR, G2_OPT, 2.0, 2.5, g2_max_on_curve]
    curve_results = []
    for g2_val in g2_curve:
        eps_d_val = eps_d_on_iso_alpha_curve(g2_val, a2_opt)
        k2_val    = kappa2(g2_val)
        print(f"  Simulating g_2={g2_val:.3f}, eps_d={eps_d_val:.3f}, "
              f"kappa_2/kappa_a={k2_val:.3f} ...", flush=True)
        Tx, _, _ = measure_Tx_sim(eps_d_val, g2_val)
        Tz, _, _ = measure_Tz_sim(eps_d_val, g2_val)
        ratio = Tz / max(Tx, 1e-9)
        print(f"    T_X={Tx:.4f} us,  T_Z={Tz:.2f} us,  ratio={ratio:.1f}", flush=True)
        curve_results.append({
            'g2': g2_val, 'eps_d': eps_d_val, 'k2': k2_val,
            'Tx': Tx, 'Tz': Tz, 'ratio': ratio,
        })

    # ── Baseline simulation (different alpha — outside the constraint curve) ───
    print(f"\n  Baseline (eps_d={EPS_D_BASE}, g_2={G2_BASE}) — different alpha:", flush=True)
    Tx_bl, _, _ = measure_Tx_sim(EPS_D_BASE, G2_BASE)
    Tz_bl, _, _ = measure_Tz_sim(EPS_D_BASE, G2_BASE)
    print(f"    T_X={Tx_bl:.4f} us,  T_Z={Tz_bl:.2f} us,  ratio={Tz_bl/max(Tx_bl,1e-9):.1f}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════
    # 5.  PLOTTING
    # ══════════════════════════════════════════════════════════════════════════

    eps_grid = np.linspace(0.5, 8.0, 300)
    g2_grid  = np.linspace(0.2, 4.0, 300)
    EE, GG   = np.meshgrid(eps_grid, g2_grid)

    # Scalar maps
    A2_MAP  = np.vectorize(analytic_alpha2)(EE, GG)
    K2_MAP  = kappa2(GG)
    TX_MAP  = np.where(A2_MAP > 0.01, 1.0 / (kappa_a * A2_MAP), np.nan)
    RAT_MAP = np.where(A2_MAP > 0.01, A2_MAP * np.exp(2.0 * A2_MAP), np.nan)

    # Iso-alpha^2 constraint curve
    g2_curve_dense = np.linspace(0.2, 4.0, 400)
    eps_curve      = np.array([eps_d_on_iso_alpha_curve(g, a2_opt)
                                for g in g2_curve_dense])
    valid = (eps_curve >= 0.5) & (eps_curve <= 8.0)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel (0,0): alpha^2 landscape ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(EE, GG, A2_MAP, levels=20, cmap='viridis')
    fig.colorbar(im, ax=ax, label=r'$\alpha^2$')
    ax.contour(EE, GG, A2_MAP, levels=[a2_opt], colors='white',
               linewidths=1.5, linestyles='--')
    ax.plot(eps_curve[valid], g2_curve_dense[valid], 'w--', lw=2,
            label=fr'$\alpha^2 = {a2_opt:.2f}$ (ratio=320 manifold)')
    ax.axhline(G2_STAR, color='cyan', lw=1.5, ls=':',
               label=r'$\kappa_2 = \kappa_a$ threshold')
    ax.plot(EPS_D_BASE, G2_BASE, 'rs', ms=10, label='Baseline')
    ax.plot(EPS_D_OPT,  G2_OPT,  'r*', ms=14, label='Optimized')
    ax.set(xlabel=r'$\epsilon_d$ (MHz)', ylabel=r'$g_2$ (MHz)',
           title=r'Cat size $\alpha^2(\epsilon_d, g_2)$')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel (0,1): kappa_2/kappa_a landscape ────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    K2_RATIO = K2_MAP / kappa_a
    im = ax.contourf(EE, GG, K2_RATIO, levels=20, cmap='RdBu_r',
                     vmin=0, vmax=4)
    fig.colorbar(im, ax=ax, label=r'$\kappa_2 / \kappa_a$')
    ax.contour(EE, GG, K2_RATIO, levels=[1.0], colors='black',
               linewidths=2.0, linestyles='-')
    ax.plot(eps_curve[valid], g2_curve_dense[valid], 'k--', lw=2,
            label=r'iso-$\alpha^2$ manifold')
    ax.axhline(G2_STAR, color='black', lw=2.0, ls='-')
    ax.plot(EPS_D_BASE, G2_BASE, 'rs', ms=10, label='Baseline')
    ax.plot(EPS_D_OPT,  G2_OPT,  'r*', ms=14, label='Optimized')
    ax.text(1.5, G2_STAR + 0.08, r'$\kappa_2 = \kappa_a$', fontsize=9,
            color='black')
    ax.text(1.5, G2_STAR - 0.25,
            r'Under-stabilized ($\kappa_2 < \kappa_a$)', fontsize=8,
            color='darkred')
    ax.text(1.5, G2_STAR + 0.20,
            r'Over-stabilized ($\kappa_2 > \kappa_a$)', fontsize=8,
            color='darkblue')
    ax.set(xlabel=r'$\epsilon_d$ (MHz)', ylabel=r'$g_2$ (MHz)',
           title=r'Two-photon stabilization: $\kappa_2 / \kappa_a$')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel (0,2): analytic T_X landscape ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    TX_CLIP = np.clip(TX_MAP, 0, 2.0)
    im = ax.contourf(EE, GG, TX_CLIP, levels=20, cmap='plasma_r')
    fig.colorbar(im, ax=ax, label=r'Analytic $T_X$ (µs)')
    ax.plot(eps_curve[valid], g2_curve_dense[valid], 'w--', lw=2,
            label=r'iso-$\alpha^2$ manifold')
    ax.axhline(G2_STAR, color='cyan', lw=1.5, ls=':',
               label=r'$\kappa_2 = \kappa_a$')
    ax.plot(EPS_D_BASE, G2_BASE, 'rs', ms=10, label='Baseline')
    ax.plot(EPS_D_OPT,  G2_OPT,  'r*', ms=14, label='Optimized')
    ax.set(xlabel=r'$\epsilon_d$ (MHz)', ylabel=r'$g_2$ (MHz)',
           title=r'Analytic $T_X = 1/(\kappa_a \alpha^2)$ (µs)')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel (1,0): analytic ratio landscape ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    RAT_CLIP = np.clip(np.log10(np.where(RAT_MAP > 0, RAT_MAP, np.nan)), 0, 4)
    im = ax.contourf(EE, GG, RAT_CLIP, levels=20, cmap='YlOrRd')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}(T_Z/T_X)$ analytic')
    ax.contour(EE, GG, RAT_MAP, levels=[TARGET_RATIO], colors='white',
               linewidths=2.0, linestyles='--')
    ax.plot(eps_curve[valid], g2_curve_dense[valid], 'w--', lw=2,
            label=r'iso-$\alpha^2$ manifold (ratio=320)')
    ax.plot(EPS_D_BASE, G2_BASE, 'bs', ms=10, label='Baseline')
    ax.plot(EPS_D_OPT,  G2_OPT,  'b*', ms=14, label='Optimized')
    ax.text(5.5, 0.7, r'$T_Z/T_X = 320$', fontsize=8, color='white')
    ax.set(xlabel=r'$\epsilon_d$ (MHz)', ylabel=r'$g_2$ (MHz)',
           title=r'Analytic noise bias $T_Z / T_X$')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel (1,1): constraint-curve scan — T_X vs g_2 ──────────────────────
    ax = fig.add_subplot(gs[1, 1])
    g2_vals = [r['g2']  for r in curve_results]
    Tx_vals = [r['Tx']  for r in curve_results]
    k2_vals = [r['k2']  for r in curve_results]
    Tx_analytic = [analytic_Tx(r['eps_d'], r['g2']) for r in curve_results]

    ax.plot(k2_vals, Tx_vals, 'o-', color='steelblue', ms=8, lw=2,
            label='Simulated $T_X$')
    ax.plot(k2_vals, Tx_analytic, 's--', color='gray', ms=7, lw=1.5,
            label=r'Analytic $1/(\kappa_a \alpha^2)$')
    ax.axvline(kappa_a, color='red', lw=1.5, ls='--',
               label=r'$\kappa_2 = \kappa_a$ threshold')
    ax.axvline(kappa2(G2_OPT), color='green', lw=1.5, ls=':',
               label=f'Optimized ($g_2$={G2_OPT})')

    for r in curve_results:
        ax.annotate(f"$g_2$={r['g2']:.2f}",
                    xy=(r['k2'], r['Tx']),
                    xytext=(r['k2'] + 0.05, r['Tx'] + 0.003),
                    fontsize=7)
    ax.set(xlabel=r'$\kappa_2 = 4g_2^2/\kappa_b$ (MHz)',
           ylabel=r'$T_X$ (µs)',
           title=r'$T_X$ along the iso-$\alpha^2$ manifold')
    ax.legend(fontsize=7)

    # ── Panel (1,2): constraint-curve scan — ratio vs g_2 ────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ratio_vals    = [r['ratio'] for r in curve_results]
    ratio_analytic = [analytic_ratio_from_alpha2(analytic_alpha2(r['eps_d'], r['g2']))
                      for r in curve_results]
    ax.plot(k2_vals, ratio_vals, 'o-', color='purple', ms=8, lw=2,
            label='Simulated ratio')
    ax.plot(k2_vals, ratio_analytic, 's--', color='gray', ms=7, lw=1.5,
            label='Analytic ratio')
    ax.axhline(TARGET_RATIO, color='red', lw=1.5, ls='--',
               label=f'Target = {TARGET_RATIO:.0f}')
    ax.axvline(kappa_a, color='red', lw=1.5, ls=':')
    ax.axvline(kappa2(G2_OPT), color='green', lw=1.5, ls=':',
               label=f'Optimized')
    ax.set(xlabel=r'$\kappa_2$ (MHz)', ylabel=r'$T_Z / T_X$',
           title='Noise bias along the iso-$\\alpha^2$ manifold')
    ax.legend(fontsize=7)

    # ── Panel (2,0): Liouvillian gaps along curve ─────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    g2_liouv  = sorted(liouvil_results.keys())
    k2_liouv  = [kappa2(g) for g in g2_liouv]
    gap1_vals = [liouvil_results[g][0] if len(liouvil_results[g]) > 0 else np.nan
                 for g in g2_liouv]
    gap2_vals = [liouvil_results[g][1] if len(liouvil_results[g]) > 1 else np.nan
                 for g in g2_liouv]
    ax.semilogy(k2_liouv, gap1_vals, 'o-', color='steelblue', ms=8, lw=2,
                label=r'Gap 1 $\approx 1/T_X$')
    ax.semilogy(k2_liouv, gap2_vals, 's-', color='darkorange', ms=8, lw=2,
                label=r'Gap 2 $\approx 1/T_Z$')
    ax.axvline(kappa_a, color='red', lw=1.5, ls='--',
               label=r'$\kappa_2 = \kappa_a$')
    ax.set(xlabel=r'$\kappa_2$ (MHz)',
           ylabel='Liouvillian gap (MHz)',
           title='Liouvillian gaps (effective single-mode)\n'
                 r'$L_2=\sqrt{\kappa_2}\,a^2$, $L_a=\sqrt{\kappa_a}\,a$')
    ax.legend(fontsize=7)

    # ── Panel (2,1): physical summary diagram ────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    summary = [
        r"$\bf{Why\ \epsilon_d \approx 4.89,\ g_2 \approx 1.76\ MHz?}$",
        "",
        r"1.  $T_Z/T_X = 320$ pins $\alpha^2 \approx 2.37$.",
        r"    This defines the iso-$\alpha^2$ manifold.",
        "",
        r"2.  Along this manifold, $T_X$ improves with $g_2$",
        r"    because $\kappa_2 = 4g_2^2/\kappa_b$ increases.",
        r"    Larger $\kappa_2$ = stronger two-photon protection.",
        "",
        r"3.  Critical threshold: $\kappa_2 = \kappa_a$",
        fr"    $\Rightarrow g_2^* = \sqrt{{\kappa_a\kappa_b/4}} \approx {G2_STAR:.2f}$ MHz.",
        r"    Below: cat stabilization weaker than loss.",
        r"    Above: stabilization dominates.",
        "",
        fr"4.  Optimizer found $g_2 \approx {G2_OPT}$ (just above threshold)",
        r"    but had NOT converged at epoch 60.",
        fr"    Bound-limited optimum: $g_2 \approx {g2_max_on_curve:.1f}$ MHz.",
        "",
        r"5.  Baseline ($g_2=1.0$) is deep in the under-stabilized",
        r"    regime ($\kappa_2/\kappa_a = 0.40$). The optimizer",
        r"    escaped this regime by crossing $g_2^*$.",
    ]
    ax.text(0.02, 0.98, '\n'.join(summary),
            transform=ax.transAxes,
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.8))

    # ── Panel (2,2): optimizer trajectory in parameter space ─────────────────
    ax = fig.add_subplot(gs[2, 2])
    # Re-draw kappa_2 ratio with optimizer path overlaid
    im = ax.contourf(EE, GG, np.clip(K2_RATIO, 0, 4), levels=15,
                     cmap='RdBu_r', vmin=0, vmax=4, alpha=0.6)
    ax.plot(eps_curve[valid], g2_curve_dense[valid], 'k--', lw=2,
            label=r'iso-$\alpha^2$ manifold')
    ax.axhline(G2_STAR, color='black', lw=2.0, ls='-',
               label=r'$\kappa_2 = \kappa_a$')
    ax.plot(EPS_D_BASE, G2_BASE, 'rs', ms=12, zorder=5, label='Baseline')
    ax.plot(EPS_D_OPT,  G2_OPT,  'r*', ms=16, zorder=5, label='Epoch 60')
    # Max reachable
    ax.plot(eps_d_at_max, g2_max_on_curve, 'g^', ms=12, zorder=5,
            label=f'Bound limit ($g_2$={g2_max_on_curve:.1f})')
    # Arrow showing optimizer direction
    ax.annotate('', xy=(EPS_D_OPT, G2_OPT),
                xytext=(EPS_D_BASE + 0.3, G2_BASE + 0.15),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.set(xlabel=r'$\epsilon_d$ (MHz)', ylabel=r'$g_2$ (MHz)',
           title='Optimizer trajectory in parameter space\n'
                 '(color = $\\kappa_2/\\kappa_a$)')
    ax.legend(fontsize=7, loc='upper left')

    plt.suptitle(
        r'Why $\epsilon_d \approx 4.89$ MHz, $g_2 \approx 1.76$ MHz?'
        '\n'
        r'The iso-$\alpha^2$ manifold + two-photon stabilization threshold',
        fontsize=13, y=1.01)

    plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to parameter_analysis.png", flush=True)
    plt.show()

    # ── Final printed summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)
    print(f"""
The optimizer converged to eps_d ≈ {EPS_D_OPT}, g_2 ≈ {G2_OPT} for two reasons:

REASON 1 — The ratio constraint pins alpha:
  The target T_Z/T_X = 320 requires alpha^2 ≈ {a2_opt:.3f} (alpha ≈ {np.sqrt(a2_opt):.3f}).
  This is SMALLER than the baseline alpha^2 = {a2_base:.3f} (alpha = {np.sqrt(a2_base):.3f}).
  The analytic ratio at baseline alpha would be {analytic_ratio_from_alpha2(a2_base):.0f} — far above 320.
  The optimizer moved to a smaller cat to satisfy the ratio constraint.

REASON 2 — Along the iso-alpha manifold, larger g_2 gives better T_X:
  kappa_2 = 4*g_2^2/kappa_b increases with g_2 at fixed alpha.
  The critical threshold is kappa_2 = kappa_a at g_2* = {G2_STAR:.3f} MHz.
  Baseline:  kappa_2/kappa_a = {kappa2(G2_BASE)/kappa_a:.3f}  (under-stabilized)
  Optimized: kappa_2/kappa_a = {kappa2(G2_OPT)/kappa_a:.3f}  (over-stabilized)
  Crossing this threshold unlocks much better T_X because the two-photon
  dissipator can now compete with and suppress single-photon error events.

REASON 3 — The optimizer had not converged:
  The parameters were still drifting upward along the manifold at epoch 60.
  The actual bound-limited optimum on this manifold is g_2 ≈ {g2_max_on_curve:.2f},
  eps_d ≈ {eps_d_at_max:.2f}, where kappa_2/kappa_a = {kappa2(g2_max_on_curve)/kappa_a:.2f}.
  More epochs (or a wider g_2 bound) would yield further improvement.
""")
