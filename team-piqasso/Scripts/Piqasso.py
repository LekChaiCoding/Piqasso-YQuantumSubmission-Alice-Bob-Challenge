"""
Cat Qubit + Noise-Biased Error Correction Simulation
=====================================================
Cat qubits are bosonic qubits encoded in coherent state superpositions:

    |0>_L = |+α>   (coherent state amplitude +α)
    |1>_L = |-α>   (coherent state amplitude -α)

Key physics: BIASED noise
    - Bit-flip  (X) rate:   ε_X ~ exp(-2|α|²)    [exponentially suppressed]
    - Phase-flip (Z) rate:  ε_Z ~ κ₁|α|²/κ₂      [grows with |α|²]
    - Noise bias:           η   = ε_Z / ε_X       [can reach 10⁶× or more]

Error correction strategy:
    Use a Z-repetition code (bias-tailored repetition code) that corrects the
    dominant Z errors while only needing X-type stabilizers. The exponential
    suppression of X errors makes this far more efficient than a standard code.

References:
    Mirrahimi et al., New J. Phys. 16, 045014 (2014)
    Guillaud & Mirrahimi, Phys. Rev. X 9, 041053 (2019)
    Chamberland et al., PRX Quantum 3, 010329 (2022)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    print("[warn] matplotlib not found — skipping plots")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FOCK SPACE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

N_FOCK = 40  # Hilbert space truncation (photon number cutoff)


def coherent_state(alpha: complex, n_fock: int = N_FOCK) -> np.ndarray:
    """
    Coherent state |α> in the truncated Fock basis.

    |α> = exp(-|α|²/2) Σ_n  α^n / sqrt(n!)  |n>
    """
    ns   = np.arange(n_fock, dtype=float)
    logp = -0.5 * abs(alpha) ** 2 + ns * np.log(abs(alpha) + 1e-300) - 0.5 * np.array(
        [math.lgamma(n + 1) for n in ns]
    )
    phase = np.exp(1j * np.angle(alpha) * ns)
    coeffs = np.exp(logp) * phase
    return coeffs / np.linalg.norm(coeffs)


def cat_qubit_logical_basis(alpha: float, n_fock: int = N_FOCK) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct orthonormal logical basis from the two coherent states.

    |0>_L = |+α>  (normalised)
    |1>_L = |-α>  orthogonalised w.r.t. |0>_L via Gram–Schmidt
    """
    ket_p = coherent_state(+alpha, n_fock)
    ket_m = coherent_state(-alpha, n_fock)
    # Gram–Schmidt
    e0 = ket_p / np.linalg.norm(ket_p)
    e1 = ket_m - np.dot(e0.conj(), ket_m) * e0
    e1 /= np.linalg.norm(e1)
    return e0, e1


def wigner_function(state: np.ndarray, x_range: float = 5.0, n_pts: int = 80,
                    n_fock: int = N_FOCK) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wigner quasi-probability distribution W(x,p) via the Laguerre formula.
    Returns meshgrid (X, P, W).
    """
    xs = np.linspace(-x_range, x_range, n_pts)
    ps = np.linspace(-x_range, x_range, n_pts)
    X, P = np.meshgrid(xs, ps)
    beta = X + 1j * P           # phase-space points
    rho  = np.outer(state, state.conj())

    W = np.zeros_like(X, dtype=float)
    for i, x in enumerate(xs):
        for j, p in enumerate(ps):
            b = x + 1j * p
            # Displacement operator D(-β)|n> in matrix form is expensive;
            # use the compact displaced-parity formula:
            #   W(β) = (2/π) Tr[D(-β) ρ D(β) P]  with P = (-1)^n̂
            D = _displacement_matrix(b, n_fock)
            P_op = np.diag((-1) ** np.arange(n_fock, dtype=float))
            W[j, i] = (2 / np.pi) * np.real(np.trace(D.conj().T @ rho @ D @ P_op))
    return X, P, W


def _displacement_matrix(beta: complex, n_fock: int) -> np.ndarray:
    """Displacement operator D(β) in Fock basis (exact, truncated)."""
    a   = np.diag(np.sqrt(np.arange(1, n_fock, dtype=float)), k=1)
    adag= a.conj().T
    G   = beta * adag - beta.conj() * a
    return _matrix_exp(G)


def _matrix_exp(M: np.ndarray) -> np.ndarray:
    """Matrix exponential (falls back to scipy if available)."""
    try:
        from scipy.linalg import expm
        return expm(M)
    except ImportError:
        # Power-series approximation (sufficient for small norms)
        result = np.eye(M.shape[0], dtype=complex)
        term   = np.eye(M.shape[0], dtype=complex)
        for k in range(1, 30):
            term = term @ M / k
            result += term
            if np.linalg.norm(term) < 1e-12:
                break
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CAT QUBIT NOISE MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CatQubitNoise:
    """
    Biased noise model for a single cat qubit.

    Physical processes:
      κ₁  single-photon loss  →  dephases the cat (Z errors)
      κ₂  two-photon pumping  →  stabilises |±α>, suppresses X errors

    Effective error rates (derived from Lindblad master equation):
      ε_X ≈ (κ₁ / κ₂) exp(-2α²)   bit-flip   (exponentially small)
      ε_Z ≈  κ₁ α² / κ₂            phase-flip (polynomial)
    """
    alpha:   float = 2.0
    kappa_1: float = 0.01   # single-photon loss rate
    kappa_2: float = 1.0    # two-photon pumping rate

    @property
    def x_error_rate(self) -> float:
        return (self.kappa_1 / self.kappa_2) * math.exp(-2 * self.alpha ** 2)

    @property
    def z_error_rate(self) -> float:
        return self.kappa_1 * self.alpha ** 2 / self.kappa_2

    @property
    def bias(self) -> float:
        """η = ε_Z / ε_X  — how many times more likely a Z error is than X."""
        return self.z_error_rate / max(self.x_error_rate, 1e-300)

    def summary(self) -> str:
        return (
            f"  α        = {self.alpha:.2f}\n"
            f"  κ₁       = {self.kappa_1:.3e}\n"
            f"  κ₂       = {self.kappa_2:.3e}\n"
            f"  ε_X      = {self.x_error_rate:.3e}  (bit-flip)\n"
            f"  ε_Z      = {self.z_error_rate:.3e}  (phase-flip)\n"
            f"  bias η   = {self.bias:.2e}×  (Z errors dominate)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PAULI UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

I2 = np.eye(2,  dtype=complex)
X  = np.array([[0, 1], [1, 0]],  dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)


def kron_at(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Tensor product  I⊗...⊗op⊗...⊗I  with `op` at position `qubit`."""
    parts = [op if i == qubit else I2 for i in range(n_qubits)]
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BIAS-TAILORED Z-REPETITION CODE
# ─────────────────────────────────────────────────────────────────────────────

class ZRepetitionCode:
    """
    [[n, 1, d]] Z-repetition code, optimal for heavily Z-biased noise.

    Encoding:
        |0>_L = |0...0>
        |1>_L = |1...1>

    Stabilizers:   Z_i ⊗ Z_{i+1}   for i = 0..n-2
    (These detect Z errors — not X errors — which is the rare event.)

    Logical operators:
        X_L = X^⊗n   (bit flip of all qubits)
        Z_L = Z_0    (phase flip of any single qubit)

    Correction threshold:  up to ⌊(n-1)/2⌋ Z errors correctable.
    X errors cause uncorrectable logical failure, but are exponentially rare
    for cat qubits — so accepting this is a deliberate bias-exploiting trade-off.
    """

    def __init__(self, n: int = 5):
        if n < 3 or n % 2 == 0:
            raise ValueError("n must be odd and >= 3")
        self.n = n

    @property
    def dim(self) -> int:
        return 2 ** self.n

    @property
    def logical_zero(self) -> np.ndarray:
        v = np.zeros(self.dim, dtype=complex)
        v[0] = 1.0   # |00...0>
        return v

    @property
    def logical_one(self) -> np.ndarray:
        v = np.zeros(self.dim, dtype=complex)
        v[-1] = 1.0  # |11...1>
        return v

    def encode(self, alpha_c: complex, beta_c: complex) -> np.ndarray:
        """α|0> + β|1>  →  α|0...0> + β|1...1>"""
        psi = alpha_c * self.logical_zero + beta_c * self.logical_one
        return psi / np.linalg.norm(psi)

    # ── stabiliser measurements ───────────────────────────────────────────────

    def _stabilizer(self, i: int) -> np.ndarray:
        """Z_i ⊗ Z_{i+1} acting on n qubits."""
        parts = [Z if j in (i, i + 1) else I2 for j in range(self.n)]
        result = parts[0]
        for p in parts[1:]:
            result = np.kron(result, p)
        return result

    def measure_syndrome(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a binary syndrome vector (length n-1).
        syndrome[i] = 1 if qubits i and i+1 disagree (error boundary detected).
        """
        rho = np.outer(state, state.conj())
        syndrome = np.zeros(self.n - 1, dtype=int)
        for i in range(self.n - 1):
            expectation = np.real(np.trace(self._stabilizer(i) @ rho))
            # +1 → agree (no boundary), -1 → disagree (boundary)
            syndrome[i] = 0 if expectation > 0 else 1
        return syndrome

    # ── error application ─────────────────────────────────────────────────────

    def apply_z(self, state: np.ndarray, qubit: int) -> np.ndarray:
        return kron_at(Z, qubit, self.n) @ state

    def apply_x(self, state: np.ndarray, qubit: int) -> np.ndarray:
        return kron_at(X, qubit, self.n) @ state

    # ── minimum-weight decoder ────────────────────────────────────────────────

    def _minimum_weight_decode(self, syndrome: np.ndarray) -> List[int]:
        """
        Minimum-weight Z-error correction for the repetition code.

        The syndrome marks "domain walls" between regions of the same value.
        Minimum-weight means we connect neighbouring domain walls with the
        shortest path and correct on the smaller side.
        """
        boundaries = [i for i, s in enumerate(syndrome) if s == 1]
        if not boundaries:
            return []

        # Pair adjacent boundaries greedily (minimum-weight matching on a path)
        corrections: List[int] = []
        i = 0
        while i < len(boundaries) - 1:
            left  = boundaries[i]
            right = boundaries[i + 1]
            # Correct the smaller segment between the two boundaries
            segment_size = right - left
            complement   = self.n - segment_size
            if segment_size <= complement:
                corrections.extend(range(left + 1, right + 1))
            else:
                corrections.extend(list(range(0, left + 1)) + list(range(right + 1, self.n)))
            i += 2

        # Handle unpaired boundary (edge case: connect to the nearest edge)
        if len(boundaries) % 2 == 1:
            last = boundaries[-1]
            if last + 1 <= self.n - 1 - last:
                corrections.extend(range(last + 1, self.n))
            else:
                corrections.extend(range(0, last + 1))

        # Collapse paired corrections (Z² = I)
        from collections import Counter
        counts = Counter(corrections)
        return [q for q, c in counts.items() if c % 2 == 1]

    def correct(self, state: np.ndarray) -> np.ndarray:
        """Apply syndrome measurement then minimum-weight Z correction."""
        syndrome    = self.measure_syndrome(state)
        error_sites = self._minimum_weight_decode(syndrome)
        corrected   = state.copy()
        for q in error_sites:
            corrected = self.apply_z(corrected, q)
        norm = np.linalg.norm(corrected)
        return corrected / norm if norm > 1e-15 else corrected

    def fidelity(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """State fidelity |<a|b>|²."""
        a = state_a / np.linalg.norm(state_a)
        b = state_b / np.linalg.norm(state_b)
        return abs(np.dot(a.conj(), b)) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_logical_error(
    noise:    CatQubitNoise,
    code:     ZRepetitionCode,
    n_trials: int = 4000,
    seed:     int = 42,
) -> dict:
    """
    Estimate logical error rates via Monte Carlo.

    Each trial:
      1. Draw a random logical state.
      2. Apply independent Z errors (rate ε_Z) and X errors (rate ε_X)
         to each physical qubit.
      3. Run the EC decoder.
      4. Measure fidelity: failure if fidelity < 0.5.

    Returns dict with rates before and after error correction.
    """
    rng   = np.random.default_rng(seed)
    eps_z = noise.z_error_rate
    eps_x = noise.x_error_rate

    fails_raw = 0
    fails_ec  = 0

    for _ in range(n_trials):
        # Random Bloch sphere point
        theta = rng.uniform(0, np.pi)
        phi   = rng.uniform(0, 2 * np.pi)
        a     = np.cos(theta / 2)
        b     = np.sin(theta / 2) * np.exp(1j * phi)
        psi_0 = code.encode(a, b)

        # Inject errors
        psi_err = psi_0.copy()
        for q in range(code.n):
            if rng.random() < eps_z:
                psi_err = code.apply_z(psi_err, q)
            if rng.random() < eps_x:
                psi_err = code.apply_x(psi_err, q)

        psi_err = psi_err / np.linalg.norm(psi_err)

        # Before EC
        if code.fidelity(psi_0, psi_err) < 0.5:
            fails_raw += 1

        # After EC
        psi_ec = code.correct(psi_err)
        if code.fidelity(psi_0, psi_ec) < 0.5:
            fails_ec += 1

    return {
        "logical_error_raw": fails_raw / n_trials,
        "logical_error_ec":  fails_ec  / n_trials,
        "suppression_factor": (fails_raw / max(fails_ec, 1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(alpha_sweep: np.ndarray, noise_base: CatQubitNoise,
             code_3: ZRepetitionCode, code_5: ZRepetitionCode,
             code_7: ZRepetitionCode) -> None:
    """Four-panel summary figure."""
    if not MATPLOTLIB:
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Cat Qubit — Noise Bias & Error Correction", fontsize=15, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel A: Wigner function of cat qubit |0>_L ──────────────────────────
    ax_w = fig.add_subplot(gs[0, 0])
    ket0, _ = cat_qubit_logical_basis(noise_base.alpha, n_fock=N_FOCK)
    print("  Computing Wigner function …", end=" ", flush=True)
    X_w, P_w, W_w = wigner_function(ket0, x_range=4.0, n_pts=60)
    print("done")
    vmax = np.max(np.abs(W_w))
    ax_w.contourf(X_w, P_w, W_w, levels=60, cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax)
    ax_w.set_title(f"|0>_L Wigner function  (α={noise_base.alpha})")
    ax_w.set_xlabel("x (quadrature)")
    ax_w.set_ylabel("p (quadrature)")
    ax_w.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax_w.axvline(0, color='k', lw=0.5, alpha=0.4)

    # ── Panel B: Error rates vs α ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[0, 1])
    x_rates, z_rates, biases = [], [], []
    for a in alpha_sweep:
        nm = CatQubitNoise(alpha=a, kappa_1=noise_base.kappa_1,
                           kappa_2=noise_base.kappa_2)
        x_rates.append(nm.x_error_rate)
        z_rates.append(nm.z_error_rate)
        biases.append(nm.bias)

    ax_e.semilogy(alpha_sweep, x_rates, 'b-o', ms=4, lw=1.5,
                  label=r'$\varepsilon_X \sim e^{-2\alpha^2}$  (bit-flip)')
    ax_e.semilogy(alpha_sweep, z_rates, 'r-s', ms=4, lw=1.5,
                  label=r'$\varepsilon_Z \sim \alpha^2$  (phase-flip)')
    ax_e.set_xlabel(r'Cat amplitude $\alpha$')
    ax_e.set_ylabel('Physical error rate')
    ax_e.set_title('Biased Noise vs α')
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)

    # ── Panel C: Noise bias η vs α ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.semilogy(alpha_sweep, biases, 'g-^', ms=4, lw=1.5,
                  label=r'$\eta = \varepsilon_Z / \varepsilon_X$')
    ax_b.set_xlabel(r'Cat amplitude $\alpha$')
    ax_b.set_ylabel(r'Noise bias $\eta$')
    ax_b.set_title(r'Noise Bias $\eta$ vs $\alpha$')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)
    ax_b.fill_between(alpha_sweep, biases, alpha=0.15, color='green')

    # ── Panel D: Logical error after EC vs α ─────────────────────────────────
    ax_l = fig.add_subplot(gs[1, 1])
    print("  Running Monte Carlo for EC panel …", flush=True)
    alpha_mc  = np.linspace(0.8, 2.8, 9)
    styles    = [('[[3,1]]', code_3, '#1f77b4', 'o'),
                 ('[[5,1]]', code_5, '#ff7f0e', 's'),
                 ('[[7,1]]', code_7, '#2ca02c', '^')]

    for label, code, color, marker in styles:
        logical_errs = []
        for a in alpha_mc:
            nm  = CatQubitNoise(alpha=a, kappa_1=noise_base.kappa_1,
                                kappa_2=noise_base.kappa_2)
            res = monte_carlo_logical_error(nm, code, n_trials=1500, seed=7)
            logical_errs.append(max(res["logical_error_ec"], 1e-5))
        ax_l.semilogy(alpha_mc, logical_errs, f'-{marker}', color=color,
                      ms=5, lw=1.5, label=label)
        print(f"    {label} done", flush=True)

    ax_l.set_xlabel(r'Cat amplitude $\alpha$')
    ax_l.set_ylabel('Logical error rate (after EC)')
    ax_l.set_title('Bias-Tailored Z-Repetition Code')
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)

    plt.savefig('cat_qubit_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: cat_qubit_simulation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "=" * 62

    print(SEP)
    print("   CAT QUBIT + NOISE-BIASED ERROR CORRECTION SIMULATION")
    print(SEP)

    # ── A. Single cat qubit ───────────────────────────────────────────────────
    alpha = 2.0
    noise = CatQubitNoise(alpha=alpha, kappa_1=0.01, kappa_2=1.0)

    print(f"\n[A] Single cat qubit properties  (α = {alpha})")
    print(noise.summary())

    overlap = math.exp(-2 * alpha ** 2)
    print(f"\n  Coherent state overlap  <+α|-α> = exp(-2α²) = {overlap:.2e}")
    print(f"  (Near-zero overlap → clean logical encoding)")

    # Fock-space logical basis
    ket0, ket1 = cat_qubit_logical_basis(alpha)
    print(f"\n  ‖|0>_L‖ = {np.linalg.norm(ket0):.8f}")
    print(f"  ‖|1>_L‖ = {np.linalg.norm(ket1):.8f}")
    print(f"  |<0|1>_L| = {abs(np.dot(ket0.conj(), ket1)):.2e}  (orthogonality)")

    # ── B. Z-repetition code ──────────────────────────────────────────────────
    print(f"\n[B] Bias-tailored Z-repetition codes")
    print(f"  Code distance d = n  →  corrects ⌊(n-1)/2⌋ Z errors")
    print(f"  X errors accepted as rare events (exponentially suppressed)")

    for n in [3, 5, 7]:
        code = ZRepetitionCode(n)
        print(f"\n  [[{n},1]] code:")
        # Encode |+>_L
        psi0 = code.encode(1 / np.sqrt(2), 1 / np.sqrt(2))

        # Inject errors on every qubit and test worst-case correction
        max_correctable = (n - 1) // 2
        for n_errs in range(max_correctable + 2):
            psi_err = psi0.copy()
            for q in range(n_errs):          # first n_errs qubits get Z errors
                psi_err = code.apply_z(psi_err, q)
            psi_corr = code.correct(psi_err)
            fid      = code.fidelity(psi0, psi_corr)
            status   = "PASS" if fid > 0.99 else "FAIL (exceeds correction capacity)"
            print(f"    {n_errs} Z error(s) → fidelity {fid:.6f}  [{status}]")

    # ── C. Syndrome decoding demo ─────────────────────────────────────────────
    print(f"\n[C] Syndrome decoding demo  ([[5,1]] code, Z error on qubit 3)")
    code5 = ZRepetitionCode(5)
    psi_in  = code5.encode(np.cos(0.7), np.sin(0.7) * np.exp(0.5j))
    psi_err = code5.apply_z(psi_in, qubit=3)
    syn     = code5.measure_syndrome(psi_err)
    psi_ec  = code5.correct(psi_err)
    fid     = code5.fidelity(psi_in, psi_ec)

    print(f"  Syndrome vector: {syn}  (1 = domain wall detected)")
    print(f"  Post-correction fidelity: {fid:.8f}  (should be 1.0)")

    # ── D. Monte Carlo simulation ─────────────────────────────────────────────
    print(f"\n[D] Monte Carlo logical error rate  ([[5,1]] code, 4000 trials)")
    code_mc = ZRepetitionCode(5)
    res = monte_carlo_logical_error(noise, code_mc, n_trials=4000)
    print(f"  Physical:  ε_X = {noise.x_error_rate:.2e},  ε_Z = {noise.z_error_rate:.2e}")
    print(f"  Logical error WITHOUT correction: {res['logical_error_raw']:.4f}")
    print(f"  Logical error WITH    correction: {res['logical_error_ec']:.4f}")
    print(f"  Error suppression:  {res['suppression_factor']:.1f}×")

    # ── E. Noise bias table ───────────────────────────────────────────────────
    print(f"\n[E] Noise bias table  (κ₁={noise.kappa_1}, κ₂={noise.kappa_2})")
    print(f"  {'α':>6}  {'ε_X':>12}  {'ε_Z':>12}  {'η (bias)':>14}")
    print(f"  {'-'*50}")
    for a in np.arange(0.5, 3.6, 0.5):
        nm = CatQubitNoise(alpha=a, kappa_1=noise.kappa_1, kappa_2=noise.kappa_2)
        print(f"  {a:>6.1f}  {nm.x_error_rate:>12.3e}  {nm.z_error_rate:>12.3e}"
              f"  {nm.bias:>12.2e}×")

    # ── F. Plots ──────────────────────────────────────────────────────────────
    print(f"\n[F] Generating plots …")
    alpha_sweep = np.linspace(0.3, 3.5, 40)
    plot_all(
        alpha_sweep = alpha_sweep,
        noise_base  = noise,
        code_3      = ZRepetitionCode(3),
        code_5      = ZRepetitionCode(5),
        code_7      = ZRepetitionCode(7),
    )

    print(f"\n{SEP}")
    print("  Simulation complete.")
    print(SEP)


if __name__ == "__main__":
    main()
