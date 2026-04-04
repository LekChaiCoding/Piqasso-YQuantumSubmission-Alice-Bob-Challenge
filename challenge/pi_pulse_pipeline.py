"""
Cat-Qubit Parallel Optimisation Comparison

Four optimisers race to find stabilisation parameters (g₂, ε_d) that
maximise lifetimes T_x, T_z while keeping noise-bias η = T_z/T_x ≈ 320.

Cost (minimised):
    L = -(w_x·T_x + w_z·T_z) + λ·(η/η_goal − 1)²
"""

import warnings
warnings.filterwarnings("ignore", message=".*SparseDIAQArray.*")

import atexit
import os
import numpy as np
import jax.numpy as jnp
import dynamiqs as dq
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from cmaes import SepCMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# ── Physics ──────────────────────────────────────────────────
NA, NB  = 15, 5
KAPPA_B = 10.0       # buffer decay rate       [MHz]
KAPPA_A = 1.0        # single-photon loss rate [MHz]

# ── Search space: [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)] ──────
BOUNDS = np.array([[0.1, 5.0], [-2.0, 2.0], [1.0, 20.0], [-5.0, 5.0]])
X0     = np.array([1.0, 0.0, 4.0, 0.0])

# ── Objective weights ────────────────────────────────────────
ETA_GOAL = 320.0
W_TX     = 1.0              # reward per µs of T_x
W_TZ     = 1.0 / ETA_GOAL   # reward per µs of T_z  (normalised → T_x scale)
LAMBDA   = 0.5              # penalty weight on relative bias deviation

# ── Simulation defaults ─────────────────────────────────────
N_TSAVE   = 100
TZ_TFINAL = 200.0    # µs – phase-flip window
TX_TFINAL = 1.0      # µs – bit-flip window

# ── Runtime ──────────────────────────────────────────────────
MAX_WORKERS     = None   # None → min(8, cpus//2)
EARLY_PATIENCE  = 8
EARLY_MIN_DELTA = 1e-3


# ═════════════════════════════════════════════════════════════
#  Worker pool (lazy, shared)
# ═════════════════════════════════════════════════════════════

_pool = None

def _n_workers():
    if MAX_WORKERS is not None:
        return max(1, int(MAX_WORKERS))
    return max(1, min(8, (os.cpu_count() or 2) // 2))

def _get_pool():
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=_n_workers())
    return _pool

atexit.register(lambda: _pool.shutdown(wait=True) if _pool else None)


# ═════════════════════════════════════════════════════════════
#  Exponential-decay fit
# ═════════════════════════════════════════════════════════════

def _fit_decay(t, y):
    """Fit y = A·exp(−t/τ) + C → return τ."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    A0   = max(float(np.ptp(y)), 1e-6)
    tau0 = max(float(np.ptp(t)) / 3, 1e-6)
    try:
        res = least_squares(
            lambda p, t, y: p[0] * np.exp(-t / p[1]) + p[2] - y,
            [A0, tau0, float(y.min())], args=(t, y),
            bounds=([0, 1e-10, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1", f_scale=0.1,
        )
        return max(float(res.x[1]), 1e-10)
    except Exception:
        return tau0


# ═════════════════════════════════════════════════════════════
#  Cat-qubit simulation
# ═════════════════════════════════════════════════════════════

def _mesolve(g2, eps_d, init, tfinal, n_tsave=N_TSAVE):
    """Evolve cat qubit, return (tsave, ⟨X_L⟩, ⟨Z_L⟩)."""
    a = dq.tensor(dq.destroy(NA), dq.eye(NB))
    b = dq.tensor(dq.eye(NA), dq.destroy(NB))

    g2c, edc = complex(g2), complex(eps_d)
    kappa2 = 4 * abs(g2c) ** 2 / KAPPA_B
    eps2   = 2 * g2c * edc / KAPPA_B
    alpha  = (float(np.sqrt(max(2 / kappa2 * (abs(eps2) - KAPPA_A / 4), 0.01)))
              if kappa2 > 1e-12 else 0.5)

    H = (np.conj(g2c) * a @ a @ b.dag()
         + g2c * a.dag() @ a.dag() @ b
         - edc * b.dag() - np.conj(edc) * b)

    cat_p = dq.coherent(NA, alpha)
    cat_m = dq.coherent(NA, -alpha)
    basis = {
        "+z": cat_p, "-z": cat_m,
        "+x": (cat_p + cat_m) / jnp.sqrt(2),
        "-x": (cat_p - cat_m) / jnp.sqrt(2),
    }

    parity = jnp.diag(jnp.array([(-1.0) ** n for n in range(NA)]))
    X_L = dq.tensor(parity, jnp.eye(NB))
    Z_L = dq.tensor(cat_p @ cat_p.dag() - cat_m @ cat_m.dag(), dq.eye(NB))

    psi0  = dq.tensor(basis[init], dq.fock(NB, 0))
    tsave = jnp.linspace(0, tfinal, n_tsave)

    res = dq.mesolve(
        H, [jnp.sqrt(KAPPA_B) * b, jnp.sqrt(KAPPA_A) * a],
        psi0, tsave, exp_ops=[X_L, Z_L],
        options=dq.Options(progress_meter=False),
    )
    return (np.array(res.tsave),
            np.array(res.expects[0].real),
            np.array(res.expects[1].real))


def measure_Tx_Tz(g2, eps_d):
    """Two mesolve runs → (T_x, T_z) via exponential fits."""
    tz_t, _,  sz = _mesolve(g2, eps_d, "+z", TZ_TFINAL)
    tx_t, sx, _  = _mesolve(g2, eps_d, "+x", TX_TFINAL)
    return _fit_decay(tx_t, sx), _fit_decay(tz_t, sz)


# ═════════════════════════════════════════════════════════════
#  Objective
# ═════════════════════════════════════════════════════════════

def cost(params):
    """L = -(w_x·T_x + w_z·T_z) + λ·(η/η_goal − 1)²"""
    g2  = complex(params[0], params[1])
    epd = complex(params[2], params[3])
    try:
        Tx, Tz = measure_Tx_Tz(g2, epd)
        Tx, Tz = max(Tx, 1e-6), max(Tz, 1e-6)
        eta = Tz / Tx
        return float(-(W_TX * Tx + W_TZ * Tz) + LAMBDA * (eta / ETA_GOAL - 1) ** 2)
    except Exception as e:
        print(f"  [!] sim failed: {e}")
        return 1e6


def _cost_worker(params):
    """Pickle-friendly wrapper for ProcessPoolExecutor."""
    return cost(params)


def batch_cost(xs, parallel=True):
    """Evaluate cost() over rows of xs, optionally in parallel."""
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return np.array([])
    if parallel and _n_workers() > 1:
        return np.array(list(_get_pool().map(_cost_worker, xs)))
    return np.array([cost(p) for p in xs])


# ═════════════════════════════════════════════════════════════
#  Early-stopping helper
# ═════════════════════════════════════════════════════════════

def _early(best, cur, ctr):
    """Returns (new_best, new_counter, should_stop)."""
    if cur < best - EARLY_MIN_DELTA:
        return cur, 0, False
    return best, ctr + 1, ctr + 1 >= EARLY_PATIENCE


# ═════════════════════════════════════════════════════════════
#  1. CMA-ES
# ═════════════════════════════════════════════════════════════

def run_cmaes(batch_size=6, n_epochs=30, seed=0, parallel=True):
    opt = SepCMA(mean=X0.copy(), sigma=0.3, bounds=BOUNDS,
                 population_size=batch_size, seed=seed)
    hist, best, pat = [], np.inf, 0
    for ep in range(n_epochs):
        xs = np.array([opt.ask() for _ in range(opt.population_size)])
        L  = batch_cost(xs, parallel)
        opt.tell([(xs[j], L[j]) for j in range(len(xs))])
        avg = float(np.mean(L))
        hist.append(avg)
        if ep % 5 == 0:
            print(f"  [CMA-ES]   ep {ep:3d} | L={avg:.4f}")
        best, pat, stop = _early(best, avg, pat)
        if stop:
            print(f"  [CMA-ES]   early stop @ ep {ep}")
            break
    return np.array(hist)


# ═════════════════════════════════════════════════════════════
#  2. PPO  (continuous, Gaussian policy)
# ═════════════════════════════════════════════════════════════

def run_ppo(batch_size=6, n_epochs=30, seed=0, parallel=True):
    rng = np.random.default_rng(seed)
    lo, hi, nd = BOUNDS[:, 0], BOUNDS[:, 1], len(X0)
    mu, log_s = X0.copy(), np.zeros(nd)
    lr_m, lr_s, clip = 0.05, 0.01, 0.2

    hist, best, pat = [], np.inf, 0
    for ep in range(n_epochs):
        sig = np.maximum(np.exp(np.clip(log_s, -7, 2)), 1e-3)
        acts = np.clip(rng.normal(mu, sig, (batch_size, nd)), lo, hi)
        L = batch_cost(acts, parallel)

        ret = -L
        adv = (ret - ret.mean()) / (ret.std() + 1e-8)
        old_lp = (-0.5 * np.sum(((acts - mu) / sig) ** 2, 1)
                  - np.sum(log_s) - nd * 0.5 * np.log(2 * np.pi))

        for _ in range(3):
            sc = np.maximum(np.exp(np.clip(log_s, -7, 2)), 1e-3)
            new_lp = (-0.5 * np.sum(((acts - mu) / sc) ** 2, 1)
                      - np.sum(log_s) - nd * 0.5 * np.log(2 * np.pi))
            r = np.exp(new_lp - old_lp)
            surr = np.minimum(r * adv, np.clip(r, 1 - clip, 1 + clip) * adv)
            mu    = np.clip(mu + lr_m * np.mean(((acts - mu) / sc ** 2) * surr[:, None], 0), lo, hi)
            log_s = np.clip(log_s + lr_s * np.mean((((acts - mu) ** 2 / sc ** 2) - 1) * surr[:, None], 0), -7, 2)

        avg = float(np.mean(L))
        hist.append(avg)
        if ep % 5 == 0:
            print(f"  [PPO]      ep {ep:3d} | L={avg:.4f}")
        best, pat, stop = _early(best, avg, pat)
        if stop:
            print(f"  [PPO]      early stop @ ep {ep}")
            break
    return np.array(hist)


# ═════════════════════════════════════════════════════════════
#  3. Bayesian Optimisation  (GP + Thompson sampling)
# ═════════════════════════════════════════════════════════════

def run_bayesopt(batch_size=6, n_epochs=30, seed=0, parallel=True):
    rng = np.random.default_rng(seed)
    nd = len(X0)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(nd), nu=2.5),
        alpha=1e-4, normalize_y=True, n_restarts_optimizer=2, random_state=seed,
    )
    X_all, y_all = np.empty((0, nd)), np.empty(0)
    n_rand = max(10, batch_size)

    hist, best, pat = [], np.inf, 0
    for ep in range(n_epochs):
        if len(X_all) < n_rand:
            cands = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (batch_size, nd))
        else:
            pool_x = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (500, nd))
            samps  = gp.sample_y(pool_x, n_samples=batch_size,
                                 random_state=rng.integers(1 << 31))
            cands = np.array([pool_x[np.argmin(samps[:, i])] for i in range(batch_size)])

        L = batch_cost(cands, parallel)
        X_all = np.vstack([X_all, cands])
        y_all = np.concatenate([y_all, L])
        gp.fit(X_all, y_all)

        avg = float(np.mean(L))
        hist.append(avg)
        if ep % 5 == 0:
            print(f"  [BayesOpt] ep {ep:3d} | L={avg:.4f}")
        best, pat, stop = _early(best, avg, pat)
        if stop:
            print(f"  [BayesOpt] early stop @ ep {ep}")
            break
    return np.array(hist)


# ═════════════════════════════════════════════════════════════
#  4. Batch SPSA
# ═════════════════════════════════════════════════════════════

class _SPSA:
    def __init__(self, x0, bounds, a=0.15, c=0.1, A=10, seed=0):
        self.x, self.bounds = x0.copy(), bounds
        self.a, self.c, self.A, self.k = a, c, A, 0
        self._d = None
        self.rng = np.random.default_rng(seed)

    def ask(self, n_pairs=3):
        ck = self.c / (self.k + 1) ** 0.101
        self._d = [self.rng.choice([-1., 1.], len(self.x)) for _ in range(n_pairs)]
        out = []
        for d in self._d:
            out.append(np.clip(self.x + ck * d, self.bounds[:, 0], self.bounds[:, 1]))
            out.append(np.clip(self.x - ck * d, self.bounds[:, 0], self.bounds[:, 1]))
        return out

    def tell(self, losses):
        ak = self.a / (self.k + 1 + self.A) ** 0.602
        ck = self.c / (self.k + 1) ** 0.101
        g = np.zeros_like(self.x)
        for i in range(len(losses) // 2):
            g += (losses[2 * i] - losses[2 * i + 1]) / (2 * ck * self._d[i])
        g /= max(len(losses) // 2, 1)
        self.x = np.clip(self.x - ak * g, self.bounds[:, 0], self.bounds[:, 1])
        self.k += 1


def run_spsa(batch_size=6, n_epochs=30, seed=0, parallel=True):
    n_pairs = max(1, batch_size // 2)
    opt = _SPSA(X0, BOUNDS, seed=seed)

    hist, best, pat = [], np.inf, 0
    for ep in range(n_epochs):
        cands = opt.ask(n_pairs)
        L = batch_cost(np.array(cands), parallel)
        opt.tell(L.tolist())
        avg = float(np.mean(L))
        hist.append(avg)
        if ep % 5 == 0:
            print(f"  [SPSA]     ep {ep:3d} | L={avg:.4f}")
        best, pat, stop = _early(best, avg, pat)
        if stop:
            print(f"  [SPSA]     early stop @ ep {ep}")
            break
    return np.array(hist)


# ═════════════════════════════════════════════════════════════
#  Plot
# ═════════════════════════════════════════════════════════════

_COLORS = {"CMA-ES": "tab:blue", "PPO": "tab:orange",
           "BayesOpt": "tab:green", "SPSA": "tab:red"}

def plot_comparison(results, n_epochs):
    plt.figure(figsize=(9, 5))
    for name, h in results.items():
        plt.plot(np.arange(len(h)), h, label=name,
                 color=_COLORS.get(name), lw=2)
    plt.xlim(0, n_epochs - 1)
    plt.xlabel("Epoch")
    plt.ylabel(r"$L$  (lower $\rightarrow$ better)")
    plt.title("Optimizer Comparison — Cat-Qubit Calibration\n"
              r"$L = -(w_x T_x + w_z T_z) + \lambda\,(\eta/\eta_{\mathrm{goal}} - 1)^2$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    BS, EPOCHS, SEED, PAR = 6, 30, 0, True
    print(f"workers={_n_workers()}  parallel={PAR}")

    runners = [
        ("CMA-ES",   run_cmaes),
        ("PPO",      run_ppo),
        ("BayesOpt", run_bayesopt),
        ("SPSA",     run_spsa),
    ]
    results = {}
    for i, (name, fn) in enumerate(runners, 1):
        print(f"\n{'=' * 50}\n{i}/4  {name}\n{'=' * 50}")
        results[name] = fn(BS, EPOCHS, SEED, PAR)

    print(f"\n{'=' * 50}\nPlotting ...\n{'=' * 50}")
    plot_comparison(results, EPOCHS)
