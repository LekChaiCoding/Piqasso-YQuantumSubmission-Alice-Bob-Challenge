"""
Cat-Qubit — Multi-Objective Online Optimisation & Calibration Pipeline

Optimises the cat-qubit stabilisation parameters (g₂, ε_d) to achieve:
  • long bit-flip lifetime  T_x
  • long phase-flip lifetime T_z
  • target noise-bias ratio  η = T_z / T_x ≈ 320

Objective (minimised):
    δ = β₁|T_x| + β₂|T_z| + ε|η - η_goal|

  With β₁, β₂ < 0 the optimiser is rewarded for longer lifetimes.
  With ε > 0 the optimiser is penalised for deviating from η_goal.

Four optimisers are compared:
    1. CMA-ES
    2. PPO  (Proximal Policy Optimization)
    3. Bayesian Optimization  (GP surrogate + Thompson sampling)
    4. Batch SPSA
"""

import warnings
warnings.filterwarnings("ignore", message=".*SparseDIAQArray.*")

import atexit
import os
import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
from cmaes import SepCMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


# ============================================================
# Physical constants
# ============================================================

NA = 15          # storage Hilbert-space dimension
NB = 5           # buffer Hilbert-space dimension
KAPPA_B = 10.0   # buffer decay rate  [MHz]
KAPPA_A = 1.0    # storage single-photon loss rate [MHz]


# ============================================================
# Objective hyper-parameters
# ============================================================

BETA_1   = -1.0    # weight on |T_x|  (negative → reward longer T_x)
BETA_2   = -0.01   # weight on |T_z|  (negative → reward longer T_z; scaled
                    #   down because T_z >> T_x in absolute value)
EPSILON  =  0.1    # weight on |η − η_goal|
ETA_GOAL = 320.0   # target bias ratio


# ============================================================
# Runtime / performance settings
# ============================================================

NUM_TSAVE = 100
ENABLE_CACHE = True
CACHE_ROUND_DECIMALS = 6

# Multi-fidelity speedup: evaluate full objective only on top-K proxy points.
USE_MULTIFIDELITY = True
PROXY_TOP_K = 2
PROXY_TZ_TFINAL = 80.0
PROXY_TX_TFINAL = 0.6
PROXY_NUM_TSAVE = 40

# If None, uses up to half the available CPUs for worker processes.
MAX_WORKERS = None

# Early-stopping options for all optimizers
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_IMPROVEMENT = 1e-3


# ============================================================
# 4 optimisation knobs: [Re(g₂), Im(g₂), Re(ε_d), Im(ε_d)]
# ============================================================

PARAM_BOUNDS = np.array([
    [0.1,  5.0],    # Re(g_2)
    [-2.0, 2.0],    # Im(g_2)
    [1.0, 20.0],    # Re(eps_d)
    [-5.0, 5.0],    # Im(eps_d)
])
PARAM_INIT = np.array([1.0, 0.0, 4.0, 0.0])   # notebook defaults


# ============================================================
# Global multiprocessing/caching state
# ============================================================

_POOL = None
_CACHE = {}


def _fidelity_signature(kw):
    tz_tfinal = float(kw.get("Tz_tfinal", 200.0))
    tx_tfinal = float(kw.get("Tx_tfinal", 1.0))
    num_tsave = int(kw.get("num_tsave", NUM_TSAVE))
    return (
        round(tz_tfinal, CACHE_ROUND_DECIMALS),
        round(tx_tfinal, CACHE_ROUND_DECIMALS),
        num_tsave,
    )


def _params_cache_key(params, kw=None):
    arr = np.asarray(params, dtype=float)
    rounded_params = tuple(np.round(arr, CACHE_ROUND_DECIMALS))
    if kw is None:
        return rounded_params
    return rounded_params + _fidelity_signature(kw)


def _resolve_workers(max_workers=None):
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu // 2))


def _ensure_pool(max_workers=None):
    global _POOL
    if _POOL is None:
        workers = _resolve_workers(max_workers)
        _POOL = ProcessPoolExecutor(max_workers=workers)
    return _POOL


def _shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
        _POOL = None


atexit.register(_shutdown_pool)


# ============================================================
# Exponential-decay fitting
# ============================================================

def _exp_model(p, t):
    A, tau, C = p
    return A * np.exp(-t / tau) + C


def _exp_residuals(p, x, y):
    return _exp_model(p, x) - y


def robust_exp_fit(x, y):
    """Fit  y = A·exp(−t/τ) + C  with robust (soft-L1) loss.  Returns τ."""
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    A0  = max(float(np.abs(y_np.max() - y_np.min())), 1e-6)
    C0  = float(y_np.min())
    tau0 = max(float(x_np.max() - x_np.min()) / 3.0, 1e-6)
    try:
        res = least_squares(
            _exp_residuals, [A0, tau0, C0],
            args=(x_np, y_np),
            bounds=([0, 1e-10, -np.inf], [np.inf, np.inf, np.inf]),
            loss="soft_l1", f_scale=0.1,
        )
        return max(float(res.x[1]), 1e-10)
    except Exception:
        return tau0


# ============================================================
# Cat-qubit simulation
# ============================================================

def measure_lifetime(g_2, eps_d, initial_state, tfinal,
                     kappa_a=KAPPA_A, kappa_b=KAPPA_B,
                     num_tsave=NUM_TSAVE):
    """Evolve the cat qubit under mesolve and return the result object.

    Parameters
    ----------
    g_2 : complex   – two-photon coupling strength  [MHz]
    eps_d : complex  – buffer drive amplitude         [MHz]
    initial_state : str – one of "+z", "-z", "+x", "-x"
    tfinal : float   – total evolution time           [μs]
    """
    na, nb = NA, NB
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    kappa_2 = 4 * abs(g_2)**2 / kappa_b
    eps_2   = 2 * g_2 * eps_d / kappa_b

    # Stabilised cat size (use magnitude for robustness with complex params)
    if kappa_2 > 1e-12:
        raw = 2.0 / kappa_2 * (abs(eps_2) - kappa_a / 4.0)
        alpha_est = float(np.sqrt(max(raw, 0.01)))
    else:
        alpha_est = 0.5

    # Hamiltonian
    g2c  = complex(g_2)
    epsd = complex(eps_d)
    H = (np.conj(g2c) * a @ a @ b.dag()
         + g2c * a.dag() @ a.dag() @ b
         - epsd * b.dag()
         - np.conj(epsd) * b)

    # Lindblad jump operators
    loss_b = jnp.sqrt(kappa_b) * b
    loss_a = jnp.sqrt(kappa_a) * a

    tsave = jnp.linspace(0, tfinal, int(num_tsave))

    # Cat-qubit logical basis
    g_state = dq.coherent(na, alpha_est)
    e_state = dq.coherent(na, -alpha_est)
    basis = {
        "+z": g_state,
        "-z": e_state,
        "+x": (g_state + e_state) / jnp.sqrt(2),
        "-x": (g_state - e_state) / jnp.sqrt(2),
    }

    # Logical observables
    # Parity operator (-1)^n built directly — avoids slow/noisy expm()
    parity_a = jnp.diag(jnp.array([(-1.0)**n for n in range(na)]))
    sx = dq.tensor(parity_a, jnp.eye(nb))               # X_L
    sz = basis["+z"] @ basis["+z"].dag() - basis["-z"] @ basis["-z"].dag()
    sz = dq.tensor(sz, dq.eye(nb))                      # Z_L

    psi0 = dq.tensor(basis[initial_state], dq.fock(nb, 0))

    return dq.mesolve(
        H, [loss_b, loss_a], psi0, tsave,
        options=dq.Options(progress_meter=False),
        exp_ops=[sx, sz],
    )


def compute_Tx_Tz(g_2, eps_d, Tz_tfinal=200.0, Tx_tfinal=1.0,
                  num_tsave=NUM_TSAVE):
    """Return (T_x, T_z) by running two mesolve simulations + exp fits."""
    # T_z — phase-flip lifetime (long)
    res_z = measure_lifetime(g_2, eps_d, "+z", Tz_tfinal,
                             num_tsave=num_tsave)
    T_z = robust_exp_fit(np.array(res_z.tsave),
                         np.array(res_z.expects[1, :].real))

    # T_x — bit-flip lifetime (short)
    res_x = measure_lifetime(g_2, eps_d, "+x", Tx_tfinal,
                             num_tsave=num_tsave)
    T_x = robust_exp_fit(np.array(res_x.tsave),
                         np.array(res_x.expects[0, :].real))

    return T_x, T_z


# ============================================================
# Objective function  δ
# ============================================================

def cat_qubit_delta(params, beta_1=BETA_1, beta_2=BETA_2,
                    epsilon=EPSILON, eta_goal=ETA_GOAL,
                    Tz_tfinal=200.0, Tx_tfinal=1.0,
                    num_tsave=NUM_TSAVE):
    r"""
    δ = β₁|T_x| + β₂|T_z| + ε|η − η_goal|   where η = T_z / T_x.

    Minimised by the optimisers.  Negative β₁, β₂ reward longer lifetimes;
    positive ε penalises deviation from the target bias.
    """
    g_2   = complex(float(params[0]), float(params[1]))
    eps_d = complex(float(params[2]), float(params[3]))
    try:
        T_x, T_z = compute_Tx_Tz(
            g_2, eps_d,
            Tz_tfinal=Tz_tfinal,
            Tx_tfinal=Tx_tfinal,
            num_tsave=num_tsave,
        )
        T_x = max(T_x, 1e-6)
        T_z = max(T_z, 1e-6)
        eta = T_z / T_x
        delta = (beta_1 * abs(T_x)
                 + beta_2 * abs(T_z)
                 + epsilon * abs(eta - eta_goal))
        return float(delta)
    except Exception as exc:
        print(f"    [warn] simulation failed: {exc}")
        return 1e6


def _cat_qubit_delta_worker(task):
    """Pickle-friendly worker entry for ProcessPoolExecutor."""
    params, kw = task
    return cat_qubit_delta(params, **kw)


def batched_cat_qubit_delta(params_batch, use_parallel=True,
                            max_workers=MAX_WORKERS,
                            use_multifidelity=USE_MULTIFIDELITY,
                            proxy_top_k=PROXY_TOP_K,
                            proxy_tz_tfinal=PROXY_TZ_TFINAL,
                            proxy_tx_tfinal=PROXY_TX_TFINAL,
                            proxy_num_tsave=PROXY_NUM_TSAVE,
                            **kw):
    """Evaluate δ for a batch.

    Uses a persistent ProcessPoolExecutor on Windows/Linux for substantial
    speedups when each objective call is expensive (mesolve + fitting).
    """
    batch = np.asarray(params_batch, dtype=float)
    n = len(batch)
    if n == 0:
        return np.array([])

    out = np.empty(n, dtype=float)
    pending_tasks = []
    pending_indices = []
    base_kw = dict(kw)

    for i, p in enumerate(batch):
        key = _params_cache_key(p, base_kw)
        if ENABLE_CACHE and key in _CACHE:
            out[i] = _CACHE[key]
        else:
            pending_indices.append(i)
            pending_tasks.append((p, base_kw))

    if not pending_tasks:
        return out

    workers = _resolve_workers(max_workers)

    def eval_tasks(tasks):
        if not tasks:
            return []
        if use_parallel and workers > 1:
            pool = _ensure_pool(workers)
            return list(pool.map(_cat_qubit_delta_worker, tasks))
        return [cat_qubit_delta(p, **kw_) for p, kw_ in tasks]

    vals = [None] * len(pending_tasks)

    if use_multifidelity and len(pending_tasks) > int(proxy_top_k):
        proxy_kw = dict(base_kw)
        proxy_kw["Tz_tfinal"] = float(proxy_tz_tfinal)
        proxy_kw["Tx_tfinal"] = float(proxy_tx_tfinal)
        proxy_kw["num_tsave"] = int(proxy_num_tsave)

        proxy_tasks = [(p, proxy_kw) for p, _ in pending_tasks]
        proxy_vals = eval_tasks(proxy_tasks)

        k = max(1, min(int(proxy_top_k), len(pending_tasks)))
        refine_local = np.argsort(proxy_vals)[:k].tolist()
        refine_set = set(refine_local)

        full_tasks = [pending_tasks[i] for i in refine_local]
        full_vals = eval_tasks(full_tasks)

        full_map = {i: v for i, v in zip(refine_local, full_vals)}
        for i in range(len(pending_tasks)):
            if i in refine_set:
                vals[i] = full_map[i]
            else:
                vals[i] = float(proxy_vals[i])
    else:
        vals = eval_tasks(pending_tasks)

    for idx, val in zip(pending_indices, vals):
        out[idx] = val
        if ENABLE_CACHE:
            _CACHE[_params_cache_key(batch[idx], base_kw)] = val

    return out


def _should_early_stop(best_loss, current_loss, patience_counter,
                       patience=EARLY_STOP_PATIENCE,
                       min_improvement=EARLY_STOP_MIN_IMPROVEMENT):
    if current_loss < best_loss - min_improvement:
        return current_loss, 0, False
    patience_counter += 1
    return best_loss, patience_counter, patience_counter >= patience


# ============================================================
# 1.  CMA-ES
# ============================================================

def run_cmaes(batch_size=6, n_epochs=30, seed=0,
              use_parallel=True, max_workers=MAX_WORKERS,
              early_stop=True, use_multifidelity=USE_MULTIFIDELITY):
    optimizer = SepCMA(
        mean=PARAM_INIT.copy(),
        sigma=0.3,
        bounds=PARAM_BOUNDS,
        population_size=batch_size,
        seed=seed,
    )
    history = []
    best_loss = np.inf
    patience_ctr = 0
    for epoch in range(n_epochs):
        xs = np.array([optimizer.ask()
                       for _ in range(optimizer.population_size)])
        losses = batched_cat_qubit_delta(
            xs,
            use_parallel=use_parallel,
            max_workers=max_workers,
            use_multifidelity=use_multifidelity,
        )
        optimizer.tell([(xs[j], losses[j]) for j in range(len(xs))])
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch % 5 == 0:
            print(f"  [CMA-ES]   Epoch {epoch:3d} | δ={epoch_loss:.4f}")

        if early_stop:
            best_loss, patience_ctr, stop = _should_early_stop(
                best_loss, epoch_loss, patience_ctr)
            if stop:
                print(f"  [CMA-ES]   Early stop at epoch {epoch:3d}")
                break
    return np.array(history)


# ============================================================
# 2.  PPO
# ============================================================

def run_ppo(batch_size=6, n_epochs=30, seed=0,
            use_parallel=True, max_workers=MAX_WORKERS,
            early_stop=True, use_multifidelity=USE_MULTIFIDELITY):
    rng = np.random.default_rng(seed)
    bounds_lo, bounds_hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    n_dim = len(PARAM_INIT)

    mu = PARAM_INIT.copy()
    log_sigma = np.zeros(n_dim)
    lr_mu, lr_sigma, clip_eps = 0.05, 0.01, 0.2
    min_sigma = 1e-3
    log_sigma_bounds = (-7.0, 2.0)

    history = []
    best_loss = np.inf
    patience_ctr = 0
    for epoch in range(n_epochs):
        log_sigma = np.clip(log_sigma, *log_sigma_bounds)
        sigma = np.maximum(np.exp(log_sigma), min_sigma)

        actions = rng.normal(mu, sigma, size=(batch_size, n_dim))
        actions_clipped = np.clip(actions, bounds_lo, bounds_hi)

        losses = batched_cat_qubit_delta(
            actions_clipped,
            use_parallel=use_parallel,
            max_workers=max_workers,
            use_multifidelity=use_multifidelity,
        )

        # PPO maximises reward → negate δ (which we minimise)
        returns = -losses
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)

        old_logp = (
            -0.5 * np.sum(((actions - mu) / sigma) ** 2, axis=1)
            - np.sum(log_sigma)
            - n_dim * 0.5 * np.log(2 * np.pi)
        )

        for _ in range(3):
            log_sigma = np.clip(log_sigma, *log_sigma_bounds)
            sc = np.maximum(np.exp(log_sigma), min_sigma)
            new_logp = (
                -0.5 * np.sum(((actions - mu) / sc) ** 2, axis=1)
                - np.sum(log_sigma)
                - n_dim * 0.5 * np.log(2 * np.pi)
            )
            ratio = np.exp(new_logp - old_logp)
            cr = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            surr = np.minimum(ratio * adv, cr * adv)

            grad_mu = np.mean(
                ((actions - mu) / sc**2) * surr[:, None], axis=0)
            grad_ls = np.mean(
                (((actions - mu)**2 / sc**2) - 1) * surr[:, None], axis=0)

            mu = np.clip(mu + lr_mu * grad_mu, bounds_lo, bounds_hi)
            log_sigma = np.clip(
                log_sigma + lr_sigma * grad_ls, *log_sigma_bounds)

        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch % 5 == 0:
            print(f"  [PPO]      Epoch {epoch:3d} | δ={epoch_loss:.4f}")

        if early_stop:
            best_loss, patience_ctr, stop = _should_early_stop(
                best_loss, epoch_loss, patience_ctr)
            if stop:
                print(f"  [PPO]      Early stop at epoch {epoch:3d}")
                break
    return np.array(history)


# ============================================================
# 3.  Bayesian Optimization (sklearn GP + Thompson sampling)
# ============================================================

def run_bayesopt(batch_size=6, n_epochs=30, seed=0,
                 use_parallel=True, max_workers=MAX_WORKERS,
                 early_stop=True, use_multifidelity=USE_MULTIFIDELITY):
    rng = np.random.default_rng(seed)
    bounds = PARAM_BOUNDS
    n_dim = len(PARAM_INIT)

    kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(n_dim),
        length_scale_bounds=(1e-2, 1e2), nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-4, normalize_y=True,
        n_restarts_optimizer=2, random_state=seed,
    )

    X_all = np.empty((0, n_dim))
    y_all = np.empty(0)
    history = []
    best_loss = np.inf
    patience_ctr = 0
    n_random = max(10, batch_size)

    for epoch in range(n_epochs):
        if len(X_all) < n_random:
            cands = rng.uniform(bounds[:, 0], bounds[:, 1],
                                size=(batch_size, n_dim))
        else:
            X_cand = rng.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(500, n_dim))
            y_samp = gp.sample_y(X_cand, n_samples=batch_size,
                                 random_state=rng.integers(1 << 31))
            cands = np.array([X_cand[np.argmin(y_samp[:, i])]
                              for i in range(batch_size)])

        losses = batched_cat_qubit_delta(
            cands,
            use_parallel=use_parallel,
            max_workers=max_workers,
            use_multifidelity=use_multifidelity,
        )
        X_all = np.vstack([X_all, cands])
        y_all = np.concatenate([y_all, losses])
        gp.fit(X_all, y_all)

        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch % 5 == 0:
            print(f"  [BayesOpt] Epoch {epoch:3d} | δ={epoch_loss:.4f}")

        if early_stop:
            best_loss, patience_ctr, stop = _should_early_stop(
                best_loss, epoch_loss, patience_ctr)
            if stop:
                print(f"  [BayesOpt] Early stop at epoch {epoch:3d}")
                break
    return np.array(history)


# ============================================================
# 4.  Batch SPSA
# ============================================================

class BatchSPSA:
    """Batch SPSA with ask/tell interface."""

    def __init__(self, x0, bounds, a=0.15, c=0.1, A=10,
                 alpha=0.602, gamma=0.101, seed=0):
        self.x = np.array(x0, dtype=float)
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(x0)
        self.a, self.c, self.A = a, c, A
        self.alpha, self.gamma = alpha, gamma
        self.k = 0
        self._deltas = None
        self.rng = np.random.default_rng(seed)

    def ask(self, batch_size=4):
        ck = self.c / (self.k + 1) ** self.gamma
        self._deltas = [self.rng.choice([-1.0, 1.0], size=self.dim)
                        for _ in range(batch_size)]
        cands = []
        for d in self._deltas:
            cands.append(np.clip(self.x + ck * d,
                                 self.bounds[:, 0], self.bounds[:, 1]))
            cands.append(np.clip(self.x - ck * d,
                                 self.bounds[:, 0], self.bounds[:, 1]))
        return cands

    def tell(self, losses):
        ak = self.a / (self.k + 1 + self.A) ** self.alpha
        ck = self.c / (self.k + 1) ** self.gamma
        g_hat = np.zeros(self.dim)
        n_pairs = len(losses) // 2
        for i in range(n_pairs):
            g_hat += ((losses[2*i] - losses[2*i+1])
                      / (2 * ck * self._deltas[i]))
        g_hat /= n_pairs
        self.x = np.clip(self.x - ak * g_hat,
                         self.bounds[:, 0], self.bounds[:, 1])
        self.k += 1


def run_spsa(batch_size=6, n_epochs=30, seed=0,
             use_parallel=True, max_workers=MAX_WORKERS,
             early_stop=True, use_multifidelity=USE_MULTIFIDELITY):
    n_pairs = max(1, batch_size // 2)
    opt = BatchSPSA(
        x0=PARAM_INIT.copy(), bounds=PARAM_BOUNDS,
        a=0.15, c=0.1, A=10, seed=seed,
    )
    history = []
    best_loss = np.inf
    patience_ctr = 0
    for epoch in range(n_epochs):
        cands = opt.ask(batch_size=n_pairs)
        losses = batched_cat_qubit_delta(
            np.array(cands),
            use_parallel=use_parallel,
            max_workers=max_workers,
            use_multifidelity=use_multifidelity,
        )
        opt.tell(losses.tolist())
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch % 5 == 0:
            print(f"  [SPSA]     Epoch {epoch:3d} | δ={epoch_loss:.4f}")

        if early_stop:
            best_loss, patience_ctr, stop = _should_early_stop(
                best_loss, epoch_loss, patience_ctr)
            if stop:
                print(f"  [SPSA]     Early stop at epoch {epoch:3d}")
                break
    return np.array(history)


# ============================================================
# Comparison plot
# ============================================================

def plot_comparison(results: dict, n_epochs: int):
    plt.figure(figsize=(9, 5))
    styles = {
        "CMA-ES":   dict(color="tab:blue",   linewidth=2),
        "PPO":      dict(color="tab:orange",  linewidth=2),
        "BayesOpt": dict(color="tab:green",   linewidth=2),
        "SPSA":     dict(color="tab:red",     linewidth=2),
    }
    for name, losses in results.items():
        # Curves may have different lengths when early stopping is enabled.
        epochs = np.arange(len(losses))
        plt.plot(epochs, losses, label=name, **styles.get(name, {}))
    plt.xlim(0, max(0, n_epochs - 1))
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(r"$\delta$ (lower is better)", fontsize=12)
    plt.title(r"Optimizer Comparison — Cat-Qubit Calibration"
              "\n"
              r"$\delta = \beta_1|T_x| + \beta_2|T_z|"
              r" + \varepsilon|\eta - \eta_{\mathrm{goal}}|$",
              fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    BATCH_SIZE = 6
    N_EPOCHS = 30
    SEED = 0
    USE_PARALLEL = True
    WORKERS = MAX_WORKERS
    USE_EARLY_STOP = True
    USE_MULTIFIDELITY_RUNTIME = USE_MULTIFIDELITY

    results = {}

    worker_count = _resolve_workers(WORKERS)
    mode = "parallel" if USE_PARALLEL and worker_count > 1 else "serial"
    fidelity_mode = "multi-fidelity" if USE_MULTIFIDELITY_RUNTIME else "full-fidelity"
    print(f"Runtime mode: {mode}, workers={worker_count}, objective={fidelity_mode}")

    print("=" * 55)
    print("1 / 4  CMA-ES")
    print("=" * 55)
    results["CMA-ES"] = run_cmaes(
        BATCH_SIZE, N_EPOCHS, SEED,
        use_parallel=USE_PARALLEL, max_workers=WORKERS,
        early_stop=USE_EARLY_STOP,
        use_multifidelity=USE_MULTIFIDELITY_RUNTIME,
    )

    print()
    print("=" * 55)
    print("2 / 4  PPO")
    print("=" * 55)
    results["PPO"] = run_ppo(
        BATCH_SIZE, N_EPOCHS, SEED,
        use_parallel=USE_PARALLEL, max_workers=WORKERS,
        early_stop=USE_EARLY_STOP,
        use_multifidelity=USE_MULTIFIDELITY_RUNTIME,
    )

    print()
    print("=" * 55)
    print("3 / 4  Bayesian Optimization")
    print("=" * 55)
    results["BayesOpt"] = run_bayesopt(
        BATCH_SIZE, N_EPOCHS, SEED,
        use_parallel=USE_PARALLEL, max_workers=WORKERS,
        early_stop=USE_EARLY_STOP,
        use_multifidelity=USE_MULTIFIDELITY_RUNTIME,
    )

    print()
    print("=" * 55)
    print("4 / 4  Batch SPSA")
    print("=" * 55)
    results["SPSA"] = run_spsa(
        BATCH_SIZE, N_EPOCHS, SEED,
        use_parallel=USE_PARALLEL, max_workers=WORKERS,
        early_stop=USE_EARLY_STOP,
        use_multifidelity=USE_MULTIFIDELITY_RUNTIME,
    )

    print()
    print("=" * 55)
    print("Plotting comparison ...")
    print("=" * 55)
    plot_comparison(results, N_EPOCHS)
