"""
PPO Optimizer for Cat-Qubit T_Z / T_X  —  Dyna / surrogate-accelerated
=======================================================================
Objectives:
  1. Maximize Tz and Tx (lifetimes as large as possible)
  2. Drive Tz / Tx → 320 and keep it there

Speed-up strategy — Dyna-style model-based RL:
  The quantum simulator (compute_vals) is the bottleneck: every call runs
  two full ODE solves.  We fit a tiny neural surrogate on all real
  observations so far, then do N_VIRTUAL cheap "imagined" rollouts per
  real one.  PPO gets (1 + N_VIRTUAL)× more gradient updates for the same
  simulation cost.  The surrogate is only used once MIN_REAL observations
  have been collected (first ~1 iteration) so it starts from real data.

Ratio stability:
  • State includes log-ratio error so agent can see and correct drift
  • β anneals 2 → 12 over training (curriculum: explore first, lock ratio late)
  • Squared log-ratio penalty: zero gradient at target, smooth pull away

Reward:
  r = 0.5·(log Tz + log(320·Tx))  −  β · (log(Tz/Tx) − log 320)²

Control parameters: eps_d_real, eps_d_im, g2_re, g2_im, delta_d  (MHz / dimensionless detuning on a†a)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

<<<<<<< HEAD:team-piqasso/Scripts/PPO_optimizer.py
from Scratch_copy import compute_vals
=======
from Scratch_nontuneable import compute_vals
>>>>>>> origin/batu:challenge/PPO_optimizer.py

# ── Constants ──────────────────────────────────────────────────────────────────
RATIO_TARGET = 320.0
LOG_TARGET   = np.log(RATIO_TARGET)
BETA_START   = 2.0
BETA_END     = 12.0

# ── Parameter bounds ──────────────────────────────────────────────────────────
PARAM_BOUNDS = np.array([
    [0.0,8.0],   # eps_d_real
    [0.0,8.0],   # eps_d_im
    [-1.0,1.0],   # g2_re
    [-1.0,1.0],   # g2_im
<<<<<<< HEAD:team-piqasso/Scripts/PPO_optimizer.py
    [-1.0,1.0],   # delta_d
=======
>>>>>>> origin/batu:challenge/PPO_optimizer.py
], dtype=np.float64)
PARAM_DIM  = PARAM_BOUNDS.shape[0]
PARAM_SPAN = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

# Order matches PARAM_BOUNDS: eps_d_re, eps_d_im, g2_re, g2_im, delta_d
INIT_PARAMS = np.array([4.0, 0.0, 1.0, 0.0, 0.0])

# Augmented state: 4 params + log_ratio_err + log_tz_norm + log_tx_norm
STATE_DIM = PARAM_DIM + 3


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize(params: np.ndarray) -> np.ndarray:
    lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    return 2.0 * (params - lo) / (hi - lo) - 1.0

def denormalize(norm_params: np.ndarray) -> np.ndarray:
    lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    return lo + (norm_params + 1.0) / 2.0 * (hi - lo)

def clip_params(params: np.ndarray) -> np.ndarray:
    return np.clip(params, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])

def build_state(params: np.ndarray, Tz: float, Tx: float, eps: float = 1e-8) -> np.ndarray:
    """7-D state: [norm_params | log_ratio_err | log_tz_norm | log_tx_norm]"""
    Tz_s = max(abs(Tz), eps)
    Tx_s = max(abs(Tx), eps)
    return np.concatenate([
        normalize(params),
        [(np.log(Tz_s / Tx_s) - LOG_TARGET) / LOG_TARGET],  # ≈0 at target
        [np.log(Tz_s)              / np.log(1e4)],
        [np.log(RATIO_TARGET*Tx_s) / np.log(1e4)],
    ]).astype(np.float32)

def reward_fn(Tz: float, Tx: float, beta: float, eps: float = 1e-8) -> float:
    Tz_s = max(abs(Tz), eps)
    Tx_s = max(abs(Tx), eps)
    log_gmean     = 0.5 * (np.log(Tz_s) + np.log(RATIO_TARGET * Tx_s))
    log_ratio_err = np.log(Tz_s / Tx_s) - LOG_TARGET
    return float(log_gmean - beta * log_ratio_err ** 2)


# Log-barrier: repels from parameter bounds so the optimizer stays interior.
# For p in [lo, hi]:  barrier(p) = log(p-lo) + log(hi-p)
# → 0 at midpoint, -∞ at either bound.  Scaled by `strength` (default 0.05).
BARRIER_STRENGTH = 0.05

def log_barrier(params: np.ndarray) -> float:
    """Sum of log-distance-to-bounds over all parameters, scaled by BARRIER_STRENGTH."""
    lo, hi  = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    d_lo    = np.maximum(params - lo, 1e-9)
    d_hi    = np.maximum(hi - params, 1e-9)
    return float(BARRIER_STRENGTH * np.sum(np.log(d_lo) + np.log(d_hi)))


def log_barrier_tensor(norm_params: torch.Tensor) -> torch.Tensor:
    """Differentiable log-barrier for the surrogate gradient search."""
    lo  = torch.tensor(PARAM_BOUNDS[:, 0], dtype=torch.float32)
    hi  = torch.tensor(PARAM_BOUNDS[:, 1], dtype=torch.float32)
    # Denormalise inside the tensor graph
    params = lo + (norm_params + 1.0) / 2.0 * (hi - lo)
    d_lo   = (params - lo).clamp(min=1e-9)
    d_hi   = (hi - params).clamp(min=1e-9)
    return BARRIER_STRENGTH * (torch.log(d_lo) + torch.log(d_hi)).sum(-1)


# ── Neural surrogate  (params → Tz, Tx) ───────────────────────────────────────
class Surrogate:
    """
    Tiny MLP that maps 4D normalised params → (log Tz, log Tx).

    Trained incrementally on all real observations.  Once MIN_REAL points
    exist, PPO uses it for N_VIRTUAL cheap rollouts between every real one.

    Training target: log-space values (more Gaussian, easier to regress).
    Architecture: 4 → 128 → 128 → 2  with SiLU — smooth and fast.
    """
    MIN_REAL = 20   # observations needed before surrogate is trusted

    def __init__(self):
        self._net = nn.Sequential(
            nn.Linear(PARAM_DIM, 128), nn.SiLU(),
            nn.Linear(128, 128),       nn.SiLU(),
            nn.Linear(128,  64),       nn.SiLU(),
            nn.Linear(64,    2),
        )
        self._opt    = optim.Adam(self._net.parameters(), lr=3e-3, weight_decay=1e-5)
        self._X: list[np.ndarray] = []   # normalised params
        self._Y: list[np.ndarray] = []   # [log_Tz, log_Tx]
        self.trained = False

    # ── data ──────────────────────────────────────────────────────────────────
    def add(self, params: np.ndarray, Tz: float, Tx: float, eps: float = 1e-8):
        self._X.append(normalize(params).astype(np.float32))
        self._Y.append(np.array([np.log(max(abs(Tz), eps)),
                                  np.log(max(abs(Tx), eps))], dtype=np.float32))

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, n_steps: int = 300):
        """Fit on entire buffer.  Called after each real rollout."""
        if len(self._X) < self.MIN_REAL:
            return
        X = torch.from_numpy(np.array(self._X))
        Y = torch.from_numpy(np.array(self._Y))
        # Normalise targets for stable training
        Y_mean = Y.mean(0, keepdim=True)
        Y_std  = Y.std(0, keepdim=True).clamp(min=1e-6)
        Yn     = (Y - Y_mean) / Y_std
        self._net.train()
        for _ in range(n_steps):
            loss = (self._net(X) - Yn).pow(2).mean()
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
        self._Y_mean = Y_mean
        self._Y_std  = Y_std
        self.trained = True

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, params: np.ndarray) -> tuple[float, float]:
        x = torch.from_numpy(normalize(params).astype(np.float32)).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            out = self._net(x).squeeze(0)
        log_Tz = float(out[0] * self._Y_std[0, 0] + self._Y_mean[0, 0])
        log_Tx = float(out[1] * self._Y_std[0, 1] + self._Y_mean[0, 1])
        return float(np.exp(log_Tz)), float(np.exp(log_Tx))

    # ── differentiable reward (for gradient optimisation) ─────────────────────
    def reward_tensor(self, norm_params: torch.Tensor, beta: float) -> torch.Tensor:
        """Reward as a differentiable function of normalised params tensor."""
        out    = self._net(norm_params)                              # (B, 2) normalised
        log_Tz = out[:, 0] * self._Y_std[0, 0] + self._Y_mean[0, 0]  # true log-scale
        log_Tx = out[:, 1] * self._Y_std[0, 1] + self._Y_mean[0, 1]
        log_gmean     = 0.5 * (log_Tz + float(np.log(RATIO_TARGET)) + log_Tx)
        log_ratio_err = log_Tz - log_Tx - float(LOG_TARGET)
        return log_gmean - beta * log_ratio_err ** 2 + log_barrier_tensor(norm_params)

    # ── gradient-based candidate search ───────────────────────────────────────
    def gradient_optimize(
        self,
        start_params: np.ndarray,
        beta:         float,
        n_steps:      int   = 300,
        n_restarts:   int   = 8,
        lr:           float = 0.02,
    ) -> list:
        """
        Gradient ascent on the surrogate reward from N random starting points.

        Each restart starts near `start_params` plus Gaussian noise, then
        does Adam gradient steps directly through the surrogate network.
        Returns a list of (predicted_reward, params) sorted best-first.

        This replaces the expensive random-walk real rollout: we get
        highly-targeted candidates in milliseconds instead of ODE solves.
        """
        s0 = normalize(start_params).astype(np.float32)
        starts = [s0.copy()]
        for _ in range(n_restarts - 1):
            noise = np.random.randn(PARAM_DIM).astype(np.float32) * 0.4
            starts.append(np.clip(s0 + noise, -1.0, 1.0))

        self._net.train()   # allow gradients to flow
        results = []
        for s in starts:
            p   = torch.tensor(s, requires_grad=True)
            opt = optim.Adam([p], lr=lr)
            for _ in range(n_steps):
                opt.zero_grad()
                r = self.reward_tensor(p.clamp(-1, 1).unsqueeze(0), beta).squeeze()
                (-r).backward()   # ascent
                opt.step()
            with torch.no_grad():
                p_clipped  = p.clamp(-1, 1)
                params_out = clip_params(denormalize(p_clipped.numpy()))
                Tz_p, Tx_p = self.predict(params_out)
                results.append((reward_fn(Tz_p, Tx_p, beta)+log_barrier(params_out), params_out.copy()))

        self._net.eval()
        results.sort(key=lambda x: x[0], reverse=True)
        return results   # [(predicted_reward, params), ...]

    @property
    def ready(self) -> bool:
        return self.trained and len(self._X) >= self.MIN_REAL

    def __len__(self) -> int:
        return len(self._X)


# ── Running reward statistics (EMA) ───────────────────────────────────────────
class RunningStats:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.mean  = 0.0
        self.var   = 1.0

    def update(self, values: np.ndarray):
        self.mean = (1-self.alpha)*self.mean + self.alpha*float(values.mean())
        self.var  = (1-self.alpha)*self.var  + self.alpha*float(values.var())

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (np.sqrt(self.var) + 1e-8)


# ── Actor-Critic ───────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.critic        = nn.Linear(128, 1)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        h   = self.shared(x)
        std = self.actor_log_std.exp().clamp(1e-4, 0.8)
        return self.actor_mean(h), std, self.critic(h)

    def get_action(self, state):
        mean, std, val = self(state)
        dist     = Normal(mean, std)
        action   = dist.sample()
        return action, dist.log_prob(action).sum(-1), val.squeeze(-1)

    def evaluate(self, states, actions):
        mean, std, val = self(states)
        dist = Normal(mean, std)
        return dist.log_prob(actions).sum(-1), val.squeeze(-1), dist.entropy().sum(-1)


# ── Environment ────────────────────────────────────────────────────────────────
class QuantumEnv:
    def __init__(
        self,
        steps_per_episode: int   = 10,
        max_delta_start:   float = 0.30,
        max_delta_end:     float = 0.04,
        n_iterations:      int   = 50,
        restart_prob:      float = 0.40,
    ):
        self.steps_per_episode = steps_per_episode
        self.max_delta_start   = max_delta_start
        self.max_delta_end     = max_delta_end
        self.n_iterations      = n_iterations
        self.restart_prob      = restart_prob
        self.max_delta         = max_delta_start
        self.best_params       = INIT_PARAMS.copy()
        self.params            = INIT_PARAMS.copy()
        self.last_Tz           = 1.0
        self.last_Tx           = 1.0 / RATIO_TARGET
        self.step_count        = 0

    def anneal(self, iteration: int):
        frac           = min(iteration / max(self.n_iterations - 1, 1), 1.0)
        self.max_delta = self.max_delta_start + frac*(self.max_delta_end - self.max_delta_start)

    def notify_best(self, params, Tz, Tx):
        self.best_params = params.copy()
        self.last_Tz = Tz
        self.last_Tx = Tx

    def reset(self, params=None) -> np.ndarray:
        if params is not None:
            self.params = params.copy()
        elif np.random.random() < self.restart_prob:
            noise = np.random.randn(PARAM_DIM) * self.max_delta * PARAM_SPAN * 0.5
            self.params = clip_params(self.best_params + noise)
        else:
            self.params = INIT_PARAMS.copy()
        self.step_count = 0
        return build_state(self.params, self.last_Tz, self.last_Tx)

    def _apply_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Shared geometry: apply clipped delta, return new params."""
        delta_norm = np.clip(delta_norm, -self.max_delta, self.max_delta)
        new_norm   = np.clip(normalize(self.params) + delta_norm, -1.0, 1.0)
        new_params = clip_params(denormalize(new_norm))
        self.params = new_params
        self.step_count += 1
        return new_params

    def step(self, delta_norm: np.ndarray, beta: float):
        """Real step — calls compute_vals."""
        new_params = self._apply_delta(delta_norm)
        try:
            Tz, Tx = compute_vals(*new_params)
            Tz, Tx = float(Tz), float(Tx)
            self.last_Tz = Tz
            self.last_Tx = Tx
            reward = reward_fn(Tz, Tx, beta) + log_barrier(new_params)
            info   = {"Tz": Tz, "Tx": Tx, "params": new_params.copy()}
        except Exception as exc:
            Tz = Tx = 1e-6
            reward = reward_fn(Tz, Tx, beta) + log_barrier(new_params)
            info   = {"error": str(exc), "params": new_params.copy(), "Tz": 0.0, "Tx": 0.0}
        return build_state(self.params, Tz, Tx), reward, self.step_count >= self.steps_per_episode, info

    def virtual_step(self, delta_norm: np.ndarray, surrogate: Surrogate, beta: float):
        """Virtual step — uses surrogate instead of real sim.  Fast."""
        new_params = self._apply_delta(delta_norm)
        Tz, Tx     = surrogate.predict(new_params)
        reward     = reward_fn(Tz, Tx, beta) + log_barrier(new_params)
        info       = {"Tz": Tz, "Tx": Tx, "params": new_params.copy(), "virtual": True}
        return build_state(self.params, Tz, Tx), reward, self.step_count >= self.steps_per_episode, info


# ── PPO agent ─────────────────────────────────────────────────────────────────
class PPO:
    def __init__(
        self,
        state_dim:   int   = STATE_DIM,
        action_dim:  int   = PARAM_DIM,
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        lam:         float = 0.95,
        clip_eps:    float = 0.2,
        ppo_epochs:  int   = 6,
        batch_size:  int   = 16,
        ent_coef:    float = 0.02,
        vf_coef:     float = 0.5,
        max_grad:    float = 0.5,
        kl_target:   float = 0.02,
        n_virtual:   int   = 3,     # virtual rollouts per real rollout
    ):
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.ent_coef   = ent_coef
        self.vf_coef    = vf_coef
        self.max_grad   = max_grad
        self.kl_target  = kl_target
        self.n_virtual  = n_virtual

        self.model        = ActorCritic(state_dim, action_dim)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=lr)
        self.reward_stats = RunningStats(alpha=0.1)
        self.surrogate    = Surrogate()

    def _beta(self, iteration, n_iterations):
        frac = min(iteration / max(n_iterations - 1, 1), 1.0)
        return BETA_START + frac * (BETA_END - BETA_START)

    def _compute_gae(self, rewards, values, dones, last_value=0.0):
        advantages, gae, next_val = [], 0.0, last_value
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.gamma * next_val * (1 - float(d)) - v
            gae   = delta + self.gamma * self.lam * (1 - float(d)) * gae
            advantages.insert(0, gae)
            next_val = v
        adv = torch.tensor(advantages, dtype=torch.float32)
        return adv + torch.tensor(values, dtype=torch.float32), adv

    # ── rollout collectors ────────────────────────────────────────────────────
    def _collect(self, env: QuantumEnv, rollout_steps: int, beta: float,
                 virtual: bool = False) -> tuple:
        states, actions, log_probs, rewards, values, dones, infos = \
            [], [], [], [], [], [], []
        state = env.reset()
        for _ in range(rollout_steps):
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, lp, val = self.model.get_action(st)
            a_np = action.cpu().numpy().flatten()

            if virtual:
                ns, r, done, info = env.virtual_step(a_np, self.surrogate, beta)
            else:
                ns, r, done, info = env.step(a_np, beta)
                # Feed real observations to surrogate buffer
                if "error" not in info:
                    self.surrogate.add(info["params"], info["Tz"], info["Tx"])

            states.append(state);  actions.append(a_np)
            log_probs.append(lp.item()); rewards.append(r)
            values.append(val.item());   dones.append(done); infos.append(info)

            tag = "[V]" if virtual else "   "
            print(f"  {tag} r={r:+.3f}  {_fmt_info(info)}")

            state = env.reset() if done else ns
        return states, actions, log_probs, rewards, values, dones, infos

    # ── PPO update ────────────────────────────────────────────────────────────
    def _update(self, states, actions, old_lps, returns, advantages):
        St = torch.tensor(np.array(states),  dtype=torch.float32)
        At = torch.tensor(np.array(actions), dtype=torch.float32)
        Ot = torch.tensor(old_lps,           dtype=torch.float32)
        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(len(St))
            kl_sum, n = 0.0, 0
            for s in range(0, len(St), self.batch_size):
                idx = perm[s:s+self.batch_size]
                lp, vals, ent = self.model.evaluate(St[idx], At[idx])
                ratio = (lp - Ot[idx]).exp()
                adv   = advantages[idx]
                loss  = (-torch.min(ratio*adv,
                                    ratio.clamp(1-self.clip_eps, 1+self.clip_eps)*adv).mean()
                         + self.vf_coef*(returns[idx]-vals).pow(2).mean()
                         - self.ent_coef*ent.mean())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
                self.optimizer.step()
                kl_sum += ((Ot[idx]-lp).exp() - 1 - (Ot[idx]-lp)).mean().item()
                n += 1
            if kl_sum/max(n,1) > 1.5*self.kl_target:
                print(f"    [KL early stop epoch {epoch+1}]");  break

    # ── main training loop ────────────────────────────────────────────────────
    def train(
        self,
        env:           QuantumEnv,
        n_iterations:  int = 50,
        rollout_steps: int = 32,
        n_verify:      int = 3,    # real sims per main-phase iter (targeted)
        plot_path:     str = "ppo_progress.png",
    ):
        """
        Two-phase training:

        WARMUP  (surrogate not ready, iter 1):
          Real rollout → build surrogate buffer → fit surrogate
          Cheap virtual rollouts for initial policy signal

        MAIN  (surrogate ready, iter 2+):
          1. Virtual-only PPO (n_virtual rollouts) — policy learning, zero real cost
          2. Surrogate gradient search — find top candidates in milliseconds
          3. Real-sim verification of top-n_verify candidates only
          4. Surrogate refit with new points

        Real-sim budget:
          Before: rollout_steps × n_iterations  =  32 × 50  = 1 600 calls
          After : rollout_steps + n_verify × (n_iterations-1)  =  32 + 3×49 ≈ 179 calls
          → ~9× fewer expensive ODE solves, targeted at highest-predicted-reward points
        """
        best_reward, best_params, best_info = -np.inf, INIT_PARAMS.copy(), {}
        hist = _empty_history()

        def _try_update_best(reward, info):
            nonlocal best_reward, best_params, best_info
            if reward > best_reward:
                best_reward = reward
                best_info   = info
                best_params = info["params"].copy()
                Tz    = info.get("Tz", float("nan"))
                Tx    = info.get("Tx", float("nan"))
                ratio = Tz / max(abs(Tx), 1e-8)
                env.notify_best(best_params, Tz, Tx)
                print(f"  *** NEW BEST  r={best_reward:+.4f}  "
                      f"Tz={Tz:.4f}  Tx={Tx:.6f}  "
                      f"ratio={ratio:.1f}  |err|={(abs(ratio-320)/320*100):.1f}%")

        def _record_epoch_best(it):
            """Snapshot the all-time best found so far into history (one point per iter)."""
            if best_info is None or not best_info:
                return
            p     = best_params
            Tz    = best_info.get("Tz", float("nan"))
            Tx    = best_info.get("Tx", float("nan"))
            ratio = Tz / max(abs(Tx), 1e-8)
            hist["best_iter"].append(it)
            hist["best_eps_d_re"].append(p[0]); hist["best_eps_d_im"].append(p[1])
            hist["best_g2_re"].append(p[2]);    hist["best_g2_im"].append(p[3])
            hist["best_Tz"].append(Tz);         hist["best_Tx"].append(Tx)
            hist["best_ratio"].append(ratio);   hist["best_reward"].append(best_reward)

        for it in range(1, n_iterations + 1):
            env.anneal(it - 1)
            beta       = self._beta(it - 1, n_iterations)
            iter_evals = []   # (reward, info) for every real eval this iteration

            # ══════════════════════════════════════════════════════════════
            # WARMUP PHASE  — build surrogate from real rollout
            # ══════════════════════════════════════════════════════════════
            if not self.surrogate.ready:
                print(f"\n=== iter {it}/{n_iterations}  β={beta:.2f}"
                      f"  Δ={env.max_delta:.3f}  [WARMUP — real rollout] ===")

                r_s, r_a, r_lp, r_r, r_v, r_d, r_info = \
                    self._collect(env, rollout_steps, beta, virtual=False)

                self.surrogate.fit(n_steps=300)
                print(f"    surrogate seeded with {len(self.surrogate)} points"
                      + ("  → ready" if self.surrogate.ready else "  → not ready yet"))

                # Seed virtual rollouts if surrogate already ready after warmup
                all_s, all_a, all_lp = list(r_s), list(r_a), list(r_lp)
                all_r, all_v, all_d  = list(r_r),  list(r_v), list(r_d)
                if self.surrogate.ready:
                    for _ in range(self.n_virtual):
                        vs, va, vlp, vr, vv, vd, _ = \
                            self._collect(env, rollout_steps, beta, virtual=True)
                        all_s += vs; all_a += va; all_lp += vlp
                        all_r += vr; all_v += vv; all_d  += vd

                r_arr = np.array(all_r, dtype=np.float64)
                self.reward_stats.update(r_arr)
                norm_r = self.reward_stats.normalize(r_arr).tolist()
                last_st = torch.tensor(all_s[-1], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, _, lv = self.model(last_st)
                returns, adv = self._compute_gae(norm_r, all_v, all_d,
                                                  lv.item() if not all_d[-1] else 0.0)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                self._update(all_s, all_a, all_lp, returns, adv)

                real_r = np.array(r_r)
                print(f"  real mean={real_r.mean():+.4f}  max={real_r.max():+.4f}")
                for idx, info in enumerate(r_info):
                    if "error" not in info:
                        iter_evals.append((r_r[idx], info))
                        _try_update_best(r_r[idx], info)
                _record_epoch_best(it)
                _save_plots(hist, plot_path)
                print(f"  plot saved → {plot_path}")
                continue

            # ══════════════════════════════════════════════════════════════
            # MAIN PHASE  — virtual PPO + targeted real verification
            # ══════════════════════════════════════════════════════════════
            print(f"\n=== iter {it}/{n_iterations}  β={beta:.2f}"
                  f"  Δ={env.max_delta:.3f}"
                  f"  surrogate={len(self.surrogate)} pts  [VIRTUAL+VERIFY] ===")

            # 1. Virtual-only rollouts → policy update (no real cost)
            all_s, all_a, all_lp = [], [], []
            all_r, all_v, all_d  = [], [], []
            for _ in range(self.n_virtual):
                vs, va, vlp, vr, vv, vd, _ = \
                    self._collect(env, rollout_steps, beta, virtual=True)
                all_s += vs; all_a += va; all_lp += vlp
                all_r += vr; all_v += vv; all_d  += vd

            r_arr = np.array(all_r, dtype=np.float64)
            self.reward_stats.update(r_arr)
            norm_r = self.reward_stats.normalize(r_arr).tolist()
            last_st = torch.tensor(all_s[-1], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, lv = self.model(last_st)
            returns, adv = self._compute_gae(norm_r, all_v, all_d,
                                              lv.item() if not all_d[-1] else 0.0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            self._update(all_s, all_a, all_lp, returns, adv)

            # 2. Surrogate gradient search — milliseconds, no ODE solves
            candidates = self.surrogate.gradient_optimize(
                env.best_params, beta, n_steps=300, n_restarts=8)
            print(f"  surrogate top predictions: "
                  + "  ".join(f"{r:+.3f}" for r, _ in candidates[:n_verify]))

            # 3. Real verification of top-n_verify candidates
            real_rewards = []
            for r_pred, cand_params in candidates[:n_verify]:
                try:
                    Tz, Tx   = compute_vals(*cand_params)
                    Tz, Tx   = float(Tz), float(Tx)
                    r_real   = reward_fn(Tz, Tx, beta) + log_barrier(cand_params)
                    info     = {"Tz": Tz, "Tx": Tx, "params": cand_params.copy()}
                    self.surrogate.add(cand_params, Tz, Tx)
                    real_rewards.append(r_real)
                    iter_evals.append((r_real, info))
                    print(f"  [verify] pred={r_pred:+.4f} → real={r_real:+.4f}"
                          f"  {_fmt_info(info)}")
                    _try_update_best(r_real, info)
                except Exception as exc:
                    print(f"  [verify failed] {str(exc)[:60]}")

            # 4. Refit surrogate + save plots
            self.surrogate.fit(n_steps=300)
            if real_rewards:
                print(f"  real mean={np.mean(real_rewards):+.4f}"
                      f"  max={max(real_rewards):+.4f}")
            _record_epoch_best(it)
            _save_plots(hist, plot_path)
            print(f"  plot saved → {plot_path}")

        return best_params, best_reward, best_info


# ── Live plotting ──────────────────────────────────────────────────────────────
def _save_plots(history: dict, out_path: str = "ppo_progress.png") -> None:
    """
    Save a 6-panel figure to *out_path* showing training progress.

    Each panel has:
      • grey dots  — every real evaluation in that iteration
      • coloured line + markers — running best-so-far
      • x-axis ticks = iteration number
    """
    iters = history["best_iter"]
    if not iters:
        return

    fig = plt.figure(figsize=(16, 10), tight_layout=True)
    fig.suptitle("PPO training progress  (best per iteration)", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(row, col, title, ylabel, target=None):
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(set(iters)))
        if target is not None:
            ax.axhline(target, color="red", lw=1.2, ls="--", label=f"target {target}")
        return ax

    # ── ε_d  ────────────────────────────────────────────────────────────────
    ax = _ax(0, 0, r"Drive amplitude $\varepsilon_d$", "MHz")
    ax.plot(history["best_iter"], history["best_eps_d_re"], "o-",
            color="steelblue",  lw=2, ms=5, label=r"$\varepsilon_{d,\mathrm{re}}$")
    ax.plot(history["best_iter"], history["best_eps_d_im"], "s-",
            color="darkorange", lw=2, ms=5, label=r"$\varepsilon_{d,\mathrm{im}}$")
    ax.legend(fontsize=8, loc="best")

    # ── g_2  ────────────────────────────────────────────────────────────────
    ax = _ax(0, 1, r"Two-photon coupling $g_2$", "MHz")
    ax.plot(history["best_iter"], history["best_g2_re"], "o-",
            color="seagreen",      lw=2, ms=5, label=r"$g_{2,\mathrm{re}}$")
    ax.plot(history["best_iter"], history["best_g2_im"], "s-",
            color="mediumpurple",  lw=2, ms=5, label=r"$g_{2,\mathrm{im}}$")
    ax.legend(fontsize=8, loc="best")

    # ── Ratio  ──────────────────────────────────────────────────────────────
    ax = _ax(1, 0, r"Ratio $T_Z / T_X$", r"$T_Z/T_X$", target=320.0)
    ax.plot(history["best_iter"], history["best_ratio"], "o-",
            color="crimson", lw=2, ms=5)

    # ── Tz  ─────────────────────────────────────────────────────────────────
    ax = _ax(1, 1, r"$T_Z$ lifetime", r"$T_Z$ (µs)")
    ax.plot(history["best_iter"], history["best_Tz"], "o-",
            color="steelblue", lw=2, ms=5)

    # ── Tx  ─────────────────────────────────────────────────────────────────
    ax = _ax(2, 0, r"$T_X$ lifetime", r"$T_X$ (µs)")
    ax.plot(history["best_iter"], history["best_Tx"], "o-",
            color="darkorange", lw=2, ms=5)

    # ── Reward  ─────────────────────────────────────────────────────────────
    ax = _ax(2, 1, "Reward", "r(Tz, Tx)")
    ax.plot(history["best_iter"], history["best_reward"], "o-",
            color="seagreen", lw=2, ms=5)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _empty_history() -> dict:
    return dict(
        best_iter=[], best_eps_d_re=[], best_eps_d_im=[],
        best_g2_re=[], best_g2_im=[], best_Tz=[], best_Tx=[],
        best_ratio=[], best_reward=[],
    )


# ── Helper ─────────────────────────────────────────────────────────────────────
def _fmt_info(info: dict) -> str:
    if "error" in info:
        return f"ERR:{info['error'][:40]}"
    p   = info.get("params", [0]*4)
    Tz  = info.get("Tz", 0.0)
    Tx  = info.get("Tx", 0.0)
    rat = Tz / max(abs(Tx), 1e-8)
    virt = " [V]" if info.get("virtual") else ""
    return (f"Tz={Tz:.3f} Tx={Tx:.5f} ratio={rat:.1f} "
            f"eps=({p[0]:.2f},{p[1]:.2f}) g2=({p[2]:.2f},{p[3]:.2f}){virt}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_ITER = 30

    env = QuantumEnv(
        steps_per_episode = 10,
        max_delta_start   = 0.30,
        max_delta_end     = 0.04,
        n_iterations      = N_ITER,
        restart_prob      = 0.40,
    )

    ppo = PPO(
        lr         = 1e-4,
        gamma      = 0.99,
        lam        = 0.95,
        clip_eps   = 0.2,
        ppo_epochs = 6,
        batch_size = 16,
        ent_coef   = 0.02,
        vf_coef    = 0.5,
        max_grad   = 0.5,
        kl_target  = 0.02,
        n_virtual  = 3,     # 3 virtual rollouts per real → ~4× cheaper per PPO update
    )

    best_params, best_reward, best_info = ppo.train(
        env,
        n_iterations  = N_ITER,
        rollout_steps = 32,
        n_verify      = 3,
        plot_path     = "ppo_progress.png",
    )

    Tz    = best_info.get("Tz", float("nan"))
    Tx    = best_info.get("Tx", float("nan"))
    ratio = Tz / max(abs(Tx), 1e-8)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print(f"  Best reward : {best_reward:+.6f}")
    print(f"  Tz          : {Tz:.4f}")
    print(f"  Tx          : {Tx:.6f}")
    print(f"  Tz/Tx       : {ratio:.2f}  (target 320,  |err|={(abs(ratio-320)/320*100):.2f}%)")
    print(f"  eps_d_real  : {best_params[0]:.4f}")
    print(f"  eps_d_im    : {best_params[1]:.4f}")
    print(f"  g2_re       : {best_params[2]:.4f}")
    print(f"  g2_im       : {best_params[3]:.4f}")
