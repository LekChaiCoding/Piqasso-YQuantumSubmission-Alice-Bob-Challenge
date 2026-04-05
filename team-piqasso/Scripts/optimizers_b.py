"""
optimizers_b.py — Agent B: PPO optimizer for cat qubit knob tuning.

Implements Proximal Policy Optimization (PPO) treating knob tuning as an MDP.

MDP formulation:
    State  s_t : [knob_0, knob_1, knob_2, knob_3, last_reward, epoch/n_epochs]  (6-dim)
    Action a_t : delta applied to knobs, clipped to [-0.3, 0.3]
    Transition : knobs_{t+1} = clip(knobs_t + a_t, KNOB_BOUNDS)
    Reward     : reward_fn(knobs, drift)

Neural networks implemented as 2-layer MLPs using pure numpy (no PyTorch/JAX).
"""

import numpy as np
from catqubit import KNOB_BOUNDS, DEFAULT_KNOBS, N_KNOBS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOB_BOUNDS_ARR = np.array(KNOB_BOUNDS, dtype=float)   # shape (4, 2)
DEFAULT_KNOBS_ARR = np.array(DEFAULT_KNOBS, dtype=float)

_ACTION_CLIP = 0.3    # max magnitude of delta per step
_STATE_DIM = 6        # [k0, k1, k2, k3, last_reward, epoch/n_epochs]
_ACTION_DIM = 4       # one delta per knob
_HIDDEN = 32          # neurons per hidden layer


# ---------------------------------------------------------------------------
# Utility: weight initialisation
# ---------------------------------------------------------------------------

def _init_weights(rng, in_dim, out_dim, scale=0.1):
    """Glorot-like init scaled by `scale`."""
    W = rng.randn(in_dim, out_dim) * scale
    b = np.zeros(out_dim)
    return W, b


# ---------------------------------------------------------------------------
# MLP forward passes (manual numpy matmul)
# ---------------------------------------------------------------------------

def _actor_forward(params, x):
    """Actor forward pass: 6 → 32 → 32 → 4  (tanh activations).

    Parameters
    ----------
    params : list of (W, b) tuples, length 3
    x      : (6,) state vector

    Returns
    -------
    mean : (4,) action mean
    """
    W1, b1 = params[0]
    W2, b2 = params[1]
    W3, b3 = params[2]

    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    mean = h2 @ W3 + b3
    return mean


def _critic_forward(params, x):
    """Critic forward pass: 6 → 32 → 32 → 1  (tanh activations).

    Parameters
    ----------
    params : list of (W, b) tuples, length 3
    x      : (6,) state vector

    Returns
    -------
    value : scalar
    """
    W1, b1 = params[0]
    W2, b2 = params[1]
    W3, b3 = params[2]

    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    value = float((h2 @ W3 + b3).squeeze())
    return value


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def _init_actor_params(rng):
    """Return list of (W, b) for actor MLP.  log_std returned separately."""
    params = [
        _init_weights(rng, _STATE_DIM, _HIDDEN, scale=0.1),
        _init_weights(rng, _HIDDEN, _HIDDEN, scale=0.1),
        _init_weights(rng, _HIDDEN, _ACTION_DIM, scale=0.1),
    ]
    log_std = np.full(_ACTION_DIM, np.log(0.2))
    return params, log_std


def _init_critic_params(rng):
    """Return list of (W, b) for critic MLP."""
    params = [
        _init_weights(rng, _STATE_DIM, _HIDDEN, scale=0.1),
        _init_weights(rng, _HIDDEN, _HIDDEN, scale=0.1),
        _init_weights(rng, _HIDDEN, 1, scale=0.1),
    ]
    return params


# ---------------------------------------------------------------------------
# Gaussian log-probability
# ---------------------------------------------------------------------------

def _log_prob(action, mean, log_std):
    """Log-prob of `action` under diagonal Gaussian N(mean, exp(log_std)^2).

    Parameters
    ----------
    action  : (4,)
    mean    : (4,)
    log_std : (4,)

    Returns
    -------
    lp : scalar
    """
    std = np.exp(log_std)
    lp = -0.5 * np.sum(((action - mean) / std) ** 2) \
         - np.sum(log_std) \
         - 0.5 * _ACTION_DIM * np.log(2.0 * np.pi)
    return float(lp)


def _sample_action(mean, log_std, rng):
    """Sample action from N(mean, exp(log_std)^2), clip to [-0.3, 0.3]."""
    std = np.exp(log_std)
    raw = mean + std * rng.randn(_ACTION_DIM)
    return np.clip(raw, -_ACTION_CLIP, _ACTION_CLIP)


# ---------------------------------------------------------------------------
# Flat parameter helpers (for finite-difference gradient)
# ---------------------------------------------------------------------------

def _flatten_actor(params, log_std):
    """Flatten actor params + log_std into a 1-D vector."""
    parts = []
    for W, b in params:
        parts.append(W.ravel())
        parts.append(b.ravel())
    parts.append(log_std.ravel())
    return np.concatenate(parts)


def _unflatten_actor(flat, layer_shapes):
    """Reconstruct actor params and log_std from flat vector.

    Parameters
    ----------
    flat         : 1-D numpy array
    layer_shapes : list of (in_dim, out_dim) for each layer
    """
    offset = 0
    params = []
    for in_d, out_d in layer_shapes:
        n_W = in_d * out_d
        n_b = out_d
        W = flat[offset: offset + n_W].reshape(in_d, out_d)
        offset += n_W
        b = flat[offset: offset + n_b].copy()
        offset += n_b
        params.append((W, b))
    log_std = flat[offset: offset + _ACTION_DIM].copy()
    return params, log_std


def _flatten_critic(params):
    """Flatten critic params into a 1-D vector."""
    parts = []
    for W, b in params:
        parts.append(W.ravel())
        parts.append(b.ravel())
    return np.concatenate(parts)


def _unflatten_critic(flat, layer_shapes):
    """Reconstruct critic params from flat vector."""
    offset = 0
    params = []
    for in_d, out_d in layer_shapes:
        n_W = in_d * out_d
        n_b = out_d
        W = flat[offset: offset + n_W].reshape(in_d, out_d)
        offset += n_W
        b = flat[offset: offset + n_b].copy()
        offset += n_b
        params.append((W, b))
    return params


# ---------------------------------------------------------------------------
# Discounted returns
# ---------------------------------------------------------------------------

def _compute_returns(rewards, gamma):
    """Compute discounted returns G_t = sum_{k>=0} gamma^k * r_{t+k}.

    Parameters
    ----------
    rewards : list or array of shape (T,)
    gamma   : float discount factor

    Returns
    -------
    returns : numpy array shape (T,)
    """
    rewards = np.asarray(rewards, dtype=float)
    T = len(rewards)
    returns = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# ---------------------------------------------------------------------------
# Running reward normaliser
# ---------------------------------------------------------------------------

class _RunningNorm:
    """Online mean/std tracker for reward normalisation."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, x):
        """Update with a batch of values (1-D array)."""
        x = np.asarray(x, dtype=float)
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (total + 1e-8)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / (total + 1e-8)
        new_var = m2 / (total + 1e-8)

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalise(self, x):
        """Normalise x using running statistics."""
        return (np.asarray(x, dtype=float) - self.mean) / (np.sqrt(self.var) + 1e-8)


# ---------------------------------------------------------------------------
# Core PPO implementation
# ---------------------------------------------------------------------------

def run_ppo(
    reward_fn,
    n_epochs=150,
    drift_fn=None,
    seed=0,
    lr=0.01,
    clip_eps=0.2,
    gamma=0.99,
    n_steps=8,
):
    """Run PPO for cat qubit knob optimisation.

    Treats knob tuning as a finite-horizon MDP and collects n_steps transitions
    per epoch using the current stochastic policy.  Parameter updates use a
    simplified REINFORCE-style gradient with a clipped probability ratio
    (PPO-clip objective), approximated via finite differences.

    Parameters
    ----------
    reward_fn : callable(knobs, drift) -> float
        Physics reward function.  `drift` is None or a (4,) drift vector.
    n_epochs  : int
        Number of training epochs (outer loop).
    drift_fn  : callable(epoch) -> np.ndarray(4,) or None
        Optional drift schedule.  Called once per epoch.
    seed      : int
        Random seed for reproducibility.
    lr        : float
        Learning rate for parameter updates.
    clip_eps  : float
        PPO clipping parameter ε (ratio clipped to [1-ε, 1+ε]).
    gamma     : float
        Discount factor for computing returns.
    n_steps   : int
        Number of environment steps per epoch (rollout length).

    Returns
    -------
    results : dict with keys
        'reward_history'    : np.ndarray (n_epochs,)
        'reward_std_history': np.ndarray (n_epochs,)
        'mean_history'      : np.ndarray (n_epochs, 4)  -- policy mean at each epoch
        'best_knobs'        : np.ndarray (4,)
        'best_reward'       : float
        'name'              : 'PPO'
    """
    rng = np.random.RandomState(seed)

    # -- layer shapes for actor and critic --
    actor_shapes = [
        (_STATE_DIM, _HIDDEN),
        (_HIDDEN, _HIDDEN),
        (_HIDDEN, _ACTION_DIM),
    ]
    critic_shapes = [
        (_STATE_DIM, _HIDDEN),
        (_HIDDEN, _HIDDEN),
        (_HIDDEN, 1),
    ]

    # -- initialise networks --
    actor_params, log_std = _init_actor_params(rng)
    critic_params = _init_critic_params(rng)

    # -- flat parameter vectors --
    flat_actor = _flatten_actor(actor_params, log_std)
    flat_critic = _flatten_critic(critic_params)

    # -- running reward normaliser --
    rew_norm = _RunningNorm()

    # -- result containers --
    reward_history = np.zeros(n_epochs)
    reward_std_history = np.zeros(n_epochs)
    mean_history = np.zeros((n_epochs, N_KNOBS))
    best_reward = -np.inf
    best_knobs = DEFAULT_KNOBS_ARR.copy()

    # -- current knob state (persists across epochs) --
    current_knobs = DEFAULT_KNOBS_ARR.copy()
    last_reward = 0.0

    # -------------------------------------------------------------------------
    # Finite-difference gradient helpers
    # -------------------------------------------------------------------------

    _FD_EPS = 1e-4  # finite-difference step size

    def _actor_loss(flat_a, states, actions, advantages, old_log_probs):
        """PPO-clip actor loss (negative because we maximise)."""
        params_a, ls = _unflatten_actor(flat_a, actor_shapes)
        total = 0.0
        for s, a, adv, old_lp in zip(states, actions, advantages, old_log_probs):
            mean = _actor_forward(params_a, s)
            new_lp = _log_prob(a, mean, ls)
            ratio = np.exp(np.clip(new_lp - old_lp, -10.0, 10.0))
            clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            obj = min(ratio * adv, clipped * adv)
            total += obj
        # Return negative loss (we want to maximise)
        return -(total / max(len(states), 1))

    def _critic_loss(flat_c, states, returns_arr):
        """MSE critic loss."""
        params_c = _unflatten_critic(flat_c, critic_shapes)
        total = 0.0
        for s, G in zip(states, returns_arr):
            v = _critic_forward(params_c, s)
            total += (v - G) ** 2
        return total / max(len(states), 1)

    def _fd_gradient_actor(flat_a, states, actions, advantages, old_log_probs):
        """Finite-difference gradient of actor loss w.r.t. flat_a."""
        grad = np.zeros_like(flat_a)
        loss0 = _actor_loss(flat_a, states, actions, advantages, old_log_probs)
        for i in range(len(flat_a)):
            flat_plus = flat_a.copy()
            flat_plus[i] += _FD_EPS
            loss_plus = _actor_loss(flat_plus, states, actions, advantages, old_log_probs)
            grad[i] = (loss_plus - loss0) / _FD_EPS
        return grad

    def _fd_gradient_critic(flat_c, states, returns_arr):
        """Finite-difference gradient of critic loss w.r.t. flat_c."""
        grad = np.zeros_like(flat_c)
        loss0 = _critic_loss(flat_c, states, returns_arr)
        for i in range(len(flat_c)):
            flat_plus = flat_c.copy()
            flat_plus[i] += _FD_EPS
            loss_plus = _critic_loss(flat_plus, states, returns_arr)
            grad[i] = (loss_plus - loss0) / _FD_EPS
        return grad

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    for epoch in range(n_epochs):
        # -- optional drift --
        drift = drift_fn(epoch) if drift_fn is not None else None

        # -- unpack current network parameters --
        actor_params, log_std = _unflatten_actor(flat_actor, actor_shapes)

        # -- build initial state for this epoch's rollout --
        epoch_frac = epoch / max(n_epochs - 1, 1)
        state = np.array([
            current_knobs[0], current_knobs[1],
            current_knobs[2], current_knobs[3],
            last_reward, epoch_frac,
        ], dtype=float)

        # -- compute policy mean at current state for mean_history --
        policy_mean_now = _actor_forward(actor_params, state)
        mean_history[epoch] = current_knobs.copy()

        # -- rollout buffer --
        states_buf = []
        actions_buf = []
        rewards_buf = []
        old_log_probs_buf = []

        rollout_knobs = current_knobs.copy()
        rollout_last_reward = last_reward

        for step in range(n_steps):
            step_epoch_frac = epoch_frac  # keep epoch fraction fixed during rollout
            s = np.array([
                rollout_knobs[0], rollout_knobs[1],
                rollout_knobs[2], rollout_knobs[3],
                rollout_last_reward, step_epoch_frac,
            ], dtype=float)

            # sample action
            mean_s = _actor_forward(actor_params, s)
            action = _sample_action(mean_s, log_std, rng)
            old_lp = _log_prob(action, mean_s, log_std)

            # apply action
            new_knobs = rollout_knobs + action
            # clip to bounds
            for k in range(N_KNOBS):
                lo, hi = KNOB_BOUNDS_ARR[k]
                new_knobs[k] = np.clip(new_knobs[k], lo, hi)

            # compute reward
            r = float(reward_fn(new_knobs, drift))

            # store transition
            states_buf.append(s.copy())
            actions_buf.append(action.copy())
            rewards_buf.append(r)
            old_log_probs_buf.append(old_lp)

            rollout_knobs = new_knobs.copy()
            rollout_last_reward = r

        # -- update current knobs to best step of rollout --
        best_step_idx = int(np.argmax(rewards_buf))
        # Re-apply the actions up to the best step to get best knobs
        best_step_knobs = current_knobs.copy()
        for step in range(best_step_idx + 1):
            delta = actions_buf[step]
            best_step_knobs = best_step_knobs + delta
            for k in range(N_KNOBS):
                lo, hi = KNOB_BOUNDS_ARR[k]
                best_step_knobs[k] = np.clip(best_step_knobs[k], lo, hi)

        current_knobs = rollout_knobs.copy()  # continue from end of rollout
        last_reward = float(rewards_buf[-1])

        # -- record epoch stats --
        epoch_rewards = np.array(rewards_buf, dtype=float)
        reward_history[epoch] = float(np.mean(epoch_rewards))
        reward_std_history[epoch] = float(np.std(epoch_rewards))

        # -- track best overall --
        epoch_best_r = float(np.max(epoch_rewards))
        if epoch_best_r > best_reward:
            best_reward = epoch_best_r
            best_knobs = best_step_knobs.copy()

        # -- compute discounted returns --
        returns_arr = _compute_returns(rewards_buf, gamma)

        # -- running normalisation of returns --
        rew_norm.update(returns_arr)
        norm_returns = rew_norm.normalise(returns_arr)

        # -- critic value estimates --
        critic_params_cur = _unflatten_critic(flat_critic, critic_shapes)
        values = np.array([_critic_forward(critic_params_cur, s) for s in states_buf])

        # -- advantages = normalised returns - baseline --
        advantages = norm_returns - values
        adv_std = float(np.std(advantages))
        advantages = (advantages - float(np.mean(advantages))) / (adv_std + 1e-8)

        # -- actor gradient step (finite differences) --
        actor_grad = _fd_gradient_actor(
            flat_actor,
            states_buf, actions_buf, advantages.tolist(), old_log_probs_buf,
        )
        # Gradient clipping
        gnorm = float(np.linalg.norm(actor_grad))
        if gnorm > 1.0:
            actor_grad = actor_grad / gnorm
        flat_actor = flat_actor - lr * actor_grad

        # -- critic gradient step (finite differences) --
        critic_grad = _fd_gradient_critic(flat_critic, states_buf, norm_returns.tolist())
        gnorm_c = float(np.linalg.norm(critic_grad))
        if gnorm_c > 1.0:
            critic_grad = critic_grad / gnorm_c
        flat_critic = flat_critic - lr * critic_grad

    # -------------------------------------------------------------------------
    # Pack and return results
    # -------------------------------------------------------------------------

    results = {
        'reward_history': reward_history,
        'reward_std_history': reward_std_history,
        'mean_history': mean_history,
        'best_knobs': best_knobs,
        'best_reward': float(best_reward),
        'name': 'PPO',
    }
    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("optimizers_b.py — PPO smoke test (5 epochs, mock reward)")
    print("=" * 60)

    # Simple mock reward: negative sum of squared deviations from target
    _TARGET = np.array([1.5, 0.2, 5.0, 0.5])

    def mock_reward(knobs, drift=None):
        k = np.asarray(knobs, dtype=float)
        if drift is not None:
            k = k + np.asarray(drift, dtype=float)
        return float(-np.sum((k - _TARGET) ** 2))

    results = run_ppo(
        reward_fn=mock_reward,
        n_epochs=5,
        drift_fn=None,
        seed=42,
        lr=0.01,
        clip_eps=0.2,
        gamma=0.99,
        n_steps=8,
    )

    print("\nResult keys:", list(results.keys()))
    print("name            :", results['name'])
    print("reward_history  :", results['reward_history'])
    print("reward_std_history:", results['reward_std_history'])
    print("mean_history    :\n", results['mean_history'])
    print("best_knobs      :", results['best_knobs'])
    print("best_reward     :", results['best_reward'])

    # Basic shape assertions
    assert results['reward_history'].shape == (5,), "reward_history wrong shape"
    assert results['reward_std_history'].shape == (5,), "reward_std_history wrong shape"
    assert results['mean_history'].shape == (5, 4), "mean_history wrong shape"
    assert results['best_knobs'].shape == (4,), "best_knobs wrong shape"
    assert isinstance(results['best_reward'], float), "best_reward should be float"
    assert results['name'] == 'PPO', "name should be 'PPO'"

    print("\n[OK] All smoke test assertions passed.")
