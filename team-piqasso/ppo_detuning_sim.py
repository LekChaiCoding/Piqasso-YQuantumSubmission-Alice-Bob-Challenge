"""CPU-only PPO over ``g2`` and ``epsilon_d`` with optional real Lindblad evolution.

JAX is forced to **CPU** via ``JAX_PLATFORMS=cpu`` before import. Choose ``--backend lindblad`` for the
storage–buffer Hamiltonian and ``dynamiqs.mesolve``. By default only **one** random candidate per
training step is simulated (see ``--sim-budget``);
the policy still samples ``--batch-size`` proposals, then a subset is evaluated. The surrogate
backend evaluates the full batch in one vectorized call unless you set ``--sim-budget``.

Detuning analysis: ``--detuning-sweep-plot out.png`` sweeps ``\\Delta \\in [0,2]`` in
``H \\to H + \\Delta a^\\dagger a``. For each ``\\Delta``, a **coarse** ``\\epsilon_d`` grid plus a
**local refine** grid (fixed seed ``g_2``) finds best reward; work is **parallelized** over ``\\Delta``.
Plots ``\\mathrm{Re}(\\epsilon_d), \\mathrm{Im}(\\epsilon_d)`` vs ``\\Delta``.

Run: ``python team-piqasso/ppo_detuning_sim.py --backend lindblad``
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from multiprocessing import cpu_count, freeze_support
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
from jax.scipy.special import gammaln
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

QUIET = False

# Allowed range for storage detuning Δ in H + Δ a†a (sweep + PPO ``--storage-detuning``).
_DELTA_MIN = 0.0
_DELTA_MAX = 2.0


@dataclass(frozen=True)
class SearchSeedParams:
    epsilon_d: complex
    g2: complex

    def to_vector(self) -> jnp.ndarray:
        return jnp.array(
            [
                float(np.real(self.g2)),
                float(np.imag(self.g2)),
                float(np.real(self.epsilon_d)),
                float(np.imag(self.epsilon_d)),
            ],
            dtype=jnp.float32,
        )

    @classmethod
    def from_vector(cls, vector: jnp.ndarray) -> SearchSeedParams:
        vector = jnp.asarray(vector, dtype=jnp.float32)
        return cls(
            epsilon_d=complex(float(vector[2]), float(vector[3])),
            g2=complex(float(vector[0]), float(vector[1])),
        )


@dataclass(frozen=True)
class ParameterBounds:
    lower: jnp.ndarray
    upper: jnp.ndarray

    @classmethod
    def default(cls) -> ParameterBounds:
        return cls(
            lower=jnp.array([0.25, -1.5, 0.5, -4.0], dtype=jnp.float32),
            upper=jnp.array([2.0, 1.5, 8.0, 4.0], dtype=jnp.float32),
        )

    @property
    def span(self) -> jnp.ndarray:
        return self.upper - self.lower

    def clip(self, vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(jnp.asarray(vector, dtype=jnp.float32), self.lower, self.upper)


@dataclass(frozen=True)
class SimulationConfig:
    na: int = 15
    nb: int = 5
    kappa_a: float = 1.0
    kappa_b: float = 10.0
    x_tfinal: float = 2.0
    z_tfinal: float = 80.0
    nsave: int = 64
    eval_x_tfinal: float = 1.0
    eval_z_tfinal: float = 24.0
    eval_nsave: int = 24
    short_window_fraction: float = 0.30
    #: Storage detuning ``Delta * a^\dagger a`` (same joint ``a`` as drive terms). Use ``[0, 2]`` in practice.
    storage_detuning: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    target_bias: float = 320.0
    eta_min: float = 100.0
    eta_max: float = 750.0
    lambda_eta: float = 2.0
    invalid_eta_reward: float = -1000.0
    delta_eps_max: float = 2.5
    delta_g2_max: float = 1.0


@dataclass(frozen=True)
class PPOConfig:
    batch_size: int = 12
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    entropy_coef: float = 0.02
    learning_rate: float = 0.03
    initial_std: float = 0.18
    min_std: float = 0.02
    max_std: float = 0.55
    seed: int = 13
    epochs: int = 200
    log_every: int = 20


@dataclass(frozen=True)
class CandidateBatch:
    actions: jnp.ndarray
    parameters: jnp.ndarray
    log_probs: jnp.ndarray


@dataclass(frozen=True)
class EvaluationBatch:
    actions: jnp.ndarray
    parameters: jnp.ndarray
    log_probs: jnp.ndarray
    rewards: jnp.ndarray
    metrics: dict[str, jnp.ndarray]


class SimulatorBackend(Protocol):
    name: str

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        ...


def progress(message: str) -> None:
    if not QUIET:
        print(f"[ppo_detuning_sim] {message}", flush=True)


def physics_informed_seed() -> SearchSeedParams:
    return SearchSeedParams(epsilon_d=4.0 + 0.0j, g2=1.0 + 0.0j)


def unpack_controls(vector: jnp.ndarray) -> tuple[complex, complex]:
    vector = jnp.asarray(vector, dtype=jnp.float32)
    g2 = complex(float(vector[0]), float(vector[1]))
    epsilon_d = complex(float(vector[2]), float(vector[3]))
    return g2, epsilon_d


def scaled_distance_from_seed(
    parameter_batch: jnp.ndarray,
    seed_vector: jnp.ndarray,
    bounds: ParameterBounds,
) -> jnp.ndarray:
    scaled = (parameter_batch - seed_vector) / jnp.maximum(bounds.span, 1e-6)
    return jnp.linalg.norm(scaled, axis=-1)


def monoexp_model(params: np.ndarray, time: np.ndarray) -> np.ndarray:
    amplitude, tau = params
    return amplitude * np.exp(-time / tau)


def decay_envelope(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    envelope = np.clip(np.abs(values), 1e-6, 1.0)
    return np.minimum.accumulate(envelope)


def robust_exp_fit(time: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray | float]:
    time = np.asarray(time, dtype=float)
    values = decay_envelope(values)

    amplitude0 = max(values[0], 1e-4)
    tau0 = max(float(time[-1] - time[0]) / 3.0, 1e-3)

    def residuals(params: np.ndarray) -> np.ndarray:
        return monoexp_model(params, time) - values

    result = least_squares(
        residuals,
        x0=np.array([amplitude0, tau0], dtype=float),
        bounds=([0.0, 1e-6], [2.0, np.inf]),
        loss="soft_l1",
        f_scale=0.05,
    )
    fit_curve = monoexp_model(result.x, time)
    rmse = float(np.sqrt(np.mean((fit_curve - values) ** 2)))
    return {
        "params": result.x,
        "tau": float(result.x[1]),
        "fit_curve": fit_curve,
        "rmse": rmse,
    }


def reference_cat_amplitude(
    g2: complex,
    epsilon_d: complex,
    *,
    kappa_a: float,
    kappa_b: float,
) -> tuple[complex, float]:
    kappa_2 = max(4.0 * abs(g2) ** 2 / kappa_b, 1e-6)
    eps_2 = 2.0 * g2 * epsilon_d / kappa_b
    alpha_sq = (2.0 * eps_2 / kappa_2) - (kappa_a / (2.0 * kappa_2))
    alpha = complex(jnp.sqrt(jnp.asarray(alpha_sq, dtype=jnp.complex64)))
    if abs(alpha) < 0.5:
        alpha = 0.5 + 0.0j
    return alpha, kappa_2


@lru_cache(maxsize=None)
def _sqrt_factorials(dim: int) -> jnp.ndarray:
    n = jnp.arange(dim, dtype=jnp.float32)
    return jnp.exp(0.5 * gammaln(n + 1.0))


def coherent_state_analytic(dim: int, alpha: complex) -> dq.QArray:
    n = jnp.arange(dim, dtype=jnp.float32)
    alpha_jnp = jnp.asarray(alpha, dtype=jnp.complex64)
    coeffs = jnp.exp(-0.5 * jnp.abs(alpha_jnp) ** 2) * jnp.power(alpha_jnp, n) / _sqrt_factorials(dim)
    ket = dq.asqarray(np.asarray(coeffs[:, None], dtype=np.complex64))
    return dq.unit(ket)


def normalized_cat_state(dim: int, alpha: complex, parity: str) -> dq.QArray:
    plus = coherent_state_analytic(dim, alpha)
    minus = coherent_state_analytic(dim, -alpha)
    if parity == "even":
        return dq.unit(plus + minus)
    if parity == "odd":
        return dq.unit(plus - minus)
    raise ValueError(f"Unknown cat parity: {parity}")


@lru_cache(maxsize=None)
def cached_static_system(na: int, nb: int) -> dict[str, dq.QArray]:
    a_storage = dq.destroy(na)
    eye_storage = dq.eye(na)
    eye_buffer = dq.eye(nb)
    buffer_vacuum = dq.fock(nb, 0)
    a = dq.tensor(a_storage, eye_buffer)
    b = dq.tensor(eye_storage, dq.destroy(nb))

    parity_diagonal = np.array([(-1) ** n for n in range(na)], dtype=np.complex64)
    parity_storage = dq.asqarray(np.diag(parity_diagonal))
    number_storage = a_storage.dag() @ a_storage

    return {
        "a_storage": a_storage,
        "a": a,
        "b": b,
        "eye_buffer": eye_buffer,
        "buffer_vacuum": buffer_vacuum,
        "logical_x_operator": parity_storage,
        "parity_operator": parity_storage,
        "number_operator": number_storage,
    }


def build_logical_operators(
    na: int,
    alpha: complex,
    logical_x_operator: dq.QArray,
    parity_operator: dq.QArray,
    number_operator: dq.QArray,
) -> dict[str, dq.QArray]:
    plus_z = coherent_state_analytic(na, alpha)
    minus_z = coherent_state_analytic(na, -alpha)
    plus_x = normalized_cat_state(na, alpha, "even")
    minus_x = normalized_cat_state(na, alpha, "odd")
    z_operator = plus_z @ plus_z.dag() - minus_z @ minus_z.dag()
    code_projector = plus_x @ plus_x.dag() + minus_x @ minus_x.dag()

    return {
        "plus_z": plus_z,
        "minus_z": minus_z,
        "plus_x": plus_x,
        "minus_x": minus_x,
        "logical_x_operator": logical_x_operator,
        "z_operator": z_operator,
        "parity_operator": parity_operator,
        "number_operator": number_operator,
        "code_projector": code_projector,
    }


def summarize_short_horizon(
    x_signal: np.ndarray,
    z_signal: np.ndarray,
    parity_signal: np.ndarray,
    code_population: np.ndarray,
    nbar_signal: np.ndarray,
    config: SimulationConfig,
) -> dict[str, float]:
    short_count = max(4, int(round(len(x_signal) * config.short_window_fraction)))
    mean_x = float(np.mean(x_signal[:short_count]))
    mean_parity = float(np.mean(np.abs(parity_signal[:short_count])))
    cat_coherence = float(np.mean(code_population[:short_count]))
    nbar = float(np.mean(nbar_signal[:short_count]))

    nonphysical = 0.0
    nonphysical += float(np.maximum(np.max(np.abs(x_signal)) - 1.0005, 0.0) ** 2)
    nonphysical += float(np.maximum(np.max(np.abs(z_signal)) - 1.0005, 0.0) ** 2)
    nonphysical += float(np.maximum(np.max(np.abs(parity_signal)) - 1.0005, 0.0) ** 2)
    subspace_loss = float(np.maximum(0.0, 0.9 - np.min(code_population)) ** 2)

    return {
        "mean_X": mean_x,
        "mean_parity": mean_parity,
        "cat_coherence": cat_coherence,
        "nbar": nbar,
        "nonphysical_penalty": nonphysical,
        "subspace_penalty": subspace_loss,
    }


def compute_action_penalty(
    g2: complex,
    epsilon_d: complex,
    *,
    seed_params: SearchSeedParams,
    reward_config: RewardConfig,
) -> float:
    delta_eps_norm = abs(epsilon_d - seed_params.epsilon_d) / max(reward_config.delta_eps_max, 1e-6)
    delta_g2_norm = abs(g2 - seed_params.g2) / max(reward_config.delta_g2_max, 1e-6)
    return float(delta_eps_norm**2 + delta_g2_norm**2)


def pack_metrics(records: list[dict[str, float]]) -> dict[str, jnp.ndarray]:
    keys = records[0].keys()
    return {
        key: jnp.asarray(np.array([record[key] for record in records], dtype=np.float32))
        for key in keys
    }


def build_batched_reward_function(config: RewardConfig):
    @jit
    def reward_fn(metrics: dict[str, jnp.ndarray]) -> jnp.ndarray:
        safe_tx = jnp.maximum(metrics["Tx"], 1e-6)
        safe_tz = jnp.maximum(metrics["Tz"], 1e-6)
        safe_eta = jnp.maximum(metrics["eta"], 1e-6)
        target_eta = jnp.maximum(jnp.asarray(config.target_bias, dtype=jnp.float32), 1e-6)
        base_reward = 0.5 * (jnp.log(safe_tx) + jnp.log(safe_tz)) - config.lambda_eta * jnp.square(
            jnp.log(safe_eta) - jnp.log(target_eta)
        )
        eta_valid = jnp.logical_and(metrics["eta"] >= config.eta_min, metrics["eta"] <= config.eta_max)
        return jnp.where(eta_valid, base_reward, jnp.full_like(base_reward, config.invalid_eta_reward))

    return reward_fn


@dataclass
class SurrogateBackend:
    seed_params: SearchSeedParams
    bounds: ParameterBounds
    reward_config: RewardConfig
    config: SimulationConfig
    name: str = "surrogate"
    target_shift: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.15, -0.08, 0.20, 0.10], dtype=jnp.float32)
    )

    def __post_init__(self) -> None:
        self.seed_vector = self.seed_params.to_vector()
        self.target_vector = self.bounds.clip(self.seed_vector + self.target_shift)

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        parameter_batch = jnp.asarray(parameter_batch, dtype=jnp.float32)
        delta = (parameter_batch - self.target_vector) / jnp.maximum(self.bounds.span, 1e-6)
        seed_distance = scaled_distance_from_seed(parameter_batch, self.seed_vector, self.bounds)
        score = jnp.exp(-6.0 * jnp.sum(jnp.square(delta), axis=-1))

        tx = 0.12 + 0.10 * (1.0 - score) + 0.02 * seed_distance
        tz = 15.0 + 45.0 * score
        eta = tz / jnp.maximum(tx, 1e-6)
        mean_x = jnp.clip(0.75 + 0.22 * score, -1.0, 1.0)
        mean_parity = jnp.clip(0.70 + 0.28 * score, -1.0, 1.0)
        cat_coherence = jnp.clip(0.65 + 0.30 * score, 0.0, 1.0)
        nbar = 2.0 + 1.4 * seed_distance + 0.2 * (1.0 - score)
        leakage = jnp.clip(0.22 - 0.18 * score + 0.05 * seed_distance, 0.0, 1.0)
        instability = 0.05 + 0.25 * seed_distance + 0.15 * (1.0 - score)

        g2 = parameter_batch[:, 0] + 1j * parameter_batch[:, 1]
        epsilon_d = parameter_batch[:, 2] + 1j * parameter_batch[:, 3]
        kappa_2 = 4.0 * jnp.abs(g2) ** 2 / self.config.kappa_b
        alpha_abs = jnp.sqrt(
            jnp.maximum(
                jnp.abs(epsilon_d / jnp.where(jnp.abs(g2) > 1e-6, jnp.conj(g2), 1.0 + 0.0j)),
                1e-6,
            )
        )
        action_penalty = jnp.square(jnp.abs(epsilon_d - self.seed_params.epsilon_d) / self.reward_config.delta_eps_max) + jnp.square(
            jnp.abs(g2 - self.seed_params.g2) / self.reward_config.delta_g2_max
        )

        return {
            "Tx": tx,
            "Tz": tz,
            "eta": eta,
            "mean_X": mean_x,
            "mean_parity": mean_parity,
            "cat_coherence": cat_coherence,
            "nbar": nbar,
            "leakage": leakage,
            "instability_penalty": instability,
            "action_penalty": action_penalty,
            "alpha_abs": alpha_abs,
            "kappa_2": kappa_2,
        }


@dataclass
class LindbladBackend:
    """Open-system evolution with the challenge storage–buffer Hamiltonian."""

    seed_params: SearchSeedParams
    bounds: ParameterBounds
    reward_config: RewardConfig
    config: SimulationConfig
    verbose: bool = True
    name: str = "lindblad"

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        parameter_batch = jnp.asarray(parameter_batch, dtype=jnp.float32)
        if self.verbose:
            progress(f"Lindblad backend evaluating batch of {parameter_batch.shape[0]} candidates.")
        records = []
        for vector in np.asarray(parameter_batch, dtype=float):
            metrics = self._evaluate_single(jnp.asarray(vector, dtype=jnp.float32))
            records.append(metrics)
        return pack_metrics(records)

    def _run_single(
        self,
        initial_state: str,
        *,
        g2: complex,
        epsilon_d: complex,
        tfinal: float,
        nsave: int,
    ) -> dict[str, object]:
        static = cached_static_system(self.config.na, self.config.nb)
        a = static["a"]
        b = static["b"]

        alpha_estimate, kappa_2 = reference_cat_amplitude(
            g2,
            epsilon_d,
            kappa_a=self.config.kappa_a,
            kappa_b=self.config.kappa_b,
        )
        operators = build_logical_operators(
            self.config.na,
            alpha_estimate,
            static["logical_x_operator"],
            static["parity_operator"],
            static["number_operator"],
        )

        # Challenge notebook drive: two-photon exchange + coherent buffer drive.
        storage_number = a.dag() @ a
        delta = jnp.asarray(self.config.storage_detuning, dtype=jnp.float32)
        hamiltonian = (
            jnp.conj(g2) * a @ a @ b.dag()
            + g2 * a.dag() @ a.dag() @ b
            - epsilon_d * b.dag()
            - jnp.conj(epsilon_d) * b
            + delta * storage_number
        )
        losses = [
            jnp.sqrt(self.config.kappa_b) * b,
            jnp.sqrt(self.config.kappa_a) * a,
        ]
        tsave = jnp.linspace(0.0, tfinal, nsave)
        state_map = {
            "+z": operators["plus_z"],
            "-z": operators["minus_z"],
            "+x": operators["plus_x"],
            "-x": operators["minus_x"],
        }
        psi0 = dq.tensor(state_map[initial_state], static["buffer_vacuum"])
        exp_ops = [
            dq.tensor(operators["logical_x_operator"], static["eye_buffer"]),
            dq.tensor(operators["z_operator"], static["eye_buffer"]),
            dq.tensor(operators["parity_operator"], static["eye_buffer"]),
            dq.tensor(operators["number_operator"], static["eye_buffer"]),
            dq.tensor(operators["code_projector"], static["eye_buffer"]),
        ]

        result = dq.mesolve(
            hamiltonian,
            losses,
            psi0,
            tsave,
            options=dq.Options(progress_meter=False),
            exp_ops=exp_ops,
        )
        return {
            "time": np.asarray(result.tsave, dtype=float),
            "logical_x": np.asarray(result.expects[0].real, dtype=float),
            "logical_z": np.asarray(result.expects[1].real, dtype=float),
            "parity": np.asarray(result.expects[2].real, dtype=float),
            "nbar": np.asarray(result.expects[3].real, dtype=float),
            "codespace_population": np.asarray(result.expects[4].real, dtype=float),
            "alpha_abs": float(abs(alpha_estimate)),
            "kappa_2": float(kappa_2),
        }

    def _evaluate_single(self, parameter_vector: jnp.ndarray) -> dict[str, float]:
        parameter_vector = self.bounds.clip(parameter_vector)
        g2, epsilon_d = unpack_controls(parameter_vector)

        try:
            z_run = self._run_single(
                "+z",
                g2=g2,
                epsilon_d=epsilon_d,
                tfinal=self.config.eval_z_tfinal,
                nsave=self.config.eval_nsave,
            )
            x_run = self._run_single(
                "+x",
                g2=g2,
                epsilon_d=epsilon_d,
                tfinal=self.config.eval_x_tfinal,
                nsave=self.config.eval_nsave,
            )

            fit_x = robust_exp_fit(x_run["time"], x_run["logical_x"])
            fit_z = robust_exp_fit(z_run["time"], z_run["logical_z"])
            tx = max(float(fit_x["tau"]), 1e-6)
            tz = max(float(fit_z["tau"]), 1e-6)
            eta = tz / tx

            x_summary = summarize_short_horizon(
                x_run["logical_x"],
                x_run["logical_z"],
                x_run["parity"],
                x_run["codespace_population"],
                x_run["nbar"],
                self.config,
            )
            z_summary = summarize_short_horizon(
                z_run["logical_x"],
                z_run["logical_z"],
                z_run["parity"],
                z_run["codespace_population"],
                z_run["nbar"],
                self.config,
            )

            mean_x = float(x_summary["mean_X"])
            mean_parity = float(0.5 * (x_summary["mean_parity"] + z_summary["mean_parity"]))
            cat_coherence = float(0.5 * (x_summary["cat_coherence"] + z_summary["cat_coherence"]))
            nbar = float(0.5 * (x_summary["nbar"] + z_summary["nbar"]))
            leakage = float(max(0.0, 1.0 - cat_coherence))
            instability_penalty = float(
                fit_x["rmse"]
                + fit_z["rmse"]
                + x_summary["nonphysical_penalty"]
                + z_summary["nonphysical_penalty"]
                + x_summary["subspace_penalty"]
                + z_summary["subspace_penalty"]
            )
            action_penalty = compute_action_penalty(
                g2,
                epsilon_d,
                seed_params=self.seed_params,
                reward_config=self.reward_config,
            )
            eta_violation = float(max(self.reward_config.eta_min - eta, 0.0))
            invalid_eta = eta < self.reward_config.eta_min or eta > self.reward_config.eta_max

            metrics = {
                "Tx": tx,
                "Tz": tz,
                "eta": eta,
                "mean_X": mean_x,
                "mean_parity": mean_parity,
                "cat_coherence": cat_coherence,
                "nbar": nbar,
                "leakage": leakage,
                "instability_penalty": instability_penalty,
                "action_penalty": action_penalty,
                "alpha_abs": float(z_run["alpha_abs"]),
                "kappa_2": float(z_run["kappa_2"]),
            }

            if invalid_eta:
                metrics["instability_penalty"] = float(metrics["instability_penalty"] + 10.0 + eta_violation)

            return metrics
        except Exception:
            return {
                "Tx": 1e-6,
                "Tz": 1e-6,
                "eta": 1.0,
                "mean_X": 0.0,
                "mean_parity": 0.0,
                "cat_coherence": 0.0,
                "nbar": 10.0,
                "leakage": 1.0,
                "instability_penalty": 25.0,
                "action_penalty": compute_action_penalty(
                    g2,
                    epsilon_d,
                    seed_params=self.seed_params,
                    reward_config=self.reward_config,
                ),
                "alpha_abs": 0.0,
                "kappa_2": 0.0,
            }


@jit
def gaussian_log_prob(actions: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    variance = jnp.square(std)
    centered = actions - mean
    return -0.5 * jnp.sum(jnp.square(centered) / variance + jnp.log(2.0 * jnp.pi * variance), axis=-1)


@jit
def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


@jit
def normalize_advantages(rewards: jnp.ndarray) -> jnp.ndarray:
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    return (rewards - rewards.mean()) / (rewards.std() + 1e-6)


def ppo_loss(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
) -> jnp.ndarray:
    std = jnp.exp(log_std)
    new_log_probs = gaussian_log_prob(actions, mean, std)
    ratios = jnp.exp(new_log_probs - old_log_probs)
    unclipped = ratios * advantages
    clipped = jnp.clip(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    surrogate = jnp.minimum(unclipped, clipped)
    return -(jnp.mean(surrogate) + entropy_coef * gaussian_entropy(log_std))


class RLParameterRefiner:
    def __init__(
        self,
        *,
        seed_params: SearchSeedParams,
        backend: SimulatorBackend,
        reward_fn,
        parameter_bounds: ParameterBounds,
        config: PPOConfig,
        simulations_per_step: int,
    ) -> None:
        self.seed_params = seed_params
        self.seed_vector = seed_params.to_vector()
        self.backend = backend
        self.reward_fn = reward_fn
        self.parameter_bounds = parameter_bounds
        self.config = config
        self.simulations_per_step = max(1, min(simulations_per_step, config.batch_size))

        self.mean = self.seed_vector.astype(jnp.float32)
        self.log_std = jnp.log(jnp.full_like(self.seed_vector, config.initial_std))
        self.rng = jax.random.PRNGKey(config.seed)
        self.best_record: dict[str, object] | None = None

    def _clip_policy_state(self) -> None:
        self.mean = jnp.clip(self.mean, self.parameter_bounds.lower, self.parameter_bounds.upper)
        self.log_std = jnp.clip(
            self.log_std,
            jnp.log(jnp.full_like(self.log_std, self.config.min_std)),
            jnp.log(jnp.full_like(self.log_std, self.config.max_std)),
        )

    def sample_candidates(self, batch_size: int) -> CandidateBatch:
        self.rng, subkey = jax.random.split(self.rng)
        noise = jax.random.normal(subkey, shape=(batch_size, self.mean.shape[0]))
        std = jnp.exp(self.log_std)
        actions = self.mean + std * noise
        parameters = self.parameter_bounds.clip(actions)
        log_probs = gaussian_log_prob(actions, self.mean, std)
        return CandidateBatch(actions=actions, parameters=parameters, log_probs=log_probs)

    def _subsample_candidates(self, candidates: CandidateBatch, n: int) -> CandidateBatch:
        if n >= candidates.actions.shape[0]:
            return candidates
        self.rng, subkey = jax.random.split(self.rng)
        perm = jax.random.permutation(subkey, candidates.actions.shape[0])
        idx = perm[:n]
        return CandidateBatch(
            actions=candidates.actions[idx],
            parameters=candidates.parameters[idx],
            log_probs=candidates.log_probs[idx],
        )

    def evaluate_candidates(self, candidates: CandidateBatch) -> EvaluationBatch:
        metrics = self.backend.evaluate_batch(candidates.parameters)
        rewards = self.reward_fn(metrics)
        evaluation = EvaluationBatch(
            actions=candidates.actions,
            parameters=candidates.parameters,
            log_probs=candidates.log_probs,
            rewards=jnp.asarray(rewards, dtype=jnp.float32),
            metrics={key: jnp.asarray(value, dtype=jnp.float32) for key, value in metrics.items()},
        )
        self._update_best(evaluation)
        return evaluation

    def _update_best(self, evaluation: EvaluationBatch) -> None:
        rewards_np = np.asarray(evaluation.rewards, dtype=float)
        best_index = int(np.argmax(rewards_np))
        best_reward = float(rewards_np[best_index])
        if self.best_record is not None and best_reward <= float(self.best_record["reward"]):
            return

        best_vector = np.asarray(evaluation.parameters[best_index], dtype=float)
        best_params = SearchSeedParams.from_vector(jnp.asarray(best_vector, dtype=jnp.float32))
        self.best_record = {
            "reward": best_reward,
            "parameters_vector": best_vector,
            "g2": best_params.g2,
            "epsilon_d": best_params.epsilon_d,
            "metrics": {key: float(np.asarray(value)[best_index]) for key, value in evaluation.metrics.items()},
        }

    def _policy_update(self, evaluation: EvaluationBatch) -> None:
        advantages = normalize_advantages(evaluation.rewards)
        mean = self.mean
        log_std = self.log_std

        for _ in range(self.config.ppo_epochs):
            _, grads = value_and_grad(ppo_loss, argnums=(0, 1))(
                mean,
                log_std,
                evaluation.actions,
                evaluation.log_probs,
                advantages,
                self.config.clip_ratio,
                self.config.entropy_coef,
            )
            mean = mean - self.config.learning_rate * grads[0]
            log_std = log_std - self.config.learning_rate * grads[1]
            self.mean = mean
            self.log_std = log_std
            self._clip_policy_state()
            mean = self.mean
            log_std = self.log_std

    def train_step(self, epoch_index: int) -> None:
        candidates = self.sample_candidates(self.config.batch_size)
        to_eval = self._subsample_candidates(candidates, self.simulations_per_step)
        evaluation = self.evaluate_candidates(to_eval)
        self._policy_update(evaluation)
        if epoch_index % self.config.log_every == 0:
            progress(
                f"epoch {epoch_index:04d} | simmed={to_eval.actions.shape[0]}/{self.config.batch_size} | "
                f"mean_reward={float(jnp.mean(evaluation.rewards)):.4f} | "
                f"best_reward={float(self.best_record['reward']):.4f} | best_eta={float(self.best_record['metrics']['eta']):.2f}"
            )

    def get_best_parameters(self) -> dict[str, object]:
        if self.best_record is None:
            raise RuntimeError("No candidate has been evaluated yet.")
        return self.best_record


def _clip_delta_interval(delta_min: float, delta_max: float) -> tuple[float, float]:
    lo = float(np.clip(delta_min, _DELTA_MIN, _DELTA_MAX))
    hi = float(np.clip(delta_max, _DELTA_MIN, _DELTA_MAX))
    if hi <= lo:
        hi = min(_DELTA_MAX, lo + 1e-6)
    return lo, hi


def _eval_epsilon_vectors(
    *,
    cfg: SimulationConfig,
    reward_config: RewardConfig,
    bounds: ParameterBounds,
    seed_params: SearchSeedParams,
    reward_fn,
    vectors: list[list[float]],
    sweep_chunk: int,
) -> tuple[float, float]:
    backend = LindbladBackend(
        seed_params=seed_params,
        bounds=bounds,
        reward_config=reward_config,
        config=cfg,
        verbose=False,
    )
    batch_all = jnp.asarray(vectors, dtype=jnp.float32)
    rewards_list: list[jnp.ndarray] = []
    n = int(batch_all.shape[0])
    chunk = max(1, sweep_chunk)
    for start in range(0, n, chunk):
        sub = batch_all[start : start + chunk]
        metrics = backend.evaluate_batch(sub)
        rewards_list.append(reward_fn(metrics))
    rewards = jnp.concatenate(rewards_list)
    j = int(jnp.argmax(rewards))
    return float(vectors[j][2]), float(vectors[j][3])


def _make_eps_grid(
    g2_re: float,
    g2_im: float,
    re_lo: float,
    re_hi: float,
    im_lo: float,
    im_hi: float,
    n_re: int,
    n_im: int,
) -> list[list[float]]:
    n_re = max(2, n_re)
    n_im = max(2, n_im)
    return [
        [g2_re, g2_im, float(er), float(ei)]
        for er in np.linspace(re_lo, re_hi, n_re)
        for ei in np.linspace(im_lo, im_hi, n_im)
    ]


def _best_eps_two_stage(
    *,
    cfg: SimulationConfig,
    reward_config: RewardConfig,
    bounds: ParameterBounds,
    seed_params: SearchSeedParams,
    reward_fn,
    coarse_n: int,
    refine_n: int,
    refine_expand: float,
    sweep_chunk: int,
) -> tuple[float, float]:
    seed_v = np.asarray(seed_params.to_vector(), dtype=np.float32)
    g2_re, g2_im = float(seed_v[0]), float(seed_v[1])
    bl = np.asarray(bounds.lower, dtype=float)
    bu = np.asarray(bounds.upper, dtype=float)
    re_lo, re_hi = float(bl[2]), float(bu[2])
    im_lo, im_hi = float(bl[3]), float(bu[3])

    v_coarse = _make_eps_grid(g2_re, g2_im, re_lo, re_hi, im_lo, im_hi, coarse_n, coarse_n)
    br, bi = _eval_epsilon_vectors(
        cfg=cfg,
        reward_config=reward_config,
        bounds=bounds,
        seed_params=seed_params,
        reward_fn=reward_fn,
        vectors=v_coarse,
        sweep_chunk=sweep_chunk,
    )

    cn = max(2, coarse_n)
    span_re = (re_hi - re_lo) / (cn - 1)
    span_im = (im_hi - im_lo) / (cn - 1)
    half_re = max(refine_expand * span_re * 0.5, 1e-6 * max(abs(re_hi), 1.0))
    half_im = max(refine_expand * span_im * 0.5, 1e-6 * max(abs(im_hi), 1.0))
    r0 = max(re_lo, br - half_re)
    r1 = min(re_hi, br + half_re)
    i0 = max(im_lo, bi - half_im)
    i1 = min(im_hi, bi + half_im)
    if r1 <= r0:
        r0, r1 = re_lo, re_hi
    if i1 <= i0:
        i0, i1 = im_lo, im_hi

    v_ref = _make_eps_grid(g2_re, g2_im, r0, r1, i0, i1, refine_n, refine_n)
    return _eval_epsilon_vectors(
        cfg=cfg,
        reward_config=reward_config,
        bounds=bounds,
        seed_params=seed_params,
        reward_fn=reward_fn,
        vectors=v_ref,
        sweep_chunk=sweep_chunk,
    )


def _detuning_delta_worker(spec: dict[str, Any]) -> tuple[float, float, float]:
    """Process-pool entry: one ``Delta`` value, returns ``(delta, best_re_eps, best_im_eps)``."""
    delta = float(spec["delta"])
    cfg = SimulationConfig(**{**spec["sim_template"], "storage_detuning": delta})
    rc = RewardConfig(**spec["reward_template"])
    bounds = ParameterBounds(
        lower=jnp.asarray(spec["bounds_lower"], dtype=jnp.float32),
        upper=jnp.asarray(spec["bounds_upper"], dtype=jnp.float32),
    )
    sp = SearchSeedParams(
        g2=complex(spec["seed_g2"][0], spec["seed_g2"][1]),
        epsilon_d=complex(spec["seed_eps"][0], spec["seed_eps"][1]),
    )
    reward_fn = build_batched_reward_function(rc)
    br, bi = _best_eps_two_stage(
        cfg=cfg,
        reward_config=rc,
        bounds=bounds,
        seed_params=sp,
        reward_fn=reward_fn,
        coarse_n=int(spec["coarse_n"]),
        refine_n=int(spec["refine_n"]),
        refine_expand=float(spec["refine_expand"]),
        sweep_chunk=int(spec["sweep_chunk"]),
    )
    return delta, br, bi


def run_detuning_sweep_plot(
    *,
    sim_config: SimulationConfig,
    reward_config: RewardConfig,
    bounds: ParameterBounds,
    seed_params: SearchSeedParams,
    delta_min: float,
    delta_max: float,
    delta_steps: int,
    coarse_n: int,
    refine_n: int,
    refine_expand: float,
    output_path: Path,
    sweep_chunk: int,
    sweep_workers: int,
) -> None:
    """Sweep ``Delta`` in ``[0, 2]``; per ``Delta``, coarse + refine ``epsilon_d`` grid; parallel over ``Delta``."""
    d_lo, d_hi = _clip_delta_interval(delta_min, delta_max)
    deltas = np.linspace(d_lo, d_hi, max(2, delta_steps))
    progress(
        f"detuning sweep: Delta in [{d_lo:.4g}, {d_hi:.4g}] ({len(deltas)} points), "
        f"eps grids coarse={coarse_n} refine={refine_n}, workers={max(1, sweep_workers)}"
    )

    sim_template = asdict(sim_config)
    reward_template = asdict(reward_config)
    bounds_lower = np.asarray(bounds.lower, dtype=np.float32).tolist()
    bounds_upper = np.asarray(bounds.upper, dtype=np.float32).tolist()
    seed_g2 = (float(np.real(seed_params.g2)), float(np.imag(seed_params.g2)))
    seed_eps = (float(np.real(seed_params.epsilon_d)), float(np.imag(seed_params.epsilon_d)))

    specs: list[dict[str, Any]] = []
    for d in deltas:
        specs.append(
            {
                "delta": float(d),
                "sim_template": sim_template,
                "reward_template": reward_template,
                "bounds_lower": bounds_lower,
                "bounds_upper": bounds_upper,
                "seed_g2": seed_g2,
                "seed_eps": seed_eps,
                "coarse_n": coarse_n,
                "refine_n": refine_n,
                "refine_expand": refine_expand,
                "sweep_chunk": max(1, sweep_chunk),
            }
        )

    workers = max(1, sweep_workers)
    results: list[tuple[float, float, float]] = []
    if workers == 1:
        reward_fn = build_batched_reward_function(reward_config)
        for i, spec in enumerate(specs):
            progress(f"detuning sweep {i + 1}/{len(specs)} | delta={spec['delta']:.4f}")
            cfg = SimulationConfig(**{**spec["sim_template"], "storage_detuning": spec["delta"]})
            br, bi = _best_eps_two_stage(
                cfg=cfg,
                reward_config=reward_config,
                bounds=bounds,
                seed_params=seed_params,
                reward_fn=reward_fn,
                coarse_n=coarse_n,
                refine_n=refine_n,
                refine_expand=refine_expand,
                sweep_chunk=max(1, sweep_chunk),
            )
            results.append((spec["delta"], br, bi))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_detuning_delta_worker, spec) for spec in specs]
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda t: t[0])

    deltas_plot = np.array([r[0] for r in results], dtype=float)
    best_re = [r[1] for r in results]
    best_im = [r[2] for r in results]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(deltas_plot, best_re, color="tab:blue", linewidth=2.0, label=r"$\mathrm{Re}(\epsilon_d)$")
    ax.plot(deltas_plot, best_im, color="tab:orange", linewidth=2.0, label=r"$\mathrm{Im}(\epsilon_d)$")
    ax.set_xlim(_DELTA_MIN, _DELTA_MAX)
    ax.set_xlabel(r"Storage detuning $\Delta$ ($a^\dagger a$ coefficient)")
    ax.set_ylabel(r"Buffer drive $\epsilon_d$ (real / imag parts)")
    ax.set_title(r"Best $\epsilon_d$ vs $\Delta$ (coarse + refine grid, fixed seed $g_2$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    progress(f"wrote {output_path.resolve()}")


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="CPU-only PPO on cat-qubit controls (Lindblad or surrogate).")
    p.add_argument("--backend", choices=["lindblad", "surrogate"], default="lindblad")
    p.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    p.add_argument("--batch-size", type=int, default=PPOConfig.batch_size)
    p.add_argument(
        "--sim-budget",
        type=int,
        default=None,
        metavar="N",
        help="Simulate at most N candidates per step after sampling --batch-size. "
        "Default: 1 for lindblad, full batch for surrogate. Use 0 for full batch on any backend.",
    )
    p.add_argument("--ppo-epochs", type=int, default=PPOConfig.ppo_epochs)
    p.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    p.add_argument("--seed", type=int, default=PPOConfig.seed)
    p.add_argument("--log-every", type=int, default=PPOConfig.log_every)
    p.add_argument("--lambda-eta", type=float, default=RewardConfig.lambda_eta)
    p.add_argument("--target-bias", type=float, default=RewardConfig.target_bias)
    p.add_argument("--na", type=int, default=SimulationConfig.na)
    p.add_argument("--nb", type=int, default=SimulationConfig.nb)
    p.add_argument("--kappa-a", type=float, default=SimulationConfig.kappa_a)
    p.add_argument("--kappa-b", type=float, default=SimulationConfig.kappa_b)
    p.add_argument("--eval-x-tfinal", type=float, default=SimulationConfig.eval_x_tfinal)
    p.add_argument("--eval-z-tfinal", type=float, default=SimulationConfig.eval_z_tfinal)
    p.add_argument("--eval-nsave", type=int, default=SimulationConfig.eval_nsave)
    p.add_argument(
        "--storage-detuning",
        type=float,
        default=0.0,
        metavar="DELTA",
        help=f"Constant storage detuning in H + Delta a†a during PPO; clamped to [{_DELTA_MIN}, {_DELTA_MAX}].",
    )
    p.add_argument(
        "--detuning-sweep-plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, skip PPO: sweep Δ and save ε_d plot (requires --backend lindblad).",
    )
    p.add_argument(
        "--delta-min",
        type=float,
        default=_DELTA_MIN,
        help=f"Sweep Delta lower bound (clamped to [{_DELTA_MIN}, {_DELTA_MAX}]).",
    )
    p.add_argument(
        "--delta-max",
        type=float,
        default=_DELTA_MAX,
        help=f"Sweep Delta upper bound (clamped to [{_DELTA_MIN}, {_DELTA_MAX}]).",
    )
    p.add_argument("--delta-steps", type=int, default=17, help="Number of Delta values along the sweep.")
    p.add_argument(
        "--sweep-coarse",
        type=int,
        default=7,
        metavar="N",
        help="Coarse epsilon_d grid size per axis (NxN mesolve batch per Delta).",
    )
    p.add_argument(
        "--sweep-refine",
        type=int,
        default=9,
        metavar="N",
        help="Refine-grid size per axis around the coarse best (NxN).",
    )
    p.add_argument(
        "--sweep-refine-expand",
        type=float,
        default=2.0,
        help="Refine window half-width ~ (expand) x coarse cell size.",
    )
    p.add_argument(
        "--sweep-workers",
        type=int,
        default=min(8, cpu_count() or 4),
        metavar="W",
        help="Parallel processes for Delta sweep (1 = serial).",
    )
    p.add_argument(
        "--sweep-chunk",
        type=int,
        default=64,
        metavar="K",
        help="Chunk size when calling evaluate_batch inside each Delta job.",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    args.ppo_cfg = PPOConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=max(1, args.log_every),
    )
    args.reward_cfg = RewardConfig(lambda_eta=args.lambda_eta, target_bias=args.target_bias)
    args.sim_cfg = SimulationConfig(
        na=args.na,
        nb=args.nb,
        kappa_a=args.kappa_a,
        kappa_b=args.kappa_b,
        eval_x_tfinal=args.eval_x_tfinal,
        eval_z_tfinal=args.eval_z_tfinal,
        eval_nsave=args.eval_nsave,
        storage_detuning=float(np.clip(args.storage_detuning, _DELTA_MIN, _DELTA_MAX)),
    )
    return args


def _simulations_per_step(sim_budget: int | None, backend_name: str, batch_size: int) -> int:
    if sim_budget is not None:
        if sim_budget <= 0:
            return batch_size
        return min(sim_budget, batch_size)
    return 1 if backend_name == "lindblad" else batch_size


def main() -> None:
    global QUIET
    args = parse_args()
    QUIET = args.quiet

    progress(f"JAX devices (expect CPU): {jax.devices()}")

    bounds = ParameterBounds.default()
    seed_params = physics_informed_seed()

    if args.detuning_sweep_plot is not None:
        if args.backend != "lindblad":
            raise SystemExit("--detuning-sweep-plot requires --backend lindblad")
        run_detuning_sweep_plot(
            sim_config=args.sim_cfg,
            reward_config=args.reward_cfg,
            bounds=bounds,
            seed_params=seed_params,
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            delta_steps=args.delta_steps,
            coarse_n=max(2, args.sweep_coarse),
            refine_n=max(2, args.sweep_refine),
            refine_expand=max(0.5, args.sweep_refine_expand),
            output_path=args.detuning_sweep_plot,
            sweep_chunk=max(1, args.sweep_chunk),
            sweep_workers=max(1, args.sweep_workers),
        )
        return

    reward_fn = build_batched_reward_function(args.reward_cfg)

    if args.backend == "surrogate":
        backend: SimulatorBackend = SurrogateBackend(
            seed_params=seed_params,
            bounds=bounds,
            reward_config=args.reward_cfg,
            config=args.sim_cfg,
        )
    else:
        backend = LindbladBackend(
            seed_params=seed_params,
            bounds=bounds,
            reward_config=args.reward_cfg,
            config=args.sim_cfg,
            verbose=not args.quiet,
        )

    n_sim = _simulations_per_step(args.sim_budget, args.backend, args.ppo_cfg.batch_size)
    progress(f"backend={backend.name} | simulations_per_step={n_sim} (batch_size={args.ppo_cfg.batch_size})")
    refiner = RLParameterRefiner(
        seed_params=seed_params,
        backend=backend,
        reward_fn=reward_fn,
        parameter_bounds=bounds,
        config=args.ppo_cfg,
        simulations_per_step=n_sim,
    )

    for epoch in range(args.ppo_cfg.epochs):
        refiner.train_step(epoch)

    best = refiner.get_best_parameters()
    progress(f"best g2={best['g2']} epsilon_d={best['epsilon_d']} metrics={best['metrics']}")


if __name__ == "__main__":
    freeze_support()
    main()
