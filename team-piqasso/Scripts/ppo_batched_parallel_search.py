from __future__ import annotations

import argparse
import os
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
from jax.scipy.special import gammaln
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

QUIET = False


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
    def from_vector(cls, vector: jnp.ndarray) -> "SearchSeedParams":
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
    def default(cls) -> "ParameterBounds":
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


@dataclass(frozen=True)
class RewardConfig:
    target_bias: float = 320.0
    eta_min: float = 100.0
    eta_max: float = 750.0
    lambda_eta: float = 2.0
    invalid_eta_reward: float = -1000.0
    w_x: float = 0.40
    w_p: float = 0.25
    w_c: float = 0.50
    w_n: float = 0.12
    w_a: float = 0.25
    w_s: float = 0.70
    delta_eps_max: float = 2.5
    delta_g2_max: float = 1.0


@dataclass(frozen=True)
class PPOConfig:
    batch_size: int = 12
    replay_capacity: int = 256
    replay_sample_size: int = 24
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    entropy_coef: float = 0.02
    learning_rate: float = 0.03
    initial_std: float = 0.18
    min_std: float = 0.02
    max_std: float = 0.55
    seed: int = 13
    epochs: int = 1000
    snapshot_every: int = 100
    log_every: int = 1
    output_dir: Path = Path("team-piqasso") / "outputs" / "ppo_batched_parallel_search"


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


@dataclass(frozen=True)
class DecayTrace:
    x_time: np.ndarray
    x_signal: np.ndarray
    x_fit: np.ndarray
    x_codespace_population: np.ndarray
    x_parity: np.ndarray
    z_time: np.ndarray
    z_signal: np.ndarray
    z_fit: np.ndarray
    z_codespace_population: np.ndarray
    z_parity: np.ndarray
    metrics: dict[str, float]


class SimulatorBackend(Protocol):
    name: str

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        ...

    def trace_candidate(self, parameter_vector: jnp.ndarray) -> DecayTrace:
        ...


def progress(message: str) -> None:
    if not QUIET:
        print(f"[ppo_batched_parallel_search] {message}", flush=True)


def physics_informed_seed() -> SearchSeedParams:
    # Same seed values used repeatedly in the challenge notebooks.
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


@dataclass
class SurrogateBackend:
    seed_params: SearchSeedParams
    bounds: ParameterBounds
    reward_config: RewardConfig
    config: SimulationConfig
    verbose: bool = True
    log_candidates: bool = False
    name: str = "surrogate"
    target_shift: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.15, -0.08, 0.20, 0.10], dtype=jnp.float32)
    )

    def __post_init__(self) -> None:
        self.seed_vector = self.seed_params.to_vector()
        self.target_vector = self.bounds.clip(self.seed_vector + self.target_shift)

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        parameter_batch = jnp.asarray(parameter_batch, dtype=jnp.float32)
        if self.verbose:
            progress(f"Surrogate backend evaluating batch of {parameter_batch.shape[0]} candidates.")
        if self.log_candidates:
            for idx, vector in enumerate(np.asarray(parameter_batch, dtype=float)):
                g2, epsilon_d = unpack_controls(jnp.asarray(vector, dtype=jnp.float32))
                progress(
                    f"  candidate {idx:02d}: g2={g2.real:+.4f}{g2.imag:+.4f}j, "
                    f"epsilon_d={epsilon_d.real:+.4f}{epsilon_d.imag:+.4f}j"
                )
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
        alpha_abs = jnp.sqrt(jnp.maximum(jnp.abs(epsilon_d / jnp.where(jnp.abs(g2) > 1e-6, jnp.conj(g2), 1.0 + 0.0j)), 1e-6))
        action_penalty = (
            jnp.square(jnp.abs(epsilon_d - self.seed_params.epsilon_d) / self.reward_config.delta_eps_max)
            + jnp.square(jnp.abs(g2 - self.seed_params.g2) / self.reward_config.delta_g2_max)
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

    def trace_candidate(self, parameter_vector: jnp.ndarray) -> DecayTrace:
        metrics = {
            key: float(value[0])
            for key, value in self.evaluate_batch(jnp.asarray(parameter_vector)[None, :]).items()
        }
        x_time = np.linspace(0.0, self.config.x_tfinal, self.config.nsave)
        z_time = np.linspace(0.0, self.config.z_tfinal, self.config.nsave)
        x_signal = np.exp(-x_time / metrics["Tx"])
        z_signal = np.exp(-z_time / metrics["Tz"])
        x_fit = x_signal.copy()
        z_fit = z_signal.copy()
        x_codespace = metrics["cat_coherence"] * np.ones_like(x_time)
        z_codespace = metrics["cat_coherence"] * np.ones_like(z_time)
        x_parity = metrics["mean_parity"] * np.ones_like(x_time)
        z_parity = metrics["mean_parity"] * np.ones_like(z_time)
        return DecayTrace(
            x_time=x_time,
            x_signal=x_signal,
            x_fit=x_fit,
            x_codespace_population=x_codespace,
            x_parity=x_parity,
            z_time=z_time,
            z_signal=z_signal,
            z_fit=z_fit,
            z_codespace_population=z_codespace,
            z_parity=z_parity,
            metrics=metrics,
        )


@dataclass
class LindbladBackend:
    seed_params: SearchSeedParams
    bounds: ParameterBounds
    reward_config: RewardConfig
    config: SimulationConfig
    verbose: bool = True
    log_candidates: bool = False
    name: str = "lindblad"

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> dict[str, jnp.ndarray]:
        parameter_batch = jnp.asarray(parameter_batch, dtype=jnp.float32)
        if self.verbose:
            progress(f"Lindblad backend evaluating batch of {parameter_batch.shape[0]} candidates.")
        records = []
        for idx, vector in enumerate(np.asarray(parameter_batch, dtype=float)):
            if self.log_candidates:
                g2, epsilon_d = unpack_controls(jnp.asarray(vector, dtype=jnp.float32))
                progress(
                    f"  candidate {idx:02d}: g2={g2.real:+.4f}{g2.imag:+.4f}j, "
                    f"epsilon_d={epsilon_d.real:+.4f}{epsilon_d.imag:+.4f}j"
                )
            metrics = self._evaluate_single(jnp.asarray(vector, dtype=jnp.float32), return_trace=False)
            records.append(metrics["metrics"])
        return pack_metrics(records)

    def trace_candidate(self, parameter_vector: jnp.ndarray) -> DecayTrace:
        return self._evaluate_single(jnp.asarray(parameter_vector, dtype=jnp.float32), return_trace=True)["trace"]

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

        hamiltonian = (
            jnp.conj(g2) * a @ a @ b.dag()
            + g2 * a.dag() @ a.dag() @ b
            - epsilon_d * b.dag()
            - jnp.conj(epsilon_d) * b
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

    def _evaluate_single(self, parameter_vector: jnp.ndarray, *, return_trace: bool) -> dict[str, object]:
        parameter_vector = self.bounds.clip(parameter_vector)
        g2, epsilon_d = unpack_controls(parameter_vector)

        try:
            x_tfinal = self.config.x_tfinal if return_trace else self.config.eval_x_tfinal
            z_tfinal = self.config.z_tfinal if return_trace else self.config.eval_z_tfinal
            nsave = self.config.nsave if return_trace else self.config.eval_nsave

            z_run = self._run_single("+z", g2=g2, epsilon_d=epsilon_d, tfinal=z_tfinal, nsave=nsave)
            x_run = self._run_single("+x", g2=g2, epsilon_d=epsilon_d, tfinal=x_tfinal, nsave=nsave)

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

            if not return_trace:
                return {"metrics": metrics}

            trace = DecayTrace(
                x_time=x_run["time"],
                x_signal=decay_envelope(x_run["logical_x"]),
                x_fit=np.asarray(fit_x["fit_curve"], dtype=float),
                x_codespace_population=x_run["codespace_population"],
                x_parity=x_run["parity"],
                z_time=z_run["time"],
                z_signal=decay_envelope(z_run["logical_z"]),
                z_fit=np.asarray(fit_z["fit_curve"], dtype=float),
                z_codespace_population=z_run["codespace_population"],
                z_parity=z_run["parity"],
                metrics=metrics,
            )
            return {"metrics": metrics, "trace": trace}
        except Exception as exc:  # pragma: no cover
            penalty_metrics = {
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
            if not return_trace:
                return {"metrics": penalty_metrics}
            zero_time_x = np.linspace(0.0, self.config.x_tfinal, self.config.nsave)
            zero_time_z = np.linspace(0.0, self.config.z_tfinal, self.config.nsave)
            zero_trace = DecayTrace(
                x_time=zero_time_x,
                x_signal=np.zeros_like(zero_time_x),
                x_fit=np.zeros_like(zero_time_x),
                x_codespace_population=np.zeros_like(zero_time_x),
                x_parity=np.zeros_like(zero_time_x),
                z_time=zero_time_z,
                z_signal=np.zeros_like(zero_time_z),
                z_fit=np.zeros_like(zero_time_z),
                z_codespace_population=np.zeros_like(zero_time_z),
                z_parity=np.zeros_like(zero_time_z),
                metrics={**penalty_metrics, "error": str(exc)},
            )
            return {"metrics": penalty_metrics, "trace": zero_trace}


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.storage: deque[dict[str, np.ndarray]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add_batch(self, evaluation: EvaluationBatch) -> None:
        metric_arrays = {key: np.asarray(value) for key, value in evaluation.metrics.items()}
        batch_size = len(np.asarray(evaluation.rewards))
        for idx in range(batch_size):
            self.storage.append(
                {
                    "actions": np.asarray(evaluation.actions[idx]),
                    "parameters": np.asarray(evaluation.parameters[idx]),
                    "log_probs": np.asarray(evaluation.log_probs[idx]),
                    "rewards": np.asarray(evaluation.rewards[idx]),
                    **{key: value[idx] for key, value in metric_arrays.items()},
                }
            )

    def sample(self, sample_size: int) -> EvaluationBatch | None:
        if not self.storage:
            return None
        size = min(sample_size, len(self.storage))
        indices = self.rng.choice(len(self.storage), size=size, replace=False)
        samples = [self.storage[int(idx)] for idx in indices]
        metric_keys = [key for key in samples[0] if key not in {"actions", "parameters", "log_probs", "rewards"}]
        return EvaluationBatch(
            actions=jnp.asarray(np.stack([sample["actions"] for sample in samples]), dtype=jnp.float32),
            parameters=jnp.asarray(np.stack([sample["parameters"] for sample in samples]), dtype=jnp.float32),
            log_probs=jnp.asarray(np.stack([sample["log_probs"] for sample in samples]), dtype=jnp.float32),
            rewards=jnp.asarray(np.stack([sample["rewards"] for sample in samples]), dtype=jnp.float32),
            metrics={
                key: jnp.asarray(np.stack([sample[key] for sample in samples]), dtype=jnp.float32)
                for key in metric_keys
            },
        )


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


def concatenate_evaluations(primary: EvaluationBatch, secondary: EvaluationBatch | None) -> EvaluationBatch:
    if secondary is None:
        return primary
    metric_keys = primary.metrics.keys()
    return EvaluationBatch(
        actions=jnp.concatenate([primary.actions, secondary.actions], axis=0),
        parameters=jnp.concatenate([primary.parameters, secondary.parameters], axis=0),
        log_probs=jnp.concatenate([primary.log_probs, secondary.log_probs], axis=0),
        rewards=jnp.concatenate([primary.rewards, secondary.rewards], axis=0),
        metrics={
            key: jnp.concatenate([primary.metrics[key], secondary.metrics[key]], axis=0)
            for key in metric_keys
        },
    )


class RLParameterRefiner:
    def __init__(
        self,
        *,
        seed_params: SearchSeedParams,
        backend: SimulatorBackend,
        reward_fn,
        parameter_bounds: ParameterBounds,
        config: PPOConfig,
    ) -> None:
        self.seed_params = seed_params
        self.seed_vector = seed_params.to_vector()
        self.backend = backend
        self.reward_fn = reward_fn
        self.parameter_bounds = parameter_bounds
        self.config = config

        self.mean = self.seed_vector.astype(jnp.float32)
        self.log_std = jnp.log(jnp.full_like(self.seed_vector, config.initial_std))
        self.rng = jax.random.PRNGKey(config.seed)
        self.replay_buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)
        self.history: list[dict[str, object]] = []
        self.best_record: dict[str, object] | None = None

    def _clip_policy_state(self) -> None:
        self.mean = jnp.clip(self.mean, self.parameter_bounds.lower, self.parameter_bounds.upper)
        self.log_std = jnp.clip(
            self.log_std,
            jnp.log(jnp.full_like(self.log_std, self.config.min_std)),
            jnp.log(jnp.full_like(self.log_std, self.config.max_std)),
        )

    def sample_candidates(self, batch_size: int | None = None) -> CandidateBatch:
        batch_size = batch_size or self.config.batch_size
        self.rng, subkey = jax.random.split(self.rng)
        noise = jax.random.normal(subkey, shape=(batch_size, self.mean.shape[0]))
        std = jnp.exp(self.log_std)
        actions = self.mean + std * noise
        parameters = self.parameter_bounds.clip(actions)
        log_probs = gaussian_log_prob(actions, self.mean, std)
        return CandidateBatch(actions=actions, parameters=parameters, log_probs=log_probs)

    def evaluate_candidates(self, candidates: CandidateBatch, *, store_in_replay: bool = True) -> EvaluationBatch:
        metrics = self.backend.evaluate_batch(candidates.parameters)
        rewards = self.reward_fn(metrics)
        evaluation = EvaluationBatch(
            actions=candidates.actions,
            parameters=candidates.parameters,
            log_probs=candidates.log_probs,
            rewards=jnp.asarray(rewards, dtype=jnp.float32),
            metrics={key: jnp.asarray(value, dtype=jnp.float32) for key, value in metrics.items()},
        )
        if store_in_replay:
            self.replay_buffer.add_batch(evaluation)
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

    def _policy_update(self, evaluation: EvaluationBatch) -> dict[str, float]:
        advantages = normalize_advantages(evaluation.rewards)
        mean = self.mean
        log_std = self.log_std
        loss_value = None

        for _ in range(self.config.ppo_epochs):
            loss_value, grads = value_and_grad(ppo_loss, argnums=(0, 1))(
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

        std = jnp.exp(self.log_std)
        new_log_probs = gaussian_log_prob(evaluation.actions, self.mean, std)
        return {
            "loss": float(loss_value),
            "approx_kl": float(jnp.mean(evaluation.log_probs - new_log_probs)),
            "entropy": float(gaussian_entropy(self.log_std)),
        }

    def train_step(self, epoch_index: int, batch_size: int | None = None) -> dict[str, float]:
        if epoch_index % self.config.log_every == 0:
            progress(f"Epoch {epoch_index:04d}: sampling {batch_size or self.config.batch_size} candidates.")
        candidates = self.sample_candidates(batch_size=batch_size)
        if epoch_index % self.config.log_every == 0:
            progress(f"Epoch {epoch_index:04d}: evaluating sampled candidates.")
        fresh_evaluation = self.evaluate_candidates(candidates, store_in_replay=True)
        if epoch_index % self.config.log_every == 0:
            progress(f"Epoch {epoch_index:04d}: drawing replay batch.")
        replay_evaluation = self.replay_buffer.sample(self.config.replay_sample_size)
        update_batch = concatenate_evaluations(fresh_evaluation, replay_evaluation)
        if epoch_index % self.config.log_every == 0:
            progress(f"Epoch {epoch_index:04d}: running PPO update on {len(update_batch.rewards)} samples.")
        update_stats = self._policy_update(update_batch)

        summary = {
            "epoch": epoch_index,
            "reward_mean": float(jnp.mean(fresh_evaluation.rewards)),
            "reward_max": float(jnp.max(fresh_evaluation.rewards)),
            "Tx_mean": float(jnp.mean(fresh_evaluation.metrics["Tx"])),
            "Tz_mean": float(jnp.mean(fresh_evaluation.metrics["Tz"])),
            "eta_mean": float(jnp.mean(fresh_evaluation.metrics["eta"])),
            "best_reward": float(self.best_record["reward"]),
            "best_Tx": float(self.best_record["metrics"]["Tx"]),
            "best_Tz": float(self.best_record["metrics"]["Tz"]),
            "best_eta": float(self.best_record["metrics"]["eta"]),
            "best_alpha_abs": float(self.best_record["metrics"]["alpha_abs"]),
            **update_stats,
        }

        self.history.append(
            {
                **summary,
                "mean": np.asarray(self.mean, dtype=float),
                "std": np.asarray(jnp.exp(self.log_std), dtype=float),
                "sampled_Tx": np.asarray(fresh_evaluation.metrics["Tx"], dtype=float),
                "sampled_Tz": np.asarray(fresh_evaluation.metrics["Tz"], dtype=float),
                "sampled_rewards": np.asarray(fresh_evaluation.rewards, dtype=float),
                "sampled_parameters": np.asarray(fresh_evaluation.parameters, dtype=float),
                "best_parameters_vector": np.asarray(self.best_record["parameters_vector"], dtype=float),
                "best_g2_real": float(np.real(self.best_record["g2"])),
                "best_g2_imag": float(np.imag(self.best_record["g2"])),
                "best_eps_real": float(np.real(self.best_record["epsilon_d"])),
                "best_eps_imag": float(np.imag(self.best_record["epsilon_d"])),
            }
        )
        return summary

    def get_best_parameters(self) -> dict[str, object]:
        if self.best_record is None:
            raise RuntimeError("No candidate has been evaluated yet.")
        return self.best_record


def save_training_plots(refiner: RLParameterRefiner, output_dir: Path) -> None:
    history = refiner.history
    epochs = np.array([entry["epoch"] for entry in history], dtype=int)
    best_tx = np.array([entry["best_Tx"] for entry in history], dtype=float)
    best_tz = np.array([entry["best_Tz"] for entry in history], dtype=float)
    best_eta = np.array([entry["best_eta"] for entry in history], dtype=float)
    best_alpha_abs = np.array([entry["best_alpha_abs"] for entry in history], dtype=float)
    best_g2_real = np.array([entry["best_g2_real"] for entry in history], dtype=float)
    best_g2_imag = np.array([entry["best_g2_imag"] for entry in history], dtype=float)
    best_eps_real = np.array([entry["best_eps_real"] for entry in history], dtype=float)
    best_eps_imag = np.array([entry["best_eps_imag"] for entry in history], dtype=float)

    def save_epoch_series(
        filename: str,
        title: str,
        ylabel: str,
        series: list[tuple[np.ndarray, str]],
    ) -> None:
        fig, ax = plt.subplots(figsize=(13, 5))
        for values, label in series:
            ax.plot(epochs, values, linewidth=2.0, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    save_epoch_series(
        "ppo_tx_vs_epoch.png",
        r"Phase-Flip Lifetime $T_X$ over Epochs",
        r"$T_X$ (us)",
        [(best_tx, r"$T_X$")],
    )
    save_epoch_series(
        "ppo_tz_vs_epoch.png",
        r"Bit-Flip Lifetime $T_Z$ over Epochs",
        r"$T_Z$ (us)",
        [(best_tz, r"$T_Z$")],
    )
    save_epoch_series(
        "ppo_eta_vs_epoch.png",
        r"Noise Bias $\eta = T_Z / T_X$ over Epochs",
        r"$\eta$",
        [(best_eta, r"$T_Z / T_X$")],
    )
    save_epoch_series(
        "ppo_control_params_vs_epoch.png",
        r"Control Parameters over Epochs",
        "Control value (MHz)",
        [
            (best_eps_real, r"Re($\epsilon_d$)"),
            (best_eps_imag, r"Im($\epsilon_d$)"),
            (best_g2_real, r"Re($g_2$)"),
            (best_g2_imag, r"Im($g_2$)"),
        ],
    )
    save_epoch_series(
        "ppo_cat_size_vs_epoch.png",
        r"Cat Size $|\alpha|$ over Epochs",
        r"$|\alpha|$",
        [(best_alpha_abs, r"$|\alpha|$")],
    )


def save_decay_snapshot(
    backend: SimulatorBackend,
    parameters_vector: np.ndarray,
    *,
    title_prefix: str,
    filename: str,
    output_dir: Path,
) -> None:
    trace = backend.trace_candidate(jnp.asarray(parameters_vector, dtype=jnp.float32))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(trace.z_time, trace.z_signal, color="tab:orange", label=r"Measured $\langle Z \rangle$")
    axes[0, 0].plot(
        trace.z_time,
        trace.z_fit,
        color="tab:orange",
        linestyle="--",
        label=rf"$T_Z$ fit, $T_Z={trace.metrics['Tz']:.2f}$ us",
    )
    axes[0, 0].plot(trace.x_time, trace.x_signal, color="tab:blue", label=r"Measured $\langle X \rangle$")
    axes[0, 0].plot(
        trace.x_time,
        trace.x_fit,
        color="tab:blue",
        linestyle="--",
        label=rf"$T_X$ fit, $T_X={trace.metrics['Tx']:.2f}$ us",
    )
    axes[0, 0].set_title(fr"{title_prefix} raw $T_X$ and $T_Z$ decay traces")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("Observable value")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.02,
        0.98,
        "Decay panel notes\n"
        "- Linear scale only\n"
        "- Time axis shown in microseconds\n"
        "- Dashed lines are exponential fits\n"
        "- Solid lines are raw expectation values",
        va="top",
        ha="left",
        family="monospace",
    )

    axes[1, 0].plot(trace.z_time, trace.z_codespace_population, label="Codespace (+z)")
    axes[1, 0].plot(trace.x_time, trace.x_codespace_population, label="Codespace (+x)")
    axes[1, 0].plot(trace.z_time, trace.z_parity, label="Parity (+z)")
    axes[1, 0].plot(trace.x_time, trace.x_parity, label="Parity (+x)")
    axes[1, 0].set_title(f"{title_prefix} cat-manifold stability")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    summary_text = "\n".join(
        [
            f"Tx = {trace.metrics['Tx']:.4f}",
            f"Tz = {trace.metrics['Tz']:.4f}",
            f"eta = {trace.metrics['eta']:.4f}",
            f"mean_X = {trace.metrics['mean_X']:.4f}",
            f"mean_parity = {trace.metrics['mean_parity']:.4f}",
            f"cat_coherence = {trace.metrics['cat_coherence']:.4f}",
            f"nbar = {trace.metrics['nbar']:.4f}",
            f"leakage = {trace.metrics['leakage']:.4f}",
            f"instability_penalty = {trace.metrics['instability_penalty']:.4f}",
            f"action_penalty = {trace.metrics['action_penalty']:.4f}",
            f"|alpha| = {trace.metrics['alpha_abs']:.4f}",
            f"kappa_2 = {trace.metrics['kappa_2']:.4f}",
        ]
    )
    axes[1, 1].text(0.02, 0.98, summary_text, va="top", ha="left", family="monospace")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def save_progress_artifacts(
    refiner: RLParameterRefiner,
    backend: SimulatorBackend,
    output_dir: Path,
    *,
    title_prefix: str,
    decay_filename: str | None = None,
) -> None:
    if not refiner.history or refiner.best_record is None:
        return
    save_training_plots(refiner, output_dir)
    if decay_filename is not None:
        save_decay_snapshot(
            backend,
            np.asarray(refiner.best_record["parameters_vector"], dtype=float),
            title_prefix=title_prefix,
            filename=decay_filename,
            output_dir=output_dir,
        )


def parse_args() -> tuple[SimulationConfig, RewardConfig, PPOConfig, str, bool, bool]:
    parser = argparse.ArgumentParser(description="Parallelizable PPO runner for cat-qubit stabilization.")
    parser.add_argument("--backend", choices=["surrogate", "lindblad"], default="lindblad")
    parser.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=PPOConfig.batch_size)
    parser.add_argument("--replay-capacity", type=int, default=PPOConfig.replay_capacity)
    parser.add_argument("--replay-sample-size", type=int, default=PPOConfig.replay_sample_size)
    parser.add_argument("--ppo-epochs", type=int, default=PPOConfig.ppo_epochs)
    parser.add_argument("--clip-ratio", type=float, default=PPOConfig.clip_ratio)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--initial-std", type=float, default=PPOConfig.initial_std)
    parser.add_argument("--min-std", type=float, default=PPOConfig.min_std)
    parser.add_argument("--max-std", type=float, default=PPOConfig.max_std)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--snapshot-every", type=int, default=PPOConfig.snapshot_every)
    parser.add_argument("--log-every", type=int, default=PPOConfig.log_every)
    parser.add_argument("--log-candidates", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--na", type=int, default=SimulationConfig.na)
    parser.add_argument("--nb", type=int, default=SimulationConfig.nb)
    parser.add_argument("--kappa-a", type=float, default=SimulationConfig.kappa_a)
    parser.add_argument("--kappa-b", type=float, default=SimulationConfig.kappa_b)
    parser.add_argument("--x-tfinal", type=float, default=SimulationConfig.x_tfinal)
    parser.add_argument("--z-tfinal", type=float, default=SimulationConfig.z_tfinal)
    parser.add_argument("--nsave", type=int, default=SimulationConfig.nsave)
    parser.add_argument("--eval-x-tfinal", type=float, default=SimulationConfig.eval_x_tfinal)
    parser.add_argument("--eval-z-tfinal", type=float, default=SimulationConfig.eval_z_tfinal)
    parser.add_argument("--eval-nsave", type=int, default=SimulationConfig.eval_nsave)
    parser.add_argument("--target-bias", type=float, default=RewardConfig.target_bias)
    parser.add_argument("--lambda-eta", type=float, default=RewardConfig.lambda_eta)
    parser.add_argument("--eta-min", type=float, default=RewardConfig.eta_min)
    parser.add_argument("--eta-max", type=float, default=RewardConfig.eta_max)
    parser.add_argument("--output-dir", type=Path, default=PPOConfig.output_dir)
    args = parser.parse_args()

    sim_config = SimulationConfig(
        na=args.na,
        nb=args.nb,
        kappa_a=args.kappa_a,
        kappa_b=args.kappa_b,
        x_tfinal=args.x_tfinal,
        z_tfinal=args.z_tfinal,
        nsave=args.nsave,
        eval_x_tfinal=args.eval_x_tfinal,
        eval_z_tfinal=args.eval_z_tfinal,
        eval_nsave=args.eval_nsave,
    )
    reward_config = RewardConfig(
        target_bias=args.target_bias,
        lambda_eta=args.lambda_eta,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
    )
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        replay_sample_size=args.replay_sample_size,
        ppo_epochs=args.ppo_epochs,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        learning_rate=args.learning_rate,
        initial_std=args.initial_std,
        min_std=args.min_std,
        max_std=args.max_std,
        seed=args.seed,
        epochs=args.epochs,
        snapshot_every=args.snapshot_every,
        log_every=max(1, args.log_every),
        output_dir=args.output_dir,
    )
    return sim_config, reward_config, ppo_config, args.backend, args.log_candidates, args.quiet


def main() -> None:
    global QUIET
    sim_config, reward_config, ppo_config, backend_name, log_candidates, quiet = parse_args()
    QUIET = quiet
    output_dir = ppo_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(f"JAX version: {jax.__version__}")
    progress(f"JAX devices: {jax.devices()}")
    progress(f"Output directory: {output_dir}")

    bounds = ParameterBounds.default()
    seed_params = physics_informed_seed()
    reward_fn = build_batched_reward_function(reward_config)

    if backend_name == "surrogate":
        backend: SimulatorBackend = SurrogateBackend(
            seed_params=seed_params,
            bounds=bounds,
            reward_config=reward_config,
            config=sim_config,
            verbose=not quiet,
            log_candidates=log_candidates,
        )
    else:
        backend = LindbladBackend(
            seed_params=seed_params,
            bounds=bounds,
            reward_config=reward_config,
            config=sim_config,
            verbose=not quiet,
            log_candidates=log_candidates,
        )

    refiner = RLParameterRefiner(
        seed_params=seed_params,
        backend=backend,
        reward_fn=reward_fn,
        parameter_bounds=bounds,
        config=ppo_config,
    )

    progress(f"Backend: {backend.name}")
    progress(f"Starting PPO search from physics seed: {seed_params}")
    progress(
        "Run configuration: "
        f"epochs={ppo_config.epochs}, batch_size={ppo_config.batch_size}, "
        f"ppo_epochs={ppo_config.ppo_epochs}, replay_sample_size={ppo_config.replay_sample_size}, "
        f"snapshot_every={ppo_config.snapshot_every}, log_every={ppo_config.log_every}, "
        f"eta_range=[{reward_config.eta_min:.1f}, {reward_config.eta_max:.1f}], "
        f"target_bias={reward_config.target_bias:.1f}, lambda_eta={reward_config.lambda_eta:.3f}"
    )
    progress(
        "Simulation resolution: "
        f"train(x_tfinal={sim_config.eval_x_tfinal}, z_tfinal={sim_config.eval_z_tfinal}, nsave={sim_config.eval_nsave}) | "
        f"snapshot(x_tfinal={sim_config.x_tfinal}, z_tfinal={sim_config.z_tfinal}, nsave={sim_config.nsave})"
    )
    try:
        for epoch in range(ppo_config.epochs):
            metrics = refiner.train_step(epoch_index=epoch)
            if epoch % ppo_config.log_every == 0:
                progress(
                    f"Epoch {epoch:04d} summary | reward_mean={metrics['reward_mean']:.4f} | "
                    f"reward_max={metrics['reward_max']:.4f} | best_reward={metrics['best_reward']:.4f} | "
                    f"Tx_mean={metrics['Tx_mean']:.4f} | Tz_mean={metrics['Tz_mean']:.4f} | eta_mean={metrics['eta_mean']:.4f}"
                )

            current_epoch = epoch + 1
            if current_epoch == 1 or current_epoch % ppo_config.log_every == 0:
                progress(f"Epoch {current_epoch:04d}: saving progress plots.")
                save_progress_artifacts(
                    refiner,
                    backend,
                    output_dir,
                    title_prefix=f"Epoch {current_epoch}",
                )

            if ppo_config.snapshot_every > 0 and current_epoch % ppo_config.snapshot_every == 0:
                progress(f"Epoch {current_epoch:04d}: saving decay snapshot for current best candidate.")
                save_progress_artifacts(
                    refiner,
                    backend,
                    output_dir,
                    title_prefix=f"Epoch {current_epoch}",
                    decay_filename=f"ppo_decay_snapshot_epoch_{current_epoch:04d}.png",
                )
    except KeyboardInterrupt:
        progress("Run interrupted. Saving the latest available plots before exiting.")
        save_progress_artifacts(
            refiner,
            backend,
            output_dir,
            title_prefix="Interrupted run",
            decay_filename="ppo_interrupted_decay_snapshot.png",
        )
        progress(f"Saved available plots to: {output_dir}")
        raise

    best = refiner.get_best_parameters()
    progress("Training complete. Writing summary plots and final decay snapshot.")
    progress(f"Best PPO-discovered g2: {best['g2']}")
    progress(f"Best PPO-discovered epsilon_d: {best['epsilon_d']}")
    progress(
        "Final control values | "
        f"Re(g2)={np.real(best['g2']):+.6f} MHz, "
        f"Im(g2)={np.imag(best['g2']):+.6f} MHz, "
        f"Re(epsilon_d)={np.real(best['epsilon_d']):+.6f} MHz, "
        f"Im(epsilon_d)={np.imag(best['epsilon_d']):+.6f} MHz"
    )
    progress("Best metric summary: " + str(best["metrics"]))
    save_progress_artifacts(
        refiner,
        backend,
        output_dir,
        title_prefix="Final best",
        decay_filename="ppo_best_decay_snapshot.png",
    )
    progress(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
