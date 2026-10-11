"""Strict configuration for the standalone VLA-OPD implementation.

Only ``batch_size`` and ``group_size`` are reported by the VLA-OPD paper.
Every other training value in the example configuration is an implementation
default and is persisted so experiments do not silently inherit settings from
MOPD or an RL trainer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when a VLA-OPD configuration is malformed."""


T = TypeVar("T")


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfigError(f"{name} must be finite and positive")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _section(cls: type[T], raw: object, name: str) -> T:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown {name} field(s): {', '.join(unknown)}")
    try:
        return cls(**dict(raw))
    except TypeError as exc:
        raise ConfigError(f"invalid {name} section: {exc}") from exc


@dataclass
class AlgorithmConfig:
    """Paper objective and on-policy collection semantics."""

    group_size: int = 8
    prompt_batch_size: int = 64
    max_episode_steps: int = 512
    score_batch_size: int = 64
    student_temperature: float = 1.0
    teacher_temperature: float = 1.0
    reduction: str = "trajectory_sum"
    reward_clip: float | None = None
    on_policy_logprob_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        _positive_int(self.group_size, "algorithm.group_size")
        _positive_int(self.prompt_batch_size, "algorithm.prompt_batch_size")
        _positive_int(self.max_episode_steps, "algorithm.max_episode_steps")
        _positive_int(self.score_batch_size, "algorithm.score_batch_size")
        self.student_temperature = _finite_positive(
            self.student_temperature, "algorithm.student_temperature"
        )
        self.teacher_temperature = _finite_positive(
            self.teacher_temperature, "algorithm.teacher_temperature"
        )
        if self.reduction not in {
            "trajectory_sum",
            "trajectory_token_mean",
            "global_token_mean",
        }:
            raise ConfigError(
                "algorithm.reduction must be 'trajectory_sum', "
                "'trajectory_token_mean', or 'global_token_mean'"
            )
        if self.reward_clip is not None:
            self.reward_clip = _finite_positive(
                self.reward_clip, "algorithm.reward_clip"
            )
        self.on_policy_logprob_tolerance = _finite_positive(
            self.on_policy_logprob_tolerance,
            "algorithm.on_policy_logprob_tolerance",
        )

    @property
    def uses_paper_objective(self) -> bool:
        """Whether optional, non-paper objective extensions are disabled."""

        return self.reduction == "trajectory_sum" and self.reward_clip is None


@dataclass
class TrainingConfig:
    """Implementation defaults absent from the paper's public specification."""

    num_updates: int = 20
    learning_rate: float = 2.0e-6
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    seed: int = 42
    output_dir: str = "outputs/vla_opd"
    log_every: int = 1
    save_every: int = 10

    def __post_init__(self) -> None:
        _positive_int(self.num_updates, "training.num_updates")
        self.learning_rate = _finite_positive(
            self.learning_rate, "training.learning_rate"
        )
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ConfigError("training.weight_decay must be finite and non-negative")
        self.weight_decay = float(self.weight_decay)
        if self.max_grad_norm is not None:
            self.max_grad_norm = _finite_positive(
                self.max_grad_norm, "training.max_grad_norm"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ConfigError("training.seed must be an integer")
        self.output_dir = _nonempty(self.output_dir, "training.output_dir")
        _positive_int(self.log_every, "training.log_every")
        _positive_int(self.save_every, "training.save_every")


@dataclass
class RuntimeConfig:
    """Factory used to connect a concrete VLA and robot environment."""

    factory: str = "vla_opd.toy:build_experiment"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.factory = _nonempty(self.factory, "runtime.factory")
        if ":" not in self.factory:
            raise ConfigError(
                "runtime.factory must use 'module.path:callable_name' syntax"
            )
        if not isinstance(self.kwargs, Mapping):
            raise ConfigError("runtime.kwargs must be a mapping")
        self.kwargs = dict(self.kwargs)


@dataclass
class VLAOPDConfig:
    algorithm: AlgorithmConfig
    training: TrainingConfig
    runtime: RuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> VLAOPDConfig:
    """Load a YAML file and reject unknown or missing top-level sections."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    expected = {"algorithm", "training", "runtime"}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"missing top-level section(s): {', '.join(missing)}")

    return VLAOPDConfig(
        algorithm=_section(AlgorithmConfig, raw["algorithm"], "algorithm"),
        training=_section(TrainingConfig, raw["training"], "training"),
        runtime=_section(RuntimeConfig, raw["runtime"], "runtime"),
    )
