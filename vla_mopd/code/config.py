"""Strict, dependency-light configuration for the MOPD reproduction.

The public entry point is :func:`load_config`.  Configuration is intentionally
split into four YAML sections so that experiment files stay readable and so a
misspelled option fails before models or datasets are loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when an MOPD configuration is malformed."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _require_string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError(f"{name} must be a non-empty domain-to-string mapping")

    result: dict[str, str] = {}
    for domain, item in value.items():
        domain = _require_nonempty_string(domain, f"{name} domain")
        result[domain] = _require_nonempty_string(item, f"{name}[{domain!r}]")
    return result


@dataclass
class ModelConfig:
    """Student and hard-routed teacher model settings."""

    student_model: str
    teacher_models: dict[str, str]
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    teacher_device: str = "student"
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        self.student_model = _require_nonempty_string(
            self.student_model, "model.student_model"
        )
        self.teacher_models = _require_string_mapping(
            self.teacher_models, "model.teacher_models"
        )

        self.torch_dtype = _require_nonempty_string(
            self.torch_dtype, "model.torch_dtype"
        ).lower()
        supported_dtypes = {
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
        }
        if self.torch_dtype not in supported_dtypes:
            raise ConfigError(
                "model.torch_dtype must be one of "
                f"{sorted(supported_dtypes)}, got {self.torch_dtype!r}"
            )

        if not isinstance(self.trust_remote_code, bool):
            raise ConfigError("model.trust_remote_code must be a boolean")
        if self.attn_implementation is not None:
            self.attn_implementation = _require_nonempty_string(
                self.attn_implementation, "model.attn_implementation"
            )
        self.teacher_device = _require_nonempty_string(
            self.teacher_device, "model.teacher_device"
        )
        if not isinstance(self.gradient_checkpointing, bool):
            raise ConfigError("model.gradient_checkpointing must be a boolean")


@dataclass
class DataConfig:
    """Domain datasets and deterministic mixture settings."""

    train_files: dict[str, str]
    domain_ratios: dict[str, float]
    max_prompt_length: int = 2048
    max_train_samples: int | None = None
    use_chat_template: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.train_files = _require_string_mapping(
            self.train_files, "data.train_files"
        )
        if not isinstance(self.domain_ratios, Mapping) or not self.domain_ratios:
            raise ConfigError(
                "data.domain_ratios must be a non-empty domain-to-number mapping"
            )

        ratios: dict[str, float] = {}
        for domain, value in self.domain_ratios.items():
            domain = _require_nonempty_string(domain, "data.domain_ratios domain")
            if not _is_number(value):
                raise ConfigError(
                    f"data.domain_ratios[{domain!r}] must be a positive number"
                )
            ratio = float(value)
            if not math.isfinite(ratio) or ratio <= 0:
                raise ConfigError(
                    f"data.domain_ratios[{domain!r}] must be finite and positive"
                )
            ratios[domain] = ratio

        ratio_sum = math.fsum(ratios.values())
        if not math.isfinite(ratio_sum) or ratio_sum <= 0:
            raise ConfigError("data.domain_ratios must have a positive finite sum")
        self.domain_ratios = {
            domain: ratio / ratio_sum for domain, ratio in ratios.items()
        }

        _require_positive_int(self.max_prompt_length, "data.max_prompt_length")
        if self.max_train_samples is not None:
            _require_positive_int(
                self.max_train_samples, "data.max_train_samples"
            )
        if not isinstance(self.use_chat_template, bool):
            raise ConfigError("data.use_chat_template must be a boolean")
        if (
            not isinstance(self.num_workers, int)
            or isinstance(self.num_workers, bool)
            or self.num_workers < 0
        ):
            raise ConfigError("data.num_workers must be a non-negative integer")


@dataclass
class AlgorithmConfig:
    """Objective and response-sampling settings."""

    loss_type: str = "pg"
    advantage_clip: float = 5.0
    topk_k: int = 64
    max_response_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if self.loss_type not in {"pg", "topk"}:
            raise ConfigError("algorithm.loss_type must be either 'pg' or 'topk'")
        if not _is_number(self.advantage_clip):
            raise ConfigError("algorithm.advantage_clip must be a number")
        self.advantage_clip = float(self.advantage_clip)
        if not math.isfinite(self.advantage_clip) or self.advantage_clip <= 0:
            raise ConfigError(
                "algorithm.advantage_clip must be finite and positive"
            )

        _require_positive_int(self.topk_k, "algorithm.topk_k")
        _require_positive_int(
            self.max_response_length, "algorithm.max_response_length"
        )
        if not _is_number(self.temperature):
            raise ConfigError("algorithm.temperature must be a positive number")
        self.temperature = float(self.temperature)
        if not math.isfinite(self.temperature) or self.temperature != 1.0:
            raise ConfigError(
                "algorithm.temperature must equal 1.0 for on-policy MOPD"
            )

        if not _is_number(self.top_p):
            raise ConfigError("algorithm.top_p must be a number in (0, 1]")
        self.top_p = float(self.top_p)
        if not math.isfinite(self.top_p) or self.top_p != 1.0:
            raise ConfigError("algorithm.top_p must equal 1.0 for on-policy MOPD")
        if not isinstance(self.enable_thinking, bool):
            raise ConfigError("algorithm.enable_thinking must be a boolean")


@dataclass
class TrainingConfig:
    """Optimizer, scheduling, checkpoint, and logging settings."""

    output_dir: str
    seed: int = 42
    max_steps: int = 1000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    logging_steps: int = 1
    save_steps: int = 100
    report_to: str | list[str] = "none"
    resume_from_checkpoint: str | None = None
    run_name: str | None = None
    wandb_project: str = "vla-mopd"

    def __post_init__(self) -> None:
        self.output_dir = _require_nonempty_string(
            self.output_dir, "training.output_dir"
        )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ConfigError("training.seed must be an integer")
        _require_positive_int(self.max_steps, "training.max_steps")
        _require_positive_int(
            self.per_device_train_batch_size,
            "training.per_device_train_batch_size",
        )
        _require_positive_int(
            self.gradient_accumulation_steps,
            "training.gradient_accumulation_steps",
        )

        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not _is_number(value) or not math.isfinite(float(value)) or value <= 0:
                raise ConfigError(f"training.{name} must be finite and positive")
            setattr(self, name, float(value))
        if (
            not _is_number(self.weight_decay)
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ConfigError("training.weight_decay must be finite and non-negative")
        self.weight_decay = float(self.weight_decay)

        if (
            not isinstance(self.warmup_steps, int)
            or isinstance(self.warmup_steps, bool)
            or self.warmup_steps < 0
        ):
            raise ConfigError("training.warmup_steps must be a non-negative integer")
        self.mixed_precision = _require_nonempty_string(
            self.mixed_precision, "training.mixed_precision"
        ).lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ConfigError(
                "training.mixed_precision must be one of 'no', 'fp16', or 'bf16'"
            )
        _require_positive_int(self.logging_steps, "training.logging_steps")
        _require_positive_int(self.save_steps, "training.save_steps")

        if isinstance(self.report_to, str):
            self.report_to = _require_nonempty_string(
                self.report_to, "training.report_to"
            ).lower()
        elif isinstance(self.report_to, list) and self.report_to:
            self.report_to = [
                _require_nonempty_string(
                    value, "training.report_to item"
                ).lower()
                for value in self.report_to
            ]
        else:
            raise ConfigError(
                "training.report_to must be a non-empty string or list of strings"
            )
        configured_reporters = (
            [self.report_to] if isinstance(self.report_to, str) else self.report_to
        )
        unsupported_reporters = sorted(
            set(configured_reporters) - {"none", "no", "wandb"}
        )
        if unsupported_reporters:
            raise ConfigError(
                "training.report_to supports only 'none' or 'wandb'; got "
                + ", ".join(unsupported_reporters)
            )
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = _require_nonempty_string(
                self.resume_from_checkpoint, "training.resume_from_checkpoint"
            )
        if self.run_name is not None:
            self.run_name = _require_nonempty_string(
                self.run_name, "training.run_name"
            )
            if (
                self.run_name in {".", ".."}
                or "/" in self.run_name
                or "\\" in self.run_name
            ):
                raise ConfigError(
                    "training.run_name must be a single safe path component"
                )
        self.wandb_project = _require_nonempty_string(
            self.wandb_project, "training.wandb_project"
        )


@dataclass
class MOPDConfig:
    """Fully validated configuration for one MOPD training run."""

    model: ModelConfig
    data: DataConfig
    algorithm: AlgorithmConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        domain_sets = {
            "model.teacher_models": set(self.model.teacher_models),
            "data.train_files": set(self.data.train_files),
            "data.domain_ratios": set(self.data.domain_ratios),
        }
        expected = domain_sets["model.teacher_models"]
        if any(domains != expected for domains in domain_sets.values()):
            details = ", ".join(
                f"{name}={sorted(domains)}"
                for name, domains in domain_sets.items()
            )
            raise ConfigError(f"domain keys must match exactly across sections: {details}")

    @property
    def domains(self) -> tuple[str, ...]:
        """Domains in teacher declaration order."""

        return tuple(self.model.teacher_models)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for metadata serialization."""

        return asdict(self)


ConfigType = TypeVar(
    "ConfigType", ModelConfig, DataConfig, AlgorithmConfig, TrainingConfig
)


def _parse_section(
    section_name: str,
    value: object,
    config_type: type[ConfigType],
) -> ConfigType:
    if not isinstance(value, Mapping):
        raise ConfigError(f"section {section_name!r} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"section {section_name!r} keys must be strings")

    allowed = {field.name for field in fields(config_type) if field.init}
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in section {section_name!r}: {sorted(unknown)}"
        )
    try:
        return config_type(**dict(value))
    except ConfigError:
        raise
    except TypeError as error:
        raise ConfigError(f"invalid section {section_name!r}: {error}") from error


def config_from_dict(value: object) -> MOPDConfig:
    """Build :class:`MOPDConfig` from a strict four-section mapping."""

    if not isinstance(value, Mapping):
        raise ConfigError("configuration root must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError("configuration section names must be strings")

    sections = {"model", "data", "algorithm", "training"}
    unknown = set(value) - sections
    missing = sections - set(value)
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {sorted(unknown)}")
    if missing:
        raise ConfigError(f"missing top-level section(s): {sorted(missing)}")

    return MOPDConfig(
        model=_parse_section("model", value["model"], ModelConfig),
        data=_parse_section("data", value["data"], DataConfig),
        algorithm=_parse_section(
            "algorithm", value["algorithm"], AlgorithmConfig
        ),
        training=_parse_section("training", value["training"], TrainingConfig),
    )


def load_config(path: str | Path) -> MOPDConfig:
    """Load and strictly validate a UTF-8 YAML experiment file."""

    config_path = Path(path).expanduser()
    try:
        with config_path.open(encoding="utf-8") as source:
            raw_config = yaml.safe_load(source)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error
    return config_from_dict(raw_config)


__all__ = [
    "AlgorithmConfig",
    "ConfigError",
    "DataConfig",
    "MOPDConfig",
    "ModelConfig",
    "TrainingConfig",
    "config_from_dict",
    "load_config",
]
