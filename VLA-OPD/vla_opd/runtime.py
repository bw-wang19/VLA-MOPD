"""Runtime boundary between the algorithm and a concrete VLA stack."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Sequence

from torch import nn

from .config import VLAOPDConfig
from .protocols import EnvironmentFactory


@dataclass
class ExperimentComponents:
    """Objects a runtime factory must provide to the generic trainer.

    ``student`` must be an ``nn.Module`` and implement the student policy
    protocol.  ``teacher`` stays frozen and implements the teacher scorer
    protocol.  ``env_factory`` must create a fresh environment for every
    trajectory so the G rollouts are independent.
    """

    student: nn.Module
    teacher: Any
    env_factory: EnvironmentFactory
    prompts: Sequence[Any]

    def __post_init__(self) -> None:
        if not isinstance(self.student, nn.Module):
            raise TypeError("runtime student must be a torch.nn.Module")
        if not callable(self.env_factory):
            raise TypeError("runtime env_factory must be callable")
        if len(self.prompts) == 0:
            raise ValueError("runtime must provide at least one prompt")


def load_experiment(config: VLAOPDConfig) -> ExperimentComponents:
    """Resolve and invoke ``runtime.factory`` without importing MOPD code."""

    module_name, callable_name = config.runtime.factory.split(":", 1)
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"cannot import runtime module {module_name!r}") from exc
    try:
        factory = getattr(module, callable_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"runtime module {module_name!r} has no callable {callable_name!r}"
        ) from exc
    if not callable(factory):
        raise RuntimeError(f"runtime target {config.runtime.factory!r} is not callable")
    components = factory(config=config, **config.runtime.kwargs)
    if not isinstance(components, ExperimentComponents):
        raise TypeError(
            "runtime factory must return vla_opd.runtime.ExperimentComponents"
        )
    return components
