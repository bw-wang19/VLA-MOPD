"""Minimal contracts for categorical VLA policies and robot environments."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from .types import ActionSample, ResetResult, StepResult


@runtime_checkable
class VLAEnvironment(Protocol):
    def reset(self, prompt: Any, *, seed: int) -> ResetResult: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class EnvironmentFactory(Protocol):
    def __call__(
        self, *, prompt: Any, prompt_index: int, group_index: int
    ) -> VLAEnvironment: ...


@runtime_checkable
class StudentPolicy(Protocol):
    def sample_action(
        self,
        observation: Any,
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> ActionSample: ...

    def score_actions(
        self,
        observations: Sequence[Any],
        action_tokens: Sequence[torch.Tensor],
        *,
        temperature: float,
    ) -> Sequence[torch.Tensor]: ...


@runtime_checkable
class TeacherPolicy(Protocol):
    def score_actions(
        self,
        observations: Sequence[Any],
        action_tokens: Sequence[torch.Tensor],
        *,
        temperature: float,
    ) -> Sequence[torch.Tensor]: ...
