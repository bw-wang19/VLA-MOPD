"""Validated data structures shared by rollout, labeling, and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Any, Mapping

import torch


def clone_snapshot(value: Any) -> Any:
    """Clone an observation so later environment mutation cannot corrupt it."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_snapshot(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception as exc:  # pragma: no cover - adapter-specific object
        raise TypeError(
            "observation cannot be snapshotted; provide deepcopy-compatible "
            "values or plain tensors/containers"
        ) from exc


def _token_vector(value: torch.Tensor, name: str, *, floating: bool) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1-D tensor")
    if floating:
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
    elif value.dtype == torch.bool or value.is_floating_point():
        raise ValueError(f"{name} must contain integer token ids")


@dataclass(frozen=True)
class ResetResult:
    observation: Any
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise TypeError("terminated and truncated must be booleans")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise TypeError("environment reward must be numeric")
        if not math.isfinite(float(self.reward)):
            raise ValueError("environment reward must be finite")


@dataclass(frozen=True)
class ActionSample:
    """One action chunk sampled from the student and executed in the env."""

    tokens: torch.Tensor
    log_probs: torch.Tensor
    mask: torch.Tensor
    env_action: Any

    def __post_init__(self) -> None:
        _token_vector(self.tokens, "action tokens", floating=False)
        _token_vector(self.log_probs, "behavior log-probabilities", floating=True)
        if self.mask.dtype != torch.bool or self.mask.ndim != 1:
            raise ValueError("action mask must be a 1-D boolean tensor")
        if self.tokens.shape != self.log_probs.shape or self.tokens.shape != self.mask.shape:
            raise ValueError("action tokens, log-probabilities, and mask must align")
        if not torch.any(self.mask):
            raise ValueError("an action sample must contain at least one valid token")


@dataclass(frozen=True)
class Transition:
    prompt_index: int
    group_index: int
    step_index: int
    observation: Any
    action_tokens: torch.Tensor
    env_action: Any
    behavior_log_probs: torch.Tensor
    token_mask: torch.Tensor
    env_reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.prompt_index, self.group_index, self.step_index) < 0:
            raise ValueError("prompt/group/step indices must be non-negative")
        if isinstance(self.env_reward, bool) or not isinstance(
            self.env_reward, (int, float)
        ):
            raise TypeError("environment reward must be numeric")
        if not math.isfinite(float(self.env_reward)):
            raise ValueError("environment reward must be finite")
        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise TypeError("terminated and truncated must be booleans")
        ActionSample(
            tokens=self.action_tokens,
            log_probs=self.behavior_log_probs,
            mask=self.token_mask,
            env_action=self.env_action,
        )


@dataclass(frozen=True)
class Trajectory:
    prompt: Any
    prompt_index: int
    group_index: int
    transitions: tuple[Transition, ...]
    final_observation: Any
    reset_info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("a trajectory must contain at least one transition")
        for step_index, transition in enumerate(self.transitions):
            if transition.prompt_index != self.prompt_index:
                raise ValueError("trajectory contains a different prompt index")
            if transition.group_index != self.group_index:
                raise ValueError("trajectory contains a different group index")
            if transition.step_index != step_index:
                raise ValueError("trajectory step indices must be contiguous from zero")

    @property
    def terminated(self) -> bool:
        return self.transitions[-1].terminated

    @property
    def truncated(self) -> bool:
        return self.transitions[-1].truncated


@dataclass(frozen=True)
class RolloutBatch:
    trajectories: tuple[Trajectory, ...]
    prompt_count: int
    group_size: int

    def __post_init__(self) -> None:
        if self.prompt_count <= 0 or self.group_size <= 0:
            raise ValueError("prompt_count and group_size must be positive")
        if len(self.trajectories) != self.prompt_count * self.group_size:
            raise ValueError("rollout batch must contain B x G trajectories")
        observed = {(t.prompt_index, t.group_index) for t in self.trajectories}
        expected = {
            (prompt_index, group_index)
            for prompt_index in range(self.prompt_count)
            for group_index in range(self.group_size)
        }
        if observed != expected:
            raise ValueError("rollout batch must contain every prompt/group pair once")


@dataclass(frozen=True)
class LabeledTrajectory:
    trajectory: Trajectory
    teacher_log_probs: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.teacher_log_probs) != len(self.trajectory.transitions):
            raise ValueError("teacher labels must align with trajectory transitions")
        for transition, teacher in zip(
            self.trajectory.transitions, self.teacher_log_probs, strict=True
        ):
            _token_vector(teacher, "teacher log-probabilities", floating=True)
            if teacher.shape != transition.action_tokens.shape:
                raise ValueError("teacher labels must align with student action tokens")


@dataclass(frozen=True)
class LabeledRolloutBatch:
    trajectories: tuple[LabeledTrajectory, ...]
    prompt_count: int
    group_size: int

    def __post_init__(self) -> None:
        RolloutBatch(
            trajectories=tuple(item.trajectory for item in self.trajectories),
            prompt_count=self.prompt_count,
            group_size=self.group_size,
        )
