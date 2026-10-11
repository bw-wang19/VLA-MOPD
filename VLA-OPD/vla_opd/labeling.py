"""Frozen-teacher forced scoring on student-visited states/actions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .protocols import TeacherPolicy
from .types import LabeledRolloutBatch, LabeledTrajectory, RolloutBatch


def label_with_teacher(
    rollouts: RolloutBatch,
    teacher: TeacherPolicy,
    *,
    temperature: float,
    score_batch_size: int | None = None,
) -> LabeledRolloutBatch:
    """Score the exact student action tokens without sampling teacher actions."""

    if temperature <= 0:
        raise ValueError("teacher temperature must be positive")
    total_steps = sum(len(item.transitions) for item in rollouts.trajectories)
    if total_steps == 0:
        raise ValueError("cannot label an empty rollout batch")
    if score_batch_size is None:
        score_batch_size = total_steps
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")

    if isinstance(teacher, nn.Module):
        teacher.eval()

    observations: list[Any] = []
    actions: list[torch.Tensor] = []
    for trajectory in rollouts.trajectories:
        for transition in trajectory.transitions:
            observations.append(transition.observation)
            actions.append(transition.action_tokens)

    flat_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, total_steps, score_batch_size):
            end = min(start + score_batch_size, total_steps)
            scored = teacher.score_actions(
                observations[start:end],
                actions[start:end],
                temperature=temperature,
            )
            if len(scored) != end - start:
                raise ValueError("teacher must return one score vector per action chunk")
            for action, log_probs in zip(actions[start:end], scored, strict=True):
                if not isinstance(log_probs, torch.Tensor):
                    raise TypeError("teacher scores must be tensors")
                if log_probs.shape != action.shape or log_probs.ndim != 1:
                    raise ValueError(
                        "teacher score shape must match the same student action tokens"
                    )
                if not log_probs.is_floating_point() or not torch.all(
                    torch.isfinite(log_probs)
                ):
                    raise ValueError("teacher log-probabilities must be finite floats")
                flat_labels.append(log_probs.detach().float().clone())

    cursor = 0
    labeled: list[LabeledTrajectory] = []
    for trajectory in rollouts.trajectories:
        count = len(trajectory.transitions)
        labeled.append(
            LabeledTrajectory(
                trajectory=trajectory,
                teacher_log_probs=tuple(flat_labels[cursor : cursor + count]),
            )
        )
        cursor += count
    if cursor != len(flat_labels):  # defensive invariant
        raise RuntimeError("teacher label reconstruction lost transitions")
    return LabeledRolloutBatch(
        trajectories=tuple(labeled),
        prompt_count=rollouts.prompt_count,
        group_size=rollouts.group_size,
    )
