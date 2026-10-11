"""Collect -> label -> single-update training loop for VLA-OPD."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import nn

from .labeling import label_with_teacher
from .losses import VLAOPDLossOutput, sampled_reverse_kl_pg_loss
from .rollout import GroupRolloutCollector
from .types import LabeledRolloutBatch, RolloutBatch


@dataclass(frozen=True)
class TrainingStepOutput:
    loss: torch.Tensor
    metrics: dict[str, float]
    rollouts: RolloutBatch


class VLAOPDTrainer:
    """Strict online trainer implementing the paper's Algorithm 1.

    Every call to :meth:`train_step` collects a fresh batch, labels it, and
    performs exactly one optimizer update.  There is no replay, critic, GAE,
    PPO ratio, GRPO normalization, or outcome reward in this class.
    """

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: Any,
        optimizer: torch.optim.Optimizer,
        collector: GroupRolloutCollector,
        teacher_temperature: float,
        reduction: str = "trajectory_sum",
        reward_clip: float | None = None,
        on_policy_logprob_tolerance: float = 1.0e-4,
        max_grad_norm: float | None = None,
        score_batch_size: int | None = None,
    ) -> None:
        if not hasattr(student, "sample_action") or not hasattr(student, "score_actions"):
            raise TypeError("student must implement sample_action and score_actions")
        if not hasattr(teacher, "score_actions"):
            raise TypeError("teacher must implement score_actions")
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive")
        if on_policy_logprob_tolerance <= 0:
            raise ValueError("on_policy_logprob_tolerance must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if score_batch_size is not None and score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive or None")

        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.collector = collector
        self.teacher_temperature = float(teacher_temperature)
        self.reduction = reduction
        self.reward_clip = reward_clip
        self.on_policy_logprob_tolerance = float(on_policy_logprob_tolerance)
        self.max_grad_norm = max_grad_norm
        self.score_batch_size = score_batch_size
        self.optimizer_steps = 0

        if isinstance(self.teacher, nn.Module):
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)

    def _score_student(
        self, labeled: LabeledRolloutBatch
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        observations: list[Any] = []
        actions: list[torch.Tensor] = []
        for item in labeled.trajectories:
            for transition in item.trajectory.transitions:
                observations.append(transition.observation)
                actions.append(transition.action_tokens)

        batch_size = self.score_batch_size or len(observations)
        flat_current: list[torch.Tensor] = []
        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            scored = self.student.score_actions(  # type: ignore[attr-defined]
                observations[start:end],
                actions[start:end],
                temperature=self.collector.temperature,
            )
            if len(scored) != end - start:
                raise ValueError("student must return one score vector per action chunk")
            for action, log_probs in zip(actions[start:end], scored, strict=True):
                if not isinstance(log_probs, torch.Tensor):
                    raise TypeError("student scores must be tensors")
                if log_probs.ndim != 1 or log_probs.shape != action.shape:
                    raise ValueError("student scores must align with sampled action tokens")
                if not log_probs.is_floating_point() or not torch.all(
                    torch.isfinite(log_probs)
                ):
                    raise ValueError("student log-probabilities must be finite floats")
                flat_current.append(log_probs)

        current: list[torch.Tensor] = []
        behavior: list[torch.Tensor] = []
        teacher: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        cursor = 0
        for item in labeled.trajectories:
            count = len(item.trajectory.transitions)
            current.append(torch.cat(flat_current[cursor : cursor + count]))
            behavior.append(
                torch.cat(
                    [t.behavior_log_probs for t in item.trajectory.transitions]
                )
            )
            teacher.append(torch.cat(item.teacher_log_probs))
            masks.append(
                torch.cat([t.token_mask for t in item.trajectory.transitions])
            )
            cursor += count
        return current, behavior, teacher, masks

    def train_step(self, prompts: Sequence[Any]) -> TrainingStepOutput:
        # Evaluation mode is deliberate: rollout and differentiable rescoring
        # must represent the same categorical policy (no dropout/BatchNorm drift).
        self.student.eval()
        rollouts = self.collector.collect(prompts, self.student)  # type: ignore[arg-type]
        labeled = label_with_teacher(
            rollouts,
            self.teacher,
            temperature=self.teacher_temperature,
            score_batch_size=self.score_batch_size,
        )

        self.optimizer.zero_grad(set_to_none=True)
        current, behavior, teacher, masks = self._score_student(labeled)

        max_drift = 0.0
        for now, old, mask in zip(current, behavior, masks, strict=True):
            old = old.to(now.device)
            mask = mask.to(now.device, dtype=torch.bool)
            drift = float((now.detach()[mask] - old[mask]).abs().max().item())
            max_drift = max(max_drift, drift)
        if not math.isfinite(max_drift) or max_drift > self.on_policy_logprob_tolerance:
            raise RuntimeError(
                "student rollout/rescore log-probabilities drifted by "
                f"{max_drift:.6g}, exceeding "
                f"{self.on_policy_logprob_tolerance:.6g}; ensure identical "
                "weights, temperature, preprocessing, and eval-mode semantics"
            )

        output: VLAOPDLossOutput = sampled_reverse_kl_pg_loss(
            current,
            behavior,
            teacher,
            masks=masks,
            reduction=self.reduction,
            reward_clip=self.reward_clip,
        )
        output.loss.backward()
        if self.max_grad_norm is None:
            squared = torch.zeros((), device=output.loss.device)
            for parameter in self.student.parameters():
                if parameter.grad is not None:
                    squared = squared + parameter.grad.detach().float().pow(2).sum()
            grad_norm = float(squared.sqrt().item())
        else:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.max_grad_norm
                ).item()
            )
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite student gradient norm")
        self.optimizer.step()
        self.optimizer_steps += 1

        environment_rewards = [
            transition.env_reward
            for trajectory in rollouts.trajectories
            for transition in trajectory.transitions
        ]
        metrics = {
            name: float(value.item()) for name, value in output.metrics.items()
        }
        metrics.update(
            {
                "loss": float(output.loss.detach().item()),
                "grad_norm": grad_norm,
                "on_policy_logprob_max_drift": max_drift,
                "environment_reward_mean_diagnostic_only": sum(environment_rewards)
                / len(environment_rewards),
                "optimizer_steps": float(self.optimizer_steps),
                "prompt_count": float(rollouts.prompt_count),
                "group_size": float(rollouts.group_size),
            }
        )
        return TrainingStepOutput(
            loss=output.loss.detach(), metrics=metrics, rollouts=rollouts
        )
