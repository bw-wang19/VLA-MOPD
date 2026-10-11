"""Paper-faithful sampled reverse-KL policy-gradient objective."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class VLAOPDLossOutput:
    loss: torch.Tensor
    raw_rewards: tuple[torch.Tensor, ...]
    rewards: tuple[torch.Tensor, ...]
    per_trajectory_loss: tuple[torch.Tensor, ...]
    metrics: dict[str, torch.Tensor]


def _validate_vectors(
    current: Sequence[torch.Tensor],
    behavior: Sequence[torch.Tensor],
    teacher: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor] | None,
) -> tuple[torch.Tensor, ...]:
    count = len(current)
    if count == 0 or len(behavior) != count or len(teacher) != count:
        raise ValueError("current, behavior, and teacher batches must be non-empty and equal")
    if masks is not None and len(masks) != count:
        raise ValueError("masks must align with trajectories")
    resolved: list[torch.Tensor] = []
    for index, (now, old, target) in enumerate(zip(current, behavior, teacher, strict=True)):
        if now.ndim != 1 or now.numel() == 0:
            raise ValueError("log-probability trajectories must be non-empty 1-D tensors")
        if now.shape != old.shape or now.shape != target.shape:
            raise ValueError("current, behavior, and teacher token vectors must align")
        if not now.is_floating_point() or not old.is_floating_point() or not target.is_floating_point():
            raise ValueError("log-probability tensors must be floating point")
        if not all(torch.all(torch.isfinite(value)) for value in (now, old, target)):
            raise ValueError("log-probability tensors must contain only finite values")
        if masks is None:
            mask = torch.ones_like(now, dtype=torch.bool)
        else:
            source_mask = masks[index]
            if not isinstance(source_mask, torch.Tensor) or source_mask.dtype != torch.bool:
                raise ValueError("each mask must be a boolean tensor")
            mask = source_mask.to(device=now.device)
        if mask.shape != now.shape:
            raise ValueError("each mask must align with its trajectory")
        if not torch.any(mask):
            raise ValueError("each trajectory must contain a valid action token")
        resolved.append(mask)
    return tuple(resolved)


def sampled_reverse_kl_pg_loss(
    current_student_log_probs: Sequence[torch.Tensor],
    behavior_student_log_probs: Sequence[torch.Tensor],
    teacher_log_probs: Sequence[torch.Tensor],
    *,
    masks: Sequence[torch.Tensor] | None = None,
    reduction: str = "trajectory_sum",
    reward_clip: float | None = None,
) -> VLAOPDLossOutput:
    """Compute VLA-OPD Eq. (6)-(7) as a minimization loss.

    The immediate reward is ``log p_teacher - log p_behavior``.  Both terms
    are detached.  The differentiable factor is the current student's
    log-probability for the exact action token sampled during rollout.

    ``trajectory_sum`` implements the paper's ``1/(B*G) * sum_t`` estimator.
    ``trajectory_token_mean`` first averages valid action tokens within each
    trajectory and then averages trajectories, so trajectories are weighted
    equally even when their valid lengths differ. ``global_token_mean``
    averages all valid tokens in the batch and therefore still gives longer
    trajectories more total weight.
    """

    if reduction not in {
        "trajectory_sum",
        "trajectory_token_mean",
        "global_token_mean",
    }:
        raise ValueError("unsupported reduction")
    if reward_clip is not None:
        if not isinstance(reward_clip, (int, float)) or isinstance(reward_clip, bool):
            raise ValueError("reward_clip must be a positive number or None")
        reward_clip = float(reward_clip)
        if not math.isfinite(reward_clip) or reward_clip <= 0:
            raise ValueError("reward_clip must be finite and positive")

    token_masks = _validate_vectors(
        current_student_log_probs,
        behavior_student_log_probs,
        teacher_log_probs,
        masks,
    )
    raw_rewards: list[torch.Tensor] = []
    used_rewards: list[torch.Tensor] = []
    trajectory_losses: list[torch.Tensor] = []
    valid_terms: list[torch.Tensor] = []
    valid_raw: list[torch.Tensor] = []
    clipped_count = 0
    valid_count = 0

    for current, behavior, target, mask in zip(
        current_student_log_probs,
        behavior_student_log_probs,
        teacher_log_probs,
        token_masks,
        strict=True,
    ):
        behavior = behavior.to(current.device)
        target = target.to(current.device)
        raw = (target.detach() - behavior.detach()).to(dtype=torch.float32)
        reward = raw if reward_clip is None else raw.clamp(-reward_clip, reward_clip)
        current_f32 = current.float()
        terms = -(current_f32[mask] * reward[mask])
        raw_rewards.append(raw)
        used_rewards.append(reward)
        trajectory_losses.append(terms.sum())
        valid_terms.append(terms)
        valid_raw.append(raw[mask])
        valid_count += int(mask.sum().item())
        if reward_clip is not None:
            clipped_count += int((raw[mask].abs() > reward_clip).sum().item())

    if reduction == "trajectory_sum":
        loss = torch.stack(trajectory_losses).mean()
    elif reduction == "trajectory_token_mean":
        loss = torch.stack([terms.mean() for terms in valid_terms]).mean()
    else:
        loss = torch.cat(valid_terms).mean()

    all_raw = torch.cat(valid_raw)
    metrics = {
        "opd_reward_mean": all_raw.mean().detach(),
        "opd_reverse_kl_mc": (-all_raw.mean()).detach(),
        "opd_reward_min": all_raw.min().detach(),
        "opd_reward_max": all_raw.max().detach(),
        "opd_clip_fraction": torch.tensor(
            clipped_count / valid_count, device=loss.device, dtype=torch.float32
        ),
        "valid_token_count": torch.tensor(valid_count, device=loss.device),
        "trajectory_count": torch.tensor(len(trajectory_losses), device=loss.device),
    }
    return VLAOPDLossOutput(
        loss=loss,
        raw_rewards=tuple(raw_rewards),
        rewards=tuple(used_rewards),
        per_trajectory_loss=tuple(trajectory_losses),
        metrics=metrics,
    )
