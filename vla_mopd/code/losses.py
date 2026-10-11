from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SampledMOPDOutput:
    loss: torch.Tensor
    raw_advantages: list[torch.Tensor]
    clipped_advantages: list[torch.Tensor]


def sampled_mopd_loss(
    student_log_probs: Sequence[torch.Tensor],
    teacher_log_probs: Sequence[torch.Tensor],
    *,
    advantage_clip: float,
    token_masks: Sequence[torch.Tensor] | None = None,
) -> SampledMOPDOutput:
    """Classic MOPD sampled-token policy-gradient loss (paper Eq. 3-4).

    Each trajectory is averaged before trajectories are averaged, matching the
    paper's ``1 / |y|`` normalization. This classic implementation is
    deliberately synchronous: the detached probabilities forming the
    advantage and the differentiable probabilities in the loss come from the
    same current policy.
    """
    if not math.isfinite(advantage_clip) or advantage_clip <= 0:
        raise ValueError("advantage_clip must be finite and positive")
    if len(student_log_probs) == 0:
        raise ValueError("at least one trajectory is required")
    if len(student_log_probs) != len(teacher_log_probs):
        raise ValueError("student and teacher batches must have equal length")

    if token_masks is not None and len(token_masks) != len(student_log_probs):
        raise ValueError("token masks and trajectories must have equal length")

    raw_advantages: list[torch.Tensor] = []
    clipped_advantages: list[torch.Tensor] = []
    sample_losses: list[torch.Tensor] = []
    for index, (current, teacher) in enumerate(
        zip(student_log_probs, teacher_log_probs, strict=True)
    ):
        if current.ndim != 1 or current.numel() == 0:
            raise ValueError("student log-probabilities must be non-empty 1-D tensors")
        if current.shape != teacher.shape:
            raise ValueError("all token log-probability tensors must align")
        teacher = teacher.to(current.device)
        raw = teacher.detach() - current.detach()
        clipped = raw.clamp(-advantage_clip, advantage_clip)
        mask = (
            torch.ones_like(current, dtype=torch.bool)
            if token_masks is None
            else token_masks[index].to(current.device, dtype=torch.bool)
        )
        if mask.shape != current.shape:
            raise ValueError("each token mask must align with its trajectory")
        if not torch.any(mask):
            raise ValueError("each trajectory must contain a valid token/action")
        raw_advantages.append(raw)
        clipped_advantages.append(clipped)
        sample_losses.append(-(clipped[mask] * current[mask]).mean())

    return SampledMOPDOutput(
        loss=torch.stack(sample_losses).mean(),
        raw_advantages=raw_advantages,
        clipped_advantages=clipped_advantages,
    )


@dataclass(frozen=True)
class TopKMOPDOutput:
    loss: torch.Tensor
    per_trajectory_loss: list[torch.Tensor]


def corrected_topk_mopd_loss(
    student_response_logits: Sequence[torch.Tensor],
    teacher_topk_indices: Sequence[torch.Tensor],
    teacher_topk_log_probs: Sequence[torch.Tensor],
    *,
    token_masks: Sequence[torch.Tensor] | None = None,
) -> TopKMOPDOutput:
    """Corrected teacher-top-k reverse-KL objective from paper Eq. 5.

    Teacher log-probabilities must retain their full-vocabulary normalization;
    they must not be re-normalized over the selected top-k support.
    """
    count = len(student_response_logits)
    if count == 0:
        raise ValueError("at least one trajectory is required")
    if count != len(teacher_topk_indices) or count != len(teacher_topk_log_probs):
        raise ValueError("student and teacher top-k batches must have equal length")
    if token_masks is not None and count != len(token_masks):
        raise ValueError("token masks and trajectories must have equal length")

    trajectory_losses: list[torch.Tensor] = []
    for index, (student_logits, indices, teacher_log_probs) in enumerate(
        zip(
            student_response_logits,
            teacher_topk_indices,
            teacher_topk_log_probs,
            strict=True,
        )
    ):
        if student_logits.ndim != 2 or student_logits.size(0) == 0:
            raise ValueError("student logits must have shape [tokens, vocabulary]")
        if indices.ndim != 2 or teacher_log_probs.shape != indices.shape:
            raise ValueError("teacher top-k tensors must align as [tokens, k]")
        if indices.size(0) != student_logits.size(0):
            raise ValueError("teacher and student response lengths must match")
        if indices.size(1) == 0:
            raise ValueError("top-k support cannot be empty")
        if indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("teacher top-k indices must be integer token ids")
        if torch.any(indices < 0) or torch.any(indices >= student_logits.size(-1)):
            raise ValueError("teacher top-k index is outside the student vocabulary")
        sorted_indices = indices.sort(dim=-1).values
        if sorted_indices.size(1) > 1 and torch.any(
            sorted_indices[:, 1:] == sorted_indices[:, :-1]
        ):
            raise ValueError("teacher top-k support contains duplicate token ids")
        if not torch.all(torch.isfinite(teacher_log_probs)):
            raise ValueError("teacher top-k log-probabilities must be finite")
        if torch.any(teacher_log_probs > 1e-6):
            raise ValueError("teacher log-probabilities cannot exceed zero")

        indices = indices.to(student_logits.device, dtype=torch.long)
        teacher_log_probs = teacher_log_probs.to(
            student_logits.device,
            dtype=torch.float32,
        ).detach()
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        selected_student_log_probs = student_log_probs.gather(-1, indices)
        student_probabilities = selected_student_log_probs.exp()
        teacher_probabilities = teacher_log_probs.exp()

        corrected_terms = (
            student_probabilities
            * (selected_student_log_probs - teacher_log_probs)
            - student_probabilities
            + teacher_probabilities
        )
        per_token_loss = corrected_terms.sum(dim=-1)
        mask = (
            torch.ones_like(per_token_loss, dtype=torch.bool)
            if token_masks is None
            else token_masks[index].to(per_token_loss.device, dtype=torch.bool)
        )
        if mask.shape != per_token_loss.shape:
            raise ValueError("each token mask must align with its trajectory")
        if not torch.any(mask):
            raise ValueError("each trajectory must contain a valid token/action")
        trajectory_losses.append(per_token_loss[mask].mean())

    return TopKMOPDOutput(
        loss=torch.stack(trajectory_losses).mean(),
        per_trajectory_loss=trajectory_losses,
    )


def categorical_entropy(
    response_logits: Sequence[torch.Tensor],
    *,
    token_masks: Sequence[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Mean policy entropy for each trajectory, used as a collapse diagnostic."""
    if token_masks is not None and len(token_masks) != len(response_logits):
        raise ValueError("token masks and trajectories must have equal length")
    entropies: list[torch.Tensor] = []
    for index, logits in enumerate(response_logits):
        log_probs = F.log_softmax(logits.detach().float(), dim=-1)
        per_token = -(log_probs.exp() * log_probs).sum(dim=-1)
        mask = (
            torch.ones_like(per_token, dtype=torch.bool)
            if token_masks is None
            else token_masks[index].to(per_token.device, dtype=torch.bool)
        )
        if mask.shape != per_token.shape:
            raise ValueError("each token mask must align with its trajectory")
        if not torch.any(mask):
            raise ValueError("each trajectory must contain a valid token/action")
        entropies.append(per_token[mask].mean())
    return entropies

