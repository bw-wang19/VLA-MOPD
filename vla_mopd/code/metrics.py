from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch


class MOPDMetricAccumulator:
    """Token- and trajectory-aware metrics with per-domain diagnostics."""

    FIELDS = (
        "teacher_logp_sum",
        "student_logp_sum",
        "advantage_sum",
        "advantage_square_sum",
        "clipped_count",
        "token_count",
        "response_length_sum",
        "trajectory_count",
        "entropy_sum",
        "loss_sum",
        "batch_count",
    )

    def __init__(self, domains: Sequence[str]) -> None:
        self.domains = tuple(domains)
        self._values: dict[str, dict[str, float]] = {}
        self.reset()

    def reset(self) -> None:
        self._values = {
            "overall": defaultdict(float),
            **{domain: defaultdict(float) for domain in self.domains},
        }

    def update(
        self,
        *,
        domains: Sequence[str],
        teacher_log_probs: Sequence[torch.Tensor],
        student_log_probs: Sequence[torch.Tensor],
        raw_advantages: Sequence[torch.Tensor],
        clipped_advantages: Sequence[torch.Tensor],
        entropies: Sequence[torch.Tensor],
        loss: torch.Tensor,
        advantage_clip: float,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> None:
        count = len(domains)
        if not all(
            len(values) == count
            for values in (
                teacher_log_probs,
                student_log_probs,
                raw_advantages,
                clipped_advantages,
                entropies,
            )
        ):
            raise ValueError("all metric inputs must align by trajectory")
        if token_masks is not None and len(token_masks) != count:
            raise ValueError("token masks and trajectories must align")

        for index, domain in enumerate(domains):
            if domain not in self._values:
                raise ValueError(f"unknown metric domain {domain!r}")
            raw = raw_advantages[index].detach().float()
            clipped = clipped_advantages[index].detach().float()
            teacher = teacher_log_probs[index].detach().float()
            student = student_log_probs[index].detach().float()
            mask = (
                torch.ones_like(raw, dtype=torch.bool)
                if token_masks is None
                else token_masks[index].to(raw.device, dtype=torch.bool)
            )
            if not (
                mask.shape == raw.shape == clipped.shape == teacher.shape == student.shape
            ):
                raise ValueError("metric tensors and token masks must align")
            if not torch.any(mask):
                raise ValueError("each trajectory must contain a valid token/action")
            raw = raw[mask]
            clipped = clipped[mask]
            teacher = teacher[mask]
            student = student[mask]
            token_count = float(mask.sum().item())
            for scope in ("overall", domain):
                values = self._values[scope]
                values["teacher_logp_sum"] += teacher.sum().item()
                values["student_logp_sum"] += student.sum().item()
                values["advantage_sum"] += raw.sum().item()
                values["advantage_square_sum"] += raw.square().sum().item()
                values["clipped_count"] += (
                    raw.abs() > advantage_clip
                ).sum().item()
                values["token_count"] += token_count
                values["response_length_sum"] += token_count
                values["trajectory_count"] += 1.0
                values["entropy_sum"] += entropies[index].detach().item()

        self._values["overall"]["loss_sum"] += loss.detach().item()
        self._values["overall"]["batch_count"] += 1.0

    def distributed_metrics(self, accelerator) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for scope, values in self._values.items():
            tensor = torch.tensor(
                [values[field] for field in self.FIELDS],
                dtype=torch.float64,
                device=accelerator.device,
            )
            tensor = accelerator.reduce(tensor, reduction="sum")
            reduced = dict(zip(self.FIELDS, tensor.tolist(), strict=True))
            tokens = max(reduced["token_count"], 1.0)
            trajectories = max(reduced["trajectory_count"], 1.0)
            advantage_mean = reduced["advantage_sum"] / tokens
            advantage_variance = max(
                reduced["advantage_square_sum"] / tokens
                - advantage_mean * advantage_mean,
                0.0,
            )
            prefix = "" if scope == "overall" else f"domain/{scope}/"
            metrics[f"{prefix}teacher_logp"] = (
                reduced["teacher_logp_sum"] / tokens
            )
            metrics[f"{prefix}student_logp"] = (
                reduced["student_logp_sum"] / tokens
            )
            metrics[f"{prefix}teacher_minus_student"] = advantage_mean
            metrics[f"{prefix}sampled_reverse_kl"] = -advantage_mean
            metrics[f"{prefix}advantage_std"] = advantage_variance**0.5
            metrics[f"{prefix}clipped_fraction"] = (
                reduced["clipped_count"] / tokens
            )
            metrics[f"{prefix}response_length"] = (
                reduced["response_length_sum"] / trajectories
            )
            metrics[f"{prefix}entropy"] = (
                reduced["entropy_sum"] / trajectories
            )
            metrics[f"{prefix}token_share_numerator"] = reduced["token_count"]
            metrics[f"{prefix}trajectory_share_numerator"] = reduced[
                "trajectory_count"
            ]

            if scope == "overall":
                batches = max(reduced["batch_count"], 1.0)
                metrics["train/loss"] = reduced["loss_sum"] / batches

        total_tokens = max(metrics.pop("token_share_numerator"), 1.0)
        total_trajectories = max(
            metrics.pop("trajectory_share_numerator"), 1.0
        )
        for domain in self.domains:
            token_key = f"domain/{domain}/token_share_numerator"
            trajectory_key = f"domain/{domain}/trajectory_share_numerator"
            metrics[f"domain/{domain}/token_share"] = (
                metrics.pop(token_key) / total_tokens
            )
            metrics[f"domain/{domain}/trajectory_share"] = (
                metrics.pop(trajectory_key) / total_trajectories
            )
        return metrics

