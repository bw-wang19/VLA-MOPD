from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .ops import forward_response_logits, gather_sampled_log_probs


DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unsupported torch_dtype={name!r}; choose one of {sorted(DTYPES)}"
        ) from error


def _config_vocab_size(config: Any) -> int:
    value = getattr(config, "vocab_size", None)
    if value is None and getattr(config, "text_config", None) is not None:
        value = getattr(config.text_config, "vocab_size", None)
    if value is None:
        raise ValueError("teacher config does not declare a token vocabulary size")
    return int(value)


def validate_shared_tokenizer(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    domain: str,
) -> None:
    """Fail fast when identical token ids would not have identical semantics."""
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            f"teacher {domain!r} tokenizer vocabulary/token-id mapping differs "
            "from the student"
        )
    for attribute in ("bos_token_id", "eos_token_id", "unk_token_id"):
        if getattr(student_tokenizer, attribute, None) != getattr(
            teacher_tokenizer, attribute, None
        ):
            raise ValueError(
                f"teacher {domain!r} {attribute} differs from the student"
            )
    student_pad = (
        student_tokenizer.pad_token_id
        if student_tokenizer.pad_token_id is not None
        else student_tokenizer.eos_token_id
    )
    teacher_pad = (
        teacher_tokenizer.pad_token_id
        if teacher_tokenizer.pad_token_id is not None
        else teacher_tokenizer.eos_token_id
    )
    if student_pad != teacher_pad:
        raise ValueError(
            f"teacher {domain!r} pad-token semantics differ from the student"
        )
    if getattr(student_tokenizer, "chat_template", None) != getattr(
        teacher_tokenizer, "chat_template", None
    ):
        raise ValueError(
            f"teacher {domain!r} chat template differs from the student"
        )


@dataclass(frozen=True)
class RoutedTeacherScores:
    sampled_log_probs: list[torch.Tensor]
    topk_indices: list[torch.Tensor] | None = None
    topk_log_probs: list[torch.Tensor] | None = None


class LocalTeacherPool:
    """Frozen, hard-routed Hugging Face teacher models.

    This local backend is intentionally simple and faithful to the algorithm.
    Paper-scale runs should replace it with asynchronous prefill services while
    preserving the same ``score_routed`` contract.
    """

    def __init__(
        self,
        model_paths: Mapping[str, str],
        *,
        device: torch.device | str,
        torch_dtype: str,
        pad_token_id: int,
        student_vocab_size: int,
        student_tokenizer: Any,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        if not model_paths:
            raise ValueError("at least one teacher model is required")
        self.device = torch.device(device)
        self.pad_token_id = int(pad_token_id)
        self.models: dict[str, Any] = {}
        models_by_path: dict[str, Any] = {}

        load_kwargs: dict[str, Any] = {
            "torch_dtype": resolve_dtype(torch_dtype),
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        for domain, path in model_paths.items():
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
            validate_shared_tokenizer(student_tokenizer, teacher_tokenizer, domain)
            config = AutoConfig.from_pretrained(
                path,
                trust_remote_code=trust_remote_code,
            )
            teacher_vocab_size = _config_vocab_size(config)
            if teacher_vocab_size != int(student_vocab_size):
                raise ValueError(
                    f"teacher {domain!r} has vocab_size={teacher_vocab_size}, "
                    f"student has vocab_size={student_vocab_size}; token-level "
                    "MOPD requires a shared action/token vocabulary"
                )
            if path in models_by_path:
                model = models_by_path[path]
            else:
                model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
                model.to(self.device)
                model.requires_grad_(False)
                model.eval()
                models_by_path[path] = model
            self.models[domain] = model

    @torch.no_grad()
    def _score_one_domain(
        self,
        domain: str,
        prompts: Sequence[Sequence[int]],
        responses: Sequence[torch.Tensor],
        *,
        topk: int | None,
        output_device: torch.device | str,
    ) -> RoutedTeacherScores:
        try:
            teacher = self.models[domain]
        except KeyError as error:
            raise KeyError(f"no teacher configured for domain {domain!r}") from error

        response_logits = forward_response_logits(
            teacher,
            prompts,
            responses,
            pad_token_id=self.pad_token_id,
            device=self.device,
        )
        sampled = [
            value.detach().to(output_device)
            for value in gather_sampled_log_probs(response_logits, responses)
        ]
        if topk is None:
            return RoutedTeacherScores(sampled_log_probs=sampled)

        if topk <= 0:
            raise ValueError("topk must be positive")
        topk_indices: list[torch.Tensor] = []
        topk_log_probs: list[torch.Tensor] = []
        for logits in response_logits:
            if topk > logits.size(-1):
                raise ValueError(
                    f"topk={topk} exceeds teacher vocabulary size "
                    f"{logits.size(-1)}"
                )
            k = topk
            full_log_probs = F.log_softmax(logits.float(), dim=-1)
            values, indices = full_log_probs.topk(k, dim=-1)
            topk_indices.append(indices.detach().to(output_device))
            topk_log_probs.append(values.detach().to(output_device))
        return RoutedTeacherScores(
            sampled_log_probs=sampled,
            topk_indices=topk_indices,
            topk_log_probs=topk_log_probs,
        )

    @torch.no_grad()
    def score_routed(
        self,
        domains: Sequence[str],
        prompts: Sequence[Sequence[int]],
        responses: Sequence[torch.Tensor],
        *,
        topk: int | None,
        output_device: torch.device | str,
    ) -> RoutedTeacherScores:
        if not (len(domains) == len(prompts) == len(responses)):
            raise ValueError("domains, prompts, and responses must align")

        routed_indices: dict[str, list[int]] = defaultdict(list)
        for index, domain in enumerate(domains):
            routed_indices[domain].append(index)

        sampled: list[torch.Tensor | None] = [None] * len(domains)
        indices_out: list[torch.Tensor | None] | None = (
            [None] * len(domains) if topk is not None else None
        )
        log_probs_out: list[torch.Tensor | None] | None = (
            [None] * len(domains) if topk is not None else None
        )

        for domain, positions in routed_indices.items():
            scores = self._score_one_domain(
                domain,
                [prompts[index] for index in positions],
                [responses[index] for index in positions],
                topk=topk,
                output_device=output_device,
            )
            for local_index, batch_index in enumerate(positions):
                sampled[batch_index] = scores.sampled_log_probs[local_index]
                if topk is not None:
                    assert indices_out is not None and log_probs_out is not None
                    assert scores.topk_indices is not None
                    assert scores.topk_log_probs is not None
                    indices_out[batch_index] = scores.topk_indices[local_index]
                    log_probs_out[batch_index] = scores.topk_log_probs[local_index]

        if any(value is None for value in sampled):
            raise RuntimeError("internal routing error: missing teacher score")
        if topk is not None:
            assert indices_out is not None and log_probs_out is not None
            if any(value is None for value in indices_out + log_probs_out):
                raise RuntimeError("internal routing error: missing top-k score")

        return RoutedTeacherScores(
            sampled_log_probs=[value for value in sampled if value is not None],
            topk_indices=(
                [value for value in indices_out if value is not None]
                if indices_out is not None
                else None
            ),
            topk_log_probs=(
                [value for value in log_probs_out if value is not None]
                if log_probs_out is not None
                else None
            ),
        )

