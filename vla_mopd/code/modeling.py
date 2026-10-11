from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .teachers import resolve_dtype


def validate_mode_invariant_student(model: torch.nn.Module) -> None:
    """Require train/eval to represent the same token policy.

    Rollout runs in eval mode so generation can use its KV cache, while the
    differentiable forced-scoring pass runs in train mode so Hugging Face
    gradient checkpointing remains active.  This is exactly on-policy only
    when the model has no mode-dependent stochastic/stateful layers.
    """
    violations: list[str] = []
    batchnorm_base = torch.nn.modules.batchnorm._BatchNorm
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    for module_name, module in model.named_modules():
        label = module_name or "<root>"
        if isinstance(module, batchnorm_base):
            violations.append(f"{label}:{type(module).__name__}")
            continue
        if isinstance(module, dropout_types) and float(module.p) > 0:
            violations.append(
                f"{label}:{type(module).__name__}(p={float(module.p):g})"
            )
            continue

        class_name = type(module).__name__.lower()
        for attribute, value in vars(module).items():
            key = attribute.lower()
            is_probability = (
                "dropout" in key
                or "drop_prob" in key
                or "layerdrop" in key
                or "drop_path" in key
                or (
                    ("drop" in class_name or "stochastic" in class_name)
                    and key == "p"
                )
            )
            if (
                is_probability
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > 0
            ):
                violations.append(f"{label}.{attribute}={float(value):g}")
                break

    if violations:
        preview = ", ".join(violations[:8])
        suffix = " ..." if len(violations) > 8 else ""
        raise ValueError(
            "strict on-policy MOPD requires a student with zero dropout and "
            "no BatchNorm/mode-dependent stochastic layers; found "
            f"{preview}{suffix}"
        )


def validate_context_window(
    model: torch.nn.Module,
    *,
    max_prompt_length: int,
    max_response_length: int,
) -> None:
    """Fail early when configured prompt plus response exceeds model context."""
    root_config = getattr(model, "config", None)
    text_config = getattr(root_config, "text_config", None)
    candidates = [value for value in (text_config, root_config) if value is not None]
    limit = None
    source = None
    for candidate in candidates:
        for attribute in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                limit = value
                source = attribute
                break
        if limit is not None:
            break
    if limit is None:
        return
    requested = max_prompt_length + max_response_length
    if requested > limit:
        raise ValueError(
            f"configured max_prompt_length + max_response_length = {requested} "
            f"exceeds model {source}={limit}"
        )


def load_student_and_tokenizer(model_config: Any):
    """Load the Stage-1 student checkpoint used to initialize MOPD."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.student_model,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "tokenizer defines neither pad_token_id nor eos_token_id"
            )
        # Reusing EOS avoids changing the same-origin action/token vocabulary.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(model_config.torch_dtype),
        "trust_remote_code": model_config.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if model_config.attn_implementation:
        load_kwargs["attn_implementation"] = model_config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_config.student_model,
        **load_kwargs,
    )
    validate_mode_invariant_student(model)

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tokenizer


def trainable_parameter_summary(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total

