from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig


@contextmanager
def _clean_model_generation_config(model: Any):
    """Prevent checkpoint generation settings from backfilling raw rollout.

    Transformers 5 merges every ``None`` field in an explicitly supplied
    config from ``model.generation_config`` and also carries custom entries.
    Temporarily replacing the model-side config is therefore required in
    addition to passing a fresh per-call config.
    """
    if not hasattr(model, "generation_config"):
        yield
        return
    original = model.generation_config
    model.generation_config = GenerationConfig()
    try:
        yield
    finally:
        model.generation_config = original


def left_pad(
    sequences: Sequence[Sequence[int] | torch.Tensor],
    pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad non-empty token sequences and return ids plus attention mask."""
    if not sequences:
        raise ValueError("cannot pad an empty collection of sequences")

    tensors = [
        torch.as_tensor(sequence, dtype=torch.long, device=device)
        for sequence in sequences
    ]
    if any(tensor.ndim != 1 or tensor.numel() == 0 for tensor in tensors):
        raise ValueError("every token sequence must be a non-empty 1-D sequence")

    width = max(tensor.numel() for tensor in tensors)
    input_ids = torch.full(
        (len(tensors), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, tensor in enumerate(tensors):
        input_ids[row, -tensor.numel() :] = tensor
        attention_mask[row, -tensor.numel() :] = 1
    return input_ids, attention_mask


def trim_after_eos(
    tokens: torch.Tensor,
    eos_token_id: int | Sequence[int] | None,
) -> torch.Tensor:
    """Keep tokens through the first EOS, without decode/re-tokenize drift."""
    if tokens.ndim != 1:
        raise ValueError("generated tokens must be a 1-D tensor")
    if eos_token_id is None:
        return tokens.clone()

    eos_ids = {
        int(token_id)
        for token_id in (
            eos_token_id
            if isinstance(eos_token_id, Sequence)
            else [eos_token_id]
        )
    }
    for index, token_id in enumerate(tokens.tolist()):
        if token_id in eos_ids:
            return tokens[: index + 1].clone()
    return tokens.clone()


def render_prompt(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_prompt_length: int,
    use_chat_template: bool,
    enable_thinking: bool,
) -> list[int]:
    """Render a prompt/messages record into the exact ids used by all policies."""
    if "messages" in record and record["messages"] is not None:
        messages = record["messages"]
    elif "prompt" in record and isinstance(record["prompt"], str):
        messages = [{"role": "user", "content": record["prompt"]}]
    else:
        raise ValueError("each record must contain either `prompt` or `messages`")

    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "use_chat_template=true but the tokenizer has no chat template"
            )
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
    else:
        if "prompt" not in record or not isinstance(record["prompt"], str):
            raise ValueError(
                "use_chat_template=false requires a string `prompt` field"
            )
        encoded = tokenizer(
            record["prompt"],
            add_special_tokens=True,
        )["input_ids"]

    prompt_ids = list(encoded)
    if not prompt_ids:
        raise ValueError("the rendered prompt is empty")
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    return prompt_ids


@torch.no_grad()
def generate_responses(
    model: Any,
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    eos_token_id: int | Sequence[int] | None,
    max_response_length: int,
    temperature: float,
    top_p: float,
    device: torch.device | str,
    synced_gpus: bool = False,
) -> list[torch.Tensor]:
    """Generate on-policy responses while preserving the sampled token ids."""
    input_ids, attention_mask = left_pad(prompts, pad_token_id, device)
    prompt_width = input_ids.size(1)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    if temperature != 1.0 or top_p != 1.0:
        raise ValueError(
            "raw-policy MOPD rollout requires temperature=1.0 and top_p=1.0"
        )

    # Never inherit checkpoint generation_config here.  In particular, many
    # chat checkpoints ship top_k/repetition/forced-token settings.  Those
    # processors would sample from a different behavior distribution while
    # forced scoring below uses the raw full-softmax policy.  A fresh config
    # gives neutral defaults; top_k=0 explicitly disables its non-neutral
    # GenerationConfig default.
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_response_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )
    with _clean_model_generation_config(model):
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
        )
    return [
        trim_after_eos(row[prompt_width:], eos_token_id)
        for row in generated
    ]


def concatenate_prompt_responses(
    prompts: Sequence[Sequence[int]],
    responses: Sequence[torch.Tensor],
    *,
    device: torch.device | str,
) -> list[torch.Tensor]:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    return [
        torch.cat(
            (
                torch.as_tensor(prompt, dtype=torch.long, device=device),
                response.detach().to(device=device, dtype=torch.long),
            )
        )
        for prompt, response in zip(prompts, responses, strict=True)
    ]


def extract_response_logits(
    logits: torch.Tensor,
    responses: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Extract causal logits predicting each response from left-padded batches.

    The input may contain full-sequence logits or only the final
    ``max_response_length + 1`` positions (via ``logits_to_keep``).
    """
    if logits.ndim != 3 or logits.size(0) != len(responses):
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")

    kept_positions = logits.size(1)
    extracted: list[torch.Tensor] = []
    for row, response in enumerate(responses):
        response_length = int(response.numel())
        if response_length <= 0:
            raise ValueError("responses must contain at least one token")
        start = kept_positions - response_length - 1
        stop = start + response_length
        if start < 0 or stop > kept_positions:
            raise ValueError(
                "logits do not retain enough causal positions for the response"
            )
        extracted.append(logits[row, start:stop])
    return extracted


def gather_sampled_log_probs(
    response_logits: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if len(response_logits) != len(responses):
        raise ValueError("response logits and responses must have equal length")
    gathered: list[torch.Tensor] = []
    for logits, response in zip(response_logits, responses, strict=True):
        if logits.size(0) != response.numel():
            raise ValueError("one logit row is required for every response token")
        token_ids = response.to(logits.device, dtype=torch.long)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gathered.append(log_probs.gather(-1, token_ids[:, None]).squeeze(-1))
    return gathered


def forward_response_logits(
    model: Any,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> list[torch.Tensor]:
    """Teacher-force responses and return their prediction logits."""
    full_sequences = concatenate_prompt_responses(
        prompts,
        responses,
        device=device,
    )
    input_ids, attention_mask = left_pad(full_sequences, pad_token_id, device)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    logits_to_keep = max(response.numel() for response in responses) + 1
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
    }
    try:
        outputs = model(logits_to_keep=logits_to_keep, **kwargs)
    except TypeError as error:
        if "logits_to_keep" not in str(error):
            raise
        outputs = model(**kwargs)
    return extract_response_logits(outputs.logits, responses)

