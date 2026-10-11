from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .config import MOPDConfig, load_config
from .data import build_training_dataset
from .losses import (
    categorical_entropy,
    corrected_topk_mopd_loss,
    sampled_mopd_loss,
)
from .metrics import MOPDMetricAccumulator
from .modeling import (
    load_student_and_tokenizer,
    trainable_parameter_summary,
    validate_context_window,
)
from .ops import (
    forward_response_logits,
    gather_sampled_log_probs,
    generate_responses,
    render_prompt,
)
from .teachers import LocalTeacherPool


LOGGER = logging.getLogger("vla_mopd.train")
PAPER_ID = "arXiv:2606.30406"


def collate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


def default_run_name(config: MOPDConfig) -> str:
    def slug(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
        return cleaned or "unnamed"

    student_name = slug(Path(config.model.student_model.rstrip("/")).name)
    domains = "-".join(slug(domain) for domain in config.domains)
    return (
        f"{student_name}-classic-mopd-{config.algorithm.loss_type}-"
        f"{domains}-step{config.training.max_steps}"
    )


def configure_run_directory(config: MOPDConfig) -> tuple[str, Path]:
    run_name = getattr(config.training, "run_name", None) or default_run_name(config)
    run_directory = (
        Path(config.training.output_dir).expanduser()
        / "mopd"
        / config.algorithm.loss_type
        / run_name
    )
    return run_name, run_directory


def _reporters(value: str | list[str]) -> list[str]:
    reporters = [value] if isinstance(value, str) else list(value)
    return [item for item in reporters if item.lower() not in {"none", "no"}]


def create_accelerator(
    config: MOPDConfig,
    *,
    project_dir: Path | None = None,
) -> Accelerator:
    reporters = _reporters(config.training.report_to)
    accumulation = GradientAccumulationPlugin(
        num_steps=config.training.gradient_accumulation_steps,
        adjust_scheduler=False,
        # This trainer deliberately cycles a finite dataset until max_steps.
        # Accumulation must therefore continue across iterator boundaries.
        sync_with_dataloader=False,
    )
    return Accelerator(
        gradient_accumulation_plugin=accumulation,
        mixed_precision=config.training.mixed_precision,
        log_with=reporters or None,
        project_dir=str(project_dir) if project_dir is not None else None,
        # max_steps below is defined in global optimizer updates.  Accelerate's
        # default scheduler wrapper otherwise advances once per rank.
        step_scheduler_with_optimizer=False,
    )


def write_run_metadata(
    config: MOPDConfig,
    run_name: str,
    run_directory: Path,
    accelerator: Accelerator,
    *,
    parent_checkpoint: Path | None = None,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as output:
        yaml.safe_dump(config.to_dict(), output, sort_keys=False, allow_unicode=True)
    metadata = {
        "run_name": run_name,
        "paper": PAPER_ID,
        "implementation": "classic MOPD Stage 3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domains": list(config.domains),
        "loss_type": config.algorithm.loss_type,
        "normalization": "mean tokens within trajectory, then mean trajectories",
        "teacher_routing": "hard route by record.domain",
        "world_size": accelerator.num_processes,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "effective_global_batch_size": (
            config.training.per_device_train_batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        ),
        "mixed_precision": accelerator.mixed_precision,
        "parent_checkpoint": (
            str(parent_checkpoint.resolve()) if parent_checkpoint is not None else None
        ),
    }
    with (run_directory / "run_metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write("\n")


_RESUME_MUTABLE_TRAINING_FIELDS = {
    "logging_steps",
    "save_steps",
    "report_to",
    "resume_from_checkpoint",
    "wandb_project",
}

_CHECKPOINT_DESTINATION_FIELDS = _RESUME_MUTABLE_TRAINING_FIELDS | {
    "output_dir",
    "run_name",
}

_MANIFEST_SUFFIXES = {
    ".arrow",
    ".bin",
    ".json",
    ".jsonl",
    ".model",
    ".ndjson",
    ".parquet",
    ".pq",
    ".pt",
    ".pth",
    ".py",
    ".safetensors",
    ".tiktoken",
    ".txt",
    ".yaml",
    ".yml",
}


def _small_file_sha256(path: Path, size: int) -> str | None:
    if size > 1_048_576:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(locator: str) -> dict[str, Any]:
    """Build a lightweight identity manifest without hashing giant shards."""
    path = Path(locator).expanduser()
    if not path.exists():
        return {"locator": locator, "local": False}
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "locator": locator,
            "local": True,
            "resolved_path": str(resolved),
            "files": [
                {
                    "path": resolved.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _small_file_sha256(resolved, stat.st_size),
                }
            ],
        }

    files: list[dict[str, Any]] = []
    for child in sorted(resolved.rglob("*")):
        if not child.is_file() or child.suffix.lower() not in _MANIFEST_SUFFIXES:
            continue
        stat = child.stat()
        files.append(
            {
                "path": str(child.relative_to(resolved)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _small_file_sha256(child, stat.st_size),
            }
        )
    return {
        "locator": locator,
        "local": True,
        "resolved_path": str(resolved),
        "files": files,
    }


def build_artifact_manifests(config: MOPDConfig) -> dict[str, Any]:
    return {
        "student_model": artifact_manifest(config.model.student_model),
        "teacher_models": {
            domain: artifact_manifest(locator)
            for domain, locator in config.model.teacher_models.items()
        },
        "train_files": {
            domain: artifact_manifest(locator)
            for domain, locator in config.data.train_files.items()
        },
    }


def validate_artifact_manifests(
    config: MOPDConfig,
    expected_artifacts: dict[str, Any],
    *,
    source: str,
) -> None:
    current_artifacts = build_artifact_manifests(config)
    if expected_artifacts != current_artifacts:
        raise ValueError(
            "model/teacher/data artifact manifest differs from "
            f"{source}"
        )


def validate_distributed_artifact_consistency(
    artifact_manifests: dict[str, Any],
    accelerator: Accelerator,
) -> None:
    """Require every rank to see the same model/teacher/data identities."""
    if accelerator.num_processes <= 1:
        return
    payload = json.dumps(
        artifact_manifests,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = torch.tensor(
        list(hashlib.sha256(payload).digest()),
        dtype=torch.int64,
        device=accelerator.device,
    ).unsqueeze(0)
    gathered = accelerator.gather(digest)
    if gathered.ndim != 2 or gathered.size(0) != accelerator.num_processes:
        raise RuntimeError("failed to gather one artifact digest from every rank")
    if not bool(torch.all(gathered == gathered[0]).item()):
        raise ValueError(
            "distributed ranks see different model/teacher/data artifact "
            "manifests; use one shared immutable snapshot"
        )


def _compare_resume_configs(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    mutable_training_fields: set[str],
    source: str,
) -> None:
    for section in ("model", "data", "algorithm"):
        if previous.get(section) != current[section]:
            raise ValueError(
                f"resume would change immutable {section!r} configuration "
                f"from {source}"
            )
    previous_training = previous.get("training")
    if not isinstance(previous_training, dict):
        raise ValueError(f"invalid prior training config in {source}")
    immutable_keys = (
        set(previous_training) | set(current["training"])
    ) - mutable_training_fields
    changed = sorted(
        key
        for key in immutable_keys
        if previous_training.get(key) != current["training"].get(key)
    )
    if changed:
        raise ValueError(
            "resume would change immutable training fields from "
            f"{source}: " + ", ".join(changed)
        )


def validate_checkpoint_contract(
    config: MOPDConfig,
    checkpoint: Path,
    accelerator: Accelerator,
) -> dict[str, Any]:
    contract_path = checkpoint / "resume_contract.json"
    if not contract_path.is_file():
        raise ValueError(
            f"checkpoint lacks resume_contract.json and cannot be safely resumed: "
            f"{checkpoint}"
        )
    with contract_path.open(encoding="utf-8") as source:
        contract = json.load(source)
    if contract.get("version") != 1:
        raise ValueError(f"unsupported checkpoint resume contract: {contract_path}")
    previous = contract.get("config")
    if not isinstance(previous, dict):
        raise ValueError(f"invalid checkpoint resume contract: {contract_path}")
    _compare_resume_configs(
        previous,
        config.to_dict(),
        mutable_training_fields=_CHECKPOINT_DESTINATION_FIELDS,
        source=str(contract_path),
    )
    previous_world_size = contract.get("world_size")
    if previous_world_size != accelerator.num_processes:
        raise ValueError(
            "resume world_size mismatch: "
            f"checkpoint={previous_world_size}, current={accelerator.num_processes}"
        )
    expected_artifacts = contract.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError(f"invalid artifact manifest in {contract_path}")
    validate_artifact_manifests(
        config,
        expected_artifacts,
        source=f"checkpoint contract: {contract_path}",
    )
    return expected_artifacts


def validate_resume_compatibility(
    config: MOPDConfig,
    run_directory: Path,
    accelerator: Accelerator,
) -> bool:
    """Validate immutable run semantics; return whether prior metadata exists."""
    resolved_path = run_directory / "resolved_config.yaml"
    if not resolved_path.is_file():
        return False
    with resolved_path.open(encoding="utf-8") as source:
        previous = yaml.safe_load(source)
    if not isinstance(previous, dict):
        raise ValueError(f"invalid prior resolved config: {resolved_path}")

    current = config.to_dict()
    _compare_resume_configs(
        previous,
        current,
        mutable_training_fields=_RESUME_MUTABLE_TRAINING_FIELDS,
        source=str(resolved_path),
    )

    metadata_path = run_directory / "run_metadata.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as source:
            metadata = json.load(source)
        previous_world_size = metadata.get("world_size")
        if (
            previous_world_size is not None
            and previous_world_size != accelerator.num_processes
        ):
            raise ValueError(
                "resume world_size mismatch: "
                f"previous={previous_world_size}, current={accelerator.num_processes}"
            )
    return True


def append_resume_metadata(
    config: MOPDConfig,
    run_directory: Path,
    accelerator: Accelerator,
    checkpoint: Path,
    global_step: int,
) -> None:
    event = {
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "global_step": global_step,
        "world_size": accelerator.num_processes,
        "mixed_precision": accelerator.mixed_precision,
        "config": config.to_dict(),
    }
    with (run_directory / "resume_history.jsonl").open(
        "a", encoding="utf-8"
    ) as output:
        output.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def protect_existing_run(
    run_directory: Path,
    resume_checkpoint: Path | None,
) -> None:
    """Refuse to silently overwrite a previous experiment."""
    if not run_directory.exists():
        return
    protected_names = {
        "resolved_config.yaml",
        "run_metadata.json",
        "final",
    }
    has_artifacts = any(
        child.name in protected_names or child.name.startswith("checkpoint-")
        for child in run_directory.iterdir()
    )
    if not has_artifacts:
        return
    if resume_checkpoint is None:
        raise FileExistsError(
            f"run directory already contains experiment artifacts: {run_directory}; "
            "set a new training.run_name or resume_from_checkpoint"
        )
    if not (run_directory / "resolved_config.yaml").is_file():
        raise FileExistsError(
            f"existing run lacks resolved_config.yaml: {run_directory}; "
            "choose a new training.run_name instead of overwriting it"
        )
    try:
        resume_checkpoint.resolve().relative_to(run_directory.resolve())
    except ValueError as error:
        raise FileExistsError(
            "an external checkpoint cannot be resumed into a target run that "
            f"already has artifacts: {run_directory}; choose a new run_name"
        ) from error
    latest_checkpoint = resolve_resume_checkpoint("latest", run_directory)
    if resume_checkpoint.resolve() != latest_checkpoint.resolve():
        raise FileExistsError(
            "an existing run can only resume from its latest complete checkpoint "
            f"({latest_checkpoint}); choose a new run_name to branch from "
            f"{resume_checkpoint}"
        )


def resolve_resume_checkpoint(value: str | None, run_directory: Path) -> Path | None:
    if value is None:
        return None
    if value != "latest":
        path = Path(value).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
        if not (path / "trainer_state.json").is_file():
            raise ValueError(
                "explicit resume checkpoint lacks the final completeness marker "
                f"trainer_state.json: {path}"
            )
        return path

    candidates: list[tuple[int, Path]] = []
    if run_directory.is_dir():
        for path in run_directory.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if not (
                match
                and path.is_dir()
                and (path / "trainer_state.json").is_file()
                and (path / "resume_contract.json").is_file()
            ):
                continue
            try:
                state = read_checkpoint_state(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            step = int(match.group(1))
            if state["global_step"] == step:
                candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(
            f"resume_from_checkpoint=latest but no checkpoint exists in {run_directory}"
        )
    return max(candidates)[1]


def read_checkpoint_state(path: Path) -> dict[str, int]:
    state_path = path / "trainer_state.json"
    if state_path.is_file():
        with state_path.open(encoding="utf-8") as source:
            raw = json.load(source)
        value = raw.get("global_step")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid global_step in {state_path}")
        consumed = raw.get("consumed_micro_batches")
        if consumed is not None and (
            not isinstance(consumed, int)
            or isinstance(consumed, bool)
            or consumed < 0
        ):
            raise ValueError(f"invalid consumed_micro_batches in {state_path}")
        result = {"global_step": value}
        if consumed is not None:
            result["consumed_micro_batches"] = consumed
        return result
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return {"global_step": int(match.group(1))}
    raise ValueError(f"cannot determine global step from checkpoint {path}")


def save_checkpoint(
    accelerator: Accelerator,
    run_directory: Path,
    global_step: int,
    consumed_micro_batches: int,
    config: MOPDConfig,
    artifact_manifests: dict[str, Any],
) -> None:
    # Never let a source-file mutation become the new provenance baseline.
    # Resume contracts inherit the original checkpoint's fixed manifest.
    validate_artifact_manifests(
        config,
        artifact_manifests,
        source="the run's fixed artifact baseline",
    )
    checkpoint_directory = run_directory / f"checkpoint-{global_step}"
    staging_directory = run_directory / f".checkpoint-{global_step}.incomplete"
    collision = torch.tensor(
        int(checkpoint_directory.exists() or staging_directory.exists()),
        dtype=torch.int32,
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        collision = accelerator.reduce(collision, reduction="max")
    if bool(collision.item()):
        raise FileExistsError(
            "refusing to overwrite an existing complete or incomplete checkpoint: "
            f"{checkpoint_directory}"
        )

    accelerator.save_state(str(staging_directory))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        contract_path = staging_directory / "resume_contract.json"
        temporary_contract_path = staging_directory / "resume_contract.json.tmp"
        with temporary_contract_path.open("w", encoding="utf-8") as output:
            json.dump(
                {
                    "version": 1,
                    "config": config.to_dict(),
                    "world_size": accelerator.num_processes,
                    "artifacts": artifact_manifests,
                },
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
        temporary_contract_path.replace(contract_path)

        # This file is deliberately written last.  Its presence is the atomic
        # marker that both the Accelerate state and the resume contract exist.
        state_path = staging_directory / "trainer_state.json"
        temporary_path = staging_directory / "trainer_state.json.tmp"
        with temporary_path.open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(
                {
                    "global_step": global_step,
                    "consumed_micro_batches": consumed_micro_batches,
                },
                output,
                indent=2,
            )
            output.write("\n")
        temporary_path.replace(state_path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        staging_directory.rename(checkpoint_directory)
    accelerator.wait_for_everyone()


def save_final_model(
    accelerator: Accelerator,
    model,
    tokenizer,
    run_directory: Path,
) -> None:
    final_directory = run_directory / "final"
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        final_directory,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_directory)


def _teacher_device(config: MOPDConfig, accelerator: Accelerator) -> torch.device:
    value = config.model.teacher_device
    if value == "student":
        return accelerator.device
    return torch.device(value)


def _generation_context(accelerator: Accelerator, model):
    # A dedicated helper keeps a future ZeRO/FSDP gather implementation local.
    return nullcontext(accelerator.unwrap_model(model))


def _run_batch(
    *,
    model,
    teacher_pool: LocalTeacherPool,
    tokenizer,
    records: list[dict[str, Any]],
    config: MOPDConfig,
    accelerator: Accelerator,
):
    domains = [str(record["domain"]) for record in records]
    unknown = set(domains) - set(config.domains)
    if unknown:
        raise ValueError(f"records contain domains without teachers: {sorted(unknown)}")

    prompts = [
        render_prompt(
            tokenizer,
            record,
            max_prompt_length=config.data.max_prompt_length,
            use_chat_template=config.data.use_chat_template,
            enable_thinking=config.algorithm.enable_thinking,
        )
        for record in records
    ]

    was_training = model.training
    model.eval()
    with _generation_context(accelerator, model) as generation_model:
        responses = generate_responses(
            generation_model,
            prompts,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_response_length=config.algorithm.max_response_length,
            temperature=config.algorithm.temperature,
            top_p=config.algorithm.top_p,
            device=accelerator.device,
            synced_gpus=accelerator.num_processes > 1,
        )

    teacher_scores = teacher_pool.score_routed(
        domains,
        prompts,
        responses,
        topk=(
            config.algorithm.topk_k
            if config.algorithm.loss_type == "topk"
            else None
        ),
        output_device=accelerator.device,
    )

    # Student loading has already rejected dropout/BatchNorm/stochastic layers,
    # so eval-mode rollout and train-mode scoring are the same policy.  Restore
    # train mode before the differentiable pass so gradient checkpointing is
    # active for large models.
    if was_training:
        model.train()
    student_response_logits = forward_response_logits(
        model,
        prompts,
        responses,
        pad_token_id=tokenizer.pad_token_id,
        device=accelerator.device,
    )
    student_log_probs = gather_sampled_log_probs(
        student_response_logits,
        responses,
    )
    diagnostic = sampled_mopd_loss(
        student_log_probs,
        teacher_scores.sampled_log_probs,
        advantage_clip=config.algorithm.advantage_clip,
    )

    if config.algorithm.loss_type == "pg":
        loss = diagnostic.loss
    else:
        assert teacher_scores.topk_indices is not None
        assert teacher_scores.topk_log_probs is not None
        loss = corrected_topk_mopd_loss(
            student_response_logits,
            teacher_scores.topk_indices,
            teacher_scores.topk_log_probs,
        ).loss

    return {
        "loss": loss,
        "domains": domains,
        "teacher_log_probs": teacher_scores.sampled_log_probs,
        "student_log_probs": student_log_probs,
        "raw_advantages": diagnostic.raw_advantages,
        "clipped_advantages": diagnostic.clipped_advantages,
        "entropies": categorical_entropy(student_response_logits),
    }


def train(config: MOPDConfig) -> Path:
    run_name, run_directory = configure_run_directory(config)
    accelerator = create_accelerator(config, project_dir=run_directory)
    set_seed(config.training.seed, device_specific=True)
    resume_value = getattr(config.training, "resume_from_checkpoint", None)
    resume_checkpoint = resolve_resume_checkpoint(resume_value, run_directory)

    if resume_checkpoint is not None:
        artifact_manifests = validate_checkpoint_contract(
            config, resume_checkpoint, accelerator
        )
    else:
        artifact_manifests = build_artifact_manifests(config)
    validate_distributed_artifact_consistency(artifact_manifests, accelerator)
    protect_existing_run(run_directory, resume_checkpoint)
    continuing_existing_run = False
    if resume_checkpoint is not None:
        continuing_existing_run = validate_resume_compatibility(
            config, run_directory, accelerator
        )
    # Ensure every rank has completed the read-only overwrite/resume checks
    # before rank 0 can eventually create metadata for this run.
    accelerator.wait_for_everyone()

    dataset = build_training_dataset(config.data, seed=config.training.seed)
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(config.training.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_records,
        pin_memory=torch.cuda.is_available(),
        # Keep worker/iterator seeding off the policy-sampling RNG stream so a
        # resumed CPU run does not perturb torch.multinomial merely by
        # reconstructing its DataLoader iterator.
        generator=data_loader_generator,
    )

    model, tokenizer = load_student_and_tokenizer(config.model)
    validate_context_window(
        model,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.algorithm.max_response_length,
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )
    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        data_loader,
        scheduler,
    )
    if len(data_loader) == 0:
        raise ValueError(
            "prepared training dataloader is empty; provide at least one sample "
            "per distributed worker"
        )
    model.train()

    unwrapped = accelerator.unwrap_model(model)
    output_embeddings = unwrapped.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("student model does not expose causal-LM output embeddings")
    student_vocab_size = int(output_embeddings.weight.shape[0])
    if (
        config.algorithm.loss_type == "topk"
        and config.algorithm.topk_k > student_vocab_size
    ):
        raise ValueError(
            f"algorithm.topk_k={config.algorithm.topk_k} exceeds student "
            f"vocabulary size {student_vocab_size}"
        )
    teacher_pool = LocalTeacherPool(
        config.model.teacher_models,
        device=_teacher_device(config, accelerator),
        torch_dtype=config.model.torch_dtype,
        pad_token_id=tokenizer.pad_token_id,
        student_vocab_size=student_vocab_size,
        student_tokenizer=tokenizer,
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=config.model.attn_implementation,
    )

    global_step = 0
    consumed_micro_batches = 0
    if resume_checkpoint is not None:
        accelerator.load_state(str(resume_checkpoint))
        checkpoint_state = read_checkpoint_state(resume_checkpoint)
        global_step = checkpoint_state["global_step"]
        consumed_micro_batches = checkpoint_state.get(
            "consumed_micro_batches",
            global_step * config.training.gradient_accumulation_steps,
        )
        if global_step > config.training.max_steps:
            raise ValueError(
                f"checkpoint step {global_step} exceeds configured max_steps "
                f"{config.training.max_steps}"
            )
        accelerator.print(f"Resumed {resume_checkpoint} at step {global_step}")

    # Persist experiment metadata only after data, all models, and any resume
    # state have loaded successfully.  A failed path/checkpoint therefore does
    # not poison a run name, and an existing run keeps its original timestamp.
    if accelerator.is_main_process:
        if resume_checkpoint is not None and continuing_existing_run:
            append_resume_metadata(
                config,
                run_directory,
                accelerator,
                resume_checkpoint,
                global_step,
            )
        else:
            write_run_metadata(
                config,
                run_name,
                run_directory,
                accelerator,
                parent_checkpoint=resume_checkpoint,
            )
    accelerator.wait_for_everyone()

    reporters = _reporters(config.training.report_to)
    if reporters:
        accelerator.init_trackers(
            project_name=config.training.wandb_project,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": run_name}} if "wandb" in reporters else None,
        )

    trainable, total = trainable_parameter_summary(unwrapped)
    accelerator.print(
        f"Run: {run_name}\n"
        f"Output: {run_directory}\n"
        f"Objective: {config.algorithm.loss_type}; domains: {', '.join(config.domains)}\n"
        f"Effective global batch: "
        f"{config.training.per_device_train_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps}\n"
        f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)"
    )

    metric_accumulator = MOPDMetricAccumulator(config.domains)
    optimizer.zero_grad(set_to_none=True)
    data_iterator = iter(data_loader)
    if resume_checkpoint is not None and len(data_loader) > 0:
        for _ in range(consumed_micro_batches % len(data_loader)):
            next(data_iterator)
    while global_step < config.training.max_steps:
        try:
            records = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            records = next(data_iterator)

        with accelerator.accumulate(model):
            batch_output = _run_batch(
                model=model,
                teacher_pool=teacher_pool,
                tokenizer=tokenizer,
                records=records,
                config=config,
                accelerator=accelerator,
            )
            loss = batch_output["loss"]
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )
            optimizer.step()
            if accelerator.sync_gradients and not optimizer.step_was_skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metric_accumulator.update(
            domains=batch_output["domains"],
            teacher_log_probs=batch_output["teacher_log_probs"],
            student_log_probs=batch_output["student_log_probs"],
            raw_advantages=batch_output["raw_advantages"],
            clipped_advantages=batch_output["clipped_advantages"],
            entropies=batch_output["entropies"],
            loss=loss,
            advantage_clip=config.algorithm.advantage_clip,
        )
        consumed_micro_batches += 1

        if not accelerator.sync_gradients:
            continue
        global_step += 1

        if global_step % config.training.logging_steps == 0:
            metrics = metric_accumulator.distributed_metrics(accelerator)
            metrics["train/learning_rate"] = scheduler.get_last_lr()[0]
            if reporters:
                accelerator.log(metrics, step=global_step)
            accelerator.print(
                json.dumps(
                    {"step": global_step, **metrics},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            metric_accumulator.reset()

        if global_step % config.training.save_steps == 0:
            save_checkpoint(
                accelerator,
                run_directory,
                global_step,
                consumed_micro_batches,
                config,
                artifact_manifests,
            )

    validate_artifact_manifests(
        config,
        artifact_manifests,
        source="the run's fixed artifact baseline",
    )
    save_final_model(accelerator, model, tokenizer, run_directory)
    if reporters:
        accelerator.end_training()
    return run_directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce classic multi-teacher on-policy distillation"
    )
    parser.add_argument("config_path", nargs="?", help="path to experiment YAML")
    parser.add_argument("--config", dest="config_option", help="path to experiment YAML")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate and print the resolved YAML without loading data/models",
    )
    args = parser.parse_args(argv)
    if bool(args.config_path) == bool(args.config_option):
        parser.error("provide exactly one of positional CONFIG or --config CONFIG")
    args.config = args.config_option or args.config_path
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.check_config:
        print(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )
    train(config)


if __name__ == "__main__":
    main()

