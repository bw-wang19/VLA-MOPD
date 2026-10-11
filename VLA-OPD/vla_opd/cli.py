"""Command-line entry point for the standalone VLA-OPD trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
import yaml

from .config import ConfigError, VLAOPDConfig, load_config
from .rollout import GroupRolloutCollector
from .runtime import load_experiment
from .trainer import VLAOPDTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train categorical VLA-OPD")
    parser.add_argument("config", type=Path, help="YAML experiment configuration")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate and print the resolved configuration without loading models",
    )
    return parser


def _save_checkpoint(
    path: Path,
    *,
    update: int,
    config: VLAOPDConfig,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "algorithm": "vla_opd_sampled_reverse_kl_pg",
            "update": update,
            "config": config.to_dict(),
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def run(config: VLAOPDConfig) -> Path:
    torch.manual_seed(config.training.seed)
    random_generator = random.Random(config.training.seed)
    components = load_experiment(config)
    optimizer = torch.optim.AdamW(
        components.student.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    collector = GroupRolloutCollector(
        components.env_factory,
        group_size=config.algorithm.group_size,
        max_episode_steps=config.algorithm.max_episode_steps,
        temperature=config.algorithm.student_temperature,
        seed=config.training.seed,
    )
    trainer = VLAOPDTrainer(
        student=components.student,
        teacher=components.teacher,
        optimizer=optimizer,
        collector=collector,
        teacher_temperature=config.algorithm.teacher_temperature,
        reduction=config.algorithm.reduction,
        reward_clip=config.algorithm.reward_clip,
        on_policy_logprob_tolerance=config.algorithm.on_policy_logprob_tolerance,
        max_grad_norm=config.training.max_grad_norm,
        score_batch_size=config.algorithm.score_batch_size,
    )

    output_dir = Path(config.training.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "implementation": "standalone VLA-OPD algorithm reproduction",
        "paper": "arXiv:2603.26666v1",
        "paper_objective": config.algorithm.uses_paper_objective,
        "environment_reward_used_for_optimization": False,
        "critic_or_group_relative_normalization": False,
        "optimizer_epochs_per_rollout_batch": 1,
        "reported_by_paper": {"batch_size": 64, "group_size": 8},
        "warning": (
            "optimizer and most rollout settings are implementation defaults; "
            "the paper does not report a complete numerical recipe"
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log_path = output_dir / "metrics.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for update in range(1, config.training.num_updates + 1):
            prompts = [
                components.prompts[
                    random_generator.randrange(len(components.prompts))
                ]
                for _ in range(config.algorithm.prompt_batch_size)
            ]
            result = trainer.train_step(prompts)
            record = {"update": update, **result.metrics}
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
            log_file.flush()
            if update % config.training.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            if update % config.training.save_every == 0:
                _save_checkpoint(
                    output_dir / f"checkpoint-{update}.pt",
                    update=update,
                    config=config,
                    student=components.student,
                    optimizer=optimizer,
                )

    _save_checkpoint(
        output_dir / "final.pt",
        update=config.training.num_updates,
        config=config,
        student=components.student,
        optimizer=optimizer,
    )
    return output_dir


def main() -> None:
    args = _parser().parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    if args.check_config:
        print(yaml.safe_dump(config.to_dict(), sort_keys=False), end="")
        return
    output_dir = run(config)
    print(f"VLA-OPD run complete: {output_dir}")


if __name__ == "__main__":
    main()
