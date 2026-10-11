from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml

from vla_opd import cli
from vla_opd.config import (
    AlgorithmConfig,
    RuntimeConfig,
    TrainingConfig,
    VLAOPDConfig,
)
from vla_opd.rollout import GroupRolloutCollector
from vla_opd.toy import (
    TinyCategoricalStudent,
    ToyExpertTeacher,
    ToyObservation,
    ToyVLAEnvironment,
)
from vla_opd.trainer import VLAOPDTrainer
from vla_opd.types import StepResult


class CountingSGD(torch.optim.SGD):
    """SGD with an observable step count for update-contract tests."""

    def __init__(self, params: Any, *, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        return super().step(closure)


class TrackingToyEnvironment(ToyVLAEnvironment):
    def __init__(
        self,
        *,
        horizon: int,
        vocab_size: int,
        reset_seeds: list[int],
    ) -> None:
        super().__init__(horizon=horizon, vocab_size=vocab_size)
        self._reset_seeds = reset_seeds

    def reset(self, prompt: Any, *, seed: int):  # type: ignore[no-untyped-def]
        self._reset_seeds.append(seed)
        return super().reset(prompt, seed=seed)


class FixedRewardToyEnvironment(ToyVLAEnvironment):
    def __init__(
        self,
        *,
        horizon: int,
        vocab_size: int,
        fixed_reward: float,
    ) -> None:
        super().__init__(horizon=horizon, vocab_size=vocab_size)
        self._fixed_reward = fixed_reward

    def step(self, action: Any) -> StepResult:
        result = super().step(action)
        return replace(result, reward=self._fixed_reward)


class DriftingStudent(TinyCategoricalStudent):
    """Reports a different policy during differentiable rescoring."""

    def score_actions(
        self,
        observations: list[ToyObservation],
        action_tokens: list[torch.Tensor],
        *,
        temperature: float,
    ) -> list[torch.Tensor]:
        scored = super().score_actions(
            observations, action_tokens, temperature=temperature
        )
        return [item + 0.25 for item in scored]


class ParametricTeacher(torch.nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.5, 0.5, vocab_size))

    def score_actions(
        self,
        observations: list[ToyObservation],
        action_tokens: list[torch.Tensor],
        *,
        temperature: float,
    ) -> list[torch.Tensor]:
        del observations
        log_probs = F.log_softmax(self.logits / temperature, dim=-1)
        return [
            log_probs.gather(0, tokens.to(dtype=torch.long, device=log_probs.device))
            for tokens in action_tokens
        ]


def _student(*, cls: type[TinyCategoricalStudent] = TinyCategoricalStudent):
    return cls(num_prompts=4, horizon=2, vocab_size=3)


def _collector(
    env_factory: Any,
    *,
    group_size: int = 2,
    seed: int = 17,
) -> GroupRolloutCollector:
    return GroupRolloutCollector(
        env_factory,
        group_size=group_size,
        max_episode_steps=2,
        temperature=1.0,
        seed=seed,
    )


def _trainer(
    *,
    student: TinyCategoricalStudent,
    teacher: Any,
    optimizer: torch.optim.Optimizer,
    collector: GroupRolloutCollector,
    tolerance: float = 1.0e-6,
) -> VLAOPDTrainer:
    return VLAOPDTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        collector=collector,
        teacher_temperature=1.0,
        on_policy_logprob_tolerance=tolerance,
    )


def test_each_train_step_collects_a_fresh_batch_and_updates_once() -> None:
    torch.manual_seed(1)
    reset_seeds: list[int] = []
    created_environments: list[ToyVLAEnvironment] = []

    def env_factory(
        *, prompt: Any, prompt_index: int, group_index: int
    ) -> ToyVLAEnvironment:
        del prompt, prompt_index, group_index
        environment = TrackingToyEnvironment(
            horizon=2, vocab_size=3, reset_seeds=reset_seeds
        )
        created_environments.append(environment)
        return environment

    student = _student()
    optimizer = CountingSGD(student.parameters(), lr=0.02)
    trainer = _trainer(
        student=student,
        teacher=ToyExpertTeacher(vocab_size=3),
        optimizer=optimizer,
        collector=_collector(env_factory),
    )

    first = trainer.train_step([0, 1])
    assert optimizer.step_calls == 1
    assert trainer.optimizer_steps == 1
    assert first.metrics["optimizer_steps"] == 1.0

    second = trainer.train_step([0, 1])
    assert optimizer.step_calls == 2
    assert trainer.optimizer_steps == 2
    assert second.metrics["optimizer_steps"] == 2.0
    assert first.rollouts is not second.rollouts

    trajectories_per_batch = 2 * 2
    assert len(created_environments) == 2 * trajectories_per_batch
    assert len({id(item) for item in created_environments}) == len(
        created_environments
    )
    assert set(reset_seeds[:trajectories_per_batch]).isdisjoint(
        reset_seeds[trajectories_per_batch:]
    )


def _single_update_with_reward(
    student: TinyCategoricalStudent, *, reward: float
) -> tuple[torch.Tensor, dict[str, float], tuple[tuple[int, ...], ...]]:
    def env_factory(
        *, prompt: Any, prompt_index: int, group_index: int
    ) -> ToyVLAEnvironment:
        del prompt, prompt_index, group_index
        return FixedRewardToyEnvironment(
            horizon=2,
            vocab_size=3,
            fixed_reward=reward,
        )

    optimizer = CountingSGD(student.parameters(), lr=0.03)
    trainer = _trainer(
        student=student,
        teacher=ToyExpertTeacher(vocab_size=3),
        optimizer=optimizer,
        collector=_collector(env_factory, seed=101),
    )
    result = trainer.train_step([0, 1])
    sampled_tokens = tuple(
        tuple(
            int(token)
            for transition in trajectory.transitions
            for token in transition.action_tokens.tolist()
        )
        for trajectory in result.rollouts.trajectories
    )
    return result.loss, result.metrics, sampled_tokens


def test_environment_reward_is_diagnostic_only_and_does_not_change_loss() -> None:
    torch.manual_seed(2)
    initial = _student()
    negative_reward_student = copy.deepcopy(initial)
    positive_reward_student = copy.deepcopy(initial)

    negative_loss, negative_metrics, negative_tokens = _single_update_with_reward(
        negative_reward_student, reward=-1000.0
    )
    positive_loss, positive_metrics, positive_tokens = _single_update_with_reward(
        positive_reward_student, reward=1000.0
    )

    assert negative_tokens == positive_tokens
    torch.testing.assert_close(negative_loss, positive_loss, rtol=0.0, atol=0.0)
    for negative, positive in zip(
        negative_reward_student.parameters(),
        positive_reward_student.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(negative, positive, rtol=0.0, atol=0.0)
    assert negative_metrics["environment_reward_mean_diagnostic_only"] == -1000.0
    assert positive_metrics["environment_reward_mean_diagnostic_only"] == 1000.0


def test_on_policy_logprob_drift_fails_before_optimizer_step() -> None:
    torch.manual_seed(3)
    student = _student(cls=DriftingStudent)
    before = [parameter.detach().clone() for parameter in student.parameters()]

    def env_factory(
        *, prompt: Any, prompt_index: int, group_index: int
    ) -> ToyVLAEnvironment:
        del prompt, prompt_index, group_index
        return ToyVLAEnvironment(horizon=2, vocab_size=3)

    optimizer = CountingSGD(student.parameters(), lr=0.1)
    trainer = _trainer(
        student=student,
        teacher=ToyExpertTeacher(vocab_size=3),
        optimizer=optimizer,
        collector=_collector(env_factory, group_size=1),
        tolerance=1.0e-4,
    )

    with pytest.raises(RuntimeError, match="rollout/rescore.*drifted"):
        trainer.train_step([0])

    assert optimizer.step_calls == 0
    assert trainer.optimizer_steps == 0
    for original, current in zip(before, student.parameters(), strict=True):
        torch.testing.assert_close(original, current, rtol=0.0, atol=0.0)


def test_teacher_module_is_frozen_and_unchanged_by_training() -> None:
    torch.manual_seed(4)
    student = _student()
    teacher = ParametricTeacher(vocab_size=3)
    teacher_before = teacher.logits.detach().clone()

    def env_factory(
        *, prompt: Any, prompt_index: int, group_index: int
    ) -> ToyVLAEnvironment:
        del prompt, prompt_index, group_index
        return ToyVLAEnvironment(horizon=2, vocab_size=3)

    trainer = _trainer(
        student=student,
        teacher=teacher,
        optimizer=CountingSGD(student.parameters(), lr=0.02),
        collector=_collector(env_factory, group_size=1),
    )

    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    trainer.train_step([0])
    assert teacher.logits.grad is None
    torch.testing.assert_close(teacher.logits, teacher_before, rtol=0.0, atol=0.0)


def _config(
    output_dir: Path,
    *,
    runtime_factory: str = "vla_opd.toy:build_experiment",
) -> VLAOPDConfig:
    return VLAOPDConfig(
        algorithm=AlgorithmConfig(
            group_size=2,
            prompt_batch_size=2,
            max_episode_steps=2,
            student_temperature=1.0,
            teacher_temperature=1.0,
            reduction="trajectory_sum",
            reward_clip=None,
            on_policy_logprob_tolerance=1.0e-6,
        ),
        training=TrainingConfig(
            num_updates=2,
            learning_rate=0.01,
            weight_decay=0.0,
            max_grad_norm=1.0,
            seed=23,
            output_dir=str(output_dir),
            log_every=1,
            save_every=1,
        ),
        runtime=RuntimeConfig(
            factory=runtime_factory,
            kwargs={"num_prompts": 4, "action_vocab_size": 3}
            if runtime_factory == "vla_opd.toy:build_experiment"
            else {},
        ),
    )


def test_toy_run_writes_metadata_logs_and_checkpoints(tmp_path: Path) -> None:
    output_dir = tmp_path / "toy-run"
    resolved = cli.run(_config(output_dir))

    assert resolved == output_dir.resolve()
    assert (resolved / "resolved_config.yaml").is_file()

    metadata = json.loads(
        (resolved / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["implementation"] == "standalone VLA-OPD algorithm reproduction"
    assert metadata["paper"] == "arXiv:2603.26666v1"
    assert metadata["paper_objective"] is True
    assert metadata["environment_reward_used_for_optimization"] is False
    assert metadata["optimizer_epochs_per_rollout_batch"] == 1

    records = [
        json.loads(line)
        for line in (resolved / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["update"] for record in records] == [1, 2]
    assert [record["optimizer_steps"] for record in records] == [1.0, 2.0]
    assert all("environment_reward_mean_diagnostic_only" in record for record in records)

    assert (resolved / "checkpoint-1.pt").is_file()
    assert (resolved / "checkpoint-2.pt").is_file()
    final = torch.load(resolved / "final.pt", map_location="cpu", weights_only=False)
    assert final["format_version"] == 1
    assert final["algorithm"] == "vla_opd_sampled_reverse_kl_pg"
    assert final["update"] == 2
    assert final["student"]
    assert final["optimizer"]


def test_check_config_does_not_load_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(
        tmp_path / "unused",
        runtime_factory="module_that_must_not_be_imported:factory",
    )
    config_path = tmp_path / "check-only.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    def forbidden_runtime_load(config: VLAOPDConfig) -> Any:
        del config
        raise AssertionError("--check-config must not load the runtime")

    monkeypatch.setattr(cli, "load_experiment", forbidden_runtime_load)
    monkeypatch.setattr(
        sys, "argv", ["vla-opd", str(config_path), "--check-config"]
    )
    cli.main()

    printed = capsys.readouterr().out
    assert "module_that_must_not_be_imported:factory" in printed
    assert "group_size: 2" in printed
