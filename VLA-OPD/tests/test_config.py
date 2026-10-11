from pathlib import Path

import pytest

from vla_opd.config import ConfigError, load_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_strict_config_and_marks_paper_objective(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
algorithm:
  group_size: 8
  prompt_batch_size: 64
  max_episode_steps: 512
  student_temperature: 1.0
  teacher_temperature: 1.0
  reduction: trajectory_sum
  reward_clip: null
  on_policy_logprob_tolerance: 0.0001
training:
  num_updates: 2
  learning_rate: 0.00001
  weight_decay: 0.0
  max_grad_norm: 1.0
  seed: 7
  output_dir: out
  log_every: 1
  save_every: 1
runtime:
  factory: vla_opd.toy:build_experiment
  kwargs: {}
""",
    )
    config = load_config(path)
    assert config.algorithm.group_size == 8
    assert config.algorithm.uses_paper_objective
    assert config.to_dict()["runtime"]["factory"].endswith("build_experiment")


def test_rejects_unknown_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
algorithm: {group_size: 8, prompt_batch_size: 64, max_episode_steps: 4,
  student_temperature: 1.0, teacher_temperature: 1.0,
  reduction: trajectory_sum, reward_clip: null,
  on_policy_logprob_tolerance: 0.001, grpo_normalize: true}
training: {num_updates: 1, learning_rate: 0.001, weight_decay: 0,
  max_grad_norm: 1, seed: 1, output_dir: out, log_every: 1, save_every: 1}
runtime: {factory: "vla_opd.toy:build_experiment", kwargs: {}}
""",
    )
    with pytest.raises(ConfigError, match="grpo_normalize"):
        load_config(path)


def test_extension_is_explicit(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
algorithm: {group_size: 2, prompt_batch_size: 2, max_episode_steps: 4,
  student_temperature: 1.0, teacher_temperature: 1.0,
  reduction: global_token_mean, reward_clip: 5,
  on_policy_logprob_tolerance: 0.001}
training: {num_updates: 1, learning_rate: 0.001, weight_decay: 0,
  max_grad_norm: null, seed: 1, output_dir: out,
  log_every: 1, save_every: 1}
runtime: {factory: "vla_opd.toy:build_experiment", kwargs: {}}
""",
    )
    assert not load_config(path).algorithm.uses_paper_objective


def test_trajectory_token_mean_is_a_valid_nonpaper_reduction(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
algorithm: {group_size: 2, prompt_batch_size: 2, max_episode_steps: 4,
  student_temperature: 1.0, teacher_temperature: 1.0,
  reduction: trajectory_token_mean, reward_clip: null,
  on_policy_logprob_tolerance: 0.001}
training: {num_updates: 1, learning_rate: 0.001, weight_decay: 0,
  max_grad_norm: null, seed: 1, output_dir: out,
  log_every: 1, save_every: 1}
runtime: {factory: "vla_opd.toy:build_experiment", kwargs: {}}
""",
    )
    config = load_config(path)
    assert config.algorithm.reduction == "trajectory_token_mean"
    assert not config.algorithm.uses_paper_objective
