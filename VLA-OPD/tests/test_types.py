from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from vla_opd.types import (
    ActionSample,
    LabeledRolloutBatch,
    LabeledTrajectory,
    RolloutBatch,
    StepResult,
    Trajectory,
    Transition,
    clone_snapshot,
)


def _transition(
    *,
    prompt_index: int = 0,
    group_index: int = 0,
    step_index: int = 0,
    observation: object | None = None,
    env_reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
) -> Transition:
    return Transition(
        prompt_index=prompt_index,
        group_index=group_index,
        step_index=step_index,
        observation={} if observation is None else observation,
        action_tokens=torch.tensor([3, 4]),
        env_action=torch.tensor([0.25]),
        behavior_log_probs=torch.tensor([-0.2, -0.7]),
        token_mask=torch.tensor([True, True]),
        env_reward=env_reward,
        terminated=terminated,
        truncated=truncated,
        info={},
    )


def _trajectory(*, prompt_index: int = 0, group_index: int = 0) -> Trajectory:
    return Trajectory(
        prompt="pick the block",
        prompt_index=prompt_index,
        group_index=group_index,
        transitions=(
            _transition(prompt_index=prompt_index, group_index=group_index),
        ),
        final_observation={"state": torch.tensor([1.0])},
    )


def test_clone_snapshot_breaks_aliases_and_detaches_nested_tensors() -> None:
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    source = {"camera": tensor, "history": [{"x": tensor}]}

    snapshot = clone_snapshot(source)
    with torch.no_grad():
        tensor.fill_(99.0)
    source["history"].append({"x": torch.tensor([3.0])})

    torch.testing.assert_close(snapshot["camera"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(snapshot["history"][0]["x"], torch.tensor([1.0, 2.0]))
    assert len(snapshot["history"]) == 1
    assert not snapshot["camera"].requires_grad


def test_action_sample_validates_token_alignment_mask_and_finiteness() -> None:
    valid = ActionSample(
        tokens=torch.tensor([1, 2]),
        log_probs=torch.tensor([-0.1, -0.2]),
        mask=torch.tensor([True, False]),
        env_action=0,
    )
    assert valid.tokens.shape == valid.log_probs.shape == valid.mask.shape

    with pytest.raises(ValueError, match="align"):
        ActionSample(
            tokens=torch.tensor([1, 2]),
            log_probs=torch.tensor([-0.1]),
            mask=torch.tensor([True, True]),
            env_action=0,
        )
    with pytest.raises(ValueError, match="boolean"):
        ActionSample(
            tokens=torch.tensor([1]),
            log_probs=torch.tensor([-0.1]),
            mask=torch.tensor([1]),
            env_action=0,
        )
    with pytest.raises(ValueError, match="valid token"):
        ActionSample(
            tokens=torch.tensor([1]),
            log_probs=torch.tensor([-0.1]),
            mask=torch.tensor([False]),
            env_action=0,
        )
    with pytest.raises(ValueError, match="finite"):
        ActionSample(
            tokens=torch.tensor([1]),
            log_probs=torch.tensor([float("inf")]),
            mask=torch.tensor([True]),
            env_action=0,
        )


def test_step_result_supports_independent_terminated_and_truncated_flags() -> None:
    terminated = StepResult(observation={}, terminated=True, truncated=False)
    truncated = StepResult(observation={}, terminated=False, truncated=True)
    assert terminated.terminated and not terminated.truncated
    assert truncated.truncated and not truncated.terminated

    with pytest.raises(TypeError, match="booleans"):
        StepResult(observation={}, terminated=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        StepResult(observation={}, reward=float("nan"))


def test_transition_and_trajectory_preserve_prompt_group_step_identity() -> None:
    first = _transition(step_index=0)
    second = _transition(step_index=1, terminated=True)
    trajectory = Trajectory(
        prompt="pick",
        prompt_index=0,
        group_index=0,
        transitions=(first, second),
        final_observation={},
    )
    assert trajectory.terminated
    assert not trajectory.truncated

    with pytest.raises(ValueError, match="contiguous"):
        Trajectory(
            prompt="pick",
            prompt_index=0,
            group_index=0,
            transitions=(first, _transition(step_index=2)),
            final_observation={},
        )
    with pytest.raises(ValueError, match="prompt index"):
        Trajectory(
            prompt="pick",
            prompt_index=1,
            group_index=0,
            transitions=(first,),
            final_observation={},
        )


def test_transition_validates_reward_and_episode_flag_types() -> None:
    with pytest.raises(ValueError, match="finite"):
        _transition(env_reward=float("nan"))
    with pytest.raises(TypeError, match="booleans"):
        _transition(terminated=1)  # type: ignore[arg-type]


def test_frozen_records_disallow_field_reassignment() -> None:
    transition = _transition()
    with pytest.raises(FrozenInstanceError):
        transition.step_index = 2  # type: ignore[misc]


def test_rollout_batch_requires_complete_prompt_group_grid() -> None:
    trajectories = tuple(
        _trajectory(prompt_index=prompt_index, group_index=group_index)
        for prompt_index in range(2)
        for group_index in range(2)
    )
    batch = RolloutBatch(trajectories=trajectories, prompt_count=2, group_size=2)
    assert len(batch.trajectories) == 4

    with pytest.raises(ValueError, match="B x G"):
        RolloutBatch(trajectories=trajectories[:-1], prompt_count=2, group_size=2)
    duplicate = trajectories[:-1] + (trajectories[0],)
    with pytest.raises(ValueError, match="every prompt/group pair"):
        RolloutBatch(trajectories=duplicate, prompt_count=2, group_size=2)


def test_teacher_labels_are_finite_and_align_to_each_step() -> None:
    trajectory = _trajectory()
    labeled = LabeledTrajectory(
        trajectory=trajectory,
        teacher_log_probs=(torch.tensor([-0.3, -0.4]),),
    )
    batch = LabeledRolloutBatch(
        trajectories=(labeled,), prompt_count=1, group_size=1
    )
    assert batch.trajectories[0].teacher_log_probs[0].shape == torch.Size([2])

    with pytest.raises(ValueError, match="align"):
        LabeledTrajectory(
            trajectory=trajectory,
            teacher_log_probs=(torch.tensor([-0.3]),),
        )
    with pytest.raises(ValueError, match="finite"):
        LabeledTrajectory(
            trajectory=trajectory,
            teacher_log_probs=(torch.tensor([float("nan"), -0.4]),),
        )
