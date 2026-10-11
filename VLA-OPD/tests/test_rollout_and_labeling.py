from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest
import torch
from torch import nn

from vla_opd.labeling import label_with_teacher
from vla_opd.rollout import GroupRolloutCollector
from vla_opd.types import ActionSample, ResetResult, StepResult


class ClosedLoopStudent:
    """A stochastic policy whose next action depends on the visited state."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.seen_states: list[int] = []

    def sample_action(
        self,
        observation: dict[str, Any],
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> ActionSample:
        assert temperature == pytest.approx(0.75)
        self.sample_count += 1
        state = int(observation["state"].item())
        self.seen_states.append(state)
        random_offset = int(torch.randint(1, 5, (), generator=generator).item())
        env_action = state + random_offset
        return ActionSample(
            tokens=torch.tensor([env_action, 999], dtype=torch.long),
            log_probs=torch.tensor([-0.25, -7.0], dtype=torch.float32),
            mask=torch.tensor([True, False]),
            env_action=env_action,
        )


class MutatingClosedLoopEnv:
    """Reuses one observation object so collector snapshots are exercised."""

    def __init__(self, *, horizon: int, prompt_index: int, group_index: int) -> None:
        self.horizon = horizon
        self.prompt_index = prompt_index
        self.group_index = group_index
        self.live_observation: dict[str, Any] = {}
        self.received_actions: list[int] = []
        self.reset_seed: int | None = None
        self.closed = False

    def reset(self, prompt: Any, *, seed: int) -> ResetResult:
        self.reset_seed = seed
        self.live_observation = {
            "state": torch.tensor(int(prompt), dtype=torch.long),
            "history": [],
        }
        return ResetResult(self.live_observation, {"seed": seed})

    def step(self, action: Any) -> StepResult:
        if self.closed:
            raise AssertionError("step called after close")
        action = int(action)
        self.received_actions.append(action)
        # Both nested values are mutated in place. Earlier Transition objects
        # must nevertheless retain the values visible before this step.
        self.live_observation["state"].fill_(action)
        self.live_observation["history"].append(action)
        return StepResult(
            observation=self.live_observation,
            reward=float(action % 2 == 0),
            terminated=len(self.received_actions) >= self.horizon,
            info={"action": action},
        )

    def close(self) -> None:
        self.closed = True


@dataclass
class RecordingFactory:
    horizon: int

    def __post_init__(self) -> None:
        self.created: list[MutatingClosedLoopEnv] = []

    def __call__(
        self, *, prompt: Any, prompt_index: int, group_index: int
    ) -> MutatingClosedLoopEnv:
        env = MutatingClosedLoopEnv(
            horizon=self.horizon,
            prompt_index=prompt_index,
            group_index=group_index,
        )
        self.created.append(env)
        return env


def _collector(
    factory: RecordingFactory,
    *,
    group_size: int = 1,
    max_steps: int = 8,
    seed: int = 123,
) -> GroupRolloutCollector:
    return GroupRolloutCollector(
        factory,
        group_size=group_size,
        max_episode_steps=max_steps,
        temperature=0.75,
        seed=seed,
    )


def _flatten_tokens(batch: Any) -> list[torch.Tensor]:
    return [
        transition.action_tokens
        for trajectory in batch.trajectories
        for transition in trajectory.transitions
    ]


def test_collects_b_times_g_with_independent_seeded_environments() -> None:
    prompts = [0, 10]
    factory = RecordingFactory(horizon=2)
    student = ClosedLoopStudent()
    batch = _collector(factory, group_size=3, seed=500).collect(prompts, student)

    assert batch.prompt_count == 2
    assert batch.group_size == 3
    assert len(batch.trajectories) == 2 * 3
    assert [
        (trajectory.prompt_index, trajectory.group_index)
        for trajectory in batch.trajectories
    ] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert len(factory.created) == 6
    assert len({id(env) for env in factory.created}) == 6
    assert [env.reset_seed for env in factory.created] == list(range(500, 506))
    assert all(env.closed for env in factory.created)

    # A separately constructed collector with the same seed reproduces every
    # sampled token, despite each group member using an independent generator.
    replay_factory = RecordingFactory(horizon=2)
    replay = _collector(replay_factory, group_size=3, seed=500).collect(
        prompts, ClosedLoopStudent()
    )
    assert len(_flatten_tokens(replay)) == len(_flatten_tokens(batch))
    for original, reproduced in zip(
        _flatten_tokens(batch), _flatten_tokens(replay), strict=True
    ):
        torch.testing.assert_close(original, reproduced)


def test_rollout_is_closed_loop_and_snapshots_survive_environment_mutation() -> None:
    factory = RecordingFactory(horizon=3)
    student = ClosedLoopStudent()
    batch = _collector(factory, seed=9).collect([0], student)
    trajectory = batch.trajectories[0]
    env = factory.created[0]

    stored_states = [
        int(transition.observation["state"].item())
        for transition in trajectory.transitions
    ]
    stored_actions = [
        int(transition.env_action) for transition in trajectory.transitions
    ]
    assert stored_states == student.seen_states
    assert stored_states[0] == 0
    assert stored_states[1:] == stored_actions[:-1]
    assert env.received_actions == stored_actions

    # All reset/step observations alias in the fake env. Mutating it after the
    # rollout must not rewrite any historical observation or final snapshot.
    expected_states = list(stored_states)
    expected_histories = [
        list(transition.observation["history"])
        for transition in trajectory.transitions
    ]
    expected_final = int(trajectory.final_observation["state"].item())
    env.live_observation["state"].fill_(100_000)
    env.live_observation["history"].append(100_000)
    assert [
        int(transition.observation["state"].item())
        for transition in trajectory.transitions
    ] == expected_states
    assert [
        transition.observation["history"]
        for transition in trajectory.transitions
    ] == expected_histories
    assert int(trajectory.final_observation["state"].item()) == expected_final


def test_terminal_transition_keeps_token_mask_and_stops_sampling() -> None:
    factory = RecordingFactory(horizon=2)
    student = ClosedLoopStudent()
    trajectory = _collector(factory, max_steps=20).collect([0], student).trajectories[0]

    assert len(trajectory.transitions) == 2
    assert student.sample_count == 2
    assert len(factory.created[0].received_actions) == 2
    assert not trajectory.transitions[0].terminated
    assert trajectory.transitions[-1].terminated
    assert not trajectory.transitions[-1].truncated
    assert torch.equal(
        trajectory.transitions[-1].token_mask,
        torch.tensor([True, False]),
    )
    assert torch.equal(
        trajectory.transitions[-1].action_tokens,
        torch.tensor([trajectory.transitions[-1].env_action, 999]),
    )
    torch.testing.assert_close(
        trajectory.transitions[-1].behavior_log_probs,
        torch.tensor([-0.25, -7.0]),
    )


def test_collector_marks_its_own_horizon_as_truncation() -> None:
    factory = RecordingFactory(horizon=100)
    student = ClosedLoopStudent()
    trajectory = _collector(factory, max_steps=2).collect([0], student).trajectories[0]

    assert len(trajectory.transitions) == 2
    assert student.sample_count == 2
    assert not trajectory.transitions[-1].terminated
    assert trajectory.transitions[-1].truncated


class RecordingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.received_observations: list[Any] = []
        self.received_actions: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        self.training_flags: list[bool] = []

    def score_actions(
        self,
        observations: Sequence[Any],
        action_tokens: Sequence[torch.Tensor],
        *,
        temperature: float,
    ) -> list[torch.Tensor]:
        assert temperature == pytest.approx(0.5)
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_flags.append(self.training)
        self.received_observations.extend(observations)
        self.received_actions.extend(action_tokens)
        return [
            -tokens.to(dtype=torch.float32) / 100.0
            for tokens in action_tokens
        ]

    def sample_action(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("teacher actions must never be sampled")

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("teacher must never step an environment")


def test_teacher_forced_scores_exact_flattened_student_actions_only() -> None:
    factory = RecordingFactory(horizon=3)
    rollouts = _collector(factory, group_size=2, seed=40).collect(
        [0, 5], ClosedLoopStudent()
    )
    env_step_counts_before = [len(env.received_actions) for env in factory.created]
    expected_transitions = [
        transition
        for trajectory in rollouts.trajectories
        for transition in trajectory.transitions
    ]
    teacher = RecordingTeacher()
    teacher.train()

    labeled = label_with_teacher(
        rollouts,
        teacher,
        temperature=0.5,
        score_batch_size=2,
    )

    assert [len(env.received_actions) for env in factory.created] == env_step_counts_before
    assert teacher.grad_enabled and not any(teacher.grad_enabled)
    assert teacher.training_flags and not any(teacher.training_flags)
    assert len(teacher.received_actions) == len(expected_transitions)
    for received, transition in zip(
        teacher.received_actions, expected_transitions, strict=True
    ):
        # Identity, not merely equality, proves labeling forwarded the stored
        # sampled tokens instead of replacing them with teacher argmax tokens.
        assert received is transition.action_tokens
    for received, transition in zip(
        teacher.received_observations, expected_transitions, strict=True
    ):
        assert received is transition.observation

    assert labeled.prompt_count == rollouts.prompt_count
    assert labeled.group_size == rollouts.group_size
    for labeled_trajectory, source in zip(
        labeled.trajectories, rollouts.trajectories, strict=True
    ):
        assert labeled_trajectory.trajectory is source
        assert len(labeled_trajectory.teacher_log_probs) == len(source.transitions)
        for score, transition in zip(
            labeled_trajectory.teacher_log_probs,
            source.transitions,
            strict=True,
        ):
            torch.testing.assert_close(
                score,
                -transition.action_tokens.to(dtype=torch.float32) / 100.0,
            )
            assert not score.requires_grad


class WrongShapeTeacher:
    def score_actions(
        self,
        observations: Sequence[Any],
        action_tokens: Sequence[torch.Tensor],
        *,
        temperature: float,
    ) -> list[torch.Tensor]:
        del observations, temperature
        return [torch.zeros(tokens.numel() + 1) for tokens in action_tokens]


def test_teacher_score_shape_mismatch_is_rejected() -> None:
    rollouts = _collector(RecordingFactory(horizon=1)).collect(
        [0], ClosedLoopStudent()
    )
    with pytest.raises(ValueError, match="shape must match"):
        label_with_teacher(
            rollouts,
            WrongShapeTeacher(),
            temperature=1.0,
        )
