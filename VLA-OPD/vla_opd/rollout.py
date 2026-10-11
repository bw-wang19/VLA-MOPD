"""Closed-loop, group-based student trajectory collection."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .protocols import EnvironmentFactory, StudentPolicy
from .types import (
    ActionSample,
    ResetResult,
    RolloutBatch,
    StepResult,
    Trajectory,
    Transition,
    clone_snapshot,
)


class GroupRolloutCollector:
    """Collect B x G independent trajectories from the current student.

    The teacher is intentionally absent from this class.  Only the student's
    ``env_action`` is ever passed to ``environment.step``.
    """

    def __init__(
        self,
        env_factory: EnvironmentFactory,
        *,
        group_size: int,
        max_episode_steps: int,
        temperature: float,
        seed: int,
    ) -> None:
        if group_size <= 0 or max_episode_steps <= 0:
            raise ValueError("group_size and max_episode_steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.env_factory = env_factory
        self.group_size = group_size
        self.max_episode_steps = max_episode_steps
        self.temperature = float(temperature)
        self.seed = seed
        self._next_rollout_seed_offset = 0

    def collect(
        self, prompts: Sequence[Any], student: StudentPolicy
    ) -> RolloutBatch:
        if len(prompts) == 0:
            raise ValueError("at least one prompt is required")
        trajectories: list[Trajectory] = []
        collection_offset = self._next_rollout_seed_offset
        self._next_rollout_seed_offset += len(prompts) * self.group_size

        for prompt_index, prompt in enumerate(prompts):
            for group_index in range(self.group_size):
                rollout_seed = (
                    self.seed
                    + collection_offset
                    + prompt_index * self.group_size
                    + group_index
                )
                env = self.env_factory(
                    prompt=prompt,
                    prompt_index=prompt_index,
                    group_index=group_index,
                )
                try:
                    reset = env.reset(prompt, seed=rollout_seed)
                    if not isinstance(reset, ResetResult):
                        raise TypeError("environment.reset must return ResetResult")
                    observation = reset.observation
                    transitions: list[Transition] = []
                    generator = torch.Generator()
                    generator.manual_seed(rollout_seed)

                    for step_index in range(self.max_episode_steps):
                        observation_snapshot = clone_snapshot(observation)
                        with torch.inference_mode():
                            sample = student.sample_action(
                                observation_snapshot,
                                temperature=self.temperature,
                                generator=generator,
                            )
                        if not isinstance(sample, ActionSample):
                            raise TypeError(
                                "student.sample_action must return ActionSample"
                            )
                        recorded_env_action = clone_snapshot(sample.env_action)
                        result = env.step(sample.env_action)
                        if not isinstance(result, StepResult):
                            raise TypeError("environment.step must return StepResult")
                        reached_horizon = step_index + 1 == self.max_episode_steps
                        terminated = result.terminated
                        truncated = result.truncated or (
                            reached_horizon and not result.terminated
                        )
                        transitions.append(
                            Transition(
                                prompt_index=prompt_index,
                                group_index=group_index,
                                step_index=step_index,
                                observation=observation_snapshot,
                                action_tokens=sample.tokens.detach().clone(),
                                env_action=recorded_env_action,
                                behavior_log_probs=sample.log_probs.detach().float().clone(),
                                token_mask=sample.mask.detach().bool().clone(),
                                env_reward=float(result.reward),
                                terminated=terminated,
                                truncated=truncated,
                                info=clone_snapshot(dict(result.info)),
                            )
                        )
                        observation = result.observation
                        if terminated or truncated:
                            break

                    trajectories.append(
                        Trajectory(
                            prompt=clone_snapshot(prompt),
                            prompt_index=prompt_index,
                            group_index=group_index,
                            transitions=tuple(transitions),
                            final_observation=clone_snapshot(observation),
                            reset_info=clone_snapshot(dict(reset.info)),
                        )
                    )
                finally:
                    env.close()

        return RolloutBatch(
            trajectories=tuple(trajectories),
            prompt_count=len(prompts),
            group_size=self.group_size,
        )
