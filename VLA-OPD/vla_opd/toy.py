"""Tiny categorical closed-loop environment used only for CPU verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import VLAOPDConfig
from .runtime import ExperimentComponents
from .types import ActionSample, ResetResult, StepResult


@dataclass(frozen=True)
class ToyObservation:
    prompt_id: int
    step: int
    previous_action: int
    visual: torch.Tensor


def _target(observation: ToyObservation, vocab_size: int) -> int:
    # An incorrect previous action changes the next target.  This makes the
    # visited state distribution genuinely student induced rather than a fixed
    # offline action-label table.
    recovery_offset = 0 if observation.previous_action < 0 else observation.previous_action
    return (observation.prompt_id + observation.step + recovery_offset) % vocab_size


class ToyVLAEnvironment:
    def __init__(self, *, horizon: int, vocab_size: int) -> None:
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.prompt_id = 0
        self.step_index = 0
        self.previous_action = -1

    def _observation(self) -> ToyObservation:
        visual = torch.tensor(
            [[self.prompt_id, self.step_index], [self.previous_action, 1]],
            dtype=torch.float32,
        )
        return ToyObservation(
            prompt_id=self.prompt_id,
            step=self.step_index,
            previous_action=self.previous_action,
            visual=visual,
        )

    def reset(self, prompt: Any, *, seed: int) -> ResetResult:
        del seed
        self.prompt_id = int(prompt)
        self.step_index = 0
        self.previous_action = -1
        return ResetResult(self._observation(), {"toy": True})

    def step(self, action: Any) -> StepResult:
        action_value = int(action)
        if not 0 <= action_value < self.vocab_size:
            raise ValueError("toy action outside vocabulary")
        before = self._observation()
        reward = float(action_value == _target(before, self.vocab_size))
        self.previous_action = action_value
        self.step_index += 1
        terminated = self.step_index >= self.horizon
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            info={"correct": bool(reward)},
        )

    def close(self) -> None:
        return None


class TinyCategoricalStudent(nn.Module):
    def __init__(
        self,
        *,
        num_prompts: int,
        horizon: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.num_prompts = num_prompts
        self.horizon = horizon
        self.vocab_size = vocab_size
        states = num_prompts * (horizon + 1) * (vocab_size + 1)
        self.logit_table = nn.Embedding(states, vocab_size)
        nn.init.normal_(self.logit_table.weight, mean=0.0, std=0.05)

    def _state_index(self, observation: ToyObservation) -> int:
        previous = observation.previous_action + 1
        return (
            observation.prompt_id * (self.horizon + 1) * (self.vocab_size + 1)
            + observation.step * (self.vocab_size + 1)
            + previous
        )

    def _log_probs(
        self, observations: Sequence[ToyObservation], *, temperature: float
    ) -> torch.Tensor:
        device = self.logit_table.weight.device
        indices = torch.tensor(
            [self._state_index(item) for item in observations],
            dtype=torch.long,
            device=device,
        )
        return F.log_softmax(self.logit_table(indices).float() / temperature, dim=-1)

    def sample_action(
        self,
        observation: ToyObservation,
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> ActionSample:
        log_probs = self._log_probs([observation], temperature=temperature)[0]
        token = torch.multinomial(
            log_probs.exp().cpu(), 1, replacement=True, generator=generator
        ).to(log_probs.device)
        selected = log_probs.gather(0, token)
        return ActionSample(
            tokens=token.to(dtype=torch.long),
            log_probs=selected,
            mask=torch.ones_like(token, dtype=torch.bool),
            env_action=int(token.item()),
        )

    def score_actions(
        self,
        observations: Sequence[ToyObservation],
        action_tokens: Sequence[torch.Tensor],
        *,
        temperature: float,
    ) -> list[torch.Tensor]:
        if len(observations) != len(action_tokens):
            raise ValueError("observations and action tokens must align")
        log_probs = self._log_probs(observations, temperature=temperature)
        result: list[torch.Tensor] = []
        for row, tokens in zip(log_probs, action_tokens, strict=True):
            result.append(row.gather(0, tokens.to(row.device, dtype=torch.long)))
        return result


class ToyExpertTeacher(nn.Module):
    def __init__(self, *, vocab_size: int, correct_probability: float = 0.92) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.correct_probability = correct_probability

    def score_actions(
        self,
        observations: Sequence[ToyObservation],
        action_tokens: Sequence[torch.Tensor],
        *,
        temperature: float,
    ) -> list[torch.Tensor]:
        if len(observations) != len(action_tokens):
            raise ValueError("observations and action tokens must align")
        low = (1.0 - self.correct_probability) / (self.vocab_size - 1)
        base_logits = torch.full((self.vocab_size,), float(torch.log(torch.tensor(low))))
        high_logit = float(torch.log(torch.tensor(self.correct_probability)))
        results: list[torch.Tensor] = []
        for observation, tokens in zip(observations, action_tokens, strict=True):
            logits = base_logits.clone()
            logits[_target(observation, self.vocab_size)] = high_logit
            teacher_log_probs = F.log_softmax(logits / temperature, dim=-1)
            results.append(
                teacher_log_probs.gather(0, tokens.cpu().to(dtype=torch.long))
            )
        return results


def build_experiment(
    *,
    config: VLAOPDConfig,
    num_prompts: int = 8,
    action_vocab_size: int = 5,
) -> ExperimentComponents:
    """Build the toy smoke-test stack; this is not a paper benchmark."""

    if num_prompts <= 0 or action_vocab_size <= 1:
        raise ValueError("toy runtime needs positive prompts and vocab_size > 1")
    horizon = min(4, config.algorithm.max_episode_steps)
    student = TinyCategoricalStudent(
        num_prompts=num_prompts,
        horizon=horizon,
        vocab_size=action_vocab_size,
    )
    teacher = ToyExpertTeacher(vocab_size=action_vocab_size)

    def env_factory(
        *, prompt: Any, prompt_index: int, group_index: int
    ) -> ToyVLAEnvironment:
        del prompt, prompt_index, group_index
        return ToyVLAEnvironment(horizon=horizon, vocab_size=action_vocab_size)

    return ExperimentComponents(
        student=student,
        teacher=teacher,
        env_factory=env_factory,
        prompts=tuple(range(num_prompts)),
    )
