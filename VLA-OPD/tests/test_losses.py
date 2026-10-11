from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vla_opd.losses import sampled_reverse_kl_pg_loss


def test_reward_sign_and_teacher_behavior_stop_gradient() -> None:
    current = torch.tensor([-2.0, -3.0], requires_grad=True)
    behavior = torch.tensor([-2.0, -3.0], requires_grad=True)
    teacher = torch.tensor([-1.0, -5.0], requires_grad=True)

    output = sampled_reverse_kl_pg_loss([current], [behavior], [teacher])

    torch.testing.assert_close(output.raw_rewards[0], torch.tensor([1.0, -2.0]))
    output.loss.backward()
    # Gradient descent raises the probability of a positive-reward token and
    # lowers the probability of a negative-reward token.
    torch.testing.assert_close(current.grad, torch.tensor([-1.0, 2.0]))
    assert behavior.grad is None
    assert teacher.grad is None
    assert not output.raw_rewards[0].requires_grad


def test_enumerated_pg_gradient_matches_exact_categorical_reverse_kl() -> None:
    student_logits = torch.tensor([0.4, -0.2, 1.0], requires_grad=True)
    teacher_logits = torch.tensor([-0.1, 0.7, 0.2])
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    exact_reverse_kl = (
        student_log_probs.exp() * (student_log_probs - teacher_log_probs)
    ).sum()
    exact_reverse_kl.backward()
    exact_gradient = student_logits.grad.detach().clone()

    pg_logits = student_logits.detach().clone().requires_grad_(True)
    pg_log_probs = F.log_softmax(pg_logits, dim=-1)
    sampling_probabilities = pg_log_probs.detach().exp()
    enumerated_objective = pg_logits.new_zeros(())
    for token_id in range(pg_logits.numel()):
        sampled = sampled_reverse_kl_pg_loss(
            [pg_log_probs[token_id : token_id + 1]],
            [pg_log_probs[token_id : token_id + 1].detach()],
            [teacher_log_probs[token_id : token_id + 1]],
        )
        enumerated_objective = (
            enumerated_objective
            + sampling_probabilities[token_id] * sampled.loss
        )

    enumerated_objective.backward()
    torch.testing.assert_close(pg_logits.grad, exact_gradient, atol=1e-6, rtol=1e-6)


def test_three_reductions_have_the_expected_length_weighting() -> None:
    short = torch.tensor([-1.0], requires_grad=True)
    long = torch.tensor([-3.0, -3.0, -3.0], requires_grad=True)
    behavior = [torch.tensor([-2.0]), torch.tensor([-4.0, -4.0, -4.0])]
    teacher = [torch.tensor([-1.0]), torch.tensor([-3.0, -3.0, -3.0])]

    paper = sampled_reverse_kl_pg_loss([short, long], behavior, teacher)
    global_mean = sampled_reverse_kl_pg_loss(
        [short, long], behavior, teacher, reduction="global_token_mean"
    )
    trajectory_mean = sampled_reverse_kl_pg_loss(
        [short, long], behavior, teacher, reduction="trajectory_token_mean"
    )

    # Every reward is +1. Token losses are [1] and [3, 3, 3].
    assert paper.loss.item() == pytest.approx((1.0 + 9.0) / 2.0)
    assert global_mean.loss.item() == pytest.approx(10.0 / 4.0)
    assert trajectory_mean.loss.item() == pytest.approx((1.0 + 3.0) / 2.0)
    assert paper.metrics["trajectory_count"].item() == 2
    assert paper.metrics["valid_token_count"].item() == 4

    trajectory_mean.loss.backward()
    # Each trajectory has total gradient magnitude 0.5 regardless of length.
    torch.testing.assert_close(short.grad, torch.tensor([-0.5]))
    torch.testing.assert_close(long.grad, torch.full((3,), -1.0 / 6.0))


def test_trajectory_token_mean_counts_only_valid_tokens() -> None:
    current = torch.tensor([-1.0, -100.0, -3.0], requires_grad=True)
    behavior = torch.tensor([-2.0, -2.0, -4.0])
    teacher = torch.tensor([-1.0, 50.0, -3.0])
    mask = torch.tensor([True, False, True])

    output = sampled_reverse_kl_pg_loss(
        [current],
        [behavior],
        [teacher],
        masks=[mask],
        reduction="trajectory_token_mean",
    )

    # The two valid token losses are 1 and 3; the masked token is not in the
    # numerator or denominator.
    assert output.loss.item() == pytest.approx(2.0)
    output.loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-0.5, 0.0, -0.5]))


def test_mask_excludes_tokens_from_loss_gradient_reduction_and_metrics() -> None:
    current = torch.tensor([-1.0, -100.0, -3.0], requires_grad=True)
    behavior = torch.tensor([-2.0, -2.0, -4.0])
    teacher = torch.tensor([-1.0, 50.0, -3.0])
    mask = torch.tensor([True, False, True])

    output = sampled_reverse_kl_pg_loss(
        [current], [behavior], [teacher], masks=[mask]
    )

    # Valid rewards are +1 and +1; the deliberately extreme masked reward is
    # retained for diagnostics-by-trajectory but excluded from scalar metrics.
    assert output.loss.item() == pytest.approx(4.0)
    assert output.metrics["opd_reward_mean"].item() == pytest.approx(1.0)
    assert output.metrics["valid_token_count"].item() == 2
    output.loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-1.0, 0.0, -1.0]))


def test_student_equal_to_teacher_is_zero_and_stationary() -> None:
    current = torch.tensor([-0.2, -1.7, -2.5], requires_grad=True)
    behavior = current.detach().clone().requires_grad_(True)
    teacher = current.detach().clone().requires_grad_(True)

    output = sampled_reverse_kl_pg_loss([current], [behavior], [teacher])

    torch.testing.assert_close(output.loss, torch.zeros_like(output.loss))
    torch.testing.assert_close(output.raw_rewards[0], torch.zeros_like(current))
    output.loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
    assert behavior.grad is None
    assert teacher.grad is None


def test_reward_clip_is_explicit_and_reports_fraction() -> None:
    current = torch.tensor([-1.0, -1.0], requires_grad=True)
    behavior = torch.tensor([-4.0, -1.0])
    teacher = torch.tensor([0.0, -3.0])

    output = sampled_reverse_kl_pg_loss(
        [current], [behavior], [teacher], reward_clip=1.5
    )

    torch.testing.assert_close(output.raw_rewards[0], torch.tensor([4.0, -2.0]))
    torch.testing.assert_close(output.rewards[0], torch.tensor([1.5, -1.5]))
    assert output.metrics["opd_clip_fraction"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("which", ["current", "behavior", "teacher"])
def test_rejects_nonfinite_log_probabilities(which: str) -> None:
    values = {
        "current": torch.tensor([-1.0]),
        "behavior": torch.tensor([-1.0]),
        "teacher": torch.tensor([-1.0]),
    }
    values[which] = torch.tensor([float("nan")])

    with pytest.raises(ValueError, match="finite"):
        sampled_reverse_kl_pg_loss(
            [values["current"]], [values["behavior"]], [values["teacher"]]
        )


@pytest.mark.parametrize(
    ("current", "behavior", "teacher"),
    [
        (torch.tensor([]), torch.tensor([]), torch.tensor([])),
        (
            torch.tensor([[-1.0]]),
            torch.tensor([[-1.0]]),
            torch.tensor([[-1.0]]),
        ),
        (torch.tensor([-1.0]), torch.tensor([-1.0, -2.0]), torch.tensor([-1.0])),
        (torch.tensor([-1]), torch.tensor([-1.0]), torch.tensor([-1.0])),
    ],
)
def test_rejects_invalid_log_probability_shape_or_dtype(
    current: torch.Tensor,
    behavior: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        sampled_reverse_kl_pg_loss([current], [behavior], [teacher])


def test_rejects_empty_or_misaligned_batches_and_masks() -> None:
    value = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="non-empty and equal"):
        sampled_reverse_kl_pg_loss([], [], [])
    with pytest.raises(ValueError, match="non-empty and equal"):
        sampled_reverse_kl_pg_loss([value], [], [value])
    with pytest.raises(ValueError, match="align"):
        sampled_reverse_kl_pg_loss(
            [value], [value], [value], masks=[torch.tensor([True, False])]
        )
    with pytest.raises(ValueError, match="valid action token"):
        sampled_reverse_kl_pg_loss(
            [value], [value], [value], masks=[torch.tensor([False])]
        )


def test_rejects_non_boolean_mask_instead_of_silently_coercing_it() -> None:
    value = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="boolean"):
        sampled_reverse_kl_pg_loss(
            [value], [value], [value], masks=[torch.tensor([1.0])]
        )


@pytest.mark.parametrize("reward_clip", [0.0, -1.0, float("inf"), True, "1"])
def test_rejects_invalid_reward_clip(reward_clip: object) -> None:
    value = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="reward_clip"):
        sampled_reverse_kl_pg_loss(
            [value], [value], [value], reward_clip=reward_clip  # type: ignore[arg-type]
        )


def test_rejects_unknown_reduction() -> None:
    value = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="reduction"):
        sampled_reverse_kl_pg_loss(
            [value], [value], [value], reduction="trajectory_mean"
        )
