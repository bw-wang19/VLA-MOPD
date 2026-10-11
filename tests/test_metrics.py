from __future__ import annotations

import unittest

import torch

from vla_mopd.code.losses import categorical_entropy
from vla_mopd.code.metrics import MOPDMetricAccumulator


class _SingleProcessAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def reduce(value, reduction):
        if reduction != "sum":
            raise AssertionError(reduction)
        return value


class MetricTests(unittest.TestCase):
    def test_action_mask_excludes_invalid_tokens_from_metrics(self) -> None:
        accumulator = MOPDMetricAccumulator(("math",))
        mask = torch.tensor([True, False])
        teacher = torch.tensor([-2.0, 100.0])
        student = torch.tensor([-1.0, -100.0])
        raw = teacher - student
        accumulator.update(
            domains=["math"],
            teacher_log_probs=[teacher],
            student_log_probs=[student],
            raw_advantages=[raw],
            clipped_advantages=[raw.clamp(-5, 5)],
            entropies=[torch.tensor(0.25)],
            loss=torch.tensor(1.0),
            advantage_clip=5.0,
            token_masks=[mask],
        )
        metrics = accumulator.distributed_metrics(_SingleProcessAccelerator())
        self.assertEqual(metrics["response_length"], 1.0)
        self.assertEqual(metrics["teacher_minus_student"], -1.0)
        self.assertEqual(metrics["sampled_reverse_kl"], 1.0)
        self.assertEqual(metrics["domain/math/token_share"], 1.0)
        self.assertEqual(metrics["domain/math/trajectory_share"], 1.0)

    def test_entropy_respects_action_mask(self) -> None:
        logits = torch.tensor([[0.0, 0.0], [20.0, -20.0]])
        entropy = categorical_entropy(
            [logits], token_masks=[torch.tensor([True, False])]
        )[0]
        self.assertAlmostEqual(entropy.item(), torch.log(torch.tensor(2.0)).item())


if __name__ == "__main__":
    unittest.main()
