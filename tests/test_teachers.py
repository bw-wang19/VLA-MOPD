from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vla_mopd.code.teachers import (
    LocalTeacherPool,
    RoutedTeacherScores,
    validate_shared_tokenizer,
)


class TeacherRoutingTests(unittest.TestCase):
    def test_hard_routing_restores_original_batch_order(self) -> None:
        pool = object.__new__(LocalTeacherPool)
        calls: list[tuple[str, list[list[int]]]] = []

        def fake_score(
            domain,
            prompts,
            responses,
            *,
            topk,
            output_device,
        ):
            calls.append((domain, list(prompts)))
            offset = {"math": 10.0, "code": 20.0}[domain]
            sampled = [torch.tensor([offset + prompt[0]]) for prompt in prompts]
            return RoutedTeacherScores(sampled_log_probs=sampled)

        pool._score_one_domain = fake_score
        scores = pool.score_routed(
            ["math", "code", "math"],
            [[1], [2], [3]],
            [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
            topk=None,
            output_device="cpu",
        )

        self.assertEqual([value.item() for value in scores.sampled_log_probs], [11, 22, 13])
        self.assertEqual(calls, [("math", [[1], [3]]), ("code", [[2]])])

    def test_unknown_domain_is_rejected(self) -> None:
        pool = object.__new__(LocalTeacherPool)
        pool.models = {}
        with self.assertRaisesRegex(KeyError, "no teacher configured"):
            pool.score_routed(
                ["unknown"],
                [[1]],
                [torch.tensor([2])],
                topk=None,
                output_device="cpu",
            )

    def test_local_scorer_uses_full_vocab_teacher_topk(self) -> None:
        class FakeTeacher:
            def __call__(self, **kwargs):
                input_ids = kwargs["input_ids"]
                logits = torch.zeros(input_ids.size(0), input_ids.size(1), 3)
                logits[:, 0, :] = torch.tensor([0.0, 2.0, 1.0])
                return SimpleNamespace(logits=logits)

        pool = object.__new__(LocalTeacherPool)
        pool.models = {"math": FakeTeacher()}
        pool.device = torch.device("cpu")
        pool.pad_token_id = 0
        scores = pool._score_one_domain(
            "math",
            [[7]],
            [torch.tensor([0])],
            topk=2,
            output_device="cpu",
        )
        expected = F.log_softmax(torch.tensor([0.0, 2.0, 1.0]), dim=-1)
        torch.testing.assert_close(
            scores.topk_indices[0], torch.tensor([[1, 2]])
        )
        torch.testing.assert_close(
            scores.topk_log_probs[0], expected[[1, 2]].unsqueeze(0)
        )
        self.assertLess(scores.topk_log_probs[0].exp().sum().item(), 1.0)


class TokenizerContractTests(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self, vocab):
            self._vocab = vocab
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.unk_token_id = 3
            self.pad_token_id = 2
            self.chat_template = "same-template"

        def get_vocab(self):
            return self._vocab

    def test_exact_token_id_mapping_is_required(self) -> None:
        student = self.FakeTokenizer({"a": 0, "b": 1})
        teacher = self.FakeTokenizer({"a": 1, "b": 0})
        with self.assertRaisesRegex(ValueError, "token-id mapping"):
            validate_shared_tokenizer(student, teacher, "math")

    def test_identical_tokenizer_contract_is_accepted(self) -> None:
        student = self.FakeTokenizer({"a": 0, "b": 1})
        teacher = self.FakeTokenizer({"a": 0, "b": 1})
        validate_shared_tokenizer(student, teacher, "math")


if __name__ == "__main__":
    unittest.main()

