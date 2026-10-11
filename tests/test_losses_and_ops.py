from __future__ import annotations

import unittest
from types import SimpleNamespace
from types import MethodType

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GenerationConfig, GPT2Config

from vla_mopd.code.losses import (
    corrected_topk_mopd_loss,
    sampled_mopd_loss,
)
from vla_mopd.code.ops import (
    extract_response_logits,
    forward_response_logits,
    gather_sampled_log_probs,
    generate_responses,
    left_pad,
    render_prompt,
    trim_after_eos,
)


class SampledMOPDLossTests(unittest.TestCase):
    def test_advantage_matches_teacher_minus_rollout_student(self) -> None:
        student = [torch.tensor([-2.0, -3.0], requires_grad=True)]
        teacher = [torch.tensor([-1.0, -5.0], requires_grad=True)]
        output = sampled_mopd_loss(
            student,
            teacher,
            advantage_clip=1.5,
        )

        torch.testing.assert_close(
            output.raw_advantages[0],
            torch.tensor([1.0, -2.0]),
        )
        torch.testing.assert_close(
            output.clipped_advantages[0],
            torch.tensor([1.0, -1.5]),
        )
        output.loss.backward()
        self.assertLess(student[0].grad[0].item(), 0.0)
        self.assertGreater(student[0].grad[1].item(), 0.0)
        self.assertIsNone(teacher[0].grad)

    def test_each_trajectory_is_length_normalized(self) -> None:
        short = torch.tensor([-1.0], requires_grad=True)
        long = torch.tensor([-3.0, -3.0, -3.0], requires_grad=True)
        teachers = [torch.tensor([0.0]), torch.tensor([-2.0, -2.0, -2.0])]
        output = sampled_mopd_loss(
            [short, long],
            teachers,
            advantage_clip=5.0,
        )
        # Both advantages are +1. Per-trajectory losses are 1 and 3, so the
        # batch loss is 2 rather than the token-weighted value 2.5.
        self.assertAlmostEqual(output.loss.item(), 2.0)

    def test_current_student_is_detached_only_inside_advantage(self) -> None:
        current = [torch.tensor([-2.0], requires_grad=True)]
        teacher = [torch.tensor([-1.0], requires_grad=True)]
        output = sampled_mopd_loss(
            current,
            teacher,
            advantage_clip=5.0,
        )
        self.assertEqual(output.raw_advantages[0].item(), 1.0)
        output.loss.backward()
        self.assertIsNotNone(current[0].grad)
        self.assertIsNone(teacher[0].grad)

    def test_action_mask_excludes_unexecuted_tokens(self) -> None:
        student = [torch.tensor([-1.0, -100.0], requires_grad=True)]
        teacher = [torch.tensor([0.0, 0.0])]
        output = sampled_mopd_loss(
            student,
            teacher,
            advantage_clip=5.0,
            token_masks=[torch.tensor([True, False])],
        )
        self.assertAlmostEqual(output.loss.item(), 1.0)
        output.loss.backward()
        self.assertNotEqual(student[0].grad[0].item(), 0.0)
        self.assertEqual(student[0].grad[1].item(), 0.0)


class CorrectedTopKLossTests(unittest.TestCase):
    def test_teacher_equality_is_zero_and_stationary(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.0, 0.0, -1.0], [0.0, 1.0, 2.0, -1.0]],
            requires_grad=True,
        )
        teacher_log_probs = F.log_softmax(logits.detach(), dim=-1)
        values, indices = teacher_log_probs.topk(2, dim=-1)
        output = corrected_topk_mopd_loss([logits], [indices], [values])

        self.assertAlmostEqual(output.loss.item(), 0.0, places=7)
        output.loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)

    def test_full_support_matches_reverse_kl(self) -> None:
        student_logits = torch.tensor([[1.2, -0.2, 0.5]], requires_grad=True)
        teacher_logits = torch.tensor([[0.1, 0.7, -0.4]])
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        indices = torch.arange(3).unsqueeze(0)
        output = corrected_topk_mopd_loss(
            [student_logits],
            [indices],
            [teacher_log_probs],
        )
        reference_logits = student_logits.detach().clone().requires_grad_(True)
        student_log_probs = F.log_softmax(reference_logits, dim=-1)
        expected = (
            student_log_probs.exp() * (student_log_probs - teacher_log_probs)
        ).sum()
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        expected.backward()
        torch.testing.assert_close(student_logits.grad, reference_logits.grad)

    def test_teacher_values_must_not_be_topk_renormalized(self) -> None:
        student_logits = torch.tensor([[1.0, 0.0, -1.0]])
        teacher_log_probs = F.log_softmax(torch.tensor([[2.0, 1.0, 0.0]]), dim=-1)
        values, indices = teacher_log_probs.topk(2, dim=-1)
        correct = corrected_topk_mopd_loss(
            [student_logits], [indices], [values]
        ).loss
        renormalized = corrected_topk_mopd_loss(
            [student_logits],
            [indices],
            [F.log_softmax(values, dim=-1)],
        ).loss
        self.assertFalse(torch.isclose(correct, renormalized))

    def test_mixed_lengths_are_normalized_per_trajectory(self) -> None:
        short_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        long_logits = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            requires_grad=True,
        )
        indices = [torch.tensor([[0]]), torch.tensor([[0], [0], [0]])]
        teacher = [
            torch.tensor([[-0.1]]),
            torch.tensor([[-2.0], [-2.0], [-2.0]]),
        ]
        output = corrected_topk_mopd_loss(
            [short_logits, long_logits], indices, teacher
        )
        short_loss = corrected_topk_mopd_loss(
            [short_logits], [indices[0]], [teacher[0]]
        ).loss
        long_loss = corrected_topk_mopd_loss(
            [long_logits], [indices[1]], [teacher[1]]
        ).loss
        torch.testing.assert_close(output.loss, (short_loss + long_loss) / 2)

    def test_topk_teacher_is_stop_gradient_and_action_mask_applies(self) -> None:
        student = torch.tensor(
            [[1.0, 0.0], [10.0, -10.0]], requires_grad=True
        )
        teacher_logits = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], requires_grad=True
        )
        teacher_full = F.log_softmax(teacher_logits, dim=-1)
        values, indices = teacher_full.topk(1, dim=-1)
        output = corrected_topk_mopd_loss(
            [student],
            [indices],
            [values],
            token_masks=[torch.tensor([True, False])],
        )
        output.loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad[0].abs().sum().item(), 0.0)
        torch.testing.assert_close(student.grad[1], torch.zeros_like(student.grad[1]))
        self.assertIsNone(teacher_logits.grad)

    def test_expected_pg_gradient_matches_full_reverse_kl(self) -> None:
        student_logits = torch.tensor([0.4, -0.2, 1.0], requires_grad=True)
        teacher_logits = torch.tensor([-0.1, 0.7, 0.2])
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        reverse_kl = (
            student_log_probs.exp()
            * (student_log_probs - teacher_log_probs)
        ).sum()
        reverse_kl.backward()
        expected_gradient = student_logits.grad.detach().clone()

        pg_logits = student_logits.detach().clone().requires_grad_(True)
        pg_log_probs = F.log_softmax(pg_logits, dim=-1)
        sampling_probabilities = pg_log_probs.detach().exp()
        enumerated_loss = torch.zeros((), dtype=pg_logits.dtype)
        for token_id in range(pg_logits.numel()):
            sampled = sampled_mopd_loss(
                [pg_log_probs[token_id : token_id + 1]],
                [teacher_log_probs[token_id : token_id + 1]],
                advantage_clip=100.0,
            )
            enumerated_loss = (
                enumerated_loss
                + sampling_probabilities[token_id] * sampled.loss
            )
        enumerated_loss.backward()
        torch.testing.assert_close(pg_logits.grad, expected_gradient)


class TokenOperationTests(unittest.TestCase):
    def test_left_pad_and_mask(self) -> None:
        ids, mask = left_pad([[1, 2, 3], [4]], 0, "cpu")
        torch.testing.assert_close(ids, torch.tensor([[1, 2, 3], [0, 0, 4]]))
        torch.testing.assert_close(mask, torch.tensor([[1, 1, 1], [0, 0, 1]]))

    def test_response_alignment_handles_mixed_lengths(self) -> None:
        logits = torch.zeros(2, 4, 5)
        responses = [torch.tensor([1, 2, 3]), torch.tensor([4])]
        logits[0, 0, 1] = 8
        logits[0, 1, 2] = 8
        logits[0, 2, 3] = 8
        logits[1, 2, 4] = 8
        response_logits = extract_response_logits(logits, responses)
        log_probs = gather_sampled_log_probs(response_logits, responses)
        self.assertEqual([len(item) for item in log_probs], [3, 1])
        self.assertTrue(all(torch.all(item > -0.01) for item in log_probs))

    def test_trim_keeps_first_eos(self) -> None:
        trimmed = trim_after_eos(torch.tensor([4, 5, 2, 9, 2]), [2, 3])
        torch.testing.assert_close(trimmed, torch.tensor([4, 5, 2]))

    def test_rollout_uses_fresh_unwarped_generation_config(self) -> None:
        class FakeModel:
            generation_config = SimpleNamespace(top_k=1, repetition_penalty=2.0)

            def generate(self, *, input_ids, generation_config, **kwargs):
                self.received = generation_config
                continuation = torch.ones(
                    input_ids.size(0), 1, dtype=torch.long, device=input_ids.device
                )
                return torch.cat((input_ids, continuation), dim=-1)

        model = FakeModel()
        responses = generate_responses(
            model,
            [[4, 5]],
            pad_token_id=0,
            eos_token_id=2,
            max_response_length=4,
            temperature=1.0,
            top_p=1.0,
            device="cpu",
        )
        self.assertEqual(model.received.top_k, 0)
        self.assertEqual(model.received.top_p, 1.0)
        self.assertIn(model.received.repetition_penalty, (None, 1.0))
        self.assertEqual(model.received.min_new_tokens, None)
        torch.testing.assert_close(responses[0], torch.tensor([1]))

    def test_checkpoint_generation_processors_cannot_backfill(self) -> None:
        model = AutoModelForCausalLM.from_config(
            GPT2Config(
                vocab_size=8,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
        malicious = GenerationConfig(
            bad_words_ids=[[1]],
            suppress_tokens=[3],
            forced_eos_token_id=4,
            min_new_tokens=3,
        )
        model.generation_config = malicious

        def capture_generate(self, *, input_ids, generation_config, **kwargs):
            prepared, _ = self._prepare_generation_config(generation_config)
            self.prepared_config = prepared
            continuation = torch.ones(
                input_ids.size(0), 1, dtype=torch.long, device=input_ids.device
            )
            return torch.cat((input_ids, continuation), dim=-1)

        model.generate = MethodType(capture_generate, model)
        generate_responses(
            model,
            [[4]],
            pad_token_id=0,
            eos_token_id=2,
            max_response_length=2,
            temperature=1.0,
            top_p=1.0,
            device="cpu",
        )
        self.assertIs(model.generation_config, malicious)
        self.assertIsNone(model.prepared_config.bad_words_ids)
        self.assertIsNone(model.prepared_config.suppress_tokens)
        self.assertIsNone(model.prepared_config.forced_eos_token_id)
        self.assertIsNone(model.prepared_config.min_new_tokens)

    def test_chat_template_fallback_without_enable_thinking(self) -> None:
        class FakeTokenizer:
            chat_template = "fake"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                return [1, 2, 3, 4]

        tokenizer = FakeTokenizer()
        rendered = render_prompt(
            tokenizer,
            {"prompt": "hello"},
            max_prompt_length=3,
            use_chat_template=True,
            enable_thinking=False,
        )
        self.assertEqual(rendered, [2, 3, 4])
        self.assertEqual(tokenizer.messages[0]["content"], "hello")

    def test_forced_scoring_uses_mask_derived_position_ids(self) -> None:
        class FakeCausalModel:
            def __call__(
                self,
                *,
                input_ids,
                attention_mask,
                position_ids,
                use_cache,
                logits_to_keep,
            ):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.position_ids = position_ids
                logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8)
                # Row 0: [5, 6, 1, 2], logits 1 and 2 predict response 1,2.
                logits[0, 1, 1] = 9
                logits[0, 2, 2] = 9
                # Row 1: [pad, pad, 7, 3], logits 2 predicts response 3.
                logits[1, 2, 3] = 9
                return SimpleNamespace(logits=logits)

        model = FakeCausalModel()
        responses = [torch.tensor([1, 2]), torch.tensor([3])]
        response_logits = forward_response_logits(
            model,
            [[5, 6], [7]],
            responses,
            pad_token_id=0,
            device="cpu",
        )
        log_probs = gather_sampled_log_probs(response_logits, responses)
        self.assertTrue(all(torch.all(value > -0.01) for value in log_probs))
        torch.testing.assert_close(
            model.position_ids,
            torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]]),
        )


if __name__ == "__main__":
    unittest.main()

