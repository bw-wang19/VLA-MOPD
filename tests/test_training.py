from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from vla_mopd.code.config import config_from_dict
from vla_mopd.code.modeling import (
    validate_context_window,
    validate_mode_invariant_student,
)
from vla_mopd.code.teachers import RoutedTeacherScores
from vla_mopd.code.train_mopd import (
    _run_batch,
    build_artifact_manifests,
    create_accelerator,
    protect_existing_run,
    resolve_resume_checkpoint,
    validate_checkpoint_contract,
    validate_distributed_artifact_consistency,
    validate_resume_compatibility,
)


class _FakeModel(torch.nn.Module):
    pass


class _FakeTeacherPool:
    def score_routed(self, domains, prompts, responses, *, topk, output_device):
        return RoutedTeacherScores(
            sampled_log_probs=[torch.tensor([-0.5]) for _ in domains]
        )


class OnPolicyModeTests(unittest.TestCase):
    def test_eval_generation_and_train_rescoring_are_mode_invariant(self) -> None:
        model = _FakeModel()
        model.train()
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            unwrap_model=lambda value: value,
        )
        config = SimpleNamespace(
            domains=("math",),
            data=SimpleNamespace(max_prompt_length=8, use_chat_template=False),
            algorithm=SimpleNamespace(
                enable_thinking=False,
                max_response_length=1,
                temperature=1.0,
                top_p=1.0,
                loss_type="pg",
                topk_k=2,
                advantage_clip=5.0,
            ),
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)

        def fake_generate(generation_model, *args, **kwargs):
            self.assertIs(generation_model, model)
            self.assertFalse(model.training)
            return [torch.tensor([1])]

        def fake_forward(scoring_model, *args, **kwargs):
            self.assertIs(scoring_model, model)
            self.assertTrue(model.training)
            return [torch.tensor([[0.0, 1.0]], requires_grad=True)]

        with (
            patch(
                "vla_mopd.code.train_mopd.render_prompt",
                return_value=[1],
            ),
            patch(
                "vla_mopd.code.train_mopd.generate_responses",
                side_effect=fake_generate,
            ),
            patch(
                "vla_mopd.code.train_mopd.forward_response_logits",
                side_effect=fake_forward,
            ),
        ):
            output = _run_batch(
                model=model,
                teacher_pool=_FakeTeacherPool(),
                tokenizer=tokenizer,
                records=[{"domain": "math", "prompt": "x"}],
                config=config,
                accelerator=accelerator,
            )

        self.assertTrue(model.training)
        self.assertTrue(output["loss"].requires_grad)

    def test_nonzero_dropout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero dropout"):
            validate_mode_invariant_student(
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
            )

    def test_zero_dropout_policy_is_accepted(self) -> None:
        validate_mode_invariant_student(
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
        )

    def test_context_overflow_is_rejected_before_rollout(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=16)
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_context_window(
                model,
                max_prompt_length=12,
                max_response_length=8,
            )


class TrainerStateTests(unittest.TestCase):
    @staticmethod
    def _config(output_dir: str):
        return config_from_dict(
            {
                "model": {
                    "student_model": "/models/sft",
                    "teacher_models": {"math": "/models/math"},
                },
                "data": {
                    "train_files": {"math": "/data/math.jsonl"},
                    "domain_ratios": {"math": 1},
                },
                "algorithm": {"loss_type": "pg"},
                "training": {"output_dir": output_dir, "max_steps": 10},
            }
        )

    def test_accumulation_crosses_dataloader_boundaries(self) -> None:
        config = SimpleNamespace(
            training=SimpleNamespace(
                report_to="none",
                gradient_accumulation_steps=3,
                mixed_precision="no",
            )
        )
        accelerator = create_accelerator(config)
        self.assertEqual(accelerator.gradient_state.num_steps, 3)
        self.assertFalse(accelerator.gradient_state.sync_with_dataloader)
        self.assertFalse(accelerator.step_scheduler_with_optimizer)

    def test_latest_skips_incomplete_checkpoint_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            incomplete = run_directory / "checkpoint-9"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text(
                json.dumps({"global_step": 9}), encoding="utf-8"
            )
            complete = run_directory / "checkpoint-3"
            complete.mkdir()
            (complete / "trainer_state.json").write_text(
                json.dumps(
                    {"global_step": 3, "consumed_micro_batches": 12}
                ),
                encoding="utf-8",
            )
            (complete / "resume_contract.json").write_text("{}\n", encoding="utf-8")
            selected = resolve_resume_checkpoint("latest", run_directory)
            self.assertEqual(selected, complete)

    def test_max_steps_cannot_change_on_same_run_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            config = self._config(directory)
            (run_directory / "resolved_config.yaml").write_text(
                yaml.safe_dump(config.to_dict(), sort_keys=False),
                encoding="utf-8",
            )
            config.training.max_steps = 20
            accelerator = SimpleNamespace(num_processes=1)
            with self.assertRaisesRegex(ValueError, "max_steps"):
                validate_resume_compatibility(config, run_directory, accelerator)

    def test_explicit_checkpoint_uses_its_own_config_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-2"
            checkpoint.mkdir()
            config = self._config(directory)
            contract = {
                "version": 1,
                "config": config.to_dict(),
                "world_size": 1,
                "artifacts": build_artifact_manifests(config),
            }
            (checkpoint / "resume_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            config.algorithm.loss_type = "topk"
            accelerator = SimpleNamespace(num_processes=1)
            with self.assertRaisesRegex(ValueError, "algorithm"):
                validate_checkpoint_contract(config, checkpoint, accelerator)

    def test_checkpoint_contract_detects_changed_local_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "math.jsonl"
            data_path.write_text('{"prompt":"one"}\n', encoding="utf-8")
            config = self._config(directory)
            config.data.train_files["math"] = str(data_path)
            checkpoint = root / "checkpoint-2"
            checkpoint.mkdir()
            contract = {
                "version": 1,
                "config": config.to_dict(),
                "world_size": 1,
                "artifacts": build_artifact_manifests(config),
            }
            (checkpoint / "resume_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            data_path.write_text('{"prompt":"two"}\n', encoding="utf-8")
            accelerator = SimpleNamespace(num_processes=1)
            with self.assertRaisesRegex(ValueError, "artifact manifest"):
                validate_checkpoint_contract(config, checkpoint, accelerator)

    def test_existing_run_rejects_branching_from_older_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            (run_directory / "resolved_config.yaml").write_text(
                "model: {}\n", encoding="utf-8"
            )
            checkpoints = []
            for step in (3, 9):
                checkpoint = run_directory / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step}), encoding="utf-8"
                )
                (checkpoint / "resume_contract.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                checkpoints.append(checkpoint)
            with self.assertRaisesRegex(FileExistsError, "latest complete"):
                protect_existing_run(run_directory, checkpoints[0])
            protect_existing_run(run_directory, checkpoints[1])

    def test_distributed_artifact_mismatch_is_rejected(self) -> None:
        manifest = {"student_model": {"locator": "same-on-rank-zero"}}

        class _MismatchedAccelerator:
            num_processes = 2
            device = torch.device("cpu")

            @staticmethod
            def gather(local_digest):
                other_digest = local_digest.clone()
                other_digest[0, 0] = (other_digest[0, 0] + 1) % 256
                return torch.cat((local_digest, other_digest), dim=0)

        with self.assertRaisesRegex(ValueError, "different.*artifact"):
            validate_distributed_artifact_consistency(
                manifest, _MismatchedAccelerator()
            )


if __name__ == "__main__":
    unittest.main()
