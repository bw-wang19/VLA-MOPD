from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from datasets import Dataset

from vla_mopd.code.config import ConfigError, config_from_dict
from vla_mopd.code.data import (
    interleave_by_domain_ratio,
    load_domain_dataset,
)


def valid_config() -> dict:
    return {
        "model": {
            "student_model": "/models/sft",
            "teacher_models": {
                "math": "/models/math",
                "code": "/models/code",
            },
        },
        "data": {
            "train_files": {
                "math": "/data/math.jsonl",
                "code": "/data/code.jsonl",
            },
            "domain_ratios": {"math": 7, "code": 3},
        },
        "algorithm": {"loss_type": "pg"},
        "training": {"output_dir": "/outputs"},
    }


class ConfigTests(unittest.TestCase):
    def test_ratios_are_normalized_and_paper_defaults_are_present(self) -> None:
        config = config_from_dict(valid_config())
        self.assertAlmostEqual(config.data.domain_ratios["math"], 0.7)
        self.assertAlmostEqual(config.data.domain_ratios["code"], 0.3)
        self.assertEqual(config.algorithm.topk_k, 64)
        self.assertEqual(config.data.num_workers, 0)
        self.assertEqual(config.training.wandb_project, "vla-mopd")

    def test_unknown_key_is_rejected(self) -> None:
        raw = valid_config()
        raw["algorithm"]["advantge_clip"] = 5.0
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_from_dict(raw)

    def test_domain_sets_must_match_exactly(self) -> None:
        raw = valid_config()
        del raw["data"]["domain_ratios"]["code"]
        with self.assertRaisesRegex(ConfigError, "domain keys must match"):
            config_from_dict(raw)

    def test_warped_behavior_distribution_is_rejected(self) -> None:
        for field, value in (("temperature", 0.8), ("top_p", 0.95)):
            raw = valid_config()
            raw["algorithm"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ConfigError, "on-policy MOPD"
            ):
                config_from_dict(raw)

    def test_advantage_clip_is_validated_for_topk_diagnostics(self) -> None:
        for value in (0.0, -1.0, float("inf")):
            raw = valid_config()
            raw["algorithm"].update(
                {"loss_type": "topk", "advantage_clip": value}
            )
            with self.subTest(value=value), self.assertRaisesRegex(
                ConfigError, "finite and positive"
            ):
                config_from_dict(raw)

    def test_resume_and_run_names_must_be_nonempty_when_present(self) -> None:
        for field in ("resume_from_checkpoint", "run_name", "wandb_project"):
            raw = valid_config()
            raw["training"][field] = "  "
            with self.subTest(field=field), self.assertRaises(ConfigError):
                config_from_dict(raw)

    def test_run_name_cannot_escape_output_directory(self) -> None:
        raw = valid_config()
        raw["training"]["run_name"] = "../other-run"
        with self.assertRaisesRegex(ConfigError, "path component"):
            config_from_dict(raw)

    def test_unknown_tracker_is_rejected(self) -> None:
        raw = valid_config()
        raw["training"]["report_to"] = "wand"
        with self.assertRaisesRegex(ConfigError, "none.*wandb"):
            config_from_dict(raw)


class DataTests(unittest.TestCase):
    def test_jsonl_loading_overwrites_stale_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = [
                {"prompt": "one", "domain": "wrong"},
                {
                    "messages": [{"role": "user", "content": "two"}],
                    "domain": "wrong",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset = load_domain_dataset(path, "math", num_workers=0)
        self.assertEqual(dataset["domain"], ["math", "math"])

    def test_interleave_has_exact_quota_and_is_deterministic(self) -> None:
        datasets = {
            "math": Dataset.from_dict(
                {"prompt": ["m0", "m1"], "domain": ["math", "math"]}
            ),
            "code": Dataset.from_dict(
                {"prompt": ["c0"], "domain": ["code"]}
            ),
        }
        first = interleave_by_domain_ratio(
            datasets,
            {"math": 0.7, "code": 0.3},
            seed=17,
            num_samples=10,
        )
        second = interleave_by_domain_ratio(
            datasets,
            {"math": 0.7, "code": 0.3},
            seed=17,
            num_samples=10,
        )
        self.assertEqual(first["domain"].count("math"), 7)
        self.assertEqual(first["domain"].count("code"), 3)
        self.assertEqual(first[:], second[:])

    def test_records_without_prompt_or_messages_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"text": "missing schema"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt.*messages"):
                load_domain_dataset(path, "math", num_workers=0)


if __name__ == "__main__":
    unittest.main()
