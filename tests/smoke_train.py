"""One-step CPU integration smoke test using a locally created tiny model.

This is intentionally separate from the fast unittest suite. Run it directly:

    CUDA_VISIBLE_DEVICES="" python tests/smoke_train.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vla_mopd.code.config import config_from_dict  # noqa: E402
from vla_mopd.code.train_mopd import train  # noqa: E402


def build_tiny_checkpoint(path: Path):
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "solve": 4,
        "code": 5,
        "follow": 6,
        "instruction": 7,
        "answer": 8,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} "
        "{% endfor %}[EOS]"
    )
    model = AutoModelForCausalLM.from_config(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return tokenizer


def write_record(path: Path, prompt: str) -> None:
    path.write_text(
        json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vla-mopd-smoke-") as directory:
        root = Path(directory)
        checkpoint = root / "tiny-stage1-student"
        tokenizer = build_tiny_checkpoint(checkpoint)
        teacher_checkpoint = root / "tiny-rl-teacher"
        teacher = AutoModelForCausalLM.from_pretrained(checkpoint)
        torch.manual_seed(7)
        with torch.no_grad():
            first_parameter = next(teacher.parameters())
            first_parameter.add_(0.02 * torch.randn_like(first_parameter))
        teacher.save_pretrained(teacher_checkpoint)
        tokenizer.save_pretrained(teacher_checkpoint)
        data_paths = {
            "math": root / "math.jsonl",
            "code": root / "code.jsonl",
            "instruction_following": root / "if.jsonl",
        }
        write_record(data_paths["math"], "solve answer")
        write_record(data_paths["code"], "code answer")
        write_record(data_paths["instruction_following"], "follow instruction")

        completed = []
        for loss_type in ("pg", "topk"):
            config = config_from_dict({
                "model": {
                    "student_model": str(checkpoint),
                    "teacher_models": {
                        domain: str(teacher_checkpoint) for domain in data_paths
                    },
                    "torch_dtype": "float32",
                    "teacher_device": "cpu",
                    "gradient_checkpointing": False,
                },
                "data": {
                    "train_files": {
                        domain: str(path) for domain, path in data_paths.items()
                    },
                    "domain_ratios": {
                        "math": 1,
                        "code": 1,
                        "instruction_following": 1,
                    },
                    "max_prompt_length": 16,
                    "max_train_samples": 3,
                    "use_chat_template": True,
                    "num_workers": 0,
                },
                "algorithm": {
                    "loss_type": loss_type,
                    "advantage_clip": 5.0,
                    "topk_k": 3,
                    "max_response_length": 2,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                "training": {
                    "output_dir": str(root / "output"),
                    "run_name": f"cpu-smoke-{loss_type}",
                    "max_steps": 1,
                    "per_device_train_batch_size": 3,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1.0e-4,
                    "mixed_precision": "no",
                    "logging_steps": 1,
                    "save_steps": 1,
                    "report_to": "none",
                },
            })
            run_directory = train(config)
            expected = run_directory / "final" / "config.json"
            if not expected.is_file():
                raise AssertionError(f"final model was not saved: {expected}")
            trainer_state = run_directory / "checkpoint-1" / "trainer_state.json"
            resume_contract = run_directory / "checkpoint-1" / "resume_contract.json"
            if not trainer_state.is_file():
                raise AssertionError(
                    f"complete checkpoint marker was not saved: {trainer_state}"
                )
            if not resume_contract.is_file():
                raise AssertionError(
                    f"resume contract was not saved: {resume_contract}"
                )
            state = json.loads(trainer_state.read_text(encoding="utf-8"))
            if state != {"global_step": 1, "consumed_micro_batches": 1}:
                raise AssertionError(f"unexpected trainer state: {state}")
            if loss_type == "pg":
                config.training.resume_from_checkpoint = "latest"
                resumed_directory = train(config)
                if resumed_directory != run_directory:
                    raise AssertionError("resume changed the run directory")
                history = run_directory / "resume_history.jsonl"
                if len(history.read_text(encoding="utf-8").splitlines()) != 1:
                    raise AssertionError("resume history was not appended exactly once")
            completed.append(str(run_directory))
        print("smoke training passed:\n" + "\n".join(completed))


if __name__ == "__main__":
    main()

