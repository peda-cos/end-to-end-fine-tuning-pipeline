"""Emit TRL DPO/GRPO YAML and a runnable preference-training script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TRL_SCRIPT = '''"""Generated preference-training runner for the real 8B stack."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer


def exact_match_reward(completions, category, priority, **kwargs):
    rewards = []
    for completion, expected_category, expected_priority in zip(completions, category, priority):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        parts = text.strip().lower().split()
        valid = len(parts) == 2 and parts[0] in {"billing", "technical", "shipping", "account"} and parts[1] in {"low", "high"}
        score = 0.5 if valid else 0.0
        if valid and parts[0] == expected_category:
            score += 0.25
        if valid and parts[0] == expected_category and parts[1] == expected_priority:
            score += 1.0
        rewards.append(score)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dpo", "grpo"], default="dpo")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = load_dataset("json", data_files=config.pop("dataset_name"), split="train")
    if args.mode == "grpo" and "prompt" not in dataset.column_names:
        dataset = dataset.map(lambda row: {"prompt": row["instruction"]})
    model_path = config.pop("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    config.pop("mode", None)
    if args.mode == "dpo":
        trainer = DPOTrainer(
            model=model,
            args=DPOConfig(**config),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            args=GRPOConfig(**config),
            train_dataset=dataset,
            reward_funcs=exact_match_reward,
            processing_class=tokenizer,
        )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
'''


def emit_trl(cfg: Any, out_dir: str | Path) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    dpo = {
        "mode": "dpo",
        "model_name_or_path": "outputs/gpu/sft",
        "dataset_name": "outputs/data/preferences_train.jsonl",
        "beta": 0.1,
        "loss_type": "sigmoid",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5.0e-7,
        "bf16": True,
        "deepspeed": "deepspeed_configs/zero3.json",
        "output_dir": "outputs/gpu/dpo",
    }
    grpo = {
        "mode": "grpo",
        "model_name_or_path": "outputs/gpu/sft",
        "dataset_name": "outputs/data/sft_train.jsonl",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-6,
        "bf16": True,
        "num_generations": 4,
        "temperature": 0.9,
        "deepspeed": "deepspeed_configs/zero3.json",
        "output_dir": "outputs/gpu/grpo",
    }
    dpo_path = target / "dpo.trl.yaml"
    grpo_path = target / "grpo.trl.yaml"
    script_path = target / "train_preference.py"
    dpo_path.write_text(yaml.safe_dump(dpo, sort_keys=False), encoding="utf-8")
    grpo_path.write_text(yaml.safe_dump(grpo, sort_keys=False), encoding="utf-8")
    script_path.write_text(TRL_SCRIPT, encoding="utf-8")
    return {"dpo": dpo_path, "grpo": grpo_path, "script": script_path}
