"""Emit real-stack Axolotl SFT configuration from the pipeline config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _base(cfg: Any, *, lora: bool) -> dict[str, Any]:
    data = {
        "base_model": cfg["project"]["base_model"],
        "model_type": "LlamaForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "strict": False,
        "datasets": [
            {
                "path": "outputs/data/sft_train.jsonl",
                "type": {
                    "field_instruction": "instruction",
                    "field_output": "response",
                    "format": "### Instruction:\n{instruction}\n### Response:\n",
                },
            }
        ],
        "val_set_size": 0.05,
        "sequence_len": 4096,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "num_epochs": int(cfg["sft"]["epochs"]),
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "learning_rate": float(cfg["sft"]["learning_rate"]) * 0.01,
        "warmup_ratio": float(cfg["sft"]["warmup_ratio"]),
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "attn_implementation": "flash_attention_3",
        "deepspeed": "deepspeed_configs/zero3.json",
        "wandb_project": f"ftpipe-{cfg['project']['name']}",
        "wandb_run_id": f"ftpipe-{cfg['project']['name']}-sft",
        "output_dir": "./outputs/gpu/sft",
        "save_steps": 200,
        "logging_steps": 10,
    }
    if lora:
        data.update(
            {
                "adapter": "lora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "output_dir": "./outputs/gpu/sft-lora",
                "wandb_run_id": f"ftpipe-{cfg['project']['name']}-sft-lora",
            }
        )
    return data


def emit_axolotl(cfg: Any, out_dir: str | Path) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    sft_path = target / "sft.axolotl.yaml"
    lora_path = target / "sft.axolotl.lora.yaml"
    sft_path.write_text(yaml.safe_dump(_base(cfg, lora=False), sort_keys=False), encoding="utf-8")
    lora_path.write_text(yaml.safe_dump(_base(cfg, lora=True), sort_keys=False), encoding="utf-8")
    return {"sft": sft_path, "sft_lora": lora_path}
