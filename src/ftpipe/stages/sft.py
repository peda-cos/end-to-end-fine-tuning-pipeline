"""Supervised fine-tuning and LoRA stages."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..manifest import parameter_sha256, sha256_json
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.lora import apply_lora, init_lora
from ..model.train import PackedBatch, evaluate_loss, pack_sft, train_model
from ..model.transformer import parameter_count
from ..seeds import stage_rng


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_sft(cfg: Any, base_checkpoint: Path) -> dict[str, Any]:
    params, config, tokenizer, _ = load_checkpoint(base_checkpoint)
    train_records = _read(Path(cfg.out_dir) / "data" / "sft_train.jsonl")
    val_records = _read(Path(cfg.out_dir) / "data" / "sft_val.jsonl")
    seq_len = int(cfg["sft"]["sequence_len"])
    pack = bool(cfg["sft"]["sample_packing"])
    train_batches = pack_sft(train_records, tokenizer, seq_len, pack=pack)
    val_batches = pack_sft(val_records, tokenizer, seq_len, pack=pack)
    cfg_steps = min(
        int(cfg["sft"]["max_steps"]),
        max(1, math.ceil(len(train_records) / int(cfg["sft"]["batch_size"])) * int(cfg["sft"]["epochs"])),
    )
    result = train_model(
        params,
        config,
        train_batches,
        learning_rate=float(cfg["sft"]["learning_rate"]),
        steps=cfg_steps,
        batch_size=int(cfg["sft"]["batch_size"]),
        warmup_ratio=float(cfg["sft"]["warmup_ratio"]),
        weight_decay=0.1,
        grad_clip=1.0,
        seed=int(cfg["run"]["seed"]) + 2,
        log_path=Path(cfg.out_dir) / "logs" / "sft.jsonl",
    )
    params = result["params"]
    val_loss = evaluate_loss(params, config, val_batches)
    out = Path(cfg.out_dir) / "models" / "sft"
    info = save_checkpoint(
        out,
        params,
        config,
        tokenizer,
        metadata={
            "stage": "sft",
            "seed": int(cfg["run"]["seed"]),
            "steps": cfg_steps,
            "epochs": int(cfg["sft"]["epochs"]),
            "val_loss": val_loss,
            "train_loss_initial": result["history"][0]["loss"] if result["history"] else None,
            "train_loss_final": result["final_loss"],
            "config_sha256": sha256_json(cfg.data),
        },
    )
    return {
        "checkpoint": out,
        "params": params,
        "config": config,
        "tokenizer": tokenizer,
        "val_loss": val_loss,
        "steps": cfg_steps,
        "train_loss_initial": result["history"][0]["loss"] if result["history"] else 0.0,
        "train_loss_final": result["final_loss"],
        "parameter_sha256": info["parameter_sha256"],
    }


def run_lora(cfg: Any, base_checkpoint: Path) -> dict[str, Any]:
    params, config, tokenizer, _ = load_checkpoint(base_checkpoint)
    train_records = _read(Path(cfg.out_dir) / "data" / "sft_train.jsonl")
    val_records = _read(Path(cfg.out_dir) / "data" / "sft_val.jsonl")
    seq_len = int(cfg["sft"]["sequence_len"])
    pack = bool(cfg["sft"]["sample_packing"])
    train_batches = pack_sft(train_records, tokenizer, seq_len, pack=pack)
    val_batches = pack_sft(val_records, tokenizer, seq_len, pack=pack)
    lora_cfg = cfg["sft"]["lora"]
    rng = stage_rng(int(cfg["run"]["seed"]), "lora")
    adapters = init_lora(params, config.n_layers, int(lora_cfg["rank"]), int(lora_cfg["alpha"]), rng)
    result = train_model(
        params,
        config,
        train_batches,
        learning_rate=float(lora_cfg["learning_rate"]),
        steps=int(lora_cfg["max_steps"]),
        batch_size=int(cfg["sft"]["batch_size"]),
        warmup_ratio=float(cfg["sft"]["warmup_ratio"]),
        weight_decay=0.0,
        grad_clip=1.0,
        seed=int(cfg["run"]["seed"]) + 3,
        lora=adapters,
        log_path=Path(cfg.out_dir) / "logs" / "lora.jsonl",
    )
    merged = apply_lora(params, result["lora"])
    val_loss = evaluate_loss(merged, config, val_batches)
    trainable = sum(value.size for name, value in result["lora"].items() if name != "_scale")
    total = parameter_count(params)
    out = Path(cfg.out_dir) / "models" / "sft_lora"
    info = save_checkpoint(
        out,
        merged,
        config,
        tokenizer,
        metadata={
            "stage": "sft_lora",
            "seed": int(cfg["run"]["seed"]),
            "steps": int(lora_cfg["max_steps"]),
            "rank": int(lora_cfg["rank"]),
            "alpha": int(lora_cfg["alpha"]),
            "trainable_parameters": trainable,
            "trainable_fraction": trainable / total if total else 0.0,
            "val_loss": val_loss,
            "base_parameter_sha256": parameter_sha256(params),
            "config_sha256": sha256_json(cfg.data),
        },
    )
    return {
        "checkpoint": out,
        "params": merged,
        "config": config,
        "tokenizer": tokenizer,
        "val_loss": val_loss,
        "steps": int(lora_cfg["max_steps"]),
        "trainable_fraction": trainable / total if total else 0.0,
        "parameter_sha256": info["parameter_sha256"],
    }
