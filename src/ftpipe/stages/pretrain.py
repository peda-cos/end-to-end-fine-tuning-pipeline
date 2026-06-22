"""Pretraining stage for the local base model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import parameter_sha256, sha256_json
from ..model.checkpoint import save_checkpoint
from ..model.tokenizer import Tokenizer
from ..model.train import evaluate_loss, pack_pretrain, train_model
from ..model.transformer import ModelConfig, init_params, parameter_count
from ..seeds import stage_rng


def run_pretrain(cfg: Any, data_files: dict[str, Path]) -> dict[str, Any]:
    from ..data.synth import read_jsonl

    clean = read_jsonl(data_files["base_corpus"])
    sft = read_jsonl(data_files["sft_train"]) + read_jsonl(data_files["sft_val"])
    texts = [row["text"] for row in clean]
    tokenizer = Tokenizer.build(texts + [row["instruction"] + " " + row["response"] for row in sft])
    out = Path(cfg.out_dir) / "models"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out / "tokenizer.json")
    config = ModelConfig.from_cfg(cfg, tokenizer.vocab_size)
    rng = stage_rng(int(cfg["run"]["seed"]), "pretrain")
    params = init_params(config, rng)
    seq_len = int(cfg["pretrain"]["sequence_len"])
    batches = pack_pretrain(texts, tokenizer, seq_len)
    if len(batches) < 2:
        raise RuntimeError("not enough pretraining batches; increase data.raw_docs")
    val_batches = batches[::10] if len(batches) >= 10 else batches[-1:]
    train_batches = [batch for index, batch in enumerate(batches) if index % 10 != 0] or batches
    steps = int(cfg["pretrain"]["max_steps"])
    result = train_model(
        params,
        config,
        train_batches,
        learning_rate=float(cfg["pretrain"]["learning_rate"]),
        steps=steps,
        batch_size=int(cfg["pretrain"]["batch_size"]),
        warmup_ratio=float(cfg["pretrain"]["warmup_ratio"]),
        weight_decay=float(cfg["pretrain"]["weight_decay"]),
        grad_clip=float(cfg["pretrain"]["grad_clip"]),
        seed=int(cfg["run"]["seed"]) + 1,
        log_path=Path(cfg.out_dir) / "logs" / "pretrain.jsonl",
    )
    params = result["params"]
    val_loss = evaluate_loss(params, config, val_batches)
    checkpoint = out / "base"
    info = save_checkpoint(
        checkpoint,
        params,
        config,
        tokenizer,
        metadata={
            "stage": "base",
            "seed": int(cfg["run"]["seed"]),
            "steps": steps,
            "val_loss": val_loss,
            "config_sha256": sha256_json(cfg.data),
            "parameter_count": parameter_count(params),
        },
    )
    (Path(cfg.out_dir) / "logs" / "pretrain-summary.json").write_text(
        json.dumps({"val_loss": val_loss, "steps": steps, "checkpoint_sha256": info["parameter_sha256"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "params": params,
        "config": config,
        "tokenizer": tokenizer,
        "val_loss": val_loss,
        "steps": steps,
        "parameter_sha256": info["parameter_sha256"],
    }
