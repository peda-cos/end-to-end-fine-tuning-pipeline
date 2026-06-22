"""Serving stage: train a draft head, benchmark HTTP, and optionally serve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..align.common import read_jsonl
from ..model.checkpoint import load_checkpoint
from ..model.draft import train_draft
from ..model.train import pack_sft
from .bench import run_bench


def run_serve(cfg: Any, checkpoint: Path) -> dict[str, Any]:
    params, config, tokenizer, _ = load_checkpoint(checkpoint)
    records = read_jsonl(Path(cfg.out_dir) / "data" / "sft_train.jsonl")
    batches = pack_sft(records, tokenizer, int(cfg["sft"]["sequence_len"]))
    draft_dir = Path(cfg.out_dir) / "models" / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    draft, history = train_draft(
        params,
        config,
        tokenizer,
        batches,
        steps=int(cfg["serve"]["draft_train_steps"]),
        batch_size=8,
        learning_rate=0.003,
        seed=int(cfg["run"]["seed"]) + 70,
        log_path=Path(cfg.out_dir) / "logs" / "draft.jsonl",
    )
    np.savez(draft_dir / "draft.npz", **draft)
    (draft_dir / "metadata.json").write_text(
        json.dumps(
            {
                "target_checkpoint": str(checkpoint),
                "steps": int(cfg["serve"]["draft_train_steps"]),
                "final_loss": history[-1]["loss"] if history else 0.0,
                "seed": int(cfg["run"]["seed"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bench = run_bench(cfg, checkpoint, draft_dir)
    return {"draft_dir": draft_dir, "draft_history": history, "bench": bench}
