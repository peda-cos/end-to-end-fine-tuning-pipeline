"""Safety evaluation stage: local guard and output shield."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..align.common import read_jsonl
from ..model.checkpoint import load_checkpoint
from ..model.generate import generate
from ..safetymodels.guard import Guard, train_guard
from ..safetymodels.shield import Shield, train_shield


def _safety_rows(cfg: Any, name: str) -> list[dict[str, Any]]:
    return read_jsonl(Path(cfg.root) / "fixtures" / "safety" / name)


def run_safety(cfg: Any, checkpoints: dict[str, Path]) -> dict[str, Any]:
    train_rows = _safety_rows(cfg, "safety_train.jsonl")
    dev_rows = _safety_rows(cfg, "safety_dev.jsonl")
    bias_rows = _safety_rows(cfg, "bias_probe.jsonl")
    out = Path(cfg.out_dir) / "safety"
    out.mkdir(parents=True, exist_ok=True)
    guard = train_guard(Path(cfg.root) / "fixtures" / "safety" / "safety_train.jsonl", cfg)
    shield = train_shield(Path(cfg.root) / "fixtures" / "safety" / "safety_train.jsonl", cfg)
    guard.save(out / "guard.json")
    shield.save(out / "shield.json")
    models = ["base", "sft", "dpo", "grpo"]
    pass_rates = {}
    refusal_rates = {}
    shield_unsafe_blocked = 0
    shield_unsafe_total = 0
    shield_benign_blocked = 0
    shield_benign_total = 0
    category_counts: dict[str, int] = {}
    loaded = {}
    for name in models:
        checkpoint = checkpoints.get(name)
        if checkpoint is None or not Path(checkpoint).exists():
            continue
        loaded[name] = load_checkpoint(checkpoint)
    for name, (params, config, tokenizer, _) in loaded.items():
        passed = 0
        refused = 0
        for row in dev_rows:
            prompt = row["text"]
            allowed, label, _ = guard.check(prompt)
            if row["safe"]:
                shield_benign_total += 1
            else:
                shield_unsafe_total += 1
            if not allowed:
                if row["safe"]:
                    shield_benign_blocked += 1
                else:
                    shield_unsafe_blocked += 1
                    passed += 1
                continue
            completion = generate(params, config, tokenizer, prompt, max_new_tokens=8, temperature=0.0).text
            output_allowed, label, _ = shield.check(completion)
            if row["safe"]:
                if output_allowed:
                    passed += 1
                else:
                    shield_benign_blocked += 1
            else:
                if output_allowed:
                    pass
                else:
                    shield_unsafe_blocked += 1
                    passed += 1
            category = row.get("category")
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
        pass_rates[name] = passed / len(dev_rows) if dev_rows else 0.0
        bias_refused = 0
        for row in bias_rows:
            allowed, _, _ = guard.check(row["text"])
            if not allowed:
                bias_refused += 1
                continue
            completion = generate(params, config, tokenizer, row["text"], max_new_tokens=8, temperature=0.0).text
            if not shield.check(completion)[0]:
                bias_refused += 1
        refusal_rates[name] = bias_refused / len(bias_rows) if bias_rows else 0.0
    bias_guard_fp = sum(1 for row in bias_rows if not guard.check(row["text"])[0]) / len(bias_rows)
    result = {
        "models": sorted(pass_rates),
        "guard_pass_rate": pass_rates,
        "shield_block_rate_unsafe": shield_unsafe_blocked / shield_unsafe_total if shield_unsafe_total else 0.0,
        "shield_false_positive_rate": shield_benign_blocked / shield_benign_total if shield_benign_total else 0.0,
        "bias_probe_guard_fp_rate": bias_guard_fp,
        "bias_probe_refusal_rate": refusal_rates,
        "category_counts": dict(sorted(category_counts.items())),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
