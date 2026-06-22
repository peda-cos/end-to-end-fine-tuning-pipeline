"""Evaluation matrix stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..align.common import preference_accuracy, read_jsonl, write_jsonl
from ..evaluate.metrics import (
    evaluate_mmlu,
    evaluate_mt_bench,
    evaluate_task,
    predict_task_rows,
)
from ..model.checkpoint import load_checkpoint

MODEL_ORDER = ("base", "sft", "sft_lora", "dpo", "grpo")


def _limit(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return rows[:count]


def run_eval(cfg: Any, checkpoints: dict[str, Path]) -> dict[str, Any]:
    fixtures = Path(cfg.root) / "fixtures"
    task_rows = _limit(read_jsonl(fixtures / "eval" / "task_test.jsonl"), int(cfg["eval"]["task_items"]))
    mmlu_rows = _limit(
        read_jsonl(fixtures / "benchmarks" / "mmlu_pro_style.jsonl"), int(cfg["eval"]["mmlu_items"])
    )
    mt_rows = _limit(
        read_jsonl(fixtures / "benchmarks" / "mt_bench_v2_style.jsonl"), int(cfg["eval"]["mt_bench_items"])
    )
    preference_rows = _limit(
        read_jsonl(fixtures / "benchmarks" / "rewardbench_2_style.jsonl"),
        int(cfg["eval"]["rewardbench_items"]),
    )
    metrics: dict[str, dict[str, float]] = {
        "task_exact_match": {},
        "task_category_accuracy": {},
        "task_format_valid": {},
        "mmlu_pro_style": {},
        "mt_bench_v2_style": {},
        "rewardbench_2_style": {},
    }
    predictions_dir = Path(cfg.out_dir) / "eval" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []
    for name in MODEL_ORDER:
        checkpoint = checkpoints.get(name)
        if checkpoint is None or not Path(checkpoint).exists():
            continue
        params, config, tokenizer, _ = load_checkpoint(checkpoint)
        task = evaluate_task(params, config, tokenizer, task_rows)
        mmlu = evaluate_mmlu(params, config, tokenizer, mmlu_rows)
        mt = evaluate_mt_bench(params, config, tokenizer, mt_rows)
        preference = preference_accuracy(params, config, tokenizer, preference_rows)
        for key, value in {**task, **mmlu, **mt}.items():
            metrics[key][name] = float(value)
        metrics["rewardbench_2_style"][name] = float(preference)
        write_jsonl(predictions_dir / f"{name}-task.jsonl", predict_task_rows(params, config, tokenizer, task_rows))
        evaluated.append(name)
    deltas = {
        name: {
            key: metrics[key][name] - metrics[key].get("base", 0.0)
            for key in (
                "task_exact_match",
                "mmlu_pro_style",
                "mt_bench_v2_style",
                "rewardbench_2_style",
            )
        }
        for name in evaluated
    }
    serving = "dpo"
    if "dpo" in evaluated and "sft" in evaluated:
        if metrics["task_exact_match"]["dpo"] < metrics["task_exact_match"]["sft"] - 0.02:
            serving = "sft"
    if serving not in evaluated:
        serving = evaluated[0] if evaluated else "base"
    result = {
        "models": evaluated,
        "metrics": metrics,
        "deltas_vs_base": deltas,
        "serving_checkpoint": serving,
    }
    out = Path(cfg.out_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
