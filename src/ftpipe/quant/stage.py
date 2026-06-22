"""Quantization stage: GPTQ, AWQ, GGUF, and dequantized accuracy checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..align.common import read_jsonl
from ..evaluate.metrics import evaluate_task
from ..manifest import sha256_json
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..quant.awq import quantize_awq
from ..quant.calibrate import capture_activations
from ..quant.gguf import read_gguf, write_gguf
from ..quant.gptq import quantize_gptq


def _task_rows(cfg: Any, limit: int = 20) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(cfg.root) / "fixtures" / "eval" / "task_test.jsonl")
    return rows[:limit]


def _evaluate(params: dict[str, np.ndarray], config: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    return evaluate_task(params, config, tokenizer, rows)


def run_quant(cfg: Any, source_checkpoint: Path) -> dict[str, Any]:
    params, config, tokenizer, _ = load_checkpoint(source_checkpoint)
    calibration_records = read_jsonl(Path(cfg.out_dir) / "data" / "sft_train.jsonl")[
        : int(cfg["quant"]["calibration_samples"])
    ]
    activations = capture_activations(params, config, tokenizer, calibration_records)
    task_rows = _task_rows(cfg)
    baseline = _evaluate(params, config, tokenizer, task_rows)
    out = Path(cfg.out_dir) / "quant"
    out.mkdir(parents=True, exist_ok=True)
    reports = {}

    gptq = quantize_gptq(
        params,
        activations,
        group_size=int(cfg["quant"]["gptq"]["group_size"]),
        damp_percent=float(cfg["quant"]["gptq"]["damp_percent"]),
        act_order=bool(cfg["quant"]["gptq"]["act_order"]),
    )
    gptq_dir = out / "gptq_int4"
    gptq_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(gptq_dir, gptq["params"], config, tokenizer, {"stage": "gptq_int4"})
    np.savez(gptq_dir / "gptq.npz", **gptq["artifacts"])
    np.savez(gptq_dir / "model.npz", **gptq["params"])
    gptq_metrics = _evaluate(gptq["params"], config, tokenizer, task_rows)
    gptq["report"].update(
        {
            "task_exact_match": gptq_metrics["task_exact_match"],
            "delta_vs_unquantized": gptq_metrics["task_exact_match"] - baseline["task_exact_match"],
            "unquantized_task_exact_match": baseline["task_exact_match"],
        }
    )
    reports["gptq_int4"] = gptq["report"]

    awq = quantize_awq(
        params,
        activations,
        group_size=int(cfg["quant"]["awq"]["group_size"]),
        alpha_grid=[float(value) for value in cfg["quant"]["awq"]["alpha_grid"]],
    )
    awq_dir = out / "awq_int4"
    awq_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(awq_dir, awq["params"], config, tokenizer, {"stage": "awq_int4"})
    np.savez(awq_dir / "awq.npz", **awq["artifacts"])
    np.savez(awq_dir / "model.npz", **awq["params"])
    awq_metrics = _evaluate(awq["params"], config, tokenizer, task_rows)
    awq["report"].update(
        {
            "task_exact_match": awq_metrics["task_exact_match"],
            "delta_vs_unquantized": awq_metrics["task_exact_match"] - baseline["task_exact_match"],
            "unquantized_task_exact_match": baseline["task_exact_match"],
        }
    )
    reports["awq_int4"] = awq["report"]

    gguf_path = out / "gguf_q4_k_m.gguf"
    gguf_report = write_gguf(
        gguf_path,
        params,
        tokenizer,
        project=str(cfg["project"]["name"]),
        config=config,
        seed=int(cfg["run"]["seed"]),
    )
    metadata, tensors = read_gguf(gguf_path)
    gguf_params = {name: value for name, value in tensors.items() if name != "lm_head"}
    gguf_metrics = _evaluate(gguf_params, config, tokenizer, task_rows)
    gguf_report.update(
        {
            "task_exact_match": gguf_metrics["task_exact_match"],
            "delta_vs_unquantized": gguf_metrics["task_exact_match"] - baseline["task_exact_match"],
            "metadata_keys": sorted(metadata),
            "quant_type": metadata.get("ftpipe.quant_type"),
            "tensor_count_read": len(tensors),
        }
    )
    reports["gguf_q4_k_m"] = gguf_report

    report = {
        "source_checkpoint": str(source_checkpoint),
        "baseline_task_exact_match": baseline["task_exact_match"],
        "reports": reports,
        "config_sha256": sha256_json(cfg.data),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, item in reports.items():
        (out / f"{name}-report.json").write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "gptq": gptq,
        "awq": awq,
        "gguf": gguf_report,
        "reports": reports,
        "baseline": baseline,
        "awq_checkpoint": awq_dir,
        "gptq_checkpoint": gptq_dir,
        "gguf_path": gguf_path,
    }
