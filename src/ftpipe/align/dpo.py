"""Direct Preference Optimization with a frozen reference and beta sweep."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import parameter_sha256, sha256_json
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.transformer import backward_logprobs, sequence_logprobs
from ..seeds import stage_rng
from .common import Adam, encode_completion, logprobs, pad_encoded, preference_accuracy, read_jsonl


def _ref_logps(
    ref_params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    chosen_values = []
    rejected_values = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        rows = []
        for pair in chunk:
            rows.append(encode_completion(tokenizer, pair["prompt"], pair["chosen"], config.max_seq_len))
            rows.append(encode_completion(tokenizer, pair["prompt"], pair["rejected"], config.max_seq_len))
        tokens, labels, segments, _ = pad_encoded(rows, tokenizer)
        logps, _ = sequence_logprobs(ref_params, config, tokens, labels, segment_ids=segments)
        for index in range(0, len(rows), 2):
            chosen_values.append(logps[index])
            rejected_values.append(logps[index + 1])
    return np.asarray(chosen_values), np.asarray(rejected_values)


def train_dpo_beta(
    sft_params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    *,
    beta: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    policy = {name: value.copy() for name, value in sft_params.items()}
    reference = {name: value.copy() for name, value in sft_params.items()}
    ref_chosen, ref_rejected = _ref_logps(reference, config, tokenizer, pairs)
    optimizer = Adam(policy, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    history = []
    log_handle = Path(log_path).open("w", encoding="utf-8")
    try:
        for step in range(steps):
            indices = rng.integers(0, len(pairs), size=batch_size)
            rows = []
            chosen_ref = []
            rejected_ref = []
            for offset, index in enumerate(indices):
                pair = pairs[int(index)]
                rows.append(encode_completion(tokenizer, pair["prompt"], pair["chosen"], config.max_seq_len))
                rows.append(encode_completion(tokenizer, pair["prompt"], pair["rejected"], config.max_seq_len))
                chosen_ref.append(ref_chosen[int(index)])
                rejected_ref.append(ref_rejected[int(index)])
            tokens, labels, segments, _ = pad_encoded(rows, tokenizer)
            logps, cache = sequence_logprobs(policy, config, tokens, labels, segment_ids=segments)
            chosen_logps = logps[0::2]
            rejected_logps = logps[1::2]
            delta = (chosen_logps - np.asarray(chosen_ref)) - (rejected_logps - np.asarray(rejected_ref))
            losses = -np.log(np.clip(1.0 / (1.0 + np.exp(-beta * delta)), 1e-12, 1.0))
            loss = float(losses.mean())
            dloss = -(beta * (1.0 - 1.0 / (1.0 + np.exp(-beta * delta)))) / len(delta)
            upstream = np.zeros(len(rows), dtype=np.float64)
            upstream[0::2] = dloss
            upstream[1::2] = -dloss
            grads = backward_logprobs(policy, cache, labels, upstream)
            norm = optimizer.step(grads, learning_rate)
            record = {
                "step": step + 1,
                "beta": beta,
                "loss": loss,
                "accuracy": float(np.mean(chosen_logps > rejected_logps)),
                "margin": float(np.mean(chosen_logps - rejected_logps)),
                "grad_norm": norm,
            }
            history.append(record)
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        log_handle.close()
    summary = {
        "beta": beta,
        "loss": float(np.mean([record["loss"] for record in history[-10:]])),
        "train_accuracy": float(np.mean([record["accuracy"] for record in history[-10:]])),
        "steps": steps,
    }
    return policy, {"summary": summary, "history": history}


def run_dpo(cfg: Any, sft_checkpoint: Path) -> dict[str, Any]:
    sft_params, config, tokenizer, metadata = load_checkpoint(sft_checkpoint)
    train_pairs = read_jsonl(Path(cfg.out_dir) / "data" / "preferences_train.jsonl")
    val_pairs = read_jsonl(Path(cfg.out_dir) / "data" / "preferences_val.jsonl")
    out = Path(cfg.out_dir) / "models"
    out.mkdir(parents=True, exist_ok=True)
    sweep = []
    candidates: dict[float, dict[str, np.ndarray]] = {}
    for index, beta in enumerate(cfg["dpo"]["beta_grid"]):
        policy, result = train_dpo_beta(
            sft_params,
            config,
            tokenizer,
            train_pairs,
            beta=float(beta),
            steps=int(cfg["dpo"]["max_steps_per_beta"]),
            batch_size=int(cfg["dpo"]["batch_size"]),
            learning_rate=float(cfg["dpo"]["learning_rate"]),
            seed=int(cfg["run"]["seed"]) + 40 + index,
            log_path=Path(cfg.out_dir) / "logs" / f"dpo-beta-{beta}.jsonl",
        )
        accuracy = preference_accuracy(policy, config, tokenizer, val_pairs)
        item = {
            "beta": float(beta),
            "preference_accuracy": accuracy,
            "loss": result["summary"]["loss"],
            "selected": False,
        }
        sweep.append(item)
        candidates[float(beta)] = policy
        save_checkpoint(
            out / f"dpo_beta_{beta}",
            policy,
            config,
            tokenizer,
            metadata={
                "stage": "dpo",
                "beta": float(beta),
                "seed": int(cfg["run"]["seed"]),
                "steps": int(cfg["dpo"]["max_steps_per_beta"]),
                "preference_accuracy": accuracy,
                "config_sha256": sha256_json(cfg.data),
            },
        )
    selected = sorted(
        sweep,
        key=lambda item: (-item["preference_accuracy"], item["loss"], item["beta"]),
    )[0]
    for item in sweep:
        item["selected"] = item["beta"] == selected["beta"]
    beta = float(selected["beta"])
    params = candidates[beta]
    accuracy = preference_accuracy(params, config, tokenizer, val_pairs)
    checkpoint = out / "dpo"
    info = save_checkpoint(
        checkpoint,
        params,
        config,
        tokenizer,
        metadata={
            "stage": "dpo",
            "beta": beta,
            "seed": int(cfg["run"]["seed"]),
            "steps": int(cfg["dpo"]["max_steps_per_beta"]),
            "preference_accuracy": accuracy,
            "beta_sweep": sweep,
            "config_sha256": sha256_json(cfg.data),
        },
    )
    (Path(cfg.out_dir) / "eval").mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "eval" / "beta-sweep.json").write_text(
        json.dumps(sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "checkpoint": checkpoint,
        "params": params,
        "config": config,
        "tokenizer": tokenizer,
        "beta": beta,
        "preference_accuracy": accuracy,
        "sweep": sweep,
        "parameter_sha256": info["parameter_sha256"],
    }
