"""Group-relative policy optimization with a verifiable structured reward."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import sha256_json
from ..model.checkpoint import load_checkpoint, save_checkpoint
from ..model.generate import generate
from ..model.transformer import backward_logprobs, forward, sequence_logprobs
from ..seeds import stage_rng
from .common import Adam, EncodedSequence, pad_encoded, read_jsonl


def parse_completion(text: str) -> tuple[bool, str | None, str | None]:
    parts = text.strip().lower().split()
    if len(parts) != 2:
        return False, None, None
    category, priority = parts
    valid = category in {"billing", "technical", "shipping", "account"} and priority in {"low", "high"}
    return valid, category if valid else None, priority if valid else None


def reward_completion(text: str, expected_category: str, expected_priority: str, weights: dict[str, float]) -> float:
    valid, category, priority = parse_completion(text)
    reward = 0.0
    if valid:
        reward += float(weights["format"])
        if category == expected_category:
            reward += float(weights["category"])
        if category == expected_category and priority == expected_priority:
            reward += float(weights["exact"])
    return reward


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    if std < 1e-4:
        return np.zeros_like(rewards, dtype=np.float64)
    return (rewards - mean) / std


def _sample(
    params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], str]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if len(prompt_ids) + max_new_tokens > config.max_seq_len:
        prompt_ids = prompt_ids[-(config.max_seq_len - max_new_tokens) :]
    ids = list(prompt_ids)
    completion: list[int] = []
    for _ in range(max_new_tokens):
        cache = forward(params, config, np.asarray([ids]), positions=np.asarray([[len(ids) - 1]]))
        logits = cache["logits"][0, 0]
        scaled = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
        if 0 < top_k < len(scaled):
            threshold = np.partition(scaled, -top_k)[-top_k]
            scaled = np.where(scaled < threshold, -np.inf, scaled)
        scaled = scaled - scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
        completion.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return prompt_ids, completion, tokenizer.decode(completion)


def _encoded_from_ids(tokenizer: Any, prompt_ids: list[int], completion: list[int]) -> EncodedSequence:
    ids = prompt_ids + completion
    tokens = ids[:-1]
    labels = [-100] * (len(prompt_ids) - 1) + ids[len(prompt_ids) :]
    labels = labels[: len(tokens)]
    return EncodedSequence(tokens=tokens, labels=labels, response_len=max(1, len(ids) - len(prompt_ids)))


def evaluate_grpo(
    params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int = 8,
) -> dict[str, float]:
    exact = 0
    format_valid = 0
    for record in records:
        completion = generate(
            params,
            config,
            tokenizer,
            record["instruction"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        ).text
        valid, category, priority = parse_completion(completion)
        format_valid += int(valid)
        exact += int(valid and category == record["category"] and priority == record["priority"])
    total = max(1, len(records))
    return {"exact_match": exact / total, "format_valid": format_valid / total}


def run_grpo(cfg: Any, sft_checkpoint: Path) -> dict[str, Any]:
    params, config, tokenizer, _ = load_checkpoint(sft_checkpoint)
    policy = {name: value.copy() for name, value in params.items()}
    reference = {name: value.copy() for name, value in params.items()}
    records = [row for row in read_jsonl(Path(cfg.out_dir) / "data" / "sft_train.jsonl") if row.get("kind") == "ticket"]
    val_records = [
        row for row in read_jsonl(Path(cfg.out_dir) / "data" / "sft_val.jsonl") if row.get("kind") == "ticket"
    ]
    if not records:
        raise RuntimeError("GRPO needs SFT ticket records")
    g_cfg = cfg["grpo"]
    weights = {name: float(value) for name, value in g_cfg["reward"].items()}
    rng = np.random.default_rng(int(cfg["run"]["seed"]) + 50)
    optimizer = Adam(policy, weight_decay=0.0)
    log_handle = (Path(cfg.out_dir) / "logs" / "grpo.jsonl").open("w", encoding="utf-8")
    history = []
    try:
        for step in range(int(g_cfg["max_steps"])):
            group_rows = []
            group_rewards = []
            group_meta = []
            indices = rng.integers(0, len(records), size=int(g_cfg["batch_prompts"]))
            for index in indices:
                record = records[int(index)]
                rewards = []
                rows = []
                for _ in range(int(g_cfg["num_generations"])):
                    prompt_ids, completion, text = _sample(
                        policy,
                        config,
                        tokenizer,
                        record["instruction"],
                        max_new_tokens=6,
                        temperature=float(g_cfg["temperature"]),
                        top_k=int(g_cfg["top_k"]),
                        rng=rng,
                    )
                    rows.append(_encoded_from_ids(tokenizer, prompt_ids, completion))
                    rewards.append(reward_completion(text, record["category"], record["priority"], weights))
                group_rows.extend(rows)
                group_rewards.extend(rewards)
                group_meta.append((float(np.mean(rewards)), float(np.std(rewards))))
            rewards_array = np.asarray(group_rewards, dtype=np.float64)
            num_generations = int(g_cfg["num_generations"])
            advantages = np.concatenate(
                [
                    group_advantages(rewards_array[start : start + num_generations])
                    for start in range(0, len(rewards_array), num_generations)
                ]
            )
            tokens, labels, segments, lengths = pad_encoded(group_rows, tokenizer)
            policy_logps, policy_cache = sequence_logprobs(
                policy, config, tokens, labels, segment_ids=segments
            )
            ref_logps, _ = sequence_logprobs(reference, config, tokens, labels, segment_ids=segments)
            upstream = (-advantages + float(g_cfg["kl_coef"]) * lengths) / max(1, len(group_rows))
            grads = backward_logprobs(policy, policy_cache, labels, upstream)
            norm = optimizer.step(grads, float(g_cfg["learning_rate"]))
            kl = float(np.mean((policy_logps - ref_logps) / np.maximum(1, lengths)))
            record = {
                "step": step + 1,
                "reward_mean": float(rewards_array.mean()),
                "reward_std": float(rewards_array.std()),
                "exact_rate": float(np.mean(rewards_array >= weights["exact"] + weights["format"] + weights["category"] - 1e-9)),
                "kl": kl,
                "grad_norm": norm,
            }
            history.append(record)
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        log_handle.close()
    evaluation = evaluate_grpo(policy, config, tokenizer, val_records)
    checkpoint = Path(cfg.out_dir) / "models" / "grpo"
    info = save_checkpoint(
        checkpoint,
        policy,
        config,
        tokenizer,
        metadata={
            "stage": "grpo",
            "seed": int(cfg["run"]["seed"]),
            "steps": int(g_cfg["max_steps"]),
            "reward": weights,
            "evaluation": evaluation,
            "config_sha256": sha256_json(cfg.data),
        },
    )
    return {
        "checkpoint": checkpoint,
        "params": policy,
        "config": config,
        "tokenizer": tokenizer,
        "evaluation": evaluation,
        "history": history,
        "parameter_sha256": info["parameter_sha256"],
    }
