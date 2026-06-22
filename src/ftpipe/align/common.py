"""Shared sequence encoding, batching, and optimization for alignment stages."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..model.tokenizer import Tokenizer
from ..model.transformer import ModelConfig, sequence_logprobs


@dataclass
class EncodedSequence:
    tokens: list[int]
    labels: list[int]
    response_len: int


def encode_completion(
    tokenizer: Tokenizer, prompt: str, response: str, max_len: int
) -> EncodedSequence:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
    ids = (prompt_ids + response_ids)[:max_len]
    tokens = ids[:-1]
    labels = [-100] * (len(prompt_ids) - 1) + ids[len(prompt_ids) :]
    labels = labels[: len(tokens)]
    if len(labels) < len(tokens):
        labels += [-100] * (len(tokens) - len(labels))
    response_len = max(1, len(ids) - len(prompt_ids))
    return EncodedSequence(tokens=tokens, labels=labels, response_len=response_len)


def pad_encoded(rows: list[EncodedSequence], tokenizer: Tokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(row.tokens) for row in rows)
    tokens = np.full((len(rows), max_len), tokenizer.pad_id, dtype=np.int64)
    labels = np.full((len(rows), max_len), -100, dtype=np.int64)
    segments = np.zeros((len(rows), max_len), dtype=np.int64)
    lengths = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        length = len(row.tokens)
        tokens[index, :length] = row.tokens
        labels[index, :length] = row.labels
        lengths[index] = row.response_len
    return tokens, labels, segments, lengths


def logprobs(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Tokenizer,
    rows: list[EncodedSequence],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    tokens, labels, segments, lengths = pad_encoded(rows, tokenizer)
    logps, cache = sequence_logprobs(params, config, tokens, labels, segment_ids=segments)
    return logps, cache, lengths


def preference_accuracy(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Tokenizer,
    pairs: list[dict[str, Any]],
    batch_size: int = 8,
) -> float:
    correct = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        rows = []
        for pair in chunk:
            chosen = encode_completion(tokenizer, pair["prompt"], pair["chosen"], config.max_seq_len)
            rejected = encode_completion(tokenizer, pair["prompt"], pair["rejected"], config.max_seq_len)
            rows.extend([chosen, rejected])
        logps, _, lengths = logprobs(params, config, tokenizer, rows)
        for index in range(0, len(rows), 2):
            chosen_score = logps[index] / max(1, lengths[index])
            rejected_score = logps[index + 1] / max(1, lengths[index + 1])
            correct += int(chosen_score > rejected_score)
    return correct / len(pairs) if pairs else 0.0


class Adam:
    def __init__(self, params: dict[str, np.ndarray], weight_decay: float = 0.0):
        self.params = params
        self.weight_decay = weight_decay
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.step_count = 0

    def step(self, grads: dict[str, np.ndarray], lr: float, max_norm: float = 1.0) -> float:
        norm = math.sqrt(sum(float(np.sum(grads[name] * grads[name])) for name in self.params))
        factor = min(1.0, max_norm / (norm + 1e-12))
        self.step_count += 1
        beta1, beta2, eps = 0.9, 0.95, 1e-8
        for name in self.params:
            grad = grads[name] * factor
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad * grad
            m_hat = self.m[name] / (1 - beta1**self.step_count)
            v_hat = self.v[name] / (1 - beta2**self.step_count)
            update = m_hat / (np.sqrt(v_hat) + eps)
            if self.weight_decay and self.params[name].ndim == 2:
                update = update + self.weight_decay * self.params[name]
            self.params[name] -= lr * update
        return norm


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
