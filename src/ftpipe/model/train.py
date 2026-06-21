"""Packing, AdamW training, evaluation, and LoRA-aware optimization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .lora import adapter_grads, apply_lora
from .tokenizer import Tokenizer
from .transformer import ModelConfig, cross_entropy, forward, loss_and_grads


@dataclass
class PackedBatch:
    tokens: np.ndarray
    labels: np.ndarray
    segment_ids: np.ndarray


def _pad_sequence(ids: list[int], labels: list[int], segment: int, seq_len: int, pad_id: int) -> PackedBatch:
    pad = seq_len - len(ids)
    return PackedBatch(
        tokens=np.asarray(ids + [pad_id] * pad, dtype=np.int64),
        labels=np.asarray(labels + [-100] * pad, dtype=np.int64),
        segment_ids=np.asarray([segment] * len(ids) + [-1] * pad, dtype=np.int64),
    )


def _pack(items: list[tuple[list[int], list[int]]], seq_len: int, pad_id: int) -> list[PackedBatch]:
    batches: list[PackedBatch] = []
    ids: list[int] = []
    labels: list[int] = []
    segments: list[int] = []
    segment = 0
    for input_ids, target_labels in items:
        input_ids = input_ids[:seq_len]
        target_labels = target_labels[:seq_len]
        if not input_ids:
            continue
        if len(ids) + len(input_ids) > seq_len:
            batches.append(
                PackedBatch(
                    tokens=np.asarray(ids + [pad_id] * (seq_len - len(ids)), dtype=np.int64),
                    labels=np.asarray(labels + [-100] * (seq_len - len(labels)), dtype=np.int64),
                    segment_ids=np.asarray(segments + [-1] * (seq_len - len(segments)), dtype=np.int64),
                )
            )
            ids, labels, segments = [], [], []
            segment = 0
        ids.extend(int(token) for token in input_ids)
        labels.extend(int(label) for label in target_labels)
        segments.extend([segment] * len(input_ids))
        segment += 1
    if ids:
        batches.append(
            PackedBatch(
                tokens=np.asarray(ids + [pad_id] * (seq_len - len(ids)), dtype=np.int64),
                labels=np.asarray(labels + [-100] * (seq_len - len(labels)), dtype=np.int64),
                segment_ids=np.asarray(segments + [-1] * (seq_len - len(segments)), dtype=np.int64),
            )
        )
    return batches


def pack_sft(
    records: Iterable[dict[str, Any]], tokenizer: Tokenizer, seq_len: int, *, pack: bool = True
) -> list[PackedBatch]:
    items: list[tuple[list[int], list[int]]] = []
    for record in records:
        prompt = str(record["instruction"])
        response = str(record["response"])
        prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
        response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
        seq = [tokenizer.bos_id] + prompt_ids + response_ids
        loss_mask = [False] * (1 + len(prompt_ids)) + [True] * len(response_ids)
        input_ids = seq[:-1]
        target = seq[1:]
        labels = [target[index] if loss_mask[index + 1] else -100 for index in range(len(target))]
        items.append((input_ids, labels))
    if pack:
        return _pack(items, seq_len, tokenizer.pad_id)
    return [
        _pad_sequence(input_ids[:seq_len], labels[:seq_len], 0, seq_len, tokenizer.pad_id)
        for input_ids, labels in items
        if input_ids
    ]


def pack_pretrain(texts: Iterable[str], tokenizer: Tokenizer, seq_len: int) -> list[PackedBatch]:
    items: list[tuple[list[int], list[int]]] = []
    for text in texts:
        ids = [tokenizer.bos_id] + tokenizer.encode(text, add_bos=False, add_eos=True)
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                continue
            items.append((chunk[:-1], chunk[1:]))
    return _pack(items, seq_len, tokenizer.pad_id)


def batch_iterator(
    batches: list[PackedBatch], batch_size: int, rng: np.random.Generator
) -> Iterator[list[PackedBatch]]:
    if not batches:
        raise ValueError("no packed batches")
    while True:
        order = rng.permutation(len(batches))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            yield [batches[int(index)] for index in indices]


def stack_batches(batch: list[PackedBatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([item.tokens for item in batch]),
        np.stack([item.labels for item in batch]),
        np.stack([item.segment_ids for item in batch]),
    )


def _lr_at(step: int, total: int, base_lr: float, warmup_ratio: float) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _clip(grads: dict[str, np.ndarray], max_norm: float) -> float:
    norm = math.sqrt(sum(float(np.sum(grad * grad)) for grad in grads.values()))
    if norm > max_norm:
        factor = max_norm / (norm + 1e-12)
        for name in grads:
            grads[name] *= factor
    return norm


def train_model(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    batches: list[PackedBatch],
    *,
    learning_rate: float,
    steps: int,
    batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    lora: dict[str, np.ndarray] | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    trainable = lora if lora is not None else params
    trainable_names = [name for name in trainable if name != "_scale"]
    m = {name: np.zeros_like(trainable[name]) for name in trainable_names}
    v = {name: np.zeros_like(trainable[name]) for name in trainable_names}
    iterator = batch_iterator(batches, batch_size, rng)
    beta1, beta2 = 0.9, 0.95
    eps = 1e-8
    history: list[dict[str, float]] = []
    log_handle = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_handle = Path(log_path).open("w", encoding="utf-8")
    last_loss = float("nan")
    try:
        for step in range(steps):
            batch = next(iterator)
            tokens, labels, segments = stack_batches(batch)
            effective = apply_lora(params, lora) if lora is not None else params
            loss, grads = loss_and_grads(effective, config, tokens, labels, segments)
            if lora is not None:
                grads = adapter_grads(grads, lora)
            norm = _clip(grads, grad_clip)
            lr = _lr_at(step, steps, learning_rate, warmup_ratio)
            for name in trainable_names:
                grad = grads[name]
                m[name] = beta1 * m[name] + (1 - beta1) * grad
                v[name] = beta2 * v[name] + (1 - beta2) * grad * grad
                m_hat = m[name] / (1 - beta1 ** (step + 1))
                v_hat = v[name] / (1 - beta2 ** (step + 1))
                update = m_hat / (np.sqrt(v_hat) + eps)
                if trainable[name].ndim == 2 and weight_decay:
                    update = update + weight_decay * trainable[name]
                trainable[name] -= lr * update
            last_loss = loss
            record = {"step": step + 1, "loss": loss, "lr": lr, "grad_norm": norm, "seed": seed}
            history.append(record)
            if log_handle is not None:
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        if log_handle is not None:
            log_handle.close()
    return {"params": params if lora is None else None, "lora": lora, "history": history, "final_loss": last_loss}


def evaluate_loss(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    batches: list[PackedBatch],
    *,
    lora: dict[str, np.ndarray] | None = None,
) -> float:
    effective = apply_lora(params, lora) if lora is not None else params
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        tokens = batch.tokens[None, :]
        labels = batch.labels[None, :]
        segments = batch.segment_ids[None, :]
        cache = forward(effective, config, tokens, segment_ids=segments)
        loss, _ = cross_entropy(cache["all_logits"], labels)
        count = int(np.sum(labels != -100))
        total_loss += loss * count
        total_tokens += count
    return total_loss / total_tokens if total_tokens else 0.0
