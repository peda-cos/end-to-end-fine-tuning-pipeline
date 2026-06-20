"""Seeded train/validation splits with cross-split near-duplicate isolation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .dedup import MinHasher, normalize_text


def prompt_hash(record: dict[str, Any]) -> str:
    text = str(record.get("instruction", record.get("prompt", record.get("text", ""))))
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def split_records(
    records: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return [], [], {"train": 0, "val": 0, "rejected_cross_split": 0}
    rng = np.random.default_rng(seed)
    order = np.arange(len(records))
    rng.shuffle(order)
    shuffled = [records[int(index)] for index in order]
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    if not train:
        train = val
        val = []

    hasher = MinHasher(ngram=3, num_hashes=16, bands=4, rows=4, seed=seed)
    def content(record: dict[str, Any]) -> str:
        return str(record.get("content", record.get("instruction", "")))

    train_sigs = [(record, hasher.signature(content(record))) for record in train]
    train_prompt_hashes = {prompt_hash(record) for record in train}
    kept_val = []
    rejected = 0
    for record in val:
        signature = hasher.signature(content(record))
        duplicate = prompt_hash(record) in train_prompt_hashes
        if not duplicate:
            duplicate = any(hasher.jaccard(signature, other) >= 0.95 for _, other in train_sigs)
        if duplicate:
            rejected += 1
        else:
            kept_val.append(record)
    if not kept_val and train:
        kept_val = [train.pop()]
    report = {
        "train": len(train),
        "val": len(kept_val),
        "rejected_cross_split": rejected,
        "seed": seed,
        "val_fraction": val_fraction,
    }
    return train, kept_val, report
