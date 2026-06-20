"""Deterministic seed offsets, one per pipeline stage."""

from __future__ import annotations

import numpy as np

STAGE_OFFSETS = {
    "data": 0,
    "pretrain": 1_000,
    "sft": 2_000,
    "lora": 3_000,
    "dpo": 4_000,
    "grpo": 5_000,
    "quant": 6_000,
    "draft": 7_000,
    "eval": 8_000,
    "safety": 9_000,
    "serve": 10_000,
}


def stage_rng(seed: int, stage: str) -> np.random.Generator:
    return np.random.default_rng(int(seed) + STAGE_OFFSETS[stage])


def stage_seed(seed: int, stage: str) -> int:
    return int(seed) + STAGE_OFFSETS[stage]
