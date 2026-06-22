"""Calibration activation capture for post-training quantization."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model.tokenizer import Tokenizer
from ..model.transformer import ModelConfig, forward


def capture_activations(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Tokenizer,
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    for record in records:
        prompt = tokenizer.encode(str(record["instruction"]), add_bos=True, add_eos=False)
        response = tokenizer.encode(str(record["response"]), add_bos=False, add_eos=True)
        ids = (prompt + response)[: config.max_seq_len]
        if len(ids) < 2:
            continue
        cache = forward(params, config, np.asarray([ids]), segment_ids=np.zeros((1, len(ids)), dtype=np.int64))
        for layer in range(config.n_layers):
            lc = cache["layers"][layer]
            prefix = f"l{layer}_"
            for target in ("wq", "wk", "wv"):
                collected.setdefault(prefix + target, []).append(lc["h1"][0])
            collected.setdefault(prefix + "wo", []).append(lc["merged"][0])
            collected.setdefault(prefix + "w1", []).append(lc["h2"][0])
            collected.setdefault(prefix + "w2", []).append(lc["f2"][0])
    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}
