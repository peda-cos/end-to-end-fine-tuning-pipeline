"""LoRA adapters over the tiny transformer's projection matrices."""

from __future__ import annotations

import numpy as np

TARGETS = ("wq", "wv", "wo", "w1", "w2")


def init_lora(
    params: dict[str, np.ndarray],
    n_layers: int,
    rank: int,
    alpha: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    scale = alpha / rank
    adapters: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        for target in TARGETS:
            weight = params[f"l{layer}_{target}"]
            out_dim, in_dim = weight.shape
            adapters[f"l{layer}_{target}_A"] = rng.normal(0.0, 0.02, size=(rank, in_dim))
            adapters[f"l{layer}_{target}_B"] = np.zeros((out_dim, rank), dtype=np.float64)
    adapters["_scale"] = np.asarray([scale], dtype=np.float64)
    return adapters


def scale_of(adapters: dict[str, np.ndarray]) -> float:
    return float(adapters["_scale"][0])


def apply_lora(params: dict[str, np.ndarray], adapters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in params.items()}
    scale = scale_of(adapters)
    for layer in range(100):
        prefix = f"l{layer}_"
        if prefix + "wq" not in result:
            break
        for target in TARGETS:
            name = prefix + target
            result[name] = result[name] + scale * (adapters[name + "_B"] @ adapters[name + "_A"])
    return result


def adapter_grads(effective_grads: dict[str, np.ndarray], adapters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scale = scale_of(adapters)
    grads: dict[str, np.ndarray] = {}
    for layer in range(100):
        prefix = f"l{layer}_"
        if prefix + "wq_A" not in adapters:
            break
        for target in TARGETS:
            base = prefix + target
            dweight = effective_grads[base]
            a = adapters[base + "_A"]
            b = adapters[base + "_B"]
            grads[base + "_A"] = scale * (b.T @ dweight)
            grads[base + "_B"] = scale * (dweight @ a.T)
    return grads


def merge(params: dict[str, np.ndarray], adapters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return apply_lora(params, adapters)
