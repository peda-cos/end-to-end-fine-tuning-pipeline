"""Activation-aware INT4 weight quantization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import dequantize_grouped, pack_int4, quantize_grouped, quantized_size_bytes, tensor_mse
from .gptq import _targets


def quantize_awq(
    params: dict[str, np.ndarray],
    activations: dict[str, np.ndarray],
    *,
    group_size: int,
    alpha_grid: list[float],
) -> dict[str, Any]:
    quantized_params = {name: value.copy() for name, value in params.items()}
    artifacts: dict[str, np.ndarray] = {}
    layer_mse = {}
    layer_alpha = {}
    total_bytes = 0
    total_weights = 0
    for name in _targets(params, activations):
        original = params[name]
        x = np.asarray(activations[name], dtype=np.float64)
        transposed = original.shape[0] == x.shape[1] and original.shape[0] != original.shape[1]
        weight = original.T if transposed else original
        if weight.shape[1] != x.shape[1]:
            raise ValueError(f"activation shape {x.shape} does not match weight {name} {original.shape}")
        best = None
        for alpha in alpha_grid:
            activation_scale = np.power(np.mean(np.abs(x), axis=0), float(alpha))
            activation_scale = np.maximum(activation_scale, 1e-4)
            activation_scale = activation_scale / np.sqrt(np.max(activation_scale) * np.min(activation_scale))
            folded = weight * activation_scale[None, :]
            codes, scales = quantize_grouped(folded, group_size)
            dequantized_folded = dequantize_grouped(codes, scales, group_size)
            candidate = dequantized_folded / activation_scale[None, :]
            output_mse = float(np.mean(((candidate - weight) @ x.T) ** 2))
            if best is None or output_mse < best[0]:
                best = (output_mse, float(alpha), candidate, codes, scales, activation_scale)
        assert best is not None
        output_mse, alpha, candidate, codes, scales, activation_scale = best
        if transposed:
            candidate = candidate.T
        quantized_params[name] = candidate
        layer_mse[name] = tensor_mse(original, candidate)
        layer_alpha[name] = alpha
        artifacts[name + ".qweight"] = pack_int4(codes)
        artifacts[name + ".scales"] = scales.astype(np.float32)
        artifacts[name + ".channel_scale"] = activation_scale.astype(np.float32)
        artifacts[name + ".shape"] = np.asarray(original.shape, dtype=np.int32)
        artifacts[name + ".transposed"] = np.asarray([1 if transposed else 0], dtype=np.int32)
        total_bytes += quantized_size_bytes(codes, scales) + activation_scale.size * 4
        total_weights += weight.size
    report = {
        "method": "awq_int4",
        "group_size": group_size,
        "alpha_grid": list(alpha_grid),
        "layer_alpha": layer_alpha,
        "quantized_layers": sorted(layer_mse),
        "layer_mse": layer_mse,
        "mean_layer_mse": float(np.mean(list(layer_mse.values()))) if layer_mse else 0.0,
        "size_bytes": total_bytes,
        "bits_per_weight": (total_bytes * 8 / total_weights) if total_weights else 0.0,
    }
    return {"params": quantized_params, "artifacts": artifacts, "report": report}


def save_awq(directory: str | Path, result: dict[str, Any]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "awq.npz", **result["artifacts"])
    np.savez(path / "model.npz", **result["params"])
