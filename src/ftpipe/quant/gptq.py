"""GPTQ-style Hessian-aware INT4 weight quantization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import dequantize_grouped, group_bounds, pack_int4, quantize_grouped, quantized_size_bytes, tensor_mse


def _targets(params: dict[str, np.ndarray], activations: dict[str, np.ndarray]) -> list[str]:
    return sorted(
        name
        for name, value in params.items()
        if value.ndim == 2 and name in activations and name not in {"tok_emb", "pos_emb", "lm_head"}
    )


def quantize_gptq(
    params: dict[str, np.ndarray],
    activations: dict[str, np.ndarray],
    *,
    group_size: int,
    damp_percent: float,
    act_order: bool,
) -> dict[str, Any]:
    quantized_params = {name: value.copy() for name, value in params.items()}
    artifacts: dict[str, np.ndarray] = {}
    layer_mse = {}
    total_bytes = 0
    total_weights = 0
    for name in _targets(params, activations):
        original = params[name]
        x = np.asarray(activations[name], dtype=np.float64)
        transposed = original.shape[0] == x.shape[1] and original.shape[0] != original.shape[1]
        weight = original.T if transposed else original
        if weight.shape[1] != x.shape[1]:
            raise ValueError(f"activation shape {x.shape} does not match weight {name} {original.shape}")
        out_features, in_features = weight.shape
        count = x.shape[0]
        hessian = (x.T @ x) / max(1, count)
        damp = max(damp_percent * float(np.mean(np.diag(hessian))), 1e-6)
        hessian = hessian + damp * np.eye(in_features)
        try:
            hinv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            hinv = np.linalg.pinv(hessian)
        perm = np.argsort(-np.diag(hessian)) if act_order else np.arange(in_features)
        work = weight[:, perm].copy()
        hperm = hinv[np.ix_(perm, perm)]
        groups = group_bounds(in_features, group_size)
        scales = np.zeros((out_features, len(groups)), dtype=np.float64)
        for group_index, (start, end) in enumerate(groups):
            scales[:, group_index] = np.maximum(np.max(np.abs(work[:, start:end]), axis=1) / 7.0, 1e-8)
        codes = np.zeros_like(work, dtype=np.int16)
        for column in range(in_features):
            group_index = column // group_size
            scale = scales[:, group_index]
            current = work[:, column]
            code = np.clip(np.rint(current / scale), -8, 7).astype(np.int16)
            codes[:, column] = code
            error = current - code * scale
            if column + 1 < in_features:
                ratio = hperm[column, column + 1 :] / hperm[column, column]
                work[:, column + 1 :] -= error[:, None] * ratio[None, :]
        quantized_permuted = dequantize_grouped(codes, scales, group_size)
        result = np.empty_like(weight)
        result[:, perm] = quantized_permuted
        if transposed:
            result = result.T
        quantized_params[name] = result
        layer_mse[name] = tensor_mse(original, result)
        packed = pack_int4(codes)
        artifacts[name + ".qweight"] = packed
        artifacts[name + ".scales"] = scales.astype(np.float32)
        artifacts[name + ".perm"] = perm.astype(np.int32)
        artifacts[name + ".shape"] = np.asarray(original.shape, dtype=np.int32)
        artifacts[name + ".transposed"] = np.asarray([1 if transposed else 0], dtype=np.int32)
        total_bytes += quantized_size_bytes(codes, scales) + perm.size * 4
        total_weights += weight.size
    report = {
        "method": "gptq_int4",
        "group_size": group_size,
        "act_order": act_order,
        "damp_percent": damp_percent,
        "quantized_layers": sorted(layer_mse),
        "layer_mse": layer_mse,
        "mean_layer_mse": float(np.mean(list(layer_mse.values()))) if layer_mse else 0.0,
        "size_bytes": total_bytes,
        "bits_per_weight": (total_bytes * 8 / total_weights) if total_weights else 0.0,
        "kernel_target": "gptq_marlin",
    }
    return {"params": quantized_params, "artifacts": artifacts, "report": report}


def save_gptq(directory: str | Path, result: dict[str, Any]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "gptq.npz", **result["artifacts"])
    np.savez(path / "model.npz", **result["params"])
