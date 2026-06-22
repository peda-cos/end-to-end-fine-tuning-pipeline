"""Group-wise INT4 quantization primitives shared by GPTQ and AWQ."""

from __future__ import annotations

import numpy as np


def group_bounds(in_features: int, group_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + group_size, in_features)) for start in range(0, in_features, group_size)]


def quantize_grouped(weight: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric INT4 quantization with one scale per input group."""
    out_features, in_features = weight.shape
    groups = group_bounds(in_features, group_size)
    scales = np.zeros((out_features, len(groups)), dtype=np.float64)
    quantized = np.zeros_like(weight, dtype=np.int16)
    for group_index, (start, end) in enumerate(groups):
        block = weight[:, start:end]
        scale = np.maximum(np.max(np.abs(block), axis=1) / 7.0, 1e-8)
        codes = np.clip(np.rint(block / scale[:, None]), -8, 7).astype(np.int16)
        quantized[:, start:end] = codes
        scales[:, group_index] = scale
    return quantized, scales


def dequantize_grouped(quantized: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    out_features, in_features = quantized.shape
    result = np.empty((out_features, in_features), dtype=np.float64)
    for group_index, (start, end) in enumerate(group_bounds(in_features, group_size)):
        result[:, start:end] = quantized[:, start:end] * scales[:, group_index : group_index + 1]
    return result


def pack_int4(quantized: np.ndarray) -> np.ndarray:
    """Pack signed INT4 codes (-8..7) two per byte, low nibble first."""
    codes = (np.asarray(quantized, dtype=np.int16) + 8).astype(np.uint8)
    flat = codes.reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, count: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = np.empty(packed.size * 2, dtype=np.int16)
    codes[0::2] = low
    codes[1::2] = high
    return (codes[:count].astype(np.int16) - 8)


def quantized_size_bytes(quantized: np.ndarray, scales: np.ndarray) -> int:
    return int(np.ceil(quantized.size / 2)) + int(scales.size * 4)


def tensor_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))
