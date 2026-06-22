"""GGUF v3 container plus Q4_K/Q6_K block codecs (toy-architecture export)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
QK_K = 256
TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q4_K = 12
TYPE_Q6_K = 14

META_UINT32 = 4
META_STRING = 8
META_ARRAY = 9


def _pack_q4_k_scales(scales: np.ndarray, mins: np.ndarray) -> bytes:
    q = bytearray(12)
    for j in range(4):
        q[j] = (int(scales[j]) & 0x3F) | (((int(scales[j + 4]) >> 4) & 0x3) << 6)
        q[j + 4] = (int(mins[j]) & 0x3F) | (((int(mins[j + 4]) >> 4) & 0x3) << 6)
        q[j + 8] = (int(scales[j + 4]) & 0xF) | ((int(mins[j + 4]) & 0xF) << 4)
    return bytes(q)


def _unpack_q4_k_scales(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    q = np.frombuffer(raw, dtype=np.uint8)
    scales = np.zeros(8, dtype=np.int64)
    mins = np.zeros(8, dtype=np.int64)
    for j in range(8):
        if j < 4:
            scales[j] = q[j] & 0x3F
            mins[j] = q[j + 4] & 0x3F
        else:
            scales[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
            mins[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return scales, mins


def encode_q4_k(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != QK_K:
        raise ValueError("Q4_K requires exactly 256 values")
    blocks = values.reshape(8, 32)
    scales_float = np.zeros(8)
    mins_float = np.zeros(8)
    for j in range(8):
        block = blocks[j]
        minimum = float(block.min())
        maximum = float(block.max())
        minimum = min(minimum, 0.0)
        scale = (maximum - minimum) / 15.0
        scales_float[j] = max(scale, 0.0)
        mins_float[j] = max(-minimum, 0.0)
    d = max(float(scales_float.max()) / 63.0, 1e-8)
    dmin = max(float(mins_float.max()) / 63.0, 1e-8)
    scales = np.clip(np.rint(scales_float / d), 0, 63).astype(np.int64)
    mins = np.clip(np.rint(mins_float / dmin), 0, 63).astype(np.int64)
    quantized = np.zeros((8, 32), dtype=np.uint8)
    for j in range(8):
        denominator = d * scales[j]
        if scales[j] == 0 or denominator == 0:
            quantized[j] = 0
            continue
        q = np.rint((blocks[j] + dmin * mins[j]) / denominator)
        quantized[j] = np.clip(q, 0, 15).astype(np.uint8)
    raw = bytearray()
    raw.extend(np.asarray([d], dtype=np.float16).tobytes())
    raw.extend(np.asarray([dmin], dtype=np.float16).tobytes())
    raw.extend(_pack_q4_k_scales(scales, mins))
    for j in range(0, 8, 2):
        raw.extend((quantized[j] | (quantized[j + 1] << 4)).astype(np.uint8).tobytes())
    if len(raw) != 144:
        raise AssertionError(f"Q4_K block size {len(raw)}")
    return bytes(raw)


def decode_q4_k(raw: bytes) -> np.ndarray:
    if len(raw) != 144:
        raise ValueError("Q4_K block must be 144 bytes")
    d = float(np.frombuffer(raw[0:2], dtype=np.float16)[0])
    dmin = float(np.frombuffer(raw[2:4], dtype=np.float16)[0])
    scales, mins = _unpack_q4_k_scales(raw[4:16])
    nibbles = np.frombuffer(raw[16:144], dtype=np.uint8)
    values = np.zeros((8, 32), dtype=np.float64)
    for pair in range(4):
        low = nibbles[pair * 32 : pair * 32 + 32] & 0x0F
        high = nibbles[pair * 32 : pair * 32 + 32] >> 4
        values[pair * 2] = d * scales[pair * 2] * low - dmin * mins[pair * 2]
        values[pair * 2 + 1] = d * scales[pair * 2 + 1] * high - dmin * mins[pair * 2 + 1]
    return values.reshape(-1)


def encode_q6_k(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != QK_K:
        raise ValueError("Q6_K requires exactly 256 values")
    scales_float = np.zeros(16, dtype=np.float64)
    quantized = np.zeros(256, dtype=np.uint8)
    for group in range(16):
        block = values[group * 16 : (group + 1) * 16]
        scales_float[group] = max(float(np.max(np.abs(block))) / 32.0, 0.0)
    d = max(float(scales_float.max()) / 127.0, 1e-8)
    scales = np.clip(np.rint(scales_float / d), -128, 127).astype(np.int8)
    for group in range(16):
        if scales[group] == 0:
            continue
        denominator = d * float(scales[group])
        q = np.rint(values[group * 16 : (group + 1) * 16] / denominator).astype(np.int64)
        quantized[group * 16 : (group + 1) * 16] = np.clip(q, -32, 31).astype(np.int64) + 32
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for half in (0, 1):
        ql_offset = half * 64
        qh_offset = half * 32
        base = half * 128
        for l in range(32):
            q1 = int(quantized[base + l])
            q2 = int(quantized[base + l + 32])
            q3 = int(quantized[base + l + 64])
            q4 = int(quantized[base + l + 96])
            ql[ql_offset + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
            ql[ql_offset + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
            qh[qh_offset + l] = (
                ((q1 >> 4) & 0x3)
                | (((q2 >> 4) & 0x3) << 2)
                | (((q3 >> 4) & 0x3) << 4)
                | (((q4 >> 4) & 0x3) << 6)
            )
    raw = bytearray()
    raw.extend(ql.tobytes())
    raw.extend(qh.tobytes())
    raw.extend(scales.tobytes())
    raw.extend(np.asarray([d], dtype=np.float16).tobytes())
    if len(raw) != 210:
        raise AssertionError(f"Q6_K block size {len(raw)}")
    return bytes(raw)


def decode_q6_k(raw: bytes) -> np.ndarray:
    if len(raw) != 210:
        raise ValueError("Q6_K block must be 210 bytes")
    ql = np.frombuffer(raw[0:128], dtype=np.uint8)
    qh = np.frombuffer(raw[128:192], dtype=np.uint8)
    scales = np.frombuffer(raw[192:208], dtype=np.int8)
    d = float(np.frombuffer(raw[208:210], dtype=np.float16)[0])
    values = np.zeros(256, dtype=np.float64)
    for half in (0, 1):
        ql_offset = half * 64
        qh_offset = half * 32
        base = half * 128
        sc_offset = half * 8
        for l in range(32):
            is_ = l // 16
            q1 = ((int(ql[ql_offset + l]) & 0xF) | (((int(qh[qh_offset + l]) >> 0) & 3) << 4)) - 32
            q2 = ((int(ql[ql_offset + l + 32]) & 0xF) | (((int(qh[qh_offset + l]) >> 2) & 3) << 4)) - 32
            q3 = ((int(ql[ql_offset + l]) >> 4) | (((int(qh[qh_offset + l]) >> 4) & 3) << 4)) - 32
            q4 = ((int(ql[ql_offset + l + 32]) >> 4) | (((int(qh[qh_offset + l]) >> 6) & 3) << 4)) - 32
            values[base + l] = d * float(scales[sc_offset + is_ + 0]) * q1
            values[base + l + 32] = d * float(scales[sc_offset + is_ + 2]) * q2
            values[base + l + 64] = d * float(scales[sc_offset + is_ + 4]) * q3
            values[base + l + 96] = d * float(scales[sc_offset + is_ + 6]) * q4
    return values


def _encode_blocks(values: np.ndarray, kind: str) -> bytes:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    padded = np.zeros((int(np.ceil(flat.size / QK_K)) * QK_K,), dtype=np.float64)
    padded[: flat.size] = flat
    encoder = encode_q4_k if kind == "Q4_K" else encode_q6_k
    return b"".join(encoder(padded[start : start + QK_K]) for start in range(0, padded.size, QK_K))


def _decode_blocks(raw: bytes, kind: str, count: int) -> np.ndarray:
    decoder = decode_q4_k if kind == "Q4_K" else decode_q6_k
    blocks = [decoder(raw[start : start + (144 if kind == "Q4_K" else 210)]) for start in range(0, len(raw), 144 if kind == "Q4_K" else 210)]
    values = np.concatenate(blocks) if blocks else np.zeros(0)
    return values[:count]


def _write_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buffer.extend(struct.pack("<Q", len(encoded)))
    buffer.extend(encoded)


def _write_metadata_value(buffer: bytearray, value: Any) -> None:
    if isinstance(value, str):
        buffer.extend(struct.pack("<I", META_STRING))
        _write_string(buffer, value)
    elif isinstance(value, list):
        buffer.extend(struct.pack("<I", META_ARRAY))
        buffer.extend(struct.pack("<I", META_STRING))
        buffer.extend(struct.pack("<Q", len(value)))
        for item in value:
            _write_string(buffer, str(item))
    elif isinstance(value, int):
        buffer.extend(struct.pack("<I", META_UINT32))
        buffer.extend(struct.pack("<I", int(value)))
    else:
        raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def write_gguf(
    path: str | Path,
    params: dict[str, np.ndarray],
    tokenizer: Any,
    *,
    project: str,
    config: Any,
    seed: int = 0,
) -> dict[str, Any]:
    tensors: list[tuple[str, int, tuple[int, ...], bytes]] = []
    for name in sorted(params):
        value = np.asarray(params[name], dtype=np.float64)
        if name == "tok_emb":
            raw = _encode_blocks(value, "Q6_K")
            tensors.append((name, TYPE_Q6_K, tuple(value.shape), raw))
        elif name == "pos_emb":
            raw = _encode_blocks(value, "Q4_K")
            tensors.append((name, TYPE_Q4_K, tuple(value.shape), raw))
        elif value.ndim == 2:
            kind = "Q6_K" if name.endswith("w2") else "Q4_K"
            raw = _encode_blocks(value, kind)
            tensors.append((name, TYPE_Q6_K if kind == "Q6_K" else TYPE_Q4_K, tuple(value.shape), raw))
        else:
            raw = value.astype(np.float32).tobytes()
            tensors.append((name, TYPE_F32, tuple(value.shape), raw))
    metadata: list[tuple[str, Any]] = [
        ("general.architecture", "ftpipe-tiny"),
        ("general.name", project),
        ("ftpipe.quant_type", "Q4_K_M"),
        ("ftpipe.context_length", int(config.max_seq_len)),
        ("ftpipe.embedding_length", int(config.d_model)),
        ("ftpipe.block_count", int(config.n_layers)),
        ("ftpipe.seed", int(seed)),
        ("tokenizer.ggml.model", "ftpipe-word"),
        ("tokenizer.ggml.tokens", list(tokenizer.tokens)),
    ]
    offsets = []
    cursor = 0
    for _, _, _, raw in tensors:
        cursor = int(np.ceil(cursor / ALIGNMENT) * ALIGNMENT)
        offsets.append(cursor)
        cursor += len(raw)
    header = bytearray()
    header.extend(GGUF_MAGIC)
    header.extend(struct.pack("<I", GGUF_VERSION))
    header.extend(struct.pack("<Q", len(tensors)))
    header.extend(struct.pack("<Q", len(metadata)))
    for key, value in metadata:
        _write_string(header, key)
        _write_metadata_value(header, value)
    for index, (name, dtype, shape, raw) in enumerate(tensors):
        _write_string(header, name)
        header.extend(struct.pack("<I", len(shape)))
        for dimension in reversed(shape):
            header.extend(struct.pack("<Q", int(dimension)))
        header.extend(struct.pack("<I", dtype))
        header.extend(struct.pack("<Q", offsets[index]))
    pad = int(np.ceil(len(header) / ALIGNMENT) * ALIGNMENT) - len(header)
    header.extend(b"\x00" * pad)
    data = bytearray(header)
    for index, (_, _, _, raw) in enumerate(tensors):
        target = len(data) - len(header) if index == 0 else offsets[index] + len(header)
        current = len(data)
        if current < target:
            data.extend(b"\x00" * (target - current))
        data.extend(raw)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(data))
    total_weights = sum(shape and int(np.prod(shape)) or 0 for _, _, shape, _ in tensors)
    report = {
        "path": str(out),
        "size_bytes": out.stat().st_size,
        "bits_per_weight": out.stat().st_size * 8 / total_weights if total_weights else 0.0,
        "tensor_count": len(tensors),
        "tensor_types": {name: ("F16" if dtype == TYPE_F16 else "F32" if dtype == TYPE_F32 else "Q4_K" if dtype == TYPE_Q4_K else "Q6_K") for name, dtype, _, _ in tensors},
    }
    return report


def read_gguf(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = Path(path).read_bytes()
    if raw[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    offset = 4
    version = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    if version != GGUF_VERSION:
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count = struct.unpack_from("<Q", raw, offset)[0]
    offset += 8
    kv_count = struct.unpack_from("<Q", raw, offset)[0]
    offset += 8

    def read_string() -> str:
        nonlocal offset
        length = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        value = raw[offset : offset + length].decode("utf-8")
        offset += length
        return value

    def read_value(value_type: int) -> Any:
        nonlocal offset
        if value_type == META_UINT32:
            value = struct.unpack_from("<I", raw, offset)[0]
            offset += 4
            return int(value)
        if value_type == META_STRING:
            return read_string()
        if value_type == META_ARRAY:
            item_type = struct.unpack_from("<I", raw, offset)[0]
            offset += 4
            count = struct.unpack_from("<Q", raw, offset)[0]
            offset += 8
            return [read_value(item_type) for _ in range(count)]
        raise ValueError(f"unsupported metadata type {value_type}")

    metadata = {}
    for _ in range(kv_count):
        key = read_string()
        value_type = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        metadata[key] = read_value(value_type)
    infos = []
    for _ in range(tensor_count):
        name = read_string()
        n_dims = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        dims = [struct.unpack_from("<Q", raw, offset + 8 * i)[0] for i in range(n_dims)]
        offset += 8 * n_dims
        dtype = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        tensor_offset = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        infos.append((name, dtype, tuple(reversed(dims)), tensor_offset))
    data_start = int(np.ceil(offset / ALIGNMENT) * ALIGNMENT)
    tensors: dict[str, np.ndarray] = {}
    for name, dtype, shape, tensor_offset in infos:
        count = int(np.prod(shape))
        start = data_start + tensor_offset
        if dtype == TYPE_F16:
            values = np.frombuffer(raw[start : start + count * 2], dtype=np.float16).astype(np.float64)
        elif dtype == TYPE_F32:
            values = np.frombuffer(raw[start : start + count * 4], dtype=np.float32).astype(np.float64)
        elif dtype == TYPE_Q4_K:
            nbytes = int(np.ceil(count / QK_K)) * 144
            values = _decode_blocks(raw[start : start + nbytes], "Q4_K", count)
        elif dtype == TYPE_Q6_K:
            nbytes = int(np.ceil(count / QK_K)) * 210
            values = _decode_blocks(raw[start : start + nbytes], "Q6_K", count)
        else:
            raise ValueError(f"unsupported tensor type {dtype}")
        tensors[name] = values.reshape(shape)
    return metadata, tensors
