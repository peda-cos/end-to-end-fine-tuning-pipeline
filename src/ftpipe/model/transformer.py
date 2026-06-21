"""Tiny decoder-only transformer with manual reverse-mode gradients in NumPy.

The local model is deliberately small; the correctness contract is a
finite-difference gradient check plus an exact packing/attention test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-5
MASK_VALUE = -1e9


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_seq_len: int = 128
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @classmethod
    def from_cfg(cls, cfg: Any, vocab_size: int) -> "ModelConfig":
        m = cfg["model"]
        return cls(
            vocab_size=vocab_size,
            d_model=int(m["d_model"]),
            n_layers=int(m["n_layers"]),
            n_heads=int(m["n_heads"]),
            d_ff=int(m["d_ff"]),
            max_seq_len=int(m["max_seq_len"]),
            tie_embeddings=bool(m["tie_embeddings"]),
        )


def init_params(config: ModelConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    d, f = config.d_model, config.d_ff
    params: dict[str, np.ndarray] = {
        "tok_emb": rng.normal(0.0, 0.02, size=(config.vocab_size, d)),
        "pos_emb": rng.normal(0.0, 0.02, size=(config.max_seq_len, d)),
    }
    if not config.tie_embeddings:
        params["lm_head"] = rng.normal(0.0, 0.02, size=(config.vocab_size, d))
    for layer in range(config.n_layers):
        prefix = f"l{layer}_"
        params[prefix + "ln1_g"] = np.ones(d, dtype=np.float64)
        params[prefix + "ln1_b"] = np.zeros(d, dtype=np.float64)
        for name in ("wq", "wk", "wv", "wo"):
            params[prefix + name] = rng.normal(0.0, 0.02, size=(d, d))
            params[prefix + "b" + name[1:]] = np.zeros(d, dtype=np.float64)
        params[prefix + "ln2_g"] = np.ones(d, dtype=np.float64)
        params[prefix + "ln2_b"] = np.zeros(d, dtype=np.float64)
        params[prefix + "w1"] = rng.normal(0.0, 0.02, size=(d, f))
        params[prefix + "b1"] = np.zeros(f, dtype=np.float64)
        params[prefix + "w2"] = rng.normal(0.0, 0.02, size=(f, d))
        params[prefix + "b2"] = np.zeros(d, dtype=np.float64)
    params["lnf_g"] = np.ones(d, dtype=np.float64)
    params["lnf_b"] = np.zeros(d, dtype=np.float64)
    return params


def _ln_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mean) * inv
    return xhat * g + b, (xhat, inv, g)


def _ln_backward(dy: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xhat, inv, g = cache
    dg = (dy * xhat).sum(axis=(0, 1))
    db = dy.sum(axis=(0, 1))
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dg, db


def _gelu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_inner)
    dy = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * c * (1.0 + 3 * 0.044715 * x**2)
    return y, dy


def build_mask(segment_ids: np.ndarray) -> np.ndarray:
    b, t = segment_ids.shape
    idx = np.arange(t)
    causal = idx[None, :] <= idx[:, None]
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    valid = segment_ids != -1
    mask = causal[None, :, :] & same
    mask = mask & valid[:, None, :] & valid[:, :, None]
    return mask


def forward(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokens: np.ndarray,
    segment_ids: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> dict[str, Any]:
    tokens = np.asarray(tokens, dtype=np.int64)
    b, t = tokens.shape
    if t > config.max_seq_len:
        raise ValueError(f"sequence length {t} exceeds max_seq_len {config.max_seq_len}")
    if segment_ids is None:
        segment_ids = np.zeros((b, t), dtype=np.int64)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    x0 = params["tok_emb"][tokens] + params["pos_emb"][None, :t, :]
    x = x0
    mask = build_mask(segment_ids)
    dh = config.head_dim
    scale = 1.0 / math.sqrt(dh)
    layers = []
    for layer in range(config.n_layers):
        prefix = f"l{layer}_"
        h1, ln1_cache = _ln_forward(x, params[prefix + "ln1_g"], params[prefix + "ln1_b"])
        q = h1 @ params[prefix + "wq"] + params[prefix + "bq"]
        k = h1 @ params[prefix + "wk"] + params[prefix + "bk"]
        v = h1 @ params[prefix + "wv"] + params[prefix + "bv"]
        q = q.reshape(b, t, config.n_heads, dh).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, config.n_heads, dh).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, config.n_heads, dh).transpose(0, 2, 1, 3)
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
        scores = np.where(mask[:, None, :, :], scores, MASK_VALUE)
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        denom = exp_scores.sum(axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        attn = exp_scores / denom
        merged = np.einsum("bhqk,bhkd->bhqd", attn, v).transpose(0, 2, 1, 3).reshape(b, t, config.d_model)
        attn_out = merged @ params[prefix + "wo"] + params[prefix + "bo"]
        x2 = x + attn_out
        h2, ln2_cache = _ln_forward(x2, params[prefix + "ln2_g"], params[prefix + "ln2_b"])
        f1 = h2 @ params[prefix + "w1"] + params[prefix + "b1"]
        f2, gelu_grad = _gelu(f1)
        f3 = f2 @ params[prefix + "w2"] + params[prefix + "b2"]
        x3 = x2 + f3
        layers.append(
            {
                "ln1_cache": ln1_cache,
                "h1": h1,
                "q": q,
                "k": k,
                "v": v,
                "attn": attn,
                "merged": merged,
                "ln2_cache": ln2_cache,
                "h2": h2,
                "f1": f1,
                "gelu_grad": gelu_grad,
                "f2": f2,
                "x_out": x3,
            }
        )
        x = x3
    hf, lnf_cache = _ln_forward(x, params["lnf_g"], params["lnf_b"])
    head = params["tok_emb"] if config.tie_embeddings else params["lm_head"]
    all_logits = hf @ head.T
    if positions is None:
        logits = all_logits
        logit_positions = None
    else:
        pos = np.asarray(positions, dtype=np.int64)
        if pos.ndim == 1:
            pos = pos[:, None]
        logits = np.take_along_axis(hf, pos[:, :, None], axis=1) @ head.T
        logit_positions = pos
    return {
        "tokens": tokens,
        "segment_ids": segment_ids,
        "x0": x0,
        "mask": mask,
        "layers": layers,
        "x_final": x,
        "hf": hf,
        "lnf_cache": lnf_cache,
        "logits": logits,
        "all_logits": all_logits,
        "logit_positions": logit_positions,
        "config": config,
    }


def backward(params: dict[str, np.ndarray], cache: dict[str, Any], dlogits: np.ndarray) -> dict[str, np.ndarray]:
    config: ModelConfig = cache["config"]
    b, t = cache["tokens"].shape
    grads = {name: np.zeros_like(value) for name, value in params.items()}
    head = params["tok_emb"] if config.tie_embeddings else params["lm_head"]
    if cache["logit_positions"] is None:
        full_dlogits = dlogits
        dhf = dlogits @ head
        dhead = dlogits.reshape(-1, config.vocab_size).T @ cache["hf"].reshape(-1, config.d_model)
    else:
        positions = cache["logit_positions"]
        full_dlogits = np.zeros_like(cache["all_logits"])
        for row in range(b):
            full_dlogits[row, positions[row], :] = dlogits[row]
        dhf = np.zeros_like(cache["hf"])
        selected_hf = cache["hf"][np.arange(b)[:, None], positions]
        dhf[np.arange(b)[:, None], positions] = dlogits @ head
        dhead = dlogits.reshape(-1, config.vocab_size).T @ selected_hf.reshape(-1, config.d_model)
    if config.tie_embeddings:
        grads["tok_emb"] += dhead
    else:
        grads["lm_head"] += dhead

    dx, dg, db = _ln_backward(dhf, cache["lnf_cache"])
    grads["lnf_g"] += dg
    grads["lnf_b"] += db
    head_dim = config.head_dim
    scale = 1.0 / math.sqrt(head_dim)
    for layer in reversed(range(config.n_layers)):
        prefix = f"l{layer}_"
        lc = cache["layers"][layer]
        # MLP path.
        dx3 = dx
        grads[prefix + "w2"] += lc["f2"].reshape(-1, config.d_ff).T @ dx3.reshape(-1, config.d_model)
        grads[prefix + "b2"] += dx3.sum(axis=(0, 1))
        df2 = dx3 @ params[prefix + "w2"].T
        df1 = df2 * lc["gelu_grad"]
        grads[prefix + "w1"] += lc["h2"].reshape(-1, config.d_model).T @ df1.reshape(-1, config.d_ff)
        grads[prefix + "b1"] += df1.sum(axis=(0, 1))
        dh2 = df1 @ params[prefix + "w1"].T
        dln2, dg2, db2 = _ln_backward(dh2, lc["ln2_cache"])
        grads[prefix + "ln2_g"] += dg2
        grads[prefix + "ln2_b"] += db2
        dx2 = dx3 + dln2

        # Attention path.
        grads[prefix + "wo"] += lc["merged"].reshape(-1, config.d_model).T @ dx2.reshape(-1, config.d_model)
        grads[prefix + "bo"] += dx2.sum(axis=(0, 1))
        dmerged = dx2 @ params[prefix + "wo"].T
        dmerged = dmerged.reshape(b, t, config.n_heads, head_dim).transpose(0, 2, 1, 3)
        attn = lc["attn"]
        dattn = np.einsum("bhqd,bhkd->bhqk", dmerged, lc["v"])
        dv = np.einsum("bhqk,bhqd->bhkd", attn, dmerged)
        dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        dq = np.einsum("bhqk,bhkd->bhqd", dscores, lc["k"]) * scale
        dk = np.einsum("bhqk,bhqd->bhkd", dscores, lc["q"]) * scale
        dq = dq.transpose(0, 2, 1, 3).reshape(b, t, config.d_model)
        dk = dk.transpose(0, 2, 1, 3).reshape(b, t, config.d_model)
        dv = dv.transpose(0, 2, 1, 3).reshape(b, t, config.d_model)
        h1_flat = lc["h1"].reshape(-1, config.d_model)
        dx1 = np.zeros_like(lc["h1"])
        for name, grad in (("wq", dq), ("wk", dk), ("wv", dv)):
            flat = grad.reshape(-1, config.d_model)
            grads[prefix + name] += h1_flat.T @ flat
            grads[prefix + "b" + name[1:]] += flat.sum(axis=0)
            dx1 += (flat @ params[prefix + name].T).reshape(b, t, config.d_model)
        dln1, dg1, db1 = _ln_backward(dx1, lc["ln1_cache"])
        grads[prefix + "ln1_g"] += dg1
        grads[prefix + "ln1_b"] += db1
        dx = dx2 + dln1

    np.add.at(grads["tok_emb"], cache["tokens"].reshape(-1), dx.reshape(-1, config.d_model))
    grads["pos_emb"][:t] += dx.sum(axis=0)
    return grads


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    if not np.any(mask):
        return 0.0, np.zeros_like(logits)
    selected = flat_logits[mask]
    targets = flat_labels[mask]
    shifted = selected - selected.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsumexp
    loss = float(-log_probs[np.arange(len(targets)), targets].mean())
    dlogits = np.zeros_like(flat_logits)
    dlogits[mask] = np.exp(log_probs)
    dlogits[mask, targets] -= 1.0
    dlogits[mask] /= len(targets)
    return loss, dlogits.reshape(logits.shape)


def sequence_logprobs(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokens: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = forward(params, config, tokens, segment_ids=segment_ids)
    logits = cache["all_logits"]
    batch = logits.shape[0]
    logps = np.zeros(batch, dtype=np.float64)
    for row in range(batch):
        mask = labels[row] != -100
        if not np.any(mask):
            continue
        selected = logits[row, mask]
        targets = labels[row, mask]
        shifted = selected - selected.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        logps[row] = float((shifted - logsumexp)[np.arange(len(targets)), targets].sum())
    return logps, cache


def backward_logprobs(
    params: dict[str, np.ndarray], cache: dict[str, Any], labels: np.ndarray, upstream: np.ndarray
) -> dict[str, np.ndarray]:
    logits = cache["all_logits"]
    batch = logits.shape[0]
    dlogits = np.zeros_like(logits)
    for row in range(batch):
        mask = labels[row] != -100
        if not np.any(mask) or upstream[row] == 0:
            continue
        selected = logits[row, mask]
        targets = labels[row, mask]
        shifted = selected - selected.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        probs = np.exp(shifted - logsumexp)
        grad = -probs
        grad[np.arange(len(targets)), targets] += 1.0
        grad *= upstream[row]
        dlogits[row, mask] = grad
    return backward(params, cache, dlogits)


def loss_and_grads(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokens: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    cache = forward(params, config, tokens, segment_ids=segment_ids)
    loss, dlogits = cross_entropy(cache["all_logits"], labels)
    grads = backward(params, cache, dlogits)
    return loss, grads


def parameter_count(params: dict[str, np.ndarray]) -> int:
    return int(sum(value.size for value in params.values()))
