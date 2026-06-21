"""EAGLE-3-style draft head trained on target hidden states (scaled down)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .tokenizer import Tokenizer
from .train import PackedBatch, batch_iterator
from .transformer import ModelConfig, cross_entropy, forward


def init_draft(config: ModelConfig, hidden_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    d = config.d_model
    features = 4 * d
    return {
        "w1": rng.normal(0.0, 0.02, size=(features, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0.0, 0.02, size=(hidden_dim, d)),
        "b2": np.zeros(d),
    }


def _gelu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    return 0.5 * x * (1.0 + t), 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * c * (1.0 + 3 * 0.044715 * x**2)


def _feature_matrix(
    target_params: dict[str, np.ndarray],
    config: ModelConfig,
    tokens: np.ndarray,
    cache: dict[str, Any] | None = None,
) -> np.ndarray:
    if cache is None:
        cache = forward(target_params, config, tokens, segment_ids=None)
    b, t = tokens.shape
    h0 = cache["layers"][0]["x_out"]
    h1 = cache["layers"][-1]["x_out"] if config.n_layers > 1 else cache["hf"]
    prev_tokens = np.concatenate([tokens[:, :1], tokens[:, :-1]], axis=1)
    token_features = target_params["tok_emb"][prev_tokens]
    position_features = np.broadcast_to(target_params["pos_emb"][None, :t, :], (b, t, config.d_model))
    return np.concatenate([h0, h1, token_features, position_features], axis=-1)


def draft_logits(
    target_params: dict[str, np.ndarray],
    config: ModelConfig,
    draft_params: dict[str, np.ndarray],
    tokens: np.ndarray,
    cache: dict[str, Any] | None = None,
) -> np.ndarray:
    features = _feature_matrix(target_params, config, tokens, cache)
    hidden, _ = _gelu(features @ draft_params["w1"] + draft_params["b1"])
    return hidden @ draft_params["w2"] + draft_params["b2"], hidden, features


def _draft_loss(
    target_params: dict[str, np.ndarray],
    config: ModelConfig,
    draft_params: dict[str, np.ndarray],
    batch: PackedBatch,
) -> tuple[float, dict[str, np.ndarray]]:
    tokens = batch.tokens[None, :]
    labels = batch.labels[None, :]
    target_cache = forward(target_params, config, tokens, segment_ids=batch.segment_ids[None, :])
    features = _feature_matrix(target_params, config, tokens, target_cache)
    hidden_pre = features @ draft_params["w1"] + draft_params["b1"]
    hidden, gelu_grad = _gelu(hidden_pre)
    logits = hidden @ draft_params["w2"] + draft_params["b2"]
    loss, dlogits = cross_entropy(logits, labels)
    grads = {name: np.zeros_like(value) for name, value in draft_params.items()}
    grads["w2"] = hidden.reshape(-1, hidden.shape[-1]).T @ dlogits.reshape(-1, config.d_model)
    grads["b2"] = dlogits.sum(axis=(0, 1))
    dhidden = dlogits @ draft_params["w2"].T
    dpre = dhidden * gelu_grad
    grads["w1"] = features.reshape(-1, features.shape[-1]).T @ dpre.reshape(-1, dpre.shape[-1])
    grads["b1"] = dpre.sum(axis=(0, 1))
    return loss, grads


def train_draft(
    target_params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Tokenizer,
    batches: list[PackedBatch],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_path: str | Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    draft = init_draft(config, hidden_dim=config.d_ff, rng=rng)
    iterator = batch_iterator(batches, batch_size, rng)
    m = {name: np.zeros_like(value) for name, value in draft.items()}
    v = {name: np.zeros_like(value) for name, value in draft.items()}
    history = []
    log_handle = Path(log_path).open("w", encoding="utf-8")
    try:
        for step in range(steps):
            group = next(iterator)
            total_loss = 0.0
            grads = {name: np.zeros_like(value) for name, value in draft.items()}
            for batch in group:
                loss, item_grads = _draft_loss(target_params, config, draft, batch)
                total_loss += loss
                for name in grads:
                    grads[name] += item_grads[name]
            for name in grads:
                grads[name] /= len(group)
                m[name] = 0.9 * m[name] + 0.1 * grads[name]
                v[name] = 0.95 * v[name] + 0.05 * grads[name] ** 2
                m_hat = m[name] / (1 - 0.9 ** (step + 1))
                v_hat = v[name] / (1 - 0.95 ** (step + 1))
                draft[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            record = {"step": step + 1, "loss": total_loss / len(group)}
            history.append(record)
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        log_handle.close()
    return draft, history


def propose(
    target_params: dict[str, np.ndarray],
    config: ModelConfig,
    draft_params: dict[str, np.ndarray],
    ids: list[int],
    *,
    count: int,
    cache: dict[str, Any] | None = None,
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
) -> list[int]:
    if not ids:
        return []
    current = list(ids)
    proposals: list[int] = []
    anchor_cache = cache if cache is not None else forward(
        target_params, config, np.asarray([current]), positions=np.asarray([[len(current) - 1]])
    )
    h0 = anchor_cache["layers"][0]["x_out"][:, -1:, :]
    h1 = anchor_cache["layers"][-1]["x_out"][:, -1:, :] if config.n_layers > 1 else anchor_cache["hf"][:, -1:, :]
    for _ in range(count):
        position = len(current)
        if position >= config.max_seq_len:
            break
        prev = current[-1]
        features = np.concatenate(
            [
                h0,
                h1,
                target_params["tok_emb"][np.asarray([[prev]])],
                target_params["pos_emb"][np.asarray([[position]])],
            ],
            axis=-1,
        )
        hidden, _ = _gelu(features @ draft_params["w1"] + draft_params["b1"])
        logits = (hidden @ draft_params["w2"] + draft_params["b2"])[0, 0]
        if deterministic:
            token = int(np.argmax(logits))
        else:
            scaled = logits / 0.8
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()
            token = int((rng or np.random.default_rng()).choice(len(probs), p=probs))
        proposals.append(token)
        current.append(token)
        if token == 2:
            break
    return proposals
