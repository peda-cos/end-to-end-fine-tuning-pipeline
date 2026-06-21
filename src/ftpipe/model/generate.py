"""Autoregressive generation for the tiny target model and draft heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transformer import ModelConfig, forward


@dataclass
class Generation:
    tokens: list[int]
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


def _last_logits(
    params: dict[str, np.ndarray], config: ModelConfig, token_ids: list[int]
) -> np.ndarray:
    tokens = np.asarray([token_ids], dtype=np.int64)
    cache = forward(params, config, tokens, positions=np.asarray([[len(token_ids) - 1]]))
    return cache["logits"][0, 0]


def sample_from_logits(
    logits: np.ndarray,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    if top_k is not None and top_k > 0 and top_k < len(scaled):
        threshold = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < threshold, -np.inf, scaled)
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.choice(len(probs), p=probs))


def generate(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> Generation:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if len(prompt_ids) + max_new_tokens > config.max_seq_len:
        prompt_ids = prompt_ids[-(config.max_seq_len - max_new_tokens) :]
    ids = list(prompt_ids)
    finish = "length"
    for _ in range(max_new_tokens):
        next_id = sample_from_logits(_last_logits(params, config, ids), temperature=temperature, top_k=top_k, rng=rng)
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            finish = "stop"
            break
    completion = ids[len(prompt_ids) :]
    return Generation(
        tokens=completion,
        text=tokenizer.decode(completion),
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(completion),
        finish_reason=finish,
    )


def logprob_of_completion(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    completion_ids = tokenizer.encode(completion, add_bos=False, add_eos=True)
    ids = (prompt_ids + completion_ids)[: config.max_seq_len]
    if len(ids) <= len(prompt_ids):
        return float("-inf")
    labels = np.full((1, len(ids) - 1), -100, dtype=np.int64)
    labels[0, len(prompt_ids) - 1 :] = np.asarray(ids[len(prompt_ids) :], dtype=np.int64)
    from .train import PackedBatch, evaluate_loss
    from .transformer import sequence_logprobs

    tokens = np.asarray([ids[:-1]], dtype=np.int64)
    logps, _ = sequence_logprobs(params, config, tokens, labels)
    length = max(1, len(ids) - len(prompt_ids))
    return float(logps[0] / length)
