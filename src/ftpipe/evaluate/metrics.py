"""Deterministic rubric judge and benchmark-shaped local evaluators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..model.generate import generate
from ..model.tokenizer import tokenize
from ..model.transformer import forward


def parse_ticket_completion(text: str) -> tuple[bool, str | None, str | None]:
    parts = text.strip().lower().split()
    if len(parts) != 2:
        return False, None, None
    category, priority = parts
    valid = category in {"billing", "technical", "shipping", "account"} and priority in {"low", "high"}
    return valid, category if valid else None, priority if valid else None


def evaluate_task(
    params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = 4,
) -> dict[str, float]:
    exact = 0
    category_correct = 0
    format_valid = 0
    for row in rows:
        completion = generate(
            params,
            config,
            tokenizer,
            row["instruction"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        ).text
        valid, category, priority = parse_ticket_completion(completion)
        format_valid += int(valid)
        category_correct += int(valid and category == row["category"])
        exact += int(valid and category == row["category"] and priority == row["priority"])
    total = max(1, len(rows))
    return {
        "task_exact_match": exact / total,
        "task_category_accuracy": category_correct / total,
        "task_format_valid": format_valid / total,
    }


def predict_task_rows(
    params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = 4,
) -> list[dict[str, Any]]:
    """Generate the exact completions reported by evaluate_task."""
    predictions = []
    for row in rows:
        output = generate(
            params,
            config,
            tokenizer,
            row["instruction"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        ).text
        valid, category, priority = parse_ticket_completion(output)
        predictions.append(
            {
                "id": row["id"],
                "instruction": row["instruction"],
                "expected": row["response"],
                "output": output,
                "format_valid": valid,
                "correct": bool(
                    valid and category == row["category"] and priority == row["priority"]
                ),
            }
        )
    return predictions


def _option_logprob(
    params: dict[str, np.ndarray], config: Any, tokenizer: Any, prompt: str, letter: str
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    tokens = np.asarray([prompt_ids], dtype=np.int64)
    cache = forward(params, config, tokens, positions=np.asarray([[len(prompt_ids) - 1]]))
    logits = cache["logits"][0, 0]
    shifted = logits - logits.max()
    log_probs = shifted - math.log(float(np.exp(shifted).sum()))
    letter_id = tokenizer.encode(letter.lower(), add_bos=False, add_eos=False)[0]
    return float(log_probs[letter_id])


def evaluate_mmlu(
    params: dict[str, np.ndarray], config: Any, tokenizer: Any, rows: list[dict[str, Any]]
) -> dict[str, float]:
    correct = 0
    for row in rows:
        scores = [
            (letter, _option_logprob(params, config, tokenizer, row["prompt"], letter))
            for letter in "ABCDEFGHIJ"
        ]
        best = max(scores, key=lambda item: item[1])[0]
        correct += int(best == row["answer"])
    return {"mmlu_pro_style": correct / max(1, len(rows))}


def rubric_score(response: str, reference: str) -> dict[str, float]:
    words = tokenize(response)
    ref_words = [word for word in tokenize(reference) if len(word) > 2]
    if not words:
        return {"total": 0.0, "format": 0.0, "reference": 0.0, "concision": 0.0, "fluency": 0.0}
    unique = len(set(words)) / len(words)
    repeated = max(words.count(word) for word in set(words)) if words else 0
    usable = "?" not in response and len(words) >= 3
    format_score = 3.0 if usable and unique >= 0.45 else (1.5 if usable else 0.0)
    overlap = sum(1 for word in ref_words if word in set(words)) / len(ref_words) if ref_words else 0.0
    reference_score = 4.0 * overlap
    if 5 <= len(words) <= 40:
        concision = 2.0
    elif 3 <= len(words) <= 80:
        concision = 1.0
    else:
        concision = 0.0
    fluency = 1.0 if unique >= 0.4 and repeated <= 3 else 0.0
    total = format_score + reference_score + concision + fluency
    return {
        "total": total,
        "format": format_score,
        "reference": reference_score,
        "concision": concision,
        "fluency": fluency,
    }


def evaluate_mt_bench(
    params: dict[str, np.ndarray],
    config: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = 24,
) -> dict[str, float]:
    scores = []
    for row in rows:
        first = generate(
            params, config, tokenizer, row["turns"][0], max_new_tokens=max_new_tokens, temperature=0.0
        ).text
        second_prompt = f"{row['turns'][0]}\nAnswer: {first}\n{row['turns'][1]}"
        second = generate(
            params, config, tokenizer, second_prompt, max_new_tokens=max_new_tokens, temperature=0.0
        ).text
        scores.append((rubric_score(first, row["reference"])["total"] + rubric_score(second, row["reference"])["total"]) / 2.0)
    return {"mt_bench_v2_style": float(np.mean(scores)) if scores else 0.0}
