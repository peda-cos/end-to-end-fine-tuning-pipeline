"""Gopher-style rules plus a learned quality classifier."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..safetymodels.classifier import LogisticRegression

SYMBOL_RE = re.compile(r"[^a-zA-Z0-9\s]")
WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def token_count(text: str) -> int:
    return len(text.split())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def symbol_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 1.0
    return len(SYMBOL_RE.findall("".join(visible))) / len(visible)


def duplicate_line_ratio(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    return 1.0 - len(set(lines)) / len(lines)


def mean_word_length(text: str) -> float:
    words = WORD_RE.findall(text)
    return sum(len(word) for word in words) / len(words) if words else 0.0


def rule_rejection(text: str, cfg: Any) -> str | None:
    q = cfg["data"]["quality"]
    if token_count(text) < int(q["min_tokens"]):
        return "too_short"
    if symbol_ratio(text) > float(q["max_symbol_ratio"]):
        return "symbol_heavy"
    if word_count(text) < int(q["min_words"]):
        return "too_few_words"
    if duplicate_line_ratio(text) > 0.3:
        return "duplicate_lines"
    if not 3.0 <= mean_word_length(text) <= 10.0:
        return "word_length"
    return None


class QualityScorer:
    def __init__(self, model: LogisticRegression, cfg: Any):
        self.model = model
        self.cfg = cfg

    def score(self, text: str) -> float:
        return self.model.predict_proba(text)

    def accept(self, text: str) -> tuple[bool, str]:
        reason = rule_rejection(text, self.cfg)
        if reason:
            return False, f"rule:{reason}"
        if self.model.predict_proba(text) < float(self.cfg["data"]["quality"]["classifier_threshold"]):
            return False, "classifier"
        return True, "accepted"


def train_quality_model(labels_path: str | Path, cfg: Any, seed: int = 0) -> QualityScorer:
    rows = [json.loads(line) for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = LogisticRegression(
        buckets=512,
        threshold=float(cfg["data"]["quality"]["classifier_threshold"]),
        steps=300,
        learning_rate=0.8,
        l2=1e-3,
    )
    model.fit([row["text"] for row in rows], [int(row["label"]) for row in rows])
    return QualityScorer(model, cfg)


def filter_documents(documents: Iterable[Any], scorer: QualityScorer) -> tuple[list[Any], list[Any], dict[str, int]]:
    kept: list[Any] = []
    removed: list[Any] = []
    reasons: dict[str, int] = {}
    for doc in documents:
        text = doc.text if not isinstance(doc, dict) else str(doc.get("text", ""))
        accepted, reason = scorer.accept(text)
        if accepted:
            kept.append(doc)
        else:
            removed.append(doc)
            reasons[reason] = reasons.get(reason, 0) + 1
    report = {
        "input_docs": len(kept) + len(removed),
        "kept_docs": len(kept),
        "removed": len(removed),
        "rejection_rate": len(removed) / (len(kept) + len(removed)) if (kept or removed) else 0.0,
        "reasons": dict(sorted(reasons.items())),
    }
    return kept, removed, report
