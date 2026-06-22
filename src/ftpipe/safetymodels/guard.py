"""Llama-Guard-shaped local input policy classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classifier import LogisticRegression


@dataclass
class Guard:
    model: LogisticRegression
    threshold: float = 0.5

    def check(self, text: str) -> tuple[bool, str, float]:
        score = self.model.predict_proba(text)
        allowed = score < self.threshold
        return allowed, "safe" if allowed else "unsafe", score

    def save(self, path: str | Path) -> None:
        self.model.threshold = self.threshold
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, cfg: Any) -> "Guard":
        model = LogisticRegression.load(path)
        return cls(model=model, threshold=float(cfg["safety"]["guard_threshold"]))


def train_guard(path: str | Path, cfg: Any) -> Guard:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = LogisticRegression(
        buckets=512,
        threshold=float(cfg["safety"]["guard_threshold"]),
        steps=400,
        learning_rate=1.0,
        l2=1e-3,
    )
    model.fit([row["text"] for row in rows], [0 if row["safe"] else 1 for row in rows])
    guard = Guard(model=model, threshold=float(cfg["safety"]["guard_threshold"]))
    return guard
