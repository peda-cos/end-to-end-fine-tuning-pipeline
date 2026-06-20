"""Small deterministic text classifier shared by quality and safety stages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


@dataclass
class LogisticRegression:
    buckets: int = 512
    threshold: float = 0.5
    weights: np.ndarray | None = None
    bias: float = 0.0
    steps: int = 200
    learning_rate: float = 0.5
    l2: float = 1e-3

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.zeros(self.buckets, dtype=np.float64)

    def _features(self, text: str) -> np.ndarray:
        vector = np.zeros(self.buckets + 1, dtype=np.float64)
        vector[-1] = 1.0
        tokens = tokenize(text)
        grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.buckets
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            vector[index] += sign
        return vector

    def fit(self, texts: list[str], labels: list[int]) -> "LogisticRegression":
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and equal length")
        x = np.stack([self._features(text) for text in texts])
        y = np.asarray(labels, dtype=np.float64)
        weights = np.zeros(self.buckets + 1, dtype=np.float64)
        for _ in range(self.steps):
            logits = x @ weights
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
            grad = x.T @ (probs - y) / len(y)
            grad[: self.buckets] += self.l2 * weights[: self.buckets]
            weights -= self.learning_rate * grad
        self.weights = weights[: self.buckets].copy()
        self.bias = float(weights[-1])
        return self

    def decision_function(self, text: str) -> float:
        vector = self._features(text)
        return float(vector[:-1] @ self.weights + self.bias)

    def predict_proba(self, text: str) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(self.decision_function(text), -60.0, 60.0))))

    def predict(self, text: str) -> bool:
        return self.predict_proba(text) >= self.threshold

    def save(self, path: str | Path) -> None:
        payload = {
            "buckets": self.buckets,
            "threshold": self.threshold,
            "weights": [float(v) for v in self.weights],
            "bias": self.bias,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegression":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            buckets=int(payload["buckets"]),
            threshold=float(payload["threshold"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
        )
        return model
