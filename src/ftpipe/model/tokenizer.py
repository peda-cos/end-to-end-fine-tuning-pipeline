"""Deterministic word-level tokenizer for the synthetic domain."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
FORCED = (
    ["billing", "technical", "shipping", "account", "low", "high", "ticket", "categories", "priorities", "answer"]
    + [str(i) for i in range(10)]
    + list("abcdefghij")
    + list(".,:;!?'-()")
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Tokenizer:
    tokens: list[str] = field(default_factory=list)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = list(SPECIALS)
        self._index = {token: index for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, texts: Iterable[str], max_vocab: int = 512) -> "Tokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        forced = [token for token in FORCED if token in counts or token in _FORCED_ALWAYS]
        tokens = list(SPECIALS)
        for token in forced:
            if token not in tokens:
                tokens.append(token)
        for token in ordered:
            if len(tokens) >= max_vocab:
                break
            if token not in tokens:
                tokens.append(token)
        return cls(tokens=tokens)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = [self._index.get(token, self.unk_id) for token in tokenize(text)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts = []
        for index in ids:
            index = int(index)
            if index in (self.pad_id, self.bos_id, self.eos_id):
                continue
            token = self.tokens[index] if 0 <= index < len(self.tokens) else "<unk>"
            if token == "<unk>":
                token = "?"
            parts.append(token)
        text = ""
        for token in parts:
            if not text:
                text = token
            elif token in ".,:;!?')-":
                text += token
            elif token == "(":
                text += " (" if text else "("
            else:
                text += " " + token
        return text

    def encode_example(self, prompt: str, response: str, max_len: int) -> tuple[list[int], list[int]]:
        prompt_ids = self.encode(prompt, add_bos=True, add_eos=False)
        response_ids = self.encode(response, add_bos=False, add_eos=True)
        ids = (prompt_ids + response_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_len]
        if len(ids) < max_len:
            pad = max_len - len(ids)
            ids = ids + [self.pad_id] * pad
            labels = labels + [-100] * pad
        return ids, labels

    def oov_rate(self, texts: Iterable[str]) -> float:
        total = 0
        unknown = 0
        for text in texts:
            for token in tokenize(text):
                total += 1
                if token not in self._index:
                    unknown += 1
        return unknown / total if total else 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"tokens": self.tokens}, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tokens=list(payload["tokens"]))


_FORCED_ALWAYS = set(SPECIALS) | set(FORCED)
