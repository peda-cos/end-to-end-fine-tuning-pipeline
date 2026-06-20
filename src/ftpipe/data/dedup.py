"""MinHash-LSH near-duplicate detection and exact dedup."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

MERSENNE = (1 << 61) - 1


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def word_shingles(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize_text(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def base_hash(shingle: tuple[str, ...]) -> int:
    digest = hashlib.blake2b(" ".join(shingle).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % MERSENNE


class MinHasher:
    def __init__(self, ngram: int, num_hashes: int, bands: int, rows: int, seed: int):
        if bands * rows != num_hashes:
            raise ValueError("bands * rows must equal num_hashes")
        self.ngram = ngram
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows = rows
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, size=num_hashes, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE, size=num_hashes, dtype=np.int64)

    def signature(self, text: str) -> np.ndarray:
        shingles = word_shingles(text, self.ngram)
        if not shingles:
            return np.zeros(self.num_hashes, dtype=np.int64)
        hashes = np.asarray([base_hash(shingle) for shingle in shingles], dtype=object)
        values = np.empty(self.num_hashes, dtype=np.int64)
        for index in range(self.num_hashes):
            values[index] = int(min((int(self.a[index]) * int(h) + int(self.b[index])) % MERSENNE for h in hashes))
        return values

    def bands_of(self, signature: np.ndarray) -> list[tuple[int, bytes]]:
        result = []
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows]
            result.append((band, chunk.tobytes()))
        return result

    @staticmethod
    def jaccard(signature_a: np.ndarray, signature_b: np.ndarray) -> float:
        return float(np.mean(signature_a == signature_b))


@dataclass
class DedupResult:
    kept: list[Any]
    removed: list[Any]
    report: dict[str, Any]


def _text(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("text", doc.get("instruction", "")))
    return str(getattr(doc, "text", ""))


def _id(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("id", ""))
    return str(getattr(doc, "id", ""))


def dedup_documents(documents: Iterable[Any], cfg: Any, seed: int) -> DedupResult:
    m = cfg["data"]["minhash"]
    hasher = MinHasher(
        ngram=int(m["ngram"]),
        num_hashes=int(m["num_hashes"]),
        bands=int(m["bands"]),
        rows=int(m["rows"]),
        seed=seed,
    )
    threshold = float(m["threshold"])
    ordered = sorted(documents, key=_id)
    kept: list[Any] = []
    removed: list[Any] = []
    exact: dict[str, str] = {}
    signatures: list[np.ndarray] = []
    buckets: dict[tuple[int, bytes], list[int]] = {}
    exact_duplicates = 0
    near_duplicates = 0

    for doc in ordered:
        text = _text(doc)
        normalized = normalize_text(text)
        if not normalized:
            removed.append(doc)
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in exact:
            removed.append(doc)
            exact_duplicates += 1
            continue
        signature = hasher.signature(text)
        candidates: set[int] = set()
        for band in hasher.bands_of(signature):
            candidates.update(buckets.get(band, []))
        duplicate = False
        if normalized and all(len(word_shingles(_text(kept[index]), hasher.ngram)) > 0 for index in candidates):
            for index in sorted(candidates):
                if hasher.jaccard(signature, signatures[index]) >= threshold:
                    duplicate = True
                    break
        if duplicate:
            removed.append(doc)
            near_duplicates += 1
            continue
        exact[digest] = _id(doc)
        signatures.append(signature)
        index = len(kept)
        kept.append(doc)
        for band in hasher.bands_of(signature):
            buckets.setdefault(band, []).append(index)

    report = {
        "input_docs": len(ordered),
        "kept_docs": len(kept),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "removed": len(removed),
        "dedup_rate": len(removed) / len(ordered) if ordered else 0.0,
        "threshold": threshold,
        "ngram": hasher.ngram,
        "num_hashes": hasher.num_hashes,
    }
    return DedupResult(kept=kept, removed=removed, report=report)
