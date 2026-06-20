"""Benchmark contamination gate using exact n-grams and MinHash Jaccard."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .dedup import MinHasher, normalize_text, word_shingles

BENCHMARK_FILES = {
    "mmlu_pro_style": "benchmarks/mmlu_pro_style.jsonl",
    "mt_bench_v2_style": "benchmarks/mt_bench_v2_style.jsonl",
    "rewardbench_2_style": "benchmarks/rewardbench_2_style.jsonl",
}


class ContaminationError(RuntimeError):
    """Raised when benchmark content is found in training data."""


def content_of(record: dict[str, Any]) -> str:
    if record.get("content"):
        return str(record["content"])
    if record.get("question"):
        return str(record["question"])
    if record.get("turns"):
        return " ".join(str(turn) for turn in record["turns"])
    if record.get("prompt"):
        return str(record["prompt"])
    return str(record.get("text", ""))


@dataclass
class BenchmarkEntry:
    benchmark: str
    id: str
    content: str
    shingles: set[tuple[str, ...]]
    signature: Any
    normalized: str = ""


@dataclass
class ContaminationReport:
    benchmark_docs: int
    scanned_docs: int
    overlap_docs: int
    rejected_docs: int
    green: bool
    matches: list[dict[str, Any]] = field(default_factory=list)
    initial_overlap_docs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_docs": self.benchmark_docs,
            "scanned_docs": self.scanned_docs,
            "overlap_docs": self.overlap_docs,
            "rejected_docs": self.rejected_docs,
            "initial_overlap_docs": self.initial_overlap_docs,
            "green": self.green,
            "matches": self.matches,
        }


def build_index(fixtures_dir: str | Path, cfg: Any) -> list[BenchmarkEntry]:
    fixtures = Path(fixtures_dir)
    m = cfg["data"]["minhash"]
    hasher = MinHasher(
        ngram=int(m["ngram"]),
        num_hashes=int(m["num_hashes"]),
        bands=int(m["bands"]),
        rows=int(m["rows"]),
        seed=int(cfg["run"]["seed"]) + 31,
    )
    n = int(cfg["data"]["contamination"]["exact_ngram"])
    entries: list[BenchmarkEntry] = []
    for benchmark, relative in BENCHMARK_FILES.items():
        path = fixtures / relative
        if not path.exists():
            raise FileNotFoundError(f"benchmark fixture missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            content = content_of(row)
            entries.append(
                BenchmarkEntry(
                    benchmark=benchmark,
                    id=str(row.get("id", "")),
                    content=content,
                    shingles=word_shingles(content, n),
                    signature=hasher.signature(content),
                    normalized=normalize_text(content),
                )
            )
    return entries


def scan_records(records: Iterable[dict[str, Any]], index: list[BenchmarkEntry], cfg: Any) -> ContaminationReport:
    m = cfg["data"]["minhash"]
    hasher = MinHasher(
        ngram=int(m["ngram"]),
        num_hashes=int(m["num_hashes"]),
        bands=int(m["bands"]),
        rows=int(m["rows"]),
        seed=int(cfg["run"]["seed"]) + 31,
    )
    n = int(cfg["data"]["contamination"]["exact_ngram"])
    threshold = float(cfg["data"]["contamination"]["threshold"])
    scanned = 0
    matches: list[dict[str, Any]] = []
    overlapping_ids: set[str] = set()
    for record in records:
        scanned += 1
        content = content_of(record)
        content_words = len(content.split())
        shingles = word_shingles(content, n)
        signature = None
        record_id = str(record.get("id", f"record-{scanned}"))
        normalized_content = normalize_text(content)
        for entry in index:
            if shingles & entry.shingles or (len(entry.normalized.split()) >= 5 and entry.normalized in normalized_content):
                similarity = 1.0
            elif content_words >= 20 and len(entry.content.split()) >= 20:
                if signature is None:
                    signature = hasher.signature(content)
                similarity = hasher.jaccard(signature, entry.signature)
            else:
                continue
            if similarity >= threshold:
                overlapping_ids.add(record_id)
                if len(matches) < 50:
                    matches.append(
                        {
                            "record_id": record_id,
                            "benchmark": entry.benchmark,
                            "benchmark_id": entry.id,
                            "similarity": round(float(similarity), 4),
                        }
                    )
                break
    return ContaminationReport(
        benchmark_docs=len(index),
        scanned_docs=scanned,
        overlap_docs=len(overlapping_ids),
        rejected_docs=0,
        green=len(overlapping_ids) == 0,
        matches=matches,
        initial_overlap_docs=len(overlapping_ids),
    )


def reject_overlaps(
    records: list[dict[str, Any]], index: list[BenchmarkEntry], cfg: Any
) -> tuple[list[dict[str, Any]], ContaminationReport]:
    """Remove contaminated records and return a green final report."""
    report = scan_records(records, index, cfg)
    bad_ids = {match["record_id"] for match in report.matches}
    # scan_records caps stored matches; recompute the id set without a cap for rejection.
    m = cfg["data"]["minhash"]
    hasher = MinHasher(
        ngram=int(m["ngram"]),
        num_hashes=int(m["num_hashes"]),
        bands=int(m["bands"]),
        rows=int(m["rows"]),
        seed=int(cfg["run"]["seed"]) + 31,
    )
    n = int(cfg["data"]["contamination"]["exact_ngram"])
    threshold = float(cfg["data"]["contamination"]["threshold"])
    bad_ids = set()
    for record in records:
        content = content_of(record)
        content_words = len(content.split())
        normalized_content = normalize_text(content)
        shingles = word_shingles(content, n)
        signature = hasher.signature(content)
        for entry in index:
            matched = bool(shingles & entry.shingles) or (
                len(entry.normalized.split()) >= 5 and entry.normalized in normalized_content
            )
            if not matched and content_words >= 20 and len(entry.content.split()) >= 20:
                matched = hasher.jaccard(signature, entry.signature) >= threshold
            if matched:
                bad_ids.add(str(record.get("id", "")))
                break
    kept = [record for record in records if str(record.get("id", "")) not in bad_ids]
    final = ContaminationReport(
        benchmark_docs=len(index),
        scanned_docs=len(kept),
        overlap_docs=0,
        rejected_docs=len(bad_ids),
        green=True,
        matches=[],
        initial_overlap_docs=len(bad_ids),
    )
    return kept, final


def assert_clean(report: ContaminationReport) -> None:
    if not report.green or report.overlap_docs:
        raise ContaminationError(
            f"benchmark contamination detected: {report.overlap_docs} overlapping docs; "
            f"first matches: {report.matches[:5]}"
        )


def check_contamination_command(cfg: Any) -> int:
    from .stage import build_data

    artifacts = build_data(cfg)
    report = artifacts.stats["contamination"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["green"] and report["overlap_docs"] == 0 else 1
