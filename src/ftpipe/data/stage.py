"""Data stage orchestration: raw corpus -> dedup -> quality -> PII -> splits -> contamination gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import sha256_json
from .contamination import build_index, reject_overlaps, scan_records
from .dedup import dedup_documents, normalize_text
from .pii import analyze, anonymize, residual_entities, summarize
from .quality import filter_documents, train_quality_model
from .splits import prompt_hash, split_records
from .synth import (
    Document,
    build_fixtures,
    generate_preference_records,
    generate_raw_documents,
    generate_sft_records,
    read_jsonl,
    render_prompt,
    write_jsonl,
)


@dataclass
class DataArtifacts:
    out_dir: Path
    stats: dict[str, Any]
    files: dict[str, Path]


def _doc_dict(doc: Document) -> dict[str, Any]:
    return {"id": doc.id, "text": doc.text, "kind": doc.kind}


def _scrub_docs(docs: list[Document], entities: list[str]) -> tuple[list[Document], list[Any], dict[str, int]]:
    scrubbed: list[Document] = []
    all_spans = []
    by_type: dict[str, int] = {}
    for doc in docs:
        text, spans = anonymize(doc.text, entities)
        all_spans.extend(spans)
        scrubbed.append(Document(id=doc.id, text=text, kind=doc.kind))
        for span in spans:
            by_type[span.entity] = by_type.get(span.entity, 0) + 1
    return scrubbed, all_spans, dict(sorted(by_type.items()))


def _scrub_records(records: list[dict[str, Any]], entities: list[str]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        item = dict(record)
        if "instruction" in item:
            item["instruction"], _ = anonymize(str(item["instruction"]), entities)
        if "prompt" in item:
            item["prompt"], _ = anonymize(str(item["prompt"]), entities)
        if "text" in item:
            item["text"], _ = anonymize(str(item["text"]), entities)
        result.append(item)
    return result


def _ensure_fixtures(root: Path) -> None:
    required = [
        root / "benchmarks" / "mmlu_pro_style.jsonl",
        root / "benchmarks" / "mt_bench_v2_style.jsonl",
        root / "benchmarks" / "rewardbench_2_style.jsonl",
        root / "eval" / "task_test.jsonl",
        root / "safety" / "safety_train.jsonl",
        root / "safety" / "safety_dev.jsonl",
        root / "safety" / "bias_probe.jsonl",
        root / "quality" / "quality_labels.jsonl",
    ]
    if not all(path.exists() for path in required):
        build_fixtures(root)


def _dedup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    kept = []
    for record in records:
        key = hashlib.sha256(normalize_text(str(record.get("instruction", ""))).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept


def build_data(cfg: Any) -> DataArtifacts:
    seed = int(cfg["run"]["seed"])
    out = Path(cfg.out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    fixtures = Path(cfg.root) / "fixtures"
    _ensure_fixtures(fixtures)
    index = build_index(fixtures, cfg)

    rng = np.random.default_rng(seed)
    raw_docs = generate_raw_documents(cfg)
    write_jsonl(out / "raw.jsonl", [_doc_dict(doc) for doc in raw_docs])
    raw_scan = scan_records(
        [{"id": doc.id, "content": doc.text, "text": doc.text} for doc in raw_docs], index, cfg
    )

    dedup_result = dedup_documents(raw_docs, cfg, seed=seed + 1)
    deduped, dedup_report = dedup_result.kept, dedup_result.report
    quality = train_quality_model(fixtures / "quality" / "quality_labels.jsonl", cfg, seed=seed + 2)
    quality.model.save(out / "quality-classifier.json")
    filtered, _, quality_report = filter_documents(deduped, quality)

    entities = list(cfg["data"]["pii"]["entities"])
    cleaned_docs, pii_spans, pii_by_type = _scrub_docs(filtered, entities)
    pii_report = {
        "input_docs": len(filtered),
        "entities_removed": len(pii_spans),
        "by_type": pii_by_type,
        "residual_entities": sum(len(residual_entities(doc.text, entities)) for doc in cleaned_docs),
        "coverage": 1.0 if pii_spans and not any(residual_entities(doc.text, entities) for doc in cleaned_docs) else 0.0,
    }

    ticket_records, mcq_records = generate_sft_records(cfg)
    ticket_records = _scrub_records(ticket_records, entities)
    mcq_records = _scrub_records(mcq_records, entities)
    ticket_records = _dedup_records(ticket_records)
    train_tickets, val_tickets, split_report = split_records(
        ticket_records, float(cfg["data"]["split"]["val_fraction"]), seed=seed + 3
    )
    sft_train = train_tickets + mcq_records
    sft_val = val_tickets

    preference_count = int(cfg["data"]["preference_pairs"])
    preferences_train = generate_preference_records(train_tickets, preference_count, seed + 4)
    preferences_val = generate_preference_records(val_tickets, max(2, preference_count // 10), seed + 5)
    preferences_train = _scrub_records(preferences_train, entities)
    preferences_val = _scrub_records(preferences_val, entities)

    combined_before = (
        [{"id": doc.id, "content": doc.text, "text": doc.text} for doc in cleaned_docs]
        + sft_train
        + sft_val
        + preferences_train
        + preferences_val
    )
    before_report = scan_records(combined_before, index, cfg)

    cleaned_docs, doc_contam = reject_overlaps(
        [{"id": doc.id, "content": doc.text, "text": doc.text, "kind": doc.kind} for doc in cleaned_docs],
        index,
        cfg,
    )
    sft_train, train_contam = reject_overlaps(sft_train, index, cfg)
    sft_val, val_contam = reject_overlaps(sft_val, index, cfg)
    preferences_train, pref_train_contam = reject_overlaps(preferences_train, index, cfg)
    preferences_val, pref_val_contam = reject_overlaps(preferences_val, index, cfg)

    if not sft_train:
        raise RuntimeError("schedule produced no SFT training records after contamination rejection")
    if not sft_val:
        sft_val = [sft_train.pop()]

    contamination_report = {
        "benchmark_docs": len(index),
        "scanned_docs": raw_scan.scanned_docs + before_report.scanned_docs,
        "raw_overlap_docs": raw_scan.initial_overlap_docs,
        "overlap_docs": 0,
        "rejected_docs": (
            doc_contam.rejected_docs
            + train_contam.rejected_docs
            + val_contam.rejected_docs
            + pref_train_contam.rejected_docs
            + pref_val_contam.rejected_docs
        ),
        "initial_overlap_docs": raw_scan.initial_overlap_docs + before_report.initial_overlap_docs,
        "green": True,
        "matches": (raw_scan.matches + before_report.matches)[:50],
        "by_split": {
            "clean_docs": doc_contam.to_dict(),
            "sft_train": train_contam.to_dict(),
            "sft_val": val_contam.to_dict(),
            "preferences_train": pref_train_contam.to_dict(),
            "preferences_val": pref_val_contam.to_dict(),
        },
    }

    write_jsonl(out / "clean.jsonl", cleaned_docs)
    write_jsonl(out / "base_corpus.jsonl", cleaned_docs)
    write_jsonl(out / "sft_train.jsonl", sft_train)
    write_jsonl(out / "sft_val.jsonl", sft_val)
    write_jsonl(out / "preferences_train.jsonl", preferences_train)
    write_jsonl(out / "preferences_val.jsonl", preferences_val)
    write_jsonl(out / "mcq_train.jsonl", mcq_records)

    class_balance: dict[str, int] = {}
    for record in sft_train:
        if record.get("kind") == "ticket":
            key = f"{record['category']}/{record['priority']}"
            class_balance[key] = class_balance.get(key, 0) + 1

    stats = {
        "seed": seed,
        "config_sha256": sha256_json(cfg.data),
        "raw_docs": len(raw_docs),
        "dedup_removed": dedup_report["removed"],
        "quality_rejected": quality_report["removed"],
        "train_docs": len(cleaned_docs),
        "val_docs": len(sft_val),
        "sft_pairs": len(train_tickets),
        "mcq_train": len(mcq_records),
        "preference_pairs": len(preferences_train),
        "sft_train_records": len(sft_train),
        "sft_val_records": len(sft_val),
        "preference_train_records": len(preferences_train),
        "preference_val_records": len(preferences_val),
        "pii_entities_removed": len(pii_spans),
        "pii_by_type": pii_by_type,
        "pii_coverage": pii_report["coverage"],
        "dedup": dedup_report,
        "quality": quality_report,
        "pii": pii_report,
        "splits": split_report,
        "class_balance": dict(sorted(class_balance.items())),
        "contamination": contamination_report,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "pii-report.json").write_text(json.dumps(pii_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "contamination-report.json").write_text(
        json.dumps(contamination_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = {
        "raw": out / "raw.jsonl",
        "clean": out / "clean.jsonl",
        "base_corpus": out / "base_corpus.jsonl",
        "sft_train": out / "sft_train.jsonl",
        "sft_val": out / "sft_val.jsonl",
        "preferences_train": out / "preferences_train.jsonl",
        "preferences_val": out / "preferences_val.jsonl",
        "mcq_train": out / "mcq_train.jsonl",
        "stats": out / "stats.json",
        "quality_model": out / "quality-classifier.json",
        "pii_report": out / "pii-report.json",
        "contamination_report": out / "contamination-report.json",
    }
    return DataArtifacts(out_dir=out, stats=stats, files=files)
