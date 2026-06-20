from __future__ import annotations

import pytest

from ftpipe.config import load_config
from ftpipe.data.contamination import (
    ContaminationError,
    assert_clean,
    build_index,
    reject_overlaps,
    scan_records,
)
from ftpipe.data.synth import read_jsonl


def test_clean_records_pass_and_exact_leak_fails():
    cfg = load_config("configs/domainx.yaml")
    index = build_index("fixtures", cfg)
    clean = [{"id": "clean", "content": "support note about a delayed parcel and a new delivery window"}]
    report = scan_records(clean, index, cfg)
    assert report.green
    assert report.overlap_docs == 0

    leaked = read_jsonl("fixtures/benchmarks/mmlu_pro_style.jsonl")[0]
    bad = [{"id": "leak", "content": leaked["question"]}]
    report = scan_records(bad, index, cfg)
    assert not report.green
    assert report.overlap_docs == 1
    with pytest.raises(ContaminationError):
        assert_clean(report)


def test_one_percent_injection_is_caught_and_rejected():
    cfg = load_config("configs/domainx.yaml")
    index = build_index("fixtures", cfg)
    benchmark_rows = read_jsonl("fixtures/benchmarks/mmlu_pro_style.jsonl")[:5]
    records = [{"id": f"clean-{i}", "content": f"unique clean policy sentence number {i} about support"} for i in range(495)]
    records += [{"id": f"leak-{i}", "content": row["question"]} for i, row in enumerate(benchmark_rows)]
    report = scan_records(records, index, cfg)
    assert report.overlap_docs == 5
    assert not report.green
    kept, final = reject_overlaps(records, index, cfg)
    assert len(kept) == 495
    assert final.green and final.overlap_docs == 0
    assert final.rejected_docs == 5
