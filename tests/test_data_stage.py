from __future__ import annotations

import json

from ftpipe.config import load_config
from ftpipe.data.stage import build_data


def test_smoke_data_stage_writes_all_artifacts(tmp_path):
    cfg = load_config("configs/smoke.yaml").with_out(tmp_path)
    artifacts = build_data(cfg)
    stats = artifacts.stats
    assert stats["raw_docs"] == 30
    assert stats["dedup_removed"] > 0
    assert stats["quality_rejected"] > 0
    assert stats["pii_entities_removed"] > 0
    assert stats["pii_coverage"] == 1.0
    assert stats["contamination"]["green"] is True
    assert stats["contamination"]["overlap_docs"] == 0
    assert stats["contamination"]["raw_overlap_docs"] > 0
    assert stats["sft_train_records"] > 0
    assert stats["sft_val_records"] > 0
    for key in ("raw", "clean", "base_corpus", "sft_train", "sft_val", "preferences_train", "stats"):
        assert artifacts.files[key].exists(), key
    written = json.loads(artifacts.files["stats"].read_text(encoding="utf-8"))
    assert written["seed"] == 7
    assert written["contamination"]["green"] is True
