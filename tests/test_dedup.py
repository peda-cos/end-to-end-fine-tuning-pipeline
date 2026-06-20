from __future__ import annotations

import copy

from ftpipe.config import load_config
from ftpipe.data.dedup import dedup_documents
from ftpipe.data.synth import Document


def test_exact_and_near_duplicates_removed():
    cfg = load_config("configs/smoke.yaml")
    docs = [
        Document("a", "the support team recorded the refund request for order 1001 today", "policy"),
        Document("b", "the support team recorded the refund request for order 1001 today", "duplicate"),
        Document("c", "the support team recorded the refund request for order 1002 today", "duplicate"),
        Document("d", "a completely different note about shipping a parcel to another city", "policy"),
    ]
    result = dedup_documents(docs, cfg, seed=7)
    kept_ids = {doc.id for doc in result.kept}
    assert "a" in kept_ids
    assert "b" not in kept_ids
    assert result.report["exact_duplicates"] >= 1
    assert result.report["removed"] >= 2
    assert len(result.kept) == 2


def test_dedup_is_deterministic():
    cfg = load_config("configs/smoke.yaml")
    docs = [
        Document("z", "refund policy note one for the billing team and the customer", "policy"),
        Document("y", "refund policy note one for the billing team and the customer", "duplicate"),
        Document("x", "shipping policy note two for the delivery team and the customer", "policy"),
    ]
    first = dedup_documents(copy.deepcopy(docs), cfg, seed=11)
    second = dedup_documents(copy.deepcopy(docs), cfg, seed=11)
    assert [doc.id for doc in first.kept] == [doc.id for doc in second.kept]
    assert first.report == second.report
