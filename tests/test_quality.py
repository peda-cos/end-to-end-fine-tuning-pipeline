from __future__ import annotations

from ftpipe.config import load_config
from ftpipe.data.quality import rule_rejection, train_quality_model


def test_rule_rejects_short_and_symbol_heavy():
    cfg = load_config("configs/smoke.yaml")
    assert rule_rejection("too short", cfg) == "too_short"
    assert rule_rejection("!!! ??? ### $$$ %%% &&& *** @@@ !!! ???", cfg) == "symbol_heavy"
    assert rule_rejection(
        "the support agent recorded the customer request and updated the account", cfg
    ) is None


def test_learned_quality_filter_separates_clean_and_garbage():
    cfg = load_config("configs/domainx.yaml")
    scorer = train_quality_model("fixtures/quality/quality_labels.jsonl", cfg)
    clean = "support policy note 4: a refund is processed within 30 days of approval. the team records each request."
    garbage = "refund refund refund refund refund refund refund refund"
    assert scorer.accept(clean)[0] is True
    assert scorer.accept(garbage)[0] is False
    assert 0.0 <= scorer.score(clean) <= 1.0
    assert scorer.score(clean) > scorer.score(garbage)
