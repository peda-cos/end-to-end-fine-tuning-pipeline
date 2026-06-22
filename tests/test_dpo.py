from __future__ import annotations

import numpy as np

from ftpipe.align.common import encode_completion, preference_accuracy
from ftpipe.align.dpo import train_dpo_beta
from ftpipe.model.tokenizer import Tokenizer
from ftpipe.model.transformer import ModelConfig, init_params, sequence_logprobs
from ftpipe.align.common import pad_encoded


def _fixture():
    tokenizer = Tokenizer.build(
        ["ticket card charged order 1001 ticket package delayed order 1002 billing high shipping low"],
        max_vocab=64,
    )
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32)
    params = init_params(config, np.random.default_rng(6))
    pairs = [
        {
            "prompt": "ticket card charged order 1001",
            "chosen": "billing high",
            "rejected": "shipping low",
        }
    ] * 4
    return tokenizer, config, params, pairs


def test_dpo_step_increases_chosen_margin(tmp_path):
    tokenizer, config, params, pairs = _fixture()
    rows = []
    for pair in pairs[:1]:
        rows += [
            encode_completion(tokenizer, pair["prompt"], pair["chosen"], config.max_seq_len),
            encode_completion(tokenizer, pair["prompt"], pair["rejected"], config.max_seq_len),
        ]
    tokens, labels, segments, _ = pad_encoded(rows, tokenizer)
    before, _ = sequence_logprobs(params, config, tokens, labels, segment_ids=segments)
    policy, _ = train_dpo_beta(
        params,
        config,
        tokenizer,
        pairs,
        beta=0.1,
        steps=5,
        batch_size=1,
        learning_rate=0.01,
        seed=11,
        log_path=tmp_path / "dpo.jsonl",
    )
    after, _ = sequence_logprobs(policy, config, tokens, labels, segment_ids=segments)
    assert (after[0] - after[1]) > (before[0] - before[1])
    assert preference_accuracy(policy, config, tokenizer, pairs) >= 0.0
