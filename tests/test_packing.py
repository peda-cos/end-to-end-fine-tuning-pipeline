from __future__ import annotations

import numpy as np

from ftpipe.model.tokenizer import Tokenizer, tokenize
from ftpipe.model.train import pack_pretrain, pack_sft, stack_batches


def _records():
    return [
        {"instruction": "Ticket: my card was charged twice for order 1001", "response": "billing high"},
        {"instruction": "Ticket: the package is delayed for order 1002", "response": "shipping low"},
        {"instruction": "Ticket: reset my password for order 1003", "response": "account low"},
    ]


def test_tokenizer_roundtrip_and_specials():
    texts = ["Ticket: my card was charged twice for order 1001", "billing high", "the package is delayed"]
    tokenizer = Tokenizer.build(texts, max_vocab=128)
    assert tokenizer.tokens[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
    ids = tokenizer.encode("billing high", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "billing high"
    assert tokenizer.oov_rate(texts) < 0.01
    assert tokenize("Order 4812!") == ["order", "4812", "!"]


def test_sft_packing_masks_prompt_and_marks_segments():
    tokenizer = Tokenizer.build(
        [record["instruction"] + " " + record["response"] for record in _records()], max_vocab=128
    )
    batches = pack_sft(_records(), tokenizer, seq_len=48)
    assert batches
    labels = np.concatenate([batch.labels for batch in batches])
    segments = np.concatenate([batch.segment_ids for batch in batches])
    assert np.any(labels == -100)
    assert np.any(labels >= 0)
    assert set(np.unique(segments)).issubset(set(range(8)) | {-1})
    for batch in batches:
        assert batch.tokens.shape == (48,)
        assert batch.labels.shape == (48,)
        assert np.all(batch.labels[batch.tokens == tokenizer.pad_id] == -100)


def test_pretrain_packing_uses_next_token_labels():
    tokenizer = Tokenizer.build(["the support team recorded the request"], max_vocab=64)
    batches = pack_pretrain(["the support team recorded the request"], tokenizer, seq_len=16)
    assert batches
    batch = batches[0]
    valid = batch.labels != -100
    assert valid.sum() > 0
    assert np.all(batch.labels[valid] < tokenizer.vocab_size)
    tokens, labels, segments = stack_batches([batch, batch])
    assert tokens.shape == labels.shape == segments.shape == (2, 16)
