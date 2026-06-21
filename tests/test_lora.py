from __future__ import annotations

import copy

import numpy as np

from ftpipe.manifest import parameter_sha256
from ftpipe.model.lora import adapter_grads, apply_lora, init_lora, scale_of
from ftpipe.model.tokenizer import Tokenizer
from ftpipe.model.train import pack_sft, train_model
from ftpipe.model.transformer import ModelConfig, forward, init_params


def test_lora_starts_as_identity_and_trains_only_adapters():
    tokenizer = Tokenizer.build(["ticket card charged order 1001 billing high"], max_vocab=64)
    records = [{"instruction": "ticket card charged order 1001", "response": "billing high"}] * 4
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32)
    params = init_params(config, np.random.default_rng(6))
    adapters = init_lora(params, config.n_layers, rank=4, alpha=8, rng=np.random.default_rng(7))
    effective = apply_lora(params, adapters)
    assert parameter_sha256(params) == parameter_sha256({name: effective[name] for name in params})
    assert scale_of(adapters) == 2.0

    base_checksum = parameter_sha256(params)
    batches = pack_sft(records, tokenizer, seq_len=24)
    train_model(
        params, config, batches, learning_rate=0.01, steps=4, batch_size=2,
        warmup_ratio=0.1, weight_decay=0.0, grad_clip=1.0, seed=8, lora=adapters,
    )
    assert parameter_sha256(params) == base_checksum
    assert any(np.any(value != 0) for name, value in adapters.items() if name.endswith("_B"))
    assert adapters["l0_wq_A"].shape == (4, 16)
