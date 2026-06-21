from __future__ import annotations

import copy

import numpy as np

from ftpipe.manifest import parameter_sha256
from ftpipe.model.checkpoint import load_checkpoint, save_checkpoint
from ftpipe.model.tokenizer import Tokenizer
from ftpipe.model.train import evaluate_loss, pack_sft, train_model
from ftpipe.model.transformer import ModelConfig, init_params


def _setup():
    tokenizer = Tokenizer.build(
        ["ticket my card was charged for order 1001 billing high", "ticket package delayed order 1002 shipping low"],
        max_vocab=80,
    )
    records = [
        {"instruction": "ticket my card was charged for order 1001", "response": "billing high"},
        {"instruction": "ticket package delayed order 1002", "response": "shipping low"},
    ] * 4
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32)
    params = init_params(config, np.random.default_rng(5))
    batches = pack_sft(records, tokenizer, seq_len=24)
    return tokenizer, config, params, batches


def test_training_reduces_loss_and_is_deterministic():
    tokenizer, config, params, batches = _setup()
    first = train_model(
        copy.deepcopy(params), config, batches, learning_rate=0.005, steps=12, batch_size=2,
        warmup_ratio=0.1, weight_decay=0.0, grad_clip=1.0, seed=7,
    )
    second = train_model(
        copy.deepcopy(params), config, batches, learning_rate=0.005, steps=12, batch_size=2,
        warmup_ratio=0.1, weight_decay=0.0, grad_clip=1.0, seed=7,
    )
    assert first["final_loss"] < first["history"][0]["loss"]
    assert parameter_sha256(first["params"]) == parameter_sha256(second["params"])


def test_checkpoint_roundtrip(tmp_path):
    tokenizer, config, params, batches = _setup()
    info = save_checkpoint(tmp_path / "model", params, config, tokenizer, {"stage": "test"})
    loaded, loaded_config, loaded_tokenizer, metadata = load_checkpoint(tmp_path / "model")
    assert parameter_sha256(loaded) == info["parameter_sha256"]
    assert loaded_config == config
    assert loaded_tokenizer.tokens == tokenizer.tokens
    assert metadata["metadata"]["stage"] == "test"
    assert np.isfinite(evaluate_loss(loaded, loaded_config, batches))
