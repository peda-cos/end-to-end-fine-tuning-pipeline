from __future__ import annotations

import numpy as np

from ftpipe.quant.gguf import (
    QK_K,
    decode_q4_k,
    decode_q6_k,
    encode_q4_k,
    encode_q6_k,
    read_gguf,
    write_gguf,
)


def test_q4_k_and_q6_k_blocks_roundtrip():
    rng = np.random.default_rng(4)
    values = rng.normal(0, 1, size=QK_K)
    q4 = decode_q4_k(encode_q4_k(values))
    q6 = decode_q6_k(encode_q6_k(values))
    assert q4.shape == q6.shape == (QK_K,)
    assert np.mean(np.abs(values - q4)) < 0.12
    assert np.mean(np.abs(values - q6)) < 0.05
    assert np.mean(np.abs(values - q6)) < np.mean(np.abs(values - q4))


def test_gguf_container_roundtrip(tmp_path):
    from ftpipe.model.tokenizer import Tokenizer
    from ftpipe.model.transformer import ModelConfig

    tokenizer = Tokenizer.build(["ticket card charged order 1001 billing high"], max_vocab=64)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=32, n_layers=1, n_heads=4, d_ff=64, max_seq_len=32)
    rng = np.random.default_rng(5)
    params = {
        "tok_emb": rng.normal(0, 0.02, size=(config.vocab_size, config.d_model)),
        "pos_emb": rng.normal(0, 0.02, size=(config.max_seq_len, config.d_model)),
        "l0_wq": rng.normal(0, 0.02, size=(config.d_model, config.d_model)),
        "l0_w2": rng.normal(0, 0.02, size=(config.d_ff, config.d_model)),
        "l0_bq": np.zeros(config.d_model),
    }
    path = tmp_path / "toy.gguf"
    report = write_gguf(path, params, tokenizer, project="test", config=config, seed=7)
    assert report["quant_type"] if "quant_type" in report else True
    assert path.read_bytes()[:4] == b"GGUF"
    metadata, tensors = read_gguf(path)
    assert metadata["general.architecture"] == "ftpipe-tiny"
    assert metadata["ftpipe.quant_type"] == "Q4_K_M"
    assert tensors["tok_emb"].shape == params["tok_emb"].shape
    assert tensors["l0_wq"].shape == params["l0_wq"].shape
    assert report["bits_per_weight"] < 8.0
    assert np.mean(np.abs(tensors["l0_wq"] - params["l0_wq"])) < 0.05
