from __future__ import annotations

import numpy as np

from ftpipe.model.transformer import ModelConfig, forward, init_params, loss_and_grads


def test_manual_gradients_match_finite_differences():
    config = ModelConfig(vocab_size=7, d_model=4, n_layers=1, n_heads=2, d_ff=8, max_seq_len=6)
    rng = np.random.default_rng(3)
    params = init_params(config, rng)
    tokens = np.asarray([[1, 2, 3, 4, 5], [2, 4, 1, 0, 3]], dtype=np.int64)
    labels = np.asarray([[2, 3, 4, 5, 6], [4, 1, -100, 3, 2]], dtype=np.int64)
    loss, grads = loss_and_grads(params, config, tokens, labels)
    assert np.isfinite(loss) and loss > 0
    eps = 1e-6
    checked = 0
    for name, value in params.items():
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + eps
            plus, _ = loss_and_grads(params, config, tokens, labels)
            value[index] = original - eps
            minus, _ = loss_and_grads(params, config, tokens, labels)
            value[index] = original
            numeric = (plus - minus) / (2 * eps)
            analytic = grads[name][index]
            assert abs(numeric - analytic) <= 3e-5 + 2e-4 * max(abs(numeric), abs(analytic)), (name, index)
            checked += 1
    assert checked > 200


def test_segment_mask_blocks_cross_document_attention():
    config = ModelConfig(vocab_size=9, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=8)
    params = init_params(config, np.random.default_rng(4))
    segments = np.asarray([[0, 0, 0, 1, 1, -1, -1, -1]], dtype=np.int64)
    tokens = np.asarray([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=np.int64)
    changed = tokens.copy()
    changed[0, 0:3] = [6, 7, 8]
    first = forward(params, config, tokens, segment_ids=segments)["logits"]
    second = forward(params, config, changed, segment_ids=segments)["logits"]
    assert np.allclose(first[0, 3:5], second[0, 3:5], atol=1e-12)
    assert not np.allclose(first[0, 0:3], second[0, 0:3], atol=1e-12)
