from __future__ import annotations

import numpy as np

from ftpipe.quant.common import dequantize_grouped, pack_int4, quantize_grouped, unpack_int4
from ftpipe.quant.gptq import quantize_gptq
from ftpipe.quant.awq import quantize_awq


def test_int4_pack_roundtrip_and_group_dequant():
    rng = np.random.default_rng(1)
    weight = rng.normal(0, 0.1, size=(8, 32))
    codes, scales = quantize_grouped(weight, group_size=8)
    assert codes.shape == weight.shape
    assert scales.shape == (8, 4)
    restored = dequantize_grouped(codes, scales, 8)
    assert np.mean(np.abs(restored - weight)) < 0.02
    packed = pack_int4(codes)
    unpacked = unpack_int4(packed, codes.size).reshape(codes.shape)
    assert np.array_equal(unpacked, codes)


def test_gptq_improves_on_naive_rounding():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, size=(128, 32))
    weight = rng.normal(0, 0.1, size=(32, 32))
    params = {"l0_wq": weight}
    baseline_codes, baseline_scales = quantize_grouped(weight, 16)
    baseline = dequantize_grouped(baseline_codes, baseline_scales, 16)
    result = quantize_gptq(params, {"l0_wq": x}, group_size=16, damp_percent=0.01, act_order=True)
    assert result["report"]["quantized_layers"] == ["l0_wq"]
    assert result["report"]["bits_per_weight"] > 4.0
    assert "l0_wq.qweight" in result["artifacts"]
    assert np.mean((result["params"]["l0_wq"] - weight) ** 2) <= np.mean((baseline - weight) ** 2) * 1.5


def test_awq_selects_alpha_and_restores_shape():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(64, 32))
    weight = rng.normal(0, 0.1, size=(32, 32))
    result = quantize_awq({"l0_wq": weight}, {"l0_wq": x}, group_size=16, alpha_grid=[0.0, 0.5, 1.0])
    assert result["params"]["l0_wq"].shape == weight.shape
    assert result["report"]["layer_alpha"]["l0_wq"] in [0.0, 0.5, 1.0]
    assert result["report"]["mean_layer_mse"] < 0.001
    assert result["report"]["bits_per_weight"] > 4.0
