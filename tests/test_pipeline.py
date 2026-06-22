from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ftpipe.config import ConfigError, load_config
from ftpipe.stages.pipeline import run_pipeline


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pipeline") / "outputs"
    cfg = load_config("configs/smoke.yaml").with_out(out)
    run_pipeline(cfg)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return cfg, metrics


def test_pipeline_writes_every_required_artifact(smoke_run):
    cfg, metrics = smoke_run
    out = Path(cfg.out_dir)
    for relative in (
        "data/stats.json",
        "models/base/params.npz",
        "models/sft/params.npz",
        "models/sft_lora/params.npz",
        "models/dpo/params.npz",
        "models/grpo/params.npz",
        "models/draft/draft.npz",
        "eval/results.json",
        "eval/predictions/dpo-task.jsonl",
        "quant/report.json",
        "quant/gguf_q4_k_m.gguf",
        "serving/bench.json",
        "safety/results.json",
        "metrics.json",
        "manifest.json",
        "model-card.md",
        "data-card.md",
    ):
        assert (out / relative).exists(), relative
    assert metrics["eval"]["models"] == ["base", "sft", "sft_lora", "dpo", "grpo"]
    assert metrics["run"]["serving_checkpoint"] in metrics["eval"]["models"]
    assert len(metrics["run"]["deterministic_sha256"]) == 64
    assert metrics["data"]["contamination"]["green"] is True
    assert set(metrics["quant"]) == {"gptq_int4", "awq_int4", "gguf_q4_k_m"}


def test_rerun_is_metric_deterministic(smoke_run):
    cfg, metrics = smoke_run
    before = metrics["run"]["deterministic_sha256"]
    run_pipeline(cfg)
    after = json.loads((Path(cfg.out_dir) / "metrics.json").read_text(encoding="utf-8"))
    assert after["run"]["deterministic_sha256"] == before
    assert after["checkpoints"] == metrics["checkpoints"]


def test_eval_predictions_are_actual_generations(smoke_run):
    cfg, _ = smoke_run
    rows = [
        json.loads(line)
        for line in (Path(cfg.out_dir) / "eval" / "predictions" / "sft-task.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    assert {"id", "instruction", "expected", "output", "correct", "format_valid"} <= set(rows[0])
    assert all(isinstance(row["output"], str) for row in rows)


def test_emitted_real_stack_configs_match_pipeline(smoke_run):
    cfg, _ = smoke_run
    gpu = Path(cfg.out_dir) / "gpu"
    axolotl = yaml.safe_load((gpu / "sft.axolotl.yaml").read_text(encoding="utf-8"))
    assert axolotl["base_model"] == cfg["project"]["base_model"]
    assert axolotl["sample_packing"] is True
    assert axolotl["attn_implementation"] == "flash_attention_3"
    assert axolotl["deepspeed"].endswith("zero3.json")
    trl = yaml.safe_load((gpu / "dpo.trl.yaml").read_text(encoding="utf-8"))
    assert trl["beta"] > 0 and trl["loss_type"] == "sigmoid"
    deployment = yaml.safe_load((gpu / "k8s" / "deployment.yaml").read_text(encoding="utf-8"))
    args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "eagle3" in " ".join(str(item) for item in args)
    hpa = yaml.safe_load((gpu / "k8s" / "hpa.yaml").read_text(encoding="utf-8"))
    metric = hpa["spec"]["metrics"][0]["external"]["metric"]["name"]
    assert metric == "vllm:num_requests_waiting"


def test_generated_cards_report_measured_values(smoke_run):
    cfg, metrics = smoke_run
    out = Path(cfg.out_dir)
    card = (out / "model-card.md").read_text(encoding="utf-8")
    data_card = (out / "data-card.md").read_text(encoding="utf-8")
    assert f"{metrics['training']['sft_val_loss']:.4f}" in card
    assert metrics["run"]["deterministic_sha256"] in card
    assert "Model Openness Framework" in card
    assert "not a GPU measurement" in card
    assert str(metrics["data"]["raw_docs"]) in data_card
    assert "green: True" in data_card


def test_stage_selection_is_validated(tmp_path):
    cfg = load_config("configs/smoke.yaml").with_out(tmp_path / "empty")
    with pytest.raises(ConfigError, match="unknown stage"):
        run_pipeline(cfg, stages=["nonsense"])
    with pytest.raises(FileNotFoundError, match="requires"):
        run_pipeline(cfg, stages=["sft"])
