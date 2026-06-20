from __future__ import annotations

import json

import pytest
import yaml

from ftpipe.config import ConfigError, load_config
from ftpipe.manifest import deterministic_view, sha256_json


def test_flagship_config_loads_and_has_expected_values():
    cfg = load_config("configs/domainx.yaml")
    assert cfg["project"]["name"] == "domainx"
    assert cfg["model"]["d_model"] == 64
    assert cfg["dpo"]["beta_grid"] == [0.05, 0.1, 0.2]
    assert cfg["serve"]["batch_sizes"] == [1, 8, 32]
    assert cfg.out_dir.name == "outputs"


def test_smoke_config_overrides_defaults():
    cfg = load_config("configs/smoke.yaml")
    assert cfg["pretrain"]["max_steps"] == 2
    assert cfg["data"]["raw_docs"] == 30
    assert cfg["serve"]["batch_sizes"] == [1, 2]


def test_unknown_key_is_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/domainx.yaml", encoding="utf-8"))
    raw["sft"]["mystery"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"sft\.mystery: unknown key"):
        load_config(path)


def test_missing_config_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_wrong_type_is_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/domainx.yaml", encoding="utf-8"))
    raw["data"]["raw_docs"] = "many"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="expected integer"):
        load_config(path)


def test_invalid_range_is_rejected(tmp_path):
    raw = yaml.safe_load(open("configs/domainx.yaml", encoding="utf-8"))
    raw["data"]["split"]["val_fraction"] = 1.5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="val_fraction"):
        load_config(path)


def test_bad_yaml_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_out_and_seed_overrides_are_immutable_copies():
    cfg = load_config("configs/domainx.yaml")
    changed = cfg.with_out("tmp/out").with_seed(11)
    assert cfg["run"]["seed"] == 7
    assert changed["run"]["seed"] == 11
    assert changed.out_dir.name == "out"
    assert sha256_json(cfg.data) != sha256_json(changed.data)


def test_deterministic_view_drops_timings():
    metrics = {
        "eval": {"task": 0.5},
        "training": {"seconds": {"total": 12.3}},
        "serving": {"batches": [{"p99_latency_ms": 9.0, "acceptance_rate": 0.75}]},
    }
    assert deterministic_view(metrics) == {
        "eval": {"task": 0.5},
        "training": {},
        "serving": {"batches": [{"acceptance_rate": 0.75}]},
    }
    assert json.dumps(deterministic_view(metrics), sort_keys=True)
