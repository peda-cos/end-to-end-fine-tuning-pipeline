"""Checkpoint save/load with canonical parameter checksums."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import parameter_sha256
from .tokenizer import Tokenizer
from .transformer import ModelConfig


def save_checkpoint(
    directory: str | Path,
    params: dict[str, np.ndarray],
    config: ModelConfig,
    tokenizer: Tokenizer,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "params.npz", **params)
    (path / "model_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    tokenizer.save(path / "tokenizer.json")
    info = {
        "parameter_sha256": parameter_sha256(params),
        "parameter_count": int(sum(value.size for value in params.values())),
        "metadata": metadata or {},
    }
    (path / "metadata.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


def load_checkpoint(directory: str | Path) -> tuple[dict[str, np.ndarray], ModelConfig, Tokenizer, dict[str, Any]]:
    path = Path(directory)
    config_payload = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    config = ModelConfig(**config_payload)
    with np.load(path / "params.npz", allow_pickle=False) as loaded:
        params = {name: loaded[name].astype(np.float64, copy=True) for name in loaded.files}
    tokenizer = Tokenizer.load(path / "tokenizer.json")
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return params, config, tokenizer, metadata


def load_params(directory: str | Path) -> dict[str, np.ndarray]:
    return load_checkpoint(directory)[0]
