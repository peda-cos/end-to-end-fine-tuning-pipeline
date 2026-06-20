"""Manifest, hashing, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(root: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def git_dirty(root: Path | None = None) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def parameter_sha256(params: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(params):
        array = np.ascontiguousarray(params[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


DROP_FOR_DETERMINISM = {
    "seconds",
    "latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
    "spec_p50_latency_ms",
    "spec_p99_latency_ms",
    "output_tokens_per_second",
    "spec_output_tokens_per_second",
    "cost_model",
    "timestamp_ms",
    "created",
    "train_seconds",
}


def deterministic_view(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: deterministic_view(v) for k, v in value.items() if k not in DROP_FOR_DETERMINISM}
    if isinstance(value, list):
        return [deterministic_view(item) for item in value]
    return value


def deterministic_sha256(metrics: dict[str, Any], params: dict[str, np.ndarray] | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(parameter_sha256(params or {}).encode("ascii"))
    digest.update(sha256_json(deterministic_view(metrics)).encode("ascii"))
    return digest.hexdigest()


@dataclass
class Manifest:
    cfg: Any
    root: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def add_artifact(self, stage: str, path: str | Path) -> dict[str, Any]:
        full = Path(path)
        if not full.is_absolute():
            full = self.root / full
        if not full.exists():
            raise FileNotFoundError(f"artifact missing: {full}")
        try:
            relative = str(full.resolve().relative_to(self.root.resolve()))
        except ValueError:
            relative = str(full)
        record = {
            "stage": stage,
            "path": relative,
            "sha256": sha256_file(full),
            "bytes": full.stat().st_size,
        }
        self.records.append(record)
        return record

    def add_tree(self, stage: str, directory: str | Path) -> list[dict[str, Any]]:
        full = Path(directory)
        if not full.is_absolute():
            full = self.root / full
        records = []
        for path in sorted(p for p in full.rglob("*") if p.is_file()):
            records.append(self.add_artifact(stage, path))
        return records

    def write(self, path: str | Path) -> dict[str, Any]:
        payload = {
            "project": self.cfg["project"]["name"],
            "seed": self.cfg["run"]["seed"],
            "config": str(self.cfg.source),
            "config_sha256": sha256_json(self.cfg.data),
            "git_sha": git_sha(self.root),
            "git_dirty": git_dirty(self.root),
            "environment": environment(),
            "artifacts": sorted(self.records, key=lambda record: record["path"]),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
