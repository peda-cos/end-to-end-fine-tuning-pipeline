"""Render Kubernetes deployment, service, HPA, and queue-wait adapter manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def emit_k8s(cfg: Any, out_dir: str | Path, *, model_path: str, speculator: str) -> dict[str, Path]:
    source = Path(__file__).resolve().parents[3] / "deploy" / "k8s"
    target = Path(out_dir) / "k8s"
    target.mkdir(parents=True, exist_ok=True)
    substitutions = {
        "{{MODEL_PATH}}": model_path,
        "{{SPECULATOR}}": speculator,
        "{{VLLM_IMAGE}}": str(cfg["serve"].get("vllm_image", "vllm/vllm-openai:latest")),
    }
    written = {}
    for template in ("deployment", "service", "hpa", "prometheus-adapter"):
        text = (source / f"{template}.yaml.tmpl").read_text(encoding="utf-8")
        for key, value in substitutions.items():
            text = text.replace(key, value)
        payload = yaml.safe_load(text)
        path = target / f"{template}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written[template] = path
    return written
