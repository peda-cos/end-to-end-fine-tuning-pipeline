"""Model Openness Framework (MOF v1) checklist and class assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Components follow MOF v1 (arXiv:2403.13784), Table 1.
COMPONENTS = [
    ("Model architecture", "III", "src/ftpipe/model/transformer.py", "MIT"),
    ("Model parameters (final checkpoints)", "III", "outputs/models/sft", "CDLA-Permissive-2.0"),
    ("Technical report or research paper", "III", "README.md", "CC-BY-4.0"),
    ("Evaluation results", "III", "outputs/eval/results.json", "CC-BY-4.0"),
    ("Model card", "III", "outputs/model-card.md", "CC-BY-4.0"),
    ("Data card", "III", "outputs/data-card.md", "CC-BY-4.0"),
    ("Sample model outputs", "III", "outputs/eval/predictions", "CC-BY-4.0"),
    ("Training, validation, and testing code", "II", "src/ftpipe/model/train.py", "MIT"),
    ("Inference code", "II", "src/ftpipe/serve/engine.py", "MIT"),
    ("Evaluation code", "II", "src/ftpipe/evaluate", "MIT"),
    ("Evaluation data", "II", "fixtures/benchmarks", "CC0-1.0"),
    ("Supporting libraries and tools", "II", "pyproject.toml", "MIT"),
    ("Research paper", "I", "paper.md", "CC-BY-4.0"),
    ("Datasets", "I", "outputs/data/raw.jsonl", "CC0-1.0"),
    ("Data preprocessing code", "I", "src/ftpipe/data", "MIT"),
    ("Model parameters (intermediate checkpoints)", "I", "outputs/models/base", "CDLA-Permissive-2.0"),
    ("Model metadata", "I", "outputs/models/base/metadata.json", "CC-BY-4.0"),
]


def assess(
    root: str | Path,
    assume_present: set[str] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    generated = Path(out_dir) if out_dir is not None else root / "outputs"
    assumed = assume_present or set()
    rows = []
    for name, level, artifact, license_name in COMPONENTS:
        if artifact.startswith("outputs/"):
            present = (generated / artifact[len("outputs/"):]).exists()
        else:
            present = (root / artifact).exists()
        rows.append(
            {
                "name": name,
                "class": level,
                "present": name in assumed or present,
                "artifact": artifact,
                "license": license_name,
            }
        )
    class_iii = all(item["present"] for item in rows if item["class"] == "III")
    class_ii = class_iii and all(item["present"] for item in rows if item["class"] == "II")
    class_i = class_ii and all(item["present"] for item in rows if item["class"] == "I")
    mof_class = "Class I — Open Science" if class_i else ("Class II — Open Tooling" if class_ii else ("Class III — Open Model" if class_iii else "Unclassified"))
    return {"class": mof_class, "components": rows}
