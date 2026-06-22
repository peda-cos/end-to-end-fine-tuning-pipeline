"""Emit llm-compressor and llama.cpp quantization assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QUANTIZE_SCRIPT = '''"""Generated llm-compressor quantizer for the real 8B stack."""
from __future__ import annotations

import argparse
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gptq", "awq"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", default="outputs/gpu/calibration.jsonl")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seq-length", type=int, default=4096)
    args = parser.parse_args()
    recipe = (
        GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
        if args.method == "gptq"
        else [AWQModifier(), QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]
    )
    oneshot(
        model=args.model,
        dataset_path=args.calibration,
        text_column="text",
        recipe=recipe,
        max_seq_length=args.seq_length,
        num_calibration_samples=args.samples,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
'''

QUANTIZE_SH = """#!/usr/bin/env bash
set -euo pipefail

python outputs/gpu/quantize.py --method gptq --model outputs/gpu/dpo --output outputs/gpu/dpo-gptq
python outputs/gpu/quantize.py --method awq --model outputs/gpu/dpo --output outputs/gpu/dpo-awq

# llama.cpp GGUF conversion (clone llama.cpp and build llama-quantize first).
python llama.cpp/convert_hf_to_gguf.py outputs/gpu/dpo --outfile outputs/gpu/target-f16.gguf --outtype f16
llama-quantize outputs/gpu/target-f16.gguf outputs/gpu/target-Q4_K_M.gguf Q4_K_M
"""


def emit_quantize(cfg: Any, out_dir: str | Path, records: list[dict[str, Any]]) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    calibration = target / "calibration.jsonl"
    with calibration.open("w", encoding="utf-8") as handle:
        for record in records[: int(cfg["quant"]["calibration_samples"])]:
            handle.write(json.dumps({"text": f"{record['instruction']}\\n{record['response']}"}, sort_keys=True) + "\\n")
    script = target / "quantize.py"
    shell = target / "quantize.sh"
    script.write_text(QUANTIZE_SCRIPT, encoding="utf-8")
    shell.write_text(QUANTIZE_SH, encoding="utf-8")
    shell.chmod(0o755)
    return {"calibration": calibration, "script": script, "shell": shell}
