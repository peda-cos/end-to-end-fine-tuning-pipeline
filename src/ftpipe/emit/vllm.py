"""Emit vLLM serving, Speculators training, safety, and lm-eval commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def emit_vllm(cfg: Any, out_dir: str | Path) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    spec = target / "speculator-train.sh"
    spec.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
# Speculators is installed in a separate environment; see the vLLM Speculators guide.
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated base weights}"
speculators prepare-data --model meta-llama/Llama-3.1-8B-Instruct --data outputs/gpu/sft-spec-data.jsonl --output outputs/gpu/spec-data --max-samples 5000 --seq-length 8192 --no-skip-token-freq
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct -- --port 8000 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
sleep 20
speculators generate-offline-data --preprocessed-data outputs/gpu/spec-data --endpoint http://localhost:8000/v1 --output outputs/gpu/spec-data/hidden_states --max-samples 5000 --concurrency 32
torchrun --standalone --nproc_per_node 2 -m speculators.train --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct --data-path outputs/gpu/spec-data --save-path outputs/gpu/spec-checkpoints --draft-vocab-size 32000 --epochs 5 --total-seq-len 8192 --vllm-endpoint http://localhost:8000/v1 --on-missing generate
""",
        encoding="utf-8",
    )
    serve = target / "vllm-serve.sh"
    serve.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
# vLLM 0.7 predates EAGLE-3; pin a validated modern vLLM release.
vllm serve meta-llama/Llama-3.1-8B-Instruct \\
  --speculative-config '{"method":"eagle3","model":"outputs/gpu/spec-checkpoints/final","num_speculative_tokens":4}' \\
  --port 8000
""",
        encoding="utf-8",
    )
    safety = target / "safety-serve.sh"
    safety.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
# Local pipeline uses classifier stand-ins; these are the real safety services.
vllm serve meta-llama/Llama-Guard-4-12B --port 8001 &
vllm serve google/shieldgemma-2b --port 8002 &
wait
""",
        encoding="utf-8",
    )
    lm_eval = target / "lm-eval.sh"
    lm_eval.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
# mmlu_pro is the lm-evaluation-harness task group; MT-Bench-v2 and RewardBench-2 have their own harnesses.
lm_eval --model local-completions \\
  --model_args base_url=http://localhost:8000/v1/completions,tokenized_requests=False \\
  --tasks mmlu_pro --num_fewshot 5 --output_path outputs/gpu/lm-eval
""",
        encoding="utf-8",
    )
    for path in (spec, serve, safety, lm_eval):
        path.chmod(0o755)
    return {"speculator": spec, "serve": serve, "safety": safety, "lm_eval": lm_eval}
