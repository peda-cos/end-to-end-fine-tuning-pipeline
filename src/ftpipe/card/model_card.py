"""Model card generation tied to measured metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mof import assess


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(metrics: dict[str, dict[str, float]], models: list[str]) -> str:
    rows = ["| Metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name, values in metrics.items():
        rows.append("| " + name + " | " + " | ".join(_fmt(values.get(model, 0.0)) for model in models) + " |")
    return "\n".join(rows)


def render_model_card(cfg: Any, metrics: dict[str, Any], root: Path) -> str:
    models = metrics["eval"]["models"]
    mof = assess(root, assume_present={"Model card", "Data card", "Sample model outputs"}, out_dir=cfg.out_dir)
    beta_rows = "\n".join(
        f"| {item['beta']} | {item['preference_accuracy']:.4f} | {item['loss']:.4f} | {'yes' if item['selected'] else 'no'} |"
        for item in metrics["eval"].get("beta_sweep", [])
    )
    quant_rows = "\n".join(
        f"| {key} | {value.get('size_bytes', 0)} | {value.get('bits_per_weight', 0.0):.2f} | "
        f"{value.get('task_exact_match', 0.0):.4f} | {value.get('delta_vs_unquantized', 0.0):+.4f} |"
        for key, value in metrics.get("quant", {}).items()
    )
    serving_rows = "\n".join(
        f"| {item['batch_size']} | {'spec' if item['speculative'] else 'target'} | "
        f"{item['output_tokens_per_second']:.2f} | {item['p50_latency_ms']:.2f} | {item['p99_latency_ms']:.2f} |"
        for item in metrics.get("serving", {}).get("batches", [])
    )
    cost = metrics.get("serving", {}).get("cost_model", {})
    comparison_rows = "\n".join(
        f"| {item['provider']} | {item['model']} | {item['usd_per_1m_output_tokens']:.2f} | {item['source']} |"
        for item in cost.get("comparisons", [])
    )
    safety = metrics.get("safety", {})
    components = "\n".join(
        f"| {item['name']} | {item['class']} | {'present' if item['present'] else 'missing'} | {item['license']} |"
        for item in mof["components"]
    )
    return f"""# Model card — {cfg['project']['name']}

## Model details

- Local model: `{metrics['run']['local_model']}` (tiny NumPy decoder-only transformer, {cfg['model']['n_layers']} layer(s), d_model {cfg['model']['d_model']}).
- Real-stack target described by this pipeline: `{cfg['project']['base_model']}` (8B class; not executed locally).
- Pipeline commit: `{metrics['run']['git_sha']}`; config hash: `{metrics['run']['config_sha256']}`.
- Seed: `{metrics['run']['seed']}`; serving checkpoint: `{metrics['run']['serving_checkpoint']}`.
- Environment: Python {metrics['run']['environment']['python']}, NumPy {metrics['run']['environment']['numpy']}, {metrics['run']['environment']['platform']}.

This card is generated from `metrics.json` in the run output directory. It reports measured local results only. The
local model is a deterministic toy used to exercise the pipeline; it is not an 8B model and its
scores are not comparable to production fine-tunes.

## Intended use

Research and portfolio demonstration of an end-to-end data → SFT → DPO/GRPO → quantize → serve → eval
pipeline: data hygiene, preference optimization, quantization math, speculative decoding, safety
filtering, and reproducible model cards. The local model is for pipeline validation, not deployment.

## Out-of-scope use

No production, safety-critical, legal, medical, or financial decisions. No claim of real MMLU-Pro,
MT-Bench-v2, RewardBench-2, or Llama Guard results; the local fixtures are synthetic stand-ins.

## Architecture

Local: {cfg['model']['n_layers']}-layer pre-LN decoder with tied word embeddings, causal multi-head attention, GELU MLP,
and manually implemented reverse-mode gradients. Real path: Llama 3.1 8B through Axolotl with
ZeRO-3, Flash Attention 3, and sample packing. See `src/ftpipe/model/transformer.py` and
`gpu/sft.axolotl.yaml`.

## Data

Synthetic support-ticket domain generated deterministically by `src/ftpipe/data/synth.py`.
Data card: `data-card.md`. Preprocessing: MinHash dedup, Gopher/learned quality filter,
Presidio-style PII scrub, seeded splits, and benchmark contamination rejection. No third-party
benchmark data or real PII is included.

## Training

| Stage | Steps | Notes |
|---|---:|---|
| Pretrain | {metrics['training']['steps'].get('pretrain', 0)} | next-token on policy/generic synthetic corpus |
| SFT | {metrics['training']['steps'].get('sft', 0)} | assistant-only packed cross-entropy |
| LoRA SFT | {metrics['training']['steps'].get('lora', 0)} | rank {cfg['sft']['lora']['rank']} adapters, fraction {metrics['training'].get('lora_trainable_fraction', 0.0):.4f} |
| DPO | {metrics['training']['steps'].get('dpo', 0)} | selected beta {metrics['training'].get('dpo_beta_selected', 0)} |
| GRPO | {metrics['training']['steps'].get('grpo', 0)} | verifiable structured reward, group-relative advantages |

Initial/final SFT loss: {_fmt(metrics['training'].get('sft_train_loss_initial', 0.0))} →
{_fmt(metrics['training'].get('sft_train_loss_final', 0.0))}. Base validation loss:
{_fmt(metrics['training'].get('base_val_loss', 0.0))}; SFT validation loss:
{_fmt(metrics['training'].get('sft_val_loss', 0.0))}. All runs use the configured seed.

## Evaluation

Local synthetic stand-ins; see README for benchmark-source caveats.

{_table(metrics['eval']['metrics'], models)}

### DPO beta sweep

| Beta | Dev preference accuracy | Loss | Selected |
|---:|---:|---:|---|
{beta_rows or '| — | — | — | — |'}

## Quantization and serving

| Artifact | Bytes | Bits/weight | Task exact match | Δ vs unquantized |
|---|---:|---:|---:|---:|
{quant_rows or '| — | — | — | — | — |'}

| Batch | Mode | Output tok/s | p50 ms | p99 ms |
|---:|---|---:|---:|---:|
{serving_rows or '| — | — | — | — | — |'}

Measured acceptance rate: {_fmt(metrics.get('serving', {}).get('speculative', {}).get('acceptance_rate', 0.0))}
(domain) / {_fmt(metrics.get('serving', {}).get('speculative', {}).get('generic_acceptance_rate', 0.0))} (generic).
The GGUF export uses the local Q4_K/Q6_K codec; it is not claimed to be llama.cpp-loadable for the toy
architecture. GPTQ/AWQ values use a reference dequantizer; CUDA Marlin/AWQ kernels were not executed.

### Modeled cost

Self-hosted cost is modeled from measured local throughput at ${cost.get('gpu_hourly_usd', 0.0):.2f}/GPU-hour:
**${cost.get('self_hosted_usd_per_1m_output_tokens', 0.0):.4f} per 1M output tokens**. This is not a GPU measurement.

| Provider | Model | USD / 1M output | Source |
|---|---|---:|---|
{comparison_rows or '| — | — | — | — |'}

## Safety

Local Llama-Guard-shaped input classifier and ShieldGemma-shaped output classifier (hashed n-gram logistic
regression), not Meta/Google models. Guard pass rates: {json.dumps(safety.get('guard_pass_rate', {}), sort_keys=True)}.
Shield unsafe block rate: {_fmt(safety.get('shield_block_rate_unsafe', 0.0))}; benign false-positive rate:
{_fmt(safety.get('shield_false_positive_rate', 0.0))}. Bias probe guard FP:
{_fmt(safety.get('bias_probe_guard_fp_rate', 0.0))}. These are small synthetic checks, not a full bias evaluation.

## Limitations

- The local model is a toy; no 8B training, GPU kernels, vLLM, Marlin, or live Kubernetes was executed.
- Benchmark-shaped fixtures are synthetic; scores are not real MMLU-Pro / MT-Bench-v2 / RewardBench-2 results.
- The MT-Bench-style judge is a deterministic rubric, not an LLM judge.
- Safety classifiers are local stand-ins, not Llama Guard 4 or ShieldGemma 2.
- Cost is modeled from local CPU throughput and vendor list prices; no provider was called.
- GGUF files target a custom toy architecture and are not loaded by llama.cpp.

## Licenses

Code: MIT. Generated data: CC0-1.0. Model parameters: CDLA-Permissive-2.0. Documentation: CC-BY-4.0.
See `LICENSES.md`.

## Model Openness Framework

Self-assessed against MOF v1 (arXiv:2403.13784): **{mof['class']}**. The spec's “MOF 2026” label is not a
separate published version used here.

| Component | Class | Status | License |
|---|---|---|---|
{components}

## Reproducibility

```bash
./verify.sh
./pipeline.sh configs/domainx.yaml
```

Deterministic metrics hash: `{metrics['run']['deterministic_sha256']}`. Timings and throughput are excluded
from this hash. Seeds, config hash, git SHA, and artifact checksums are recorded in `outputs/manifest.json`.
"""
