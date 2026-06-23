# End-to-End Fine-Tuning Pipeline (Data → SFT → DPO/GRPO → Quantize → Serve)

A reproducible, one-command fine-tuning pipeline for a domain task: synthetic support-ticket
triage. It runs data hygiene (dedup, quality filtering, PII scrubbing, benchmark-contamination
gates), pretraining, SFT, LoRA SFT, DPO with a beta sweep, GRPO with a verifiable reward,
GPTQ/AWQ/GGUF quantization, speculative serving behind an OpenAI-compatible HTTP API,
an evaluation matrix, safety classifiers, a Model Openness Framework self-assessment, and a
TypeScript results dashboard.

The local core is a small NumPy decoder-only transformer so the whole mechanism runs
deterministically on CPU in about three minutes. The pipeline also emits the YAML, scripts, and
Kubernetes manifests for the real 8B stack (Axolotl/TRL/vLLM/llama.cpp); those GPU steps are not
executed here and are explicitly marked as such. No result below is copied from a vendor
benchmark or from the source specification.

## Quickstart

Requirements: Python 3.14 + `uv`, Node/Bun for the dashboard, git. No GPU, API key, or network
service is needed at runtime (dependency installation is the only network step).

```bash
git clone <this-repository> end-to-end-fine-tuning-pipeline
cd end-to-end-fine-tuning-pipeline

./verify.sh                          # pinned Python + Bun deps, tests, typecheck, contamination gate
./pipeline.sh configs/smoke.yaml     # ~4 s end-to-end smoke run into outputs/smoke/
./pipeline.sh configs/domainx.yaml   # measured run: 1200 raw docs, 5 checkpoints, ~3 min on CPU
```

Inspect the artifacts:

```bash
cat outputs/metrics.json
cat outputs/model-card.md
cat outputs/data-card.md
cat outputs/serving/bench.json

bun run dashboard outputs/metrics.json outputs/dashboard.html
# open outputs/dashboard.html

# Serve the selected checkpoint (DPO for the measured run) over the OpenAI-compatible API.
uv run --frozen python -m ftpipe serve configs/domainx.yaml \
  --checkpoint outputs/models/dpo --port 8000
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"domainx","messages":[{"role":"user","content":"Ticket: my card was charged twice order 4812\nAnswer with category and priority."}],"max_tokens":8}'
```

`./pipeline.sh configs/domainx.yaml --serve` also works: it reruns the pipeline and leaves the
server running. The standalone `serve` command reuses an existing checkpoint without retraining.

## Architecture

```
configs/*.yaml
    |
    v
[data]  deterministic corpus -> exact + MinHash-LSH dedup -> quality rules + learned classifier
        -> PII recognizers/anonymization -> seeded splits -> exact n-gram + MinHash contamination gate
    |
    v
[pretrain]  tiny decoder-only transformer, next-token prediction on cleaned corpus
    |
    v
[SFT]  assistant-only packed cross-entropy; full fine-tune + LoRA alternative
    |
    v
[DPO]  frozen reference, sigmoid loss, beta sweep on held-out preferences
[GRPO] verifiable structured reward, group-relative advantages, KL penalty
    |
    v
[eval] base vs SFT vs LoRA vs DPO vs GRPO on task/MCQ/rubric/preference suites
    |
    v
[quant] GPTQ-INT4 (Hessian/act-order) + AWQ-INT4 + GGUF Q4_K_M/Q6_K codec
    |
    v
[serve] draft head training -> EAGLE-style proposals -> target verification ->
        longest-accepted-prefix decode -> batch scheduler -> OpenAI-compatible HTTP
    |
    v
[safety] guard (input) + shield (output) classifiers -> pass rate, FPR, bias probes
    |
    v
[emit] Axolotl YAML, TRL runner, llm-compressor/llama.cpp scripts, Speculators script,
       vLLM serve script, lm-eval script, Kubernetes Deployment/Service/HPA
    |
    v
[card] metrics.json + model-card.md (MOF v1 self-assessment) + data-card.md
```

Every stage writes JSON/markdown artifacts and a checksummed `manifest.json`. A full rerun removes
previous generated stage directories first, so stale beta checkpoints cannot leak into a report.

### File layout

```
src/ftpipe/
  data/         synth corpus, MinHash dedup, quality filter, PII, splits, contamination gate
  model/        NumPy transformer, manual gradients, tokenizer, packing, LoRA, draft head
  align/        DPO beta sweep, GRPO, shared Adam/sequence helpers
  stages/       pretrain, SFT, LoRA, eval, safety, pipeline orchestration
  quant/        GPTQ, AWQ, GGUF codec, calibration, quant stage
  serve/        engine, scheduler, HTTP server, benchmark/cost model
  evaluate/     task/MCQ/MT-Bench-style rubric/preference metrics
  safetymodels/ guard and shield classifiers
  card/         model card, data card, MOF checklist
  emit/         Axolotl, TRL, llm-compressor, Speculators, vLLM, Kubernetes emitters
fixtures/       deterministic synthetic data shaped like benchmark rows (not copied datasets)
dashboard/      TypeScript static report renderer + Bun tests
deploy/k8s/     deployment, service, HPA, Prometheus adapter templates
configs/        smoke and flagship YAML
```

## What runs offline

Everything except `uv sync` / `bun install`:

- Data generation, dedup, quality classification, PII scrubbing, split hygiene, contamination gate.
- Pretraining, SFT, LoRA, DPO, GRPO, quantization, speculative decoding, evaluation, safety checks.
- Local HTTP serving and benchmarking on loopback only.
- Model/data cards, manifest, deterministic metrics hash.
- TypeScript dashboard unit tests and the dashboard CLI.

No test calls an external network service. The server tests use `127.0.0.1` with an ephemeral port.

## Measured results

Measured locally on an Intel Core i5-1235U (12 threads, CPU only), Linux x86_64, Python 3.14.10,
NumPy 2.5.5. Total pipeline runtime: **188 s**. Timings and throughput are excluded from the
deterministic metrics hash. These are local synthetic-fixture measurements, not production
benchmarks and not 8B-model results.

### Data hygiene

| Metric | Measured |
|---|---:|
| Raw documents | 1,200 |
| Removed by exact + MinHash-LSH dedup | 171 (14.25%; 60 exact, 111 near) |
| Removed by quality rules/classifier | 175 (17.0%) |
| PII entities removed | 101 (coverage 1.0, 0 residual) |
| SFT train / validation records | 744 / 78 |
| Preference train / validation pairs | 600 / 60 |
| Contamination scan | 66 benchmark-shaped docs, 3,536 records scanned |
| Raw overlap detected | 2 |
| Final overlap / gate | **0 / green** |

The injected benchmark-shaped bait is detected by exact n-gram/MinHash; the final split scans are
green. The contamination rejection path is also covered by an explicit 1% injection test.

### Training

| Stage | Steps | Result |
|---|---:|---|
| Pretrain | 350 | validation loss 2.0815 |
| SFT | 279 | train loss 8.0932 → 0.0059; held-out loss 0.0156 |
| LoRA SFT | 200 | rank 8; 11.37% trainable; held-out loss 1.5736 |
| DPO | 80 per beta | β=0.20 selected; dev preference accuracy 1.0000 |
| GRPO | 60 | verifiable-reward validation exact 1.0000, format 1.0000 |

DPO beta sweep (same 60 held-out preference pairs):

| Beta | Dev preference accuracy | Loss | Selected |
|---:|---:|---:|:---:|
| 0.05 | 1.0000 | 0.6039 | no |
| 0.10 | 1.0000 | 0.5694 | no |
| 0.20 | 1.0000 | 0.4449 | **yes** |

### Evaluation matrix

All benchmark labels are `-style`: the fixtures are synthetic rows shaped like the named
benchmarks, never the real datasets. `MT-Bench-style` is a deterministic rubric out of 10, not an
LLM judge. `task_*` is the domain ticket-triage suite.

| Metric | base | SFT | LoRA | SFT+DPO | SFT+GRPO |
|---|---:|---:|---:|---:|---:|
| Task exact match | 0.0000 | 0.6000 | 0.1667 | **0.6167** | 0.6000 |
| Task category accuracy | 0.0000 | 0.6000 | 0.4000 | 0.6167 | **0.6333** |
| Task format valid | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MMLU-Pro-style | 0.0667 | 0.0667 | **0.1000** | 0.0667 | 0.0667 |
| MT-Bench-v2-style (0-10) | **5.3620** | 0.2454 | 0.2083 | 0.2269 | 0.2435 |
| RewardBench-2-style preference | 0.3333 | **1.0000** | 0.6667 | 0.9583 | 0.9167 |

Findings, stated plainly:

- Domain SFT moves exact match from 0 to 0.60 and format validity to 1.00.
- DPO adds +1.67 percentage points of exact match over SFT and wins the serving selection.
- GRPO is best on category accuracy but ties SFT on exact match.
- The large MT-Bench-style drop is **domain specialization/forgetting**: after SFT the model
  emits terse labels instead of open-ended prose. The rubric rewards reference overlap and
  length, so this is expected and is a limitation of the stand-in evaluation, not evidence that
  prose quality improved.
- LoRA at 11.37% trainable parameters does not close the quality gap at this budget
  (0.1667 exact vs 0.6000 full SFT); this is a measured negative result, not a tuned claim.

### Quantization

| Artifact | Size | Bits/weight | Task exact (20-item subset) | Delta vs unquantized |
|---|---:|---:|---:|---:|
| GPTQ-INT4 | 44,544 B | 5.44 | 0.5500 | +0.0000 |
| AWQ-INT4 | 44,544 B | 5.44 | 0.5500 | +0.0000 |
| GGUF Q4_K_M | 86,848 B | 6.43 | 0.5500 | +0.0000 |

Bits/weight include scales/permutation metadata and unquantized embeddings/biases, so they are
above 4. GPTQ uses a Hessian inverse and optional activation ordering; AWQ selects a per-channel
scaling exponent from the configured grid by output MSE. GGUF is a real container/codec written
and read back by this repository, but it is a toy-architecture export, not a llama.cpp-loadable
Llama file.

### Serving and modeled cost

Speculative greedy output is verified token-for-token against target-only greedy output for the
comparison prompts; `spec_output_equals_target` is **true**. Acceptance is 0.40 on domain prompts
and 0.5556 on generic corpus prompts.

| Batch | Mode | Output tok/s | p50 ms | p99 ms |
|---:|---|---:|---:|---:|
| 1 | target | 176.34 | 12.40 | 40.90 |
| 1 | speculative | 199.07 | 14.80 | 17.59 |
| 8 | target | 533.97 | 39.94 | 42.71 |
| 8 | speculative | 210.77 | 105.82 | 110.24 |
| 32 | target | 33.36 | 45.71 | 1073.66 |
| 32 | speculative | 220.97 | 144.88 | 156.80 |

At this toy scale, speculative decoding helps batch 1 slightly, loses at batch 8 due to
verification overhead, and is more stable at batch 32 where the batched target path is worst.
The measured acceptance rate is far below the 0.74 illustrative figure in the spec. No 2-3x
speedup is claimed.

Modeled self-hosted cost at the configured **$3.99/GPU-hour** and the best measured speculative
throughput (220.97 output tok/s):

**$5.0158 per 1M output tokens** — a model derived from local CPU throughput, **not a GPU
measurement**. Vendor list prices checked at the cited pricing pages: OpenAI `gpt-5.4-mini`
$4.50/1M output, Anthropic `claude-haiku-5.5` $0.50/1M output. These compare prices, not matched
quality; no provider was called.

### Safety

| Check | Measured |
|---|---:|
| Guard pass rate, base | 0.9833 |
| Guard pass rate, SFT / DPO / GRPO | 1.0000 |
| Shield unsafe block rate | 1.0000 |
| Shield benign false-positive rate | 0.0083 |
| Bias-probe guard false-positive rate | 0.0000 |
| Bias-probe refusal rate, base / aligned | 0.0833 / 0.0000 |

The guard/shield are local hashed n-gram logistic classifiers trained on synthetic safety data,
not Llama Guard 4 or ShieldGemma 2. They exist to exercise the input/output filtering path and its
metrics; they are not a production safety evaluation.

### Reproducibility

- Stage-specific seeds are derived from `run.seed` (7) with fixed offsets; all optimization uses
  NumPy generators.
- `outputs/metrics.json` records the config SHA-256, git SHA, environment, checkpoint parameter
  hashes, and a deterministic metrics hash: `1f849198e8cd936278fd034a5ec2e9199fd61ee7f71bc4251d7edeca6480ac1e`.
- `outputs/manifest.json` checksums 212 artifacts.
- Rerunning the same config and seed reproduces the deterministic hash; the integration test
  asserts this.
- Output directories are reset at the start of a full run; partial-stage runs (`--stages`) reuse
  existing artifacts and validate their presence.

## Configuration

`configs/smoke.yaml` runs in seconds and is the test/quickstart config.
`configs/domainx.yaml` is the measured 1200-document run. All keys are validated strictly: unknown
keys, wrong types, out-of-range values, and inconsistent MinHash band/row settings fail before any
stage runs. Delete `run.output_dir` to change the artifact location; `--out` does the same without
editing YAML.

Key knobs: `data.minhash`, `data.quality`, `data.pii.entities`, `sft.sample_packing`,
`sft.lora.*`, `dpo.beta_grid`, `grpo.reward`, `quant.*`, `serve.batch_sizes`, `serve.draft_tokens`,
`serve.draft_train_steps`, `serve.comparison_models`.

## Optional real GPU / provider mode

The pipeline emits, under `outputs/gpu/`:

- `sft.axolotl.yaml` / `sft.axolotl.lora.yaml` — Llama 3.1 8B, ZeRO-3, Flash Attention 3,
  sample packing, cosine schedule, W&B project/run IDs.
- `dpo.trl.yaml`, `grpo.trl.yaml`, `train_preference.py` — TRL DPO/GRPO runner with a verifiable
  reward function.
- `quantize.py`, `quantize.sh`, `calibration.jsonl` — llm-compressor GPTQ/AWQ recipes and
  llama.cpp GGUF conversion commands.
- `speculator-train.sh` — Speculators data preparation, hidden-state generation, and EAGLE-3
  training commands from the current Speculators tutorial.
- `vllm-serve.sh`, `safety-serve.sh`, `lm-eval.sh` — serving, real safety-model, and
  lm-evaluation-harness commands.
- `k8s/deployment.yaml`, `service.yaml`, `hpa.yaml`, `prometheus-adapter.yaml` — vLLM deployment,
  HPA on `vllm:num_requests_waiting`, and the external-metric adapter.

Typical handoff after a local run:

```bash
axolotl train outputs/gpu/sft.axolotl.yaml
python outputs/gpu/train_preference.py --config outputs/gpu/dpo.trl.yaml --mode dpo
bash outputs/gpu/quantize.sh
bash outputs/gpu/speculator-train.sh
bash outputs/gpu/vllm-serve.sh
bash outputs/gpu/lm-eval.sh
kubectl apply -f outputs/gpu/k8s/
```

These commands require gated base weights, an `HF_TOKEN`, multiple H100-class GPUs, and pinned
upstream releases. **None of them were executed in this repository's verification.** The spec's
"vLLM 0.7 + EAGLE-3" pairing predates current EAGLE-3/Speculators support; the emitted scripts
therefore target current Speculators and must be pinned to a validated vLLM release. The emitted
Deployment uses `vllm/vllm-openai:latest` unless you change it; pin an image digest before
deploying. The local adapter boundary is exactly this directory: fixtures + NumPy core for
deterministic validation, emitted real-stack assets for a GPU environment.

## Design decisions

- **Real mechanisms, small scale.** The transformer has manual forward/backward passes validated
  by finite differences; DPO and GRPO update real parameters against a frozen reference; GPTQ and
  AWQ perform actual quantization math; speculative decoding verifies with the target model.
- **Deterministic fixtures instead of paid services.** Synthetic tickets have rule-derived labels,
  so SFT/DPO/eval data can be regenerated exactly. No HF dataset or paid API is required.
- **Strict config validation.** A typo or inconsistent MinHash shape fails fast rather than
  silently training with a different pipeline.
- **Outputs are disposable, source is authoritative.** Generated data, checkpoints, quantized
  files, cards, and served traces live under ignored `outputs/`; the YAML plus seed is the
  reproducibility contract.
- **Honest evaluation naming.** Benchmark-shaped fixtures are always suffixed `-style`, the MT-Bench
  judge is a rubric, and quantization/cost/safety limitations are stated in the generated card.
- **MOF v1, not an invented "MOF 2026".** The generated card self-assesses against the published
  Model Openness Framework (arXiv:2403.13784) and reports **Class II — Open Tooling**; no separate
  2026 MOF version is claimed.

## Limitations

- The local model is 108,032 parameters, not Llama 3.1 8B. No 8B weights were downloaded or trained.
- All eval fixtures are synthetic and small; scores are not real MMLU-Pro, MT-Bench-v2,
  RewardBench-2, or task-leaderboard numbers.
- The MT-Bench-style judge is a deterministic rubric, not an LLM judge; its prose scores are
  depressed by domain-specialized terse outputs.
- Safety models are local classifiers, not Llama Guard 4 or ShieldGemma 2; bias probes are 12
  synthetic sentences, not a bias benchmark.
- GPTQ/AWQ dequantized weights are evaluated in NumPy; Marlin/AWQ CUDA kernels were not run.
- GGUF is a repository-specific container/codec for the toy architecture; llama.cpp cannot load
  it as a Llama model.
- Speculative decoding uses a small trained linear draft head, not EAGLE-3 target hidden-state
  heads; acceptance and throughput are local CPU measurements.
- Cost is modeled from local throughput at a configured GPU hourly rate; no GPU or vendor API was
  used, so the "$/1M tokens" figure is not a cloud bill.
- Real Axolotl/TRL/vLLM/Kubernetes assets are emitted but unexecuted; versions must be pinned.
- LoRA memory savings were not measured directly; trainable parameter fraction is the proxy.

## References

- Capstone specification — <https://aiengineeringfromscratch.com/learn/capstone-projects/end-to-end-fine-tuning-pipeline>
- Model Openness Framework, White et al., arXiv:2403.13784 — <https://arxiv.org/abs/2403.13784>
- Axolotl configuration reference and TRL `DPOTrainer`/`DPOConfig` documentation —
  <https://docs.axolotl.ai/docs/config-reference.html>,
  <https://huggingface.co/docs/trl/en/dpo_trainer>
- vLLM Speculators EAGLE-3 algorithm and training tutorial —
  <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/>,
  <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/>
- Vendor list prices used in the comparison table — <https://developers.openai.com/api/docs/pricing>,
  <https://platform.claude.com/docs/en/about-claude/pricing>
