/** Static HTML report for a completed ftpipe run. No network, no framework. */

export interface RunSummary {
  project: string;
  local_model: string;
  base_model: string;
  seed: number;
  config_sha256: string;
  git_sha: string;
  deterministic_sha256: string;
  serving_checkpoint: string;
}

export interface DataSummary {
  raw_docs: number;
  dedup_removed: number;
  quality_rejected: number;
  pii_entities_removed: number;
  sft_train_records: number;
  preference_pairs: number;
  contamination: { green: boolean; raw_overlap_docs: number; overlap_docs: number };
}

export interface TrainingSummary {
  steps: Record<string, number>;
  base_val_loss: number;
  sft_val_loss: number;
  sft_train_loss_initial: number;
  sft_train_loss_final: number;
  lora_trainable_fraction: number;
  dpo_beta_selected: number;
}

export interface EvalSummary {
  models: string[];
  metrics: Record<string, Record<string, number>>;
  deltas_vs_base: Record<string, Record<string, number>>;
}

export interface QuantReport {
  size_bytes: number;
  bits_per_weight: number;
  task_exact_match: number;
  delta_vs_unquantized: number;
}

export interface ServingBatch {
  batch_size: number;
  speculative: boolean;
  output_tokens_per_second: number;
  p50_latency_ms: number;
  p99_latency_ms: number;
}

export interface ServingSummary {
  speculative: { acceptance_rate: number; generic_acceptance_rate: number; spec_output_equals_target: boolean };
  batches: ServingBatch[];
  cost_model: {
    gpu_hourly_usd: number;
    self_hosted_usd_per_1m_output_tokens: number;
    comparisons: { provider: string; model: string; usd_per_1m_output_tokens: number; source: string }[];
  };
}

export interface SafetySummary {
  guard_pass_rate: Record<string, number>;
  shield_block_rate_unsafe: number;
  shield_false_positive_rate: number;
  bias_probe_guard_fp_rate: number;
}

export interface Metrics {
  run: RunSummary;
  data: DataSummary;
  training: TrainingSummary;
  eval: EvalSummary;
  quant: Record<string, QuantReport>;
  serving: ServingSummary;
  safety: SafetySummary;
}

type Json = Record<string, unknown>;

function record(value: unknown, label: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label}: expected object`);
  }
  return value as Json;
}

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numberMap(value: unknown): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(record(value ?? {}, "numberMap"))) {
    result[key] = num(item);
  }
  return result;
}

function nestedNumbers(value: unknown): Record<string, Record<string, number>> {
  const result: Record<string, Record<string, number>> = {};
  for (const [key, item] of Object.entries(record(value ?? {}, "nestedNumbers"))) {
    result[key] = numberMap(item);
  }
  return result;
}

export function parseMetrics(raw: unknown): Metrics {
  const root = record(raw, "metrics");
  const run = record(root.run, "run");
  const data = record(root.data, "data");
  const training = record(root.training ?? {}, "training");
  const evaluation = record(root.eval, "eval");
  const contamination = record(data.contamination ?? {}, "data.contamination");
  const quant: Record<string, QuantReport> = {};
  for (const [name, item] of Object.entries(record(root.quant ?? {}, "quant"))) {
    const report = record(item, `quant.${name}`);
    quant[name] = {
      size_bytes: num(report.size_bytes),
      bits_per_weight: num(report.bits_per_weight),
      task_exact_match: num(report.task_exact_match),
      delta_vs_unquantized: num(report.delta_vs_unquantized),
    };
  }
  const servingRaw = record(root.serving ?? {}, "serving");
  const speculative = record(servingRaw.speculative ?? {}, "serving.speculative");
  const cost = record(servingRaw.cost_model ?? {}, "serving.cost_model");
  const batches = Array.isArray(servingRaw.batches) ? servingRaw.batches : [];
  const safetyRaw = record(root.safety ?? {}, "safety");
  const models = Array.isArray(evaluation.models) ? evaluation.models.map((item) => text(item)) : [];
  return {
    run: {
      project: text(run.project, "ftpipe"),
      local_model: text(run.local_model),
      base_model: text(run.base_model),
      seed: num(run.seed),
      config_sha256: text(run.config_sha256),
      git_sha: text(run.git_sha),
      deterministic_sha256: text(run.deterministic_sha256),
      serving_checkpoint: text(run.serving_checkpoint),
    },
    data: {
      raw_docs: num(data.raw_docs),
      dedup_removed: num(data.dedup_removed),
      quality_rejected: num(data.quality_rejected),
      pii_entities_removed: num(data.pii_entities_removed),
      sft_train_records: num(data.sft_train_records),
      preference_pairs: num(data.preference_pairs),
      contamination: {
        green: contamination.green === true,
        raw_overlap_docs: num(contamination.raw_overlap_docs),
        overlap_docs: num(contamination.overlap_docs),
      },
    },
    training: {
      steps: numberMap(training.steps),
      base_val_loss: num(training.base_val_loss),
      sft_val_loss: num(training.sft_val_loss),
      sft_train_loss_initial: num(training.sft_train_loss_initial),
      sft_train_loss_final: num(training.sft_train_loss_final),
      lora_trainable_fraction: num(training.lora_trainable_fraction),
      dpo_beta_selected: num(training.dpo_beta_selected),
    },
    eval: {
      models,
      metrics: nestedNumbers(evaluation.metrics),
      deltas_vs_base: nestedNumbers(evaluation.deltas_vs_base),
    },
    quant,
    serving: {
      speculative: {
        acceptance_rate: num(speculative.acceptance_rate),
        generic_acceptance_rate: num(speculative.generic_acceptance_rate),
        spec_output_equals_target: speculative.spec_output_equals_target === true,
      },
      batches: batches.map((item, index) => {
        const batch = record(item, `serving.batches[${index}]`);
        return {
          batch_size: num(batch.batch_size),
          speculative: batch.speculative === true,
          output_tokens_per_second: num(batch.output_tokens_per_second),
          p50_latency_ms: num(batch.p50_latency_ms),
          p99_latency_ms: num(batch.p99_latency_ms),
        };
      }),
      cost_model: {
        gpu_hourly_usd: num(cost.gpu_hourly_usd),
        self_hosted_usd_per_1m_output_tokens: num(cost.self_hosted_usd_per_1m_output_tokens),
        comparisons: Array.isArray(cost.comparisons)
          ? cost.comparisons.map((item, index) => {
              const comparison = record(item, `serving.cost_model.comparisons[${index}]`);
              return {
                provider: text(comparison.provider),
                model: text(comparison.model),
                usd_per_1m_output_tokens: num(comparison.usd_per_1m_output_tokens),
                source: text(comparison.source),
              };
            })
          : [],
      },
    },
    safety: {
      guard_pass_rate: numberMap(safetyRaw.guard_pass_rate),
      shield_block_rate_unsafe: num(safetyRaw.shield_block_rate_unsafe),
      shield_false_positive_rate: num(safetyRaw.shield_false_positive_rate),
      bias_probe_guard_fp_rate: num(safetyRaw.bias_probe_guard_fp_rate),
    },
  };
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fixed(value: number, digits = 4): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function metricTable(metrics: Metrics): string {
  const { models } = metrics.eval;
  const header = ["Metric", ...models].map((cell) => `<th>${escapeHtml(cell)}</th>`).join("");
  const rows = Object.entries(metrics.eval.metrics)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, values]) => {
      const cells = models.map((model) => `<td>${fixed(values[model] ?? Number.NaN)}</td>`).join("");
      return `<tr><th>${escapeHtml(name)}</th>${cells}</tr>`;
    })
    .join("\n");
  return `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
}

function deltaTable(metrics: Metrics): string {
  const names = ["task_exact_match", "mmlu_pro_style", "mt_bench_v2_style", "rewardbench_2_style"];
  const models = metrics.eval.models.filter((model) => model !== "base");
  const rows = names
    .map((name) => {
      const cells = models
        .map((model) => {
          const delta = metrics.eval.deltas_vs_base[model]?.[name];
          return `<td>${delta === undefined ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`}</td>`;
        })
        .join("");
      return `<tr><th>${escapeHtml(name)}</th>${cells}</tr>`;
    })
    .join("\n");
  const header = ["Metric", ...models].map((cell) => `<th>${escapeHtml(cell)}</th>`).join("");
  return `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
}

function quantTable(metrics: Metrics): string {
  return Object.entries(metrics.quant)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(
      ([name, report]) =>
        `<tr><th>${escapeHtml(name)}</th><td>${(report.size_bytes / 1024).toFixed(1)} KiB</td>` +
        `<td>${fixed(report.bits_per_weight, 2)}</td><td>${fixed(report.task_exact_match)}</td>` +
        `<td>${report.delta_vs_unquantized >= 0 ? "+" : ""}${report.delta_vs_unquantized.toFixed(4)}</td></tr>`,
    )
    .join("\n");
}

function servingTable(metrics: Metrics): string {
  return metrics.serving.batches
    .map(
      (batch) =>
        `<tr><td>${batch.batch_size}</td><td>${batch.speculative ? "speculative" : "target"}</td>` +
        `<td>${batch.output_tokens_per_second.toFixed(1)}</td><td>${batch.p50_latency_ms.toFixed(2)}</td>` +
        `<td>${batch.p99_latency_ms.toFixed(2)}</td></tr>`,
    )
    .join("\n");
}

function safetyTable(metrics: Metrics): string {
  return Object.entries(metrics.safety.guard_pass_rate)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([model, rate]) => `<tr><th>${escapeHtml(model)}</th><td>${percent(rate)}</td></tr>`)
    .join("\n");
}

export function renderDashboard(metrics: Metrics): string {
  const data = metrics.data;
  const contamination = data.contamination.green
    ? `<span class="ok">green</span> (${data.contamination.raw_overlap_docs} raw overlaps rejected)`
    : `<span class="bad">failed</span> (${data.contamination.overlap_docs} remaining)`;
  const batches = servingTable(metrics) || `<tr><td colspan="5">no serving measurements</td></tr>`;
  const quant = quantTable(metrics) || `<tr><td colspan="5">no quantized artifacts</td></tr>`;
  const guards = safetyTable(metrics) || `<tr><td colspan="2">no safety measurements</td></tr>`;
  const cost = metrics.serving.cost_model;
  const comparisons = cost.comparisons
    .map(
      (item) =>
        `<tr><th>${escapeHtml(item.provider)}</th><td>${escapeHtml(item.model)}</td>` +
        `<td>$${item.usd_per_1m_output_tokens.toFixed(2)}</td>` +
        `<td><a href="${escapeHtml(item.source)}">pricing</a></td></tr>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(metrics.run.project)} — fine-tuning report</title>
<style>
:root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }
body { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; }
th, td { border-bottom: 1px solid #8884; padding: 0.35rem 0.5rem; text-align: left; }
td { text-align: right; font-variant-numeric: tabular-nums; }
tbody th { font-weight: 600; }
code { overflow-wrap: anywhere; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
.card { border: 1px solid #8884; border-radius: 8px; padding: 0.75rem 1rem; }
.card b { display: block; font-size: 1.5rem; }
.ok { color: #1a7f37; font-weight: 700; }
.bad { color: #b42318; font-weight: 700; }
</style>
</head>
<body>
<h1>${escapeHtml(metrics.run.project)} — fine-tuning report</h1>
<p>Local model <code>${escapeHtml(metrics.run.local_model)}</code>, real-stack target
<code>${escapeHtml(metrics.run.base_model)}</code>, seed ${metrics.run.seed},
serving checkpoint <code>${escapeHtml(metrics.run.serving_checkpoint)}</code>.</p>

<h2>Data hygiene</h2>
<div class="grid">
  <div class="card">raw documents<b>${data.raw_docs}</b></div>
  <div class="card">dedup removed<b>${data.dedup_removed}</b></div>
  <div class="card">quality rejected<b>${data.quality_rejected}</b></div>
  <div class="card">PII entities removed<b>${data.pii_entities_removed}</b></div>
  <div class="card">SFT records<b>${data.sft_train_records}</b></div>
  <div class="card">preference pairs<b>${data.preference_pairs}</b></div>
</div>
<p>Contamination gate: ${contamination}</p>

<h2>Training</h2>
<p>Base val loss ${fixed(metrics.training.base_val_loss)}, SFT val loss
${fixed(metrics.training.sft_val_loss)}; SFT train loss
${fixed(metrics.training.sft_train_loss_initial)} → ${fixed(metrics.training.sft_train_loss_final)};
selected DPO β ${metrics.training.dpo_beta_selected}; LoRA trainable
${percent(metrics.training.lora_trainable_fraction)}.</p>
<p>${Object.entries(metrics.training.steps)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([stage, steps]) => `${escapeHtml(stage)} ${steps}`)
    .join(" · ")}</p>

<h2>Evaluation</h2>
${metricTable(metrics)}
<h3>Delta vs base</h3>
${deltaTable(metrics)}

<h2>Quantization</h2>
<table><thead><tr><th>Artifact</th><th>Size</th><th>Bits/weight</th><th>Task exact</th><th>Δ</th></tr></thead>
<tbody>${quant}</tbody></table>

<h2>Serving</h2>
<p>Speculative acceptance: domain ${percent(metrics.serving.speculative.acceptance_rate)},
generic ${percent(metrics.serving.speculative.generic_acceptance_rate)};
output parity ${metrics.serving.speculative.spec_output_equals_target ? "verified" : "failed"}.</p>
<table><thead><tr><th>Batch</th><th>Mode</th><th>Output tok/s</th><th>p50 ms</th><th>p99 ms</th></tr></thead>
<tbody>${batches}</tbody></table>
<p>Modeled self-hosted cost at $${cost.gpu_hourly_usd.toFixed(2)}/GPU-hour:
<strong>$${cost.self_hosted_usd_per_1m_output_tokens.toFixed(4)} per 1M output tokens</strong>.</p>
<table><thead><tr><th>Provider</th><th>Model</th><th>USD / 1M output</th><th>Source</th></tr></thead>
<tbody>${comparisons || `<tr><td colspan="4">no comparison prices configured</td></tr>`}</tbody></table>

<h2>Safety</h2>
<table><thead><tr><th>Model</th><th>Guard pass rate</th></tr></thead><tbody>${guards}</tbody></table>
<p>Shield unsafe block rate ${percent(metrics.safety.shield_block_rate_unsafe)};
benign false-positive rate ${percent(metrics.safety.shield_false_positive_rate)};
bias-probe guard false-positive rate ${percent(metrics.safety.bias_probe_guard_fp_rate)}.</p>

<footer>
<p>Built from <code>outputs/metrics.json</code>. Deterministic metrics hash
<code>${escapeHtml(metrics.run.deterministic_sha256)}</code>; config
<code>${escapeHtml(metrics.run.config_sha256)}</code>; commit
<code>${escapeHtml(metrics.run.git_sha)}</code>.</p>
<p>Benchmark-shaped fixtures are synthetic; local model results are not production benchmarks.</p>
</footer>
</body>
</html>
`;
}
