import { describe, expect, test } from "bun:test";

import { escapeHtml, parseMetrics, renderDashboard } from "../src/report";

function fixture(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    run: {
      project: "domainx",
      local_model: "tiny-2l-64d",
      base_model: "meta-llama/Llama-3.1-8B-Instruct",
      seed: 7,
      config_sha256: "configsha",
      git_sha: "deadbeef",
      deterministic_sha256: "detsha",
      serving_checkpoint: "dpo",
    },
    data: {
      raw_docs: 1200,
      dedup_removed: 120,
      quality_rejected: 35,
      pii_entities_removed: 100,
      sft_train_records: 720,
      preference_pairs: 540,
      contamination: { green: true, raw_overlap_docs: 3, overlap_docs: 0 },
    },
    training: {
      steps: { pretrain: 350, sft: 300, lora: 200, dpo: 80, grpo: 60 },
      base_val_loss: 4.2,
      sft_val_loss: 1.4,
      sft_train_loss_initial: 3.1,
      sft_train_loss_final: 1.2,
      lora_trainable_fraction: 0.021,
      dpo_beta_selected: 0.05,
    },
    eval: {
      models: ["base", "sft", "dpo"],
      metrics: {
        task_exact_match: { base: 0.1, sft: 0.6, dpo: 0.7 },
        mmlu_pro_style: { base: 0.2, sft: 0.25, dpo: 0.27 },
      },
      deltas_vs_base: {
        sft: { task_exact_match: 0.5, mmlu_pro_style: 0.05 },
        dpo: { task_exact_match: 0.6, mmlu_pro_style: 0.07 },
      },
    },
    quant: {
      gptq_int4: { size_bytes: 1048576, bits_per_weight: 4.4, task_exact_match: 0.69, delta_vs_unquantized: -0.01 },
    },
    serving: {
      speculative: { acceptance_rate: 0.74, generic_acceptance_rate: 0.41, spec_output_equals_target: true },
      batches: [
        { batch_size: 8, speculative: true, output_tokens_per_second: 1200, p50_latency_ms: 30, p99_latency_ms: 90 },
      ],
      cost_model: {
        gpu_hourly_usd: 3.99,
        self_hosted_usd_per_1m_output_tokens: 0.92,
        comparisons: [
          {
            provider: "OpenAI",
            model: "gpt-5.4-mini",
            usd_per_1m_output_tokens: 4.5,
            source: "https://developers.openai.com/api/docs/pricing",
          },
        ],
      },
    },
    safety: {
      guard_pass_rate: { base: 0.9, dpo: 0.93 },
      shield_block_rate_unsafe: 1.0,
      shield_false_positive_rate: 0.02,
      bias_probe_guard_fp_rate: 0.0,
    },
    ...overrides,
  };
}

describe("dashboard report", () => {
  test("escapes hostile strings from metrics", () => {
    const raw = fixture();
    (raw.run as Record<string, unknown>).project = `<script>alert("x")</script>`;
    const html = renderDashboard(parseMetrics(raw));
    expect(html).not.toContain("<script>alert");
    expect(html).toContain("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;");
    expect(escapeHtml("a&b'c")).toBe("a&amp;b&#39;c");
  });

  test("renders every evaluated model and metric", () => {
    const html = renderDashboard(parseMetrics(fixture()));
    for (const cell of ["base", "sft", "dpo", "task_exact_match", "0.7000"]) {
      expect(html).toContain(cell);
    }
    expect(html).toContain("+0.6000");
  });

  test("reports data hygiene and contamination status", () => {
    const html = renderDashboard(parseMetrics(fixture()));
    expect(html).toContain("dedup removed");
    expect(html).toContain("<b>120</b>");
    expect(html).toContain("green");
    expect(html).toContain("3 raw overlaps rejected");
  });

  test("renders quantization and serving economics", () => {
    const html = renderDashboard(parseMetrics(fixture()));
    expect(html).toContain("gptq_int4");
    expect(html).toContain("1024.0 KiB");
    expect(html).toContain("Speculative acceptance: domain 74.0%");
    expect(html).toContain("$0.9200 per 1M output tokens");
    expect(html).toContain("output parity verified");
  });

  test("surfaces safety rates and empty artifact collections", () => {
    const raw = fixture({ quant: {}, serving: { ...(fixture().serving as object), batches: [] } });
    const html = renderDashboard(parseMetrics(raw));
    expect(html).toContain("no quantized artifacts");
    expect(html).toContain("benign false-positive rate 2.0%");
    expect(html).toContain("bias-probe guard false-positive rate 0.0%");
  });

  test("rejects malformed metrics instead of rendering misleading output", () => {
    expect(() => parseMetrics(null)).toThrow("metrics: expected object");
    expect(() => parseMetrics({ run: {}, data: {} })).toThrow("eval: expected object");
    expect(() => parseMetrics({ run: {}, data: {}, eval: { models: [] } })).not.toThrow();
  });
});
