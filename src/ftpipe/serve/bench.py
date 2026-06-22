"""HTTP benchmark and cost model for the local serving stack."""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..align.common import read_jsonl
from .engine import Engine
from .server import create_server


def _post(port: int, prompt: str, max_tokens: int) -> tuple[float, dict[str, Any]]:
    payload = json.dumps(
        {"model": "ftpipe", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    return (time.perf_counter() - started) * 1000, body


def _benchmark_batch(
    cfg: Any,
    checkpoint: Path,
    draft_path: Path | None,
    batch_size: int,
    speculative: bool,
    prompts: list[str],
) -> dict[str, Any]:
    engine = Engine(
        checkpoint,
        draft_path=draft_path,
        speculative=speculative,
        guard=None,
        shield=None,
        draft_tokens=int(cfg["serve"]["draft_tokens"]),
    )
    server = create_server(
        engine,
        port=0,
        model_name=str(cfg["project"]["name"]),
        trace_path=Path(cfg.out_dir) / "serving" / f"trace-{batch_size}-{int(speculative)}.jsonl",
        max_batch=batch_size,
        window_ms=int(cfg["serve"]["batch_window_ms"]),
    )
    port = int(server.server_address[1])
    requests_per_batch = int(cfg["serve"]["requests_per_batch"])
    selected = [prompts[index % len(prompts)] for index in range(requests_per_batch)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        results = list(pool.map(lambda prompt: _post(port, prompt, int(cfg["serve"]["max_new_tokens"])), selected))
    elapsed = time.perf_counter() - started
    latencies = [item[0] for item in results]
    completion_tokens = sum(item[1]["usage"]["completion_tokens"] for item in results)
    with engine.stats.lock:
        acceptance = engine.stats.accepted / engine.stats.proposed if engine.stats.proposed else 0.0
    server.shutdown()
    server.scheduler.shutdown()  # type: ignore[attr-defined]
    return {
        "batch_size": batch_size,
        "speculative": speculative,
        "requests": len(selected),
        "output_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": completion_tokens / elapsed if elapsed else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p99_latency_ms": sorted(latencies)[min(len(latencies) - 1, math.ceil(0.99 * len(latencies)) - 1)] if latencies else 0.0,
        "acceptance_rate": acceptance,
    }


def acceptance_on_prompts(engine: Engine, prompts: list[str]) -> float:
    before = engine.stats.accepted, engine.stats.proposed
    for prompt in prompts:
        engine.generate(prompt, max_new_tokens=8, temperature=0.0)
    with engine.stats.lock:
        accepted = engine.stats.accepted - before[0]
        proposed = engine.stats.proposed - before[1]
    return accepted / proposed if proposed else 0.0


def run_bench(cfg: Any, checkpoint: Path, draft_path: Path | None) -> dict[str, Any]:
    task_rows = read_jsonl(Path(cfg.root) / "fixtures" / "eval" / "task_test.jsonl")
    generic_rows = read_jsonl(Path(cfg.out_dir) / "data" / "base_corpus.jsonl")
    domain_prompts = [row["instruction"] for row in task_rows[:12]]
    generic_prompts = [row["text"] for row in generic_rows[:12]]
    batches = []
    for batch_size in cfg["serve"]["batch_sizes"]:
        for speculative in (False, True):
            if speculative and draft_path is None:
                continue
            batches.append(
                _benchmark_batch(
                    cfg,
                    checkpoint,
                    draft_path,
                    int(batch_size),
                    speculative,
                    domain_prompts,
                )
            )
    engine = Engine(
        checkpoint,
        draft_path=draft_path,
        speculative=True,
        draft_tokens=int(cfg["serve"]["draft_tokens"]),
    )
    target_engine = Engine(checkpoint, draft_path=None, speculative=False)
    comparison_prompts = domain_prompts[:3]
    equal = True
    for prompt in comparison_prompts:
        spec_output = engine.generate(prompt, max_new_tokens=8, temperature=0.0)
        target_output = target_engine.generate(prompt, max_new_tokens=8, temperature=0.0)
        equal = equal and spec_output.tokens == target_output.tokens
    if not equal:
        raise RuntimeError("speculative greedy generation diverged from target-only greedy generation")
    domain_acceptance = acceptance_on_prompts(engine, domain_prompts)
    generic_acceptance = acceptance_on_prompts(engine, generic_prompts)
    spec_batches = [item for item in batches if item["speculative"]]
    best = max(spec_batches, key=lambda item: item["output_tokens_per_second"]) if spec_batches else None
    throughput = best["output_tokens_per_second"] if best else 0.0
    hourly = float(cfg["serve"]["gpu_hourly_usd"])
    cost_per_million = (hourly / 3600.0) / throughput * 1_000_000 if throughput else 0.0
    comparisons = [
        {
            "provider": item["provider"],
            "model": item["model"],
            "usd_per_1m_output_tokens": item["output_usd_per_mtok"],
            "source": item["source"],
        }
        for item in cfg["serve"]["comparison_models"]
    ]
    cost_model = {
        "gpu_hourly_usd": hourly,
        "self_hosted_usd_per_1m_output_tokens": cost_per_million,
        "basis": "measured local output throughput at the configured GPU hourly rate; not a GPU measurement",
        "comparisons": comparisons,
    }
    balanced = {
        "output_tokens_per_second": {},
        "spec_output_tokens_per_second": {},
        "p50_latency_ms": {},
        "p99_latency_ms": {},
        "spec_p50_latency_ms": {},
        "spec_p99_latency_ms": {},
        "acceptance_rate": {},
    }
    for batch_size in cfg["serve"]["batch_sizes"]:
        base = next((item for item in batches if item["batch_size"] == batch_size and not item["speculative"]), None)
        spec = next((item for item in batches if item["batch_size"] == batch_size and item["speculative"]), None)
        if base:
            balanced["output_tokens_per_second"][str(batch_size)] = base["output_tokens_per_second"]
            balanced["p50_latency_ms"][str(batch_size)] = base["p50_latency_ms"]
            balanced["p99_latency_ms"][str(batch_size)] = base["p99_latency_ms"]
        if spec:
            balanced["spec_output_tokens_per_second"][str(batch_size)] = spec["output_tokens_per_second"]
            balanced["spec_p50_latency_ms"][str(batch_size)] = spec["p50_latency_ms"]
            balanced["spec_p99_latency_ms"][str(batch_size)] = spec["p99_latency_ms"]
            balanced["acceptance_rate"][str(batch_size)] = spec["acceptance_rate"]
    result = {
        "speculative": {
            "acceptance_rate": domain_acceptance,
            "domain_acceptance_rate": domain_acceptance,
            "generic_acceptance_rate": generic_acceptance,
            "spec_output_equals_target": equal,
        },
        "batches": batches,
        "tables": balanced,
        "cost_model": cost_model,
    }
    out = Path(cfg.out_dir) / "serving"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def bench_command(cfg: Any, checkpoint: Path) -> None:
    draft_path = Path(cfg.out_dir) / "models" / "draft"
    result = run_bench(cfg, checkpoint, draft_path if draft_path.exists() else None)
    print(json.dumps(result["tables"], indent=2, sort_keys=True))
