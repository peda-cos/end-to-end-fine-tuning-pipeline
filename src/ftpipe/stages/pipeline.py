"""One-command pipeline: data, training, alignment, quantization, serving, eval, card."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..align.common import read_jsonl
from ..align.dpo import run_dpo
from ..align.grpo import run_grpo
from ..card.data_card import render_data_card
from ..card.model_card import render_model_card
from ..config import ConfigError
from ..data.stage import build_data
from ..emit.axolotl import emit_axolotl
from ..emit.k8s import emit_k8s
from ..emit.quantize import emit_quantize
from ..emit.trl import emit_trl
from ..emit.vllm import emit_vllm
from ..manifest import Manifest, deterministic_sha256, environment, git_sha, sha256_json
from ..serve.stage import run_serve as run_serving_stage
from .evaluate import MODEL_ORDER, run_eval
from .pretrain import run_pretrain
from .safety import run_safety
from .sft import run_lora, run_sft

STAGES = (
    "data",
    "pretrain",
    "sft",
    "lora",
    "dpo",
    "grpo",
    "eval",
    "quant",
    "serve",
    "safety",
    "emit",
    "card",
    "manifest",
)


def _model_path(cfg: Any, name: str) -> Path:
    return Path(cfg.out_dir) / "models" / name


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_metadata(cfg: Any, name: str) -> dict[str, Any]:
    path = _model_path(cfg, name) / "metadata.json"
    if not path.exists():
        return {}
    return _read_json(path)


def _data_files(cfg: Any) -> dict[str, Path]:
    out = Path(cfg.out_dir) / "data"
    files = {
        key: out / f"{key}.jsonl"
        for key in (
            "raw",
            "clean",
            "base_corpus",
            "sft_train",
            "sft_val",
            "preferences_train",
            "preferences_val",
            "mcq_train",
        )
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"data stage artifacts missing; run the data stage first: {missing[:3]}")
    return files


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{stage} requires {path}; run that stage first")
    return path


def _stage_data(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    artifacts = build_data(cfg)
    stats = artifacts.stats
    print(
        f"[data] {stats['raw_docs']} raw, {stats['dedup_removed']} deduped, "
        f"{stats['quality_rejected']} filtered, {stats['sft_train_records']} SFT + "
        f"{stats['preference_pairs']} preference records (seed={stats['seed']})",
        flush=True,
    )
    return {"artifacts": artifacts}


def _stage_pretrain(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    result = run_pretrain(cfg, _data_files(cfg))
    print(
        f"[pretrain] {result['steps']} steps, params={sum(value.size for value in result['params'].values())}, "
        f"val_loss={result['val_loss']:.4f}",
        flush=True,
    )
    return result


def _stage_sft(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    base = _require(_model_path(cfg, "base"), "sft")
    result = run_sft(cfg, base)
    print(
        f"[SFT] {result['steps']} steps, val_loss {result['train_loss_initial']:.4f} -> "
        f"{result['train_loss_final']:.4f} (held-out {result['val_loss']:.4f})",
        flush=True,
    )
    return result


def _stage_lora(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    if not bool(cfg["sft"]["lora"]["enabled"]):
        print("[LoRA] disabled in config", flush=True)
        return {}
    base = _require(_model_path(cfg, "base"), "lora")
    result = run_lora(cfg, base)
    print(
        f"[LoRA] rank {cfg['sft']['lora']['rank']}, {result['trainable_fraction'] * 100:.2f}% trainable, "
        f"val_loss={result['val_loss']:.4f}",
        flush=True,
    )
    return result


def _stage_dpo(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    sft = _require(_model_path(cfg, "sft"), "dpo")
    result = run_dpo(cfg, sft)
    print(
        f"[DPO] selected beta={result['beta']}, dev preference accuracy={result['preference_accuracy']:.4f}, "
        f"{cfg['dpo']['max_steps_per_beta']} steps per beta",
        flush=True,
    )
    return result


def _stage_grpo(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    sft = _require(_model_path(cfg, "sft"), "grpo")
    result = run_grpo(cfg, sft)
    evaluation = result["evaluation"]
    print(
        f"[GRPO] {cfg['grpo']['max_steps']} steps, val exact={evaluation['exact_match']:.4f}, "
        f"format={evaluation['format_valid']:.4f}",
        flush=True,
    )
    return result


def _checkpoints(cfg: Any) -> dict[str, Path]:
    paths = {name: _model_path(cfg, name) for name in MODEL_ORDER}
    return {name: path for name, path in paths.items() if path.exists()}


def _stage_eval(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    checkpoints = _checkpoints(cfg)
    if not checkpoints:
        raise FileNotFoundError("eval requires at least one trained checkpoint")
    result = run_eval(cfg, checkpoints)
    best = result["serving_checkpoint"]
    deltas = result["deltas_vs_base"].get(best, {})
    print(
        f"[eval] models={','.join(result['models'])}; serving={best}; "
        f"task {result['metrics']['task_exact_match'].get(best, 0.0):.4f} "
        f"({deltas.get('task_exact_match', 0.0):+.4f} vs base)",
        flush=True,
    )
    return result


def _serving_checkpoint(cfg: Any) -> Path:
    results_path = Path(cfg.out_dir) / "eval" / "results.json"
    if results_path.exists():
        name = str(_read_json(results_path)["serving_checkpoint"])
        candidate = _model_path(cfg, name)
        if candidate.exists():
            return candidate
    fallback = _model_path(cfg, "dpo")
    return _require(fallback, "serve")


def _stage_quant(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    from ..quant.stage import run_quant

    source = _serving_checkpoint(cfg)
    result = run_quant(cfg, source)
    reports = result["reports"]
    sizes = ", ".join(
        f"{name} {report['size_bytes'] / (1024 * 1024):.3f} MiB" for name, report in reports.items()
    )
    print(f"[quant] source={source.name}; {sizes}", flush=True)
    return result


def _stage_serve(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    source = _serving_checkpoint(cfg)
    result = run_serving_stage(cfg, source)
    spec = result["bench"]["speculative"]
    print(
        f"[serve] target={source.name}; acceptance domain={spec['domain_acceptance_rate']:.4f} "
        f"generic={spec['generic_acceptance_rate']:.4f}; spec==target={spec['spec_output_equals_target']}",
        flush=True,
    )
    return result


def _stage_safety(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    checkpoints = {
        name: path for name, path in _checkpoints(cfg).items() if name in {"base", "sft", "dpo", "grpo"}
    }
    result = run_safety(cfg, checkpoints)
    rates = result["guard_pass_rate"]
    summary = ", ".join(f"{name} {value:.4f}" for name, value in sorted(rates.items()))
    print(f"[safety] guard pass rate: {summary}", flush=True)
    return result


def _stage_emit(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    out = Path(cfg.out_dir) / "gpu"
    records = read_jsonl(Path(cfg.out_dir) / "data" / "sft_train.jsonl")
    written = {}
    written.update(emit_axolotl(cfg, out))
    written.update(emit_trl(cfg, out))
    written.update(emit_quantize(cfg, out, records))
    written.update(emit_vllm(cfg, out))
    written.update(
        emit_k8s(
            cfg,
            out,
            model_path="outputs/gpu/dpo",
            speculator="outputs/gpu/spec-checkpoints/final",
        )
    )
    print(f"[emit] real-stack configs: {len(written)} files under {out}", flush=True)
    return written


def _training_metrics(cfg: Any) -> dict[str, Any]:
    base = _checkpoint_metadata(cfg, "base").get("metadata", {})
    sft = _checkpoint_metadata(cfg, "sft").get("metadata", {})
    lora = _checkpoint_metadata(cfg, "sft_lora").get("metadata", {})
    dpo = _checkpoint_metadata(cfg, "dpo").get("metadata", {})
    grpo = _checkpoint_metadata(cfg, "grpo").get("metadata", {})
    return {
        "steps": {
            "pretrain": base.get("steps", 0),
            "sft": sft.get("steps", 0),
            "lora": lora.get("steps", 0),
            "dpo": dpo.get("steps", 0),
            "grpo": grpo.get("steps", 0),
        },
        "base_val_loss": base.get("val_loss", 0.0),
        "sft_val_loss": sft.get("val_loss", 0.0),
        "sft_train_loss_initial": sft.get("train_loss_initial", 0.0),
        "sft_train_loss_final": sft.get("train_loss_final", 0.0),
        "lora_trainable_fraction": lora.get("trainable_fraction", 0.0),
        "dpo_beta_selected": dpo.get("beta", 0.0),
        "dpo_val_preference_accuracy": dpo.get("preference_accuracy", 0.0),
        "grpo_eval": grpo.get("evaluation", {}),
    }


def _assemble_metrics(cfg: Any) -> dict[str, Any]:
    out = Path(cfg.out_dir)
    data_stats = _read_json(out / "data" / "stats.json")
    eval_result = _read_json(out / "eval" / "results.json")
    beta_path = out / "eval" / "beta-sweep.json"
    if beta_path.exists():
        eval_result["beta_sweep"] = _read_json(beta_path)
    quant = _read_json(out / "quant" / "report.json")["reports"]
    serving = _read_json(out / "serving" / "bench.json")
    safety = _read_json(out / "safety" / "results.json")
    checkpoints = {
        name: _checkpoint_metadata(cfg, name).get("parameter_sha256", "")
        for name in MODEL_ORDER
        if _model_path(cfg, name).exists()
    }
    metrics: dict[str, Any] = {
        "run": {
            "project": cfg["project"]["name"],
            "local_model": cfg["project"]["local_model"],
            "base_model": cfg["project"]["base_model"],
            "seed": int(cfg["run"]["seed"]),
            "config_sha256": sha256_json(cfg.data),
            "git_sha": git_sha(cfg.root),
            "environment": environment(),
            "serving_checkpoint": eval_result["serving_checkpoint"],
        },
        "data": data_stats,
        "training": _training_metrics(cfg),
        "checkpoints": checkpoints,
        "eval": eval_result,
        "quant": quant,
        "serving": serving,
        "safety": safety,
    }
    deterministic = deterministic_sha256(
        {
            "data": data_stats,
            "training": metrics["training"],
            "checkpoints": checkpoints,
            "eval": eval_result,
            "quant": quant,
            "serving": serving,
            "safety": safety,
        }
    )
    metrics["run"]["deterministic_sha256"] = deterministic
    return metrics


def _stage_card(cfg: Any, _: dict[str, Any]) -> dict[str, Any]:
    metrics = _assemble_metrics(cfg)
    out = Path(cfg.out_dir)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "model-card.md").write_text(render_model_card(cfg, metrics, cfg.root), encoding="utf-8")
    (out / "data-card.md").write_text(render_data_card(cfg, metrics), encoding="utf-8")
    print(
        f"[card] model-card.md + data-card.md under {out}; "
        f"deterministic hash {metrics['run']['deterministic_sha256'][:16]}",
        flush=True,
    )
    return metrics


def _stage_manifest(cfg: Any, timings: dict[str, float]) -> dict[str, Any]:
    out = Path(cfg.out_dir)
    (out / "timings.json").write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = Manifest(cfg=cfg, root=cfg.root)
    (out / "manifest.json").unlink(missing_ok=True)
    manifest.add_tree("outputs", out)
    payload = manifest.write(out / "manifest.json")
    print(f"[manifest] {len(payload['artifacts'])} artifacts under {out}", flush=True)
    return payload


HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "data": _stage_data,
    "pretrain": _stage_pretrain,
    "sft": _stage_sft,
    "lora": _stage_lora,
    "dpo": _stage_dpo,
    "grpo": _stage_grpo,
    "eval": _stage_eval,
    "quant": _stage_quant,
    "serve": _stage_serve,
    "safety": _stage_safety,
    "emit": _stage_emit,
    "card": _stage_card,
}


def _selected_stages(stages: list[str] | None) -> list[str]:
    if stages is None:
        return list(STAGES)
    requested = list(dict.fromkeys(stages))
    unknown = sorted(set(requested) - set(STAGES))
    if unknown:
        raise ConfigError(f"unknown stage(s): {', '.join(unknown)}; known: {', '.join(STAGES)}")
    return [name for name in STAGES if name in requested]


def _reset_outputs(cfg: Any) -> None:
    out = Path(cfg.out_dir)
    for name in ("data", "models", "logs", "eval", "quant", "serving", "safety", "gpu"):
        shutil.rmtree(out / name, ignore_errors=True)
    for name in ("metrics.json", "model-card.md", "data-card.md", "manifest.json", "timings.json"):
        (out / name).unlink(missing_ok=True)


def run_pipeline(cfg: Any, *, stages: list[str] | None = None, serve: bool = False, port: int = 8000) -> dict[str, Any]:
    selected = _selected_stages(stages)
    if stages is None:
        _reset_outputs(cfg)
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for name in selected:
        started = time.perf_counter()
        if name == "manifest":
            results[name] = _stage_manifest(cfg, timings)
        else:
            results[name] = HANDLERS[name](cfg, results)
        timings[name] = round(time.perf_counter() - started, 3)
    total = sum(timings.values())
    print(f"[pipeline] {len(selected)} stages in {total:.1f}s; artifacts in {cfg.out_dir}", flush=True)
    if serve:
        from ..serve.server import serve_checkpoint

        checkpoint = _serving_checkpoint(cfg)
        serve_checkpoint(cfg=cfg, checkpoint=checkpoint, port=port, speculative=bool(cfg["serve"]["speculative"]))
    return results
