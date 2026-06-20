"""YAML configuration loading, strict validation, and path resolution."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a pipeline config is missing, unknown, or out of range."""


DEFAULTS: dict[str, Any] = {
    "run": {"seed": 7, "output_dir": "outputs", "stage_timeout_s": 900},
    "project": {
        "name": "domainx",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "local_model": "tiny-2l-64d",
    },
    "data": {
        "raw_docs": 1200,
        "sft_pairs": 800,
        "preference_pairs": 600,
        "minhash": {"ngram": 5, "num_hashes": 64, "bands": 16, "rows": 4, "threshold": 0.5},
        "quality": {
            "min_tokens": 12,
            "min_words": 8,
            "max_symbol_ratio": 0.15,
            "classifier_threshold": 0.5,
        },
        "pii": {"entities": ["EMAIL", "PHONE", "CREDIT_CARD", "SSN", "IPV4", "PERSON"]},
        "split": {"val_fraction": 0.1},
        "contamination": {"threshold": 0.5, "exact_ngram": 8},
    },
    "model": {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "max_seq_len": 128,
        "tie_embeddings": True,
        "dropout": 0.0,
    },
    "pretrain": {
        "max_steps": 350,
        "batch_size": 8,
        "sequence_len": 64,
        "learning_rate": 0.003,
        "warmup_ratio": 0.05,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
    },
    "sft": {
        "epochs": 3,
        "max_steps": 300,
        "batch_size": 8,
        "sequence_len": 96,
        "learning_rate": 0.001,
        "warmup_ratio": 0.05,
        "sample_packing": True,
        "lora": {
            "enabled": True,
            "rank": 8,
            "alpha": 16,
            "max_steps": 200,
            "learning_rate": 0.002,
        },
    },
    "dpo": {
        "epochs": 1,
        "max_steps_per_beta": 80,
        "beta_grid": [0.05, 0.1, 0.2],
        "batch_size": 4,
        "learning_rate": 0.0005,
        "loss_type": "sigmoid",
    },
    "grpo": {
        "max_steps": 60,
        "num_generations": 4,
        "batch_prompts": 4,
        "learning_rate": 0.0002,
        "temperature": 0.9,
        "top_k": 20,
        "kl_coef": 0.02,
        "reward": {"exact": 1.0, "format": 0.5, "category": 0.25},
    },
    "quant": {
        "calibration_samples": 64,
        "gptq": {"group_size": 32, "damp_percent": 0.01, "act_order": True},
        "awq": {"group_size": 32, "alpha_grid": [0.0, 0.5, 1.0]},
        "gguf": {"embedding_dtype": "F16", "head_dtype": "Q6_K"},
    },
    "serve": {
        "max_new_tokens": 16,
        "draft_tokens": 4,
        "speculative": True,
        "guard": True,
        "max_batch": 8,
        "batch_window_ms": 5,
        "batch_sizes": [1, 8, 32],
        "requests_per_batch": 12,
        "gpu_hourly_usd": 3.99,
        "comparison_models": [
            {
                "provider": "OpenAI",
                "model": "gpt-5.4-mini",
                "input_usd_per_mtok": 0.75,
                "output_usd_per_mtok": 4.5,
                "source": "https://developers.openai.com/api/docs/pricing",
            },
            {
                "provider": "Anthropic",
                "model": "claude-haiku-4.5",
                "input_usd_per_mtok": 1.0,
                "output_usd_per_mtok": 5.0,
                "source": "https://platform.claude.com/docs/en/about-claude/pricing",
            },
        ],
    },
    "eval": {
        "task_items": 60,
        "mmlu_items": 30,
        "mt_bench_items": 12,
        "rewardbench_items": 24,
        "judge": "deterministic-rubric",
    },
    "safety": {
        "guard_threshold": 0.5,
        "shield_threshold": 0.5,
        "guard_model": "meta-llama/Llama-Guard-4-12B",
        "shield_model": "google/shieldgemma-2b",
    },
    "card": {
        "mof_version": "1.0",
        "code_license": "MIT",
        "data_license": "CC0-1.0",
        "weights_license": "CDLA-Permissive-2.0",
    },
}


_COMPARISON_KEYS = {"provider", "model", "input_usd_per_mtok", "output_usd_per_mtok", "source"}


def _schema() -> dict[str, Any]:
    """Leaf types plus nested dict schemas; lists use a one-element type tuple."""

    return {
        "run": {
            "seed": int,
            "output_dir": str,
            "stage_timeout_s": int,
        },
        "project": {"name": str, "base_model": str, "local_model": str},
        "data": {
            "raw_docs": int,
            "sft_pairs": int,
            "preference_pairs": int,
            "minhash": {
                "ngram": int,
                "num_hashes": int,
                "bands": int,
                "rows": int,
                "threshold": float,
            },
            "quality": {
                "min_tokens": int,
                "min_words": int,
                "max_symbol_ratio": float,
                "classifier_threshold": float,
            },
            "pii": {"entities": (str,)},
            "split": {"val_fraction": float},
            "contamination": {"threshold": float, "exact_ngram": int},
        },
        "model": {
            "d_model": int,
            "n_layers": int,
            "n_heads": int,
            "d_ff": int,
            "max_seq_len": int,
            "tie_embeddings": bool,
            "dropout": float,
        },
        "pretrain": {
            "max_steps": int,
            "batch_size": int,
            "sequence_len": int,
            "learning_rate": float,
            "warmup_ratio": float,
            "weight_decay": float,
            "grad_clip": float,
        },
        "sft": {
            "epochs": int,
            "max_steps": int,
            "batch_size": int,
            "sequence_len": int,
            "learning_rate": float,
            "warmup_ratio": float,
            "sample_packing": bool,
            "lora": {
                "enabled": bool,
                "rank": int,
                "alpha": int,
                "max_steps": int,
                "learning_rate": float,
            },
        },
        "dpo": {
            "epochs": int,
            "max_steps_per_beta": int,
            "beta_grid": (float,),
            "batch_size": int,
            "learning_rate": float,
            "loss_type": str,
        },
        "grpo": {
            "max_steps": int,
            "num_generations": int,
            "batch_prompts": int,
            "learning_rate": float,
            "temperature": float,
            "top_k": int,
            "kl_coef": float,
            "reward": {"exact": float, "format": float, "category": float},
        },
        "quant": {
            "calibration_samples": int,
            "gptq": {"group_size": int, "damp_percent": float, "act_order": bool},
            "awq": {"group_size": int, "alpha_grid": (float,)},
            "gguf": {"embedding_dtype": str, "head_dtype": str},
        },
        "serve": {
            "max_new_tokens": int,
            "draft_tokens": int,
            "speculative": bool,
            "guard": bool,
            "max_batch": int,
            "batch_window_ms": int,
            "batch_sizes": (int,),
            "requests_per_batch": int,
            "gpu_hourly_usd": float,
            "comparison_models": list,
        },
        "eval": {
            "task_items": int,
            "mmlu_items": int,
            "mt_bench_items": int,
            "rewardbench_items": int,
            "judge": str,
        },
        "safety": {
            "guard_threshold": float,
            "shield_threshold": float,
            "guard_model": str,
            "shield_model": str,
        },
        "card": {
            "mof_version": str,
            "code_license": str,
            "data_license": str,
            "weights_license": str,
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return f"list[{_type_name(expected[0])}]"
    if isinstance(expected, dict):
        return "mapping"
    return expected.__name__


def _validate_node(value: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping, got {type(value).__name__}")
        for key in value:
            if key not in schema:
                raise ConfigError(f"{path}.{key}: unknown key")
        for key, child in schema.items():
            if key not in value:
                raise ConfigError(f"{path}.{key}: missing required key")
            _validate_node(value[key], child, f"{path}.{key}")
        return
    if isinstance(schema, tuple):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
        for index, item in enumerate(value):
            _validate_node(item, schema[0], f"{path}[{index}]")
        return
    if schema is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected integer, got {type(value).__name__}")
        return
    if schema is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected number, got {type(value).__name__}")
        return
    if schema is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected boolean, got {type(value).__name__}")
        return
    if schema is str:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{path}: expected non-empty string")
        return
    if schema is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
        return
    raise ConfigError(f"{path}: internal schema error for {schema!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_ranges(cfg: dict[str, Any]) -> None:
    d = cfg["data"]
    _require(d["raw_docs"] > 0, "data.raw_docs: must be > 0")
    _require(d["sft_pairs"] > 0, "data.sft_pairs: must be > 0")
    _require(d["preference_pairs"] > 0, "data.preference_pairs: must be > 0")
    m = d["minhash"]
    _require(m["ngram"] >= 3, "data.minhash.ngram: must be >= 3")
    _require(m["num_hashes"] > 0, "data.minhash.num_hashes: must be > 0")
    _require(m["bands"] > 0 and m["rows"] > 0, "data.minhash: bands/rows must be > 0")
    _require(m["bands"] * m["rows"] == m["num_hashes"], "data.minhash: bands * rows must equal num_hashes")
    _require(0.0 < m["threshold"] <= 1.0, "data.minhash.threshold: must be in (0, 1]")
    q = d["quality"]
    _require(q["min_tokens"] >= 1 and q["min_words"] >= 1, "data.quality: limits must be >= 1")
    _require(0.0 <= q["max_symbol_ratio"] <= 1.0, "data.quality.max_symbol_ratio: must be in [0, 1]")
    _require(0.0 <= q["classifier_threshold"] <= 1.0, "data.quality.classifier_threshold: must be in [0, 1]")
    _require(0.0 < d["split"]["val_fraction"] < 1.0, "data.split.val_fraction: must be in (0, 1)")
    _require(0.0 < d["contamination"]["threshold"] <= 1.0, "data.contamination.threshold: must be in (0, 1]")
    _require(d["contamination"]["exact_ngram"] >= 3, "data.contamination.exact_ngram: must be >= 3")
    _require(bool(d["pii"]["entities"]), "data.pii.entities: must not be empty")

    model = cfg["model"]
    _require(model["d_model"] > 0 and model["d_ff"] > 0, "model: dimensions must be > 0")
    _require(model["d_model"] % model["n_heads"] == 0, "model.d_model must be divisible by model.n_heads")
    _require(model["n_layers"] >= 1 and model["n_heads"] >= 1, "model: layers/heads must be >= 1")
    _require(0.0 <= model["dropout"] < 1.0, "model.dropout: must be in [0, 1)")
    _require(model["max_seq_len"] >= 8, "model.max_seq_len: must be >= 8")

    for section in ("pretrain", "sft"):
        s = cfg[section]
        _require(s["batch_size"] > 0, f"{section}.batch_size: must be > 0")
        _require(s["sequence_len"] > 0, f"{section}.sequence_len: must be > 0")
        _require(s["learning_rate"] > 0, f"{section}.learning_rate: must be > 0")
        _require(0.0 <= s["warmup_ratio"] < 1.0, f"{section}.warmup_ratio: must be in [0, 1)")
        _require(s["max_steps"] > 0, f"{section}.max_steps: must be > 0")
    _require(cfg["sft"]["epochs"] > 0, "sft.epochs: must be > 0")
    _require(cfg["sft"]["sequence_len"] <= model["max_seq_len"], "sft.sequence_len exceeds model.max_seq_len")
    _require(cfg["pretrain"]["sequence_len"] <= model["max_seq_len"], "pretrain.sequence_len exceeds model.max_seq_len")
    _require(cfg["pretrain"]["weight_decay"] >= 0, "pretrain.weight_decay: must be >= 0")
    _require(cfg["pretrain"]["grad_clip"] > 0, "pretrain.grad_clip: must be > 0")
    lora = cfg["sft"]["lora"]
    _require(lora["rank"] > 0 and lora["alpha"] > 0, "sft.lora: rank/alpha must be > 0")
    _require(lora["max_steps"] > 0, "sft.lora.max_steps: must be > 0")
    _require(lora["learning_rate"] > 0, "sft.lora.learning_rate: must be > 0")

    dpo = cfg["dpo"]
    _require(dpo["epochs"] > 0, "dpo.epochs: must be > 0")
    _require(dpo["max_steps_per_beta"] > 0, "dpo.max_steps_per_beta: must be > 0")
    _require(bool(dpo["beta_grid"]) and all(b > 0 for b in dpo["beta_grid"]), "dpo.beta_grid: betas must be > 0")
    _require(dpo["batch_size"] > 0, "dpo.batch_size: must be > 0")
    _require(dpo["learning_rate"] > 0, "dpo.learning_rate: must be > 0")
    _require(dpo["loss_type"] == "sigmoid", "dpo.loss_type: only 'sigmoid' is implemented")

    grpo = cfg["grpo"]
    _require(grpo["max_steps"] > 0, "grpo.max_steps: must be > 0")
    _require(grpo["num_generations"] >= 2, "grpo.num_generations: must be >= 2")
    _require(grpo["batch_prompts"] > 0, "grpo.batch_prompts: must be > 0")
    _require(grpo["learning_rate"] > 0, "grpo.learning_rate: must be > 0")
    _require(0.0 < grpo["temperature"] <= 2.0, "grpo.temperature: must be in (0, 2]")
    _require(grpo["top_k"] >= 1, "grpo.top_k: must be >= 1")
    _require(grpo["kl_coef"] >= 0, "grpo.kl_coef: must be >= 0")
    _require(all(v >= 0 for v in grpo["reward"].values()), "grpo.reward: weights must be >= 0")

    quant = cfg["quant"]
    _require(quant["calibration_samples"] > 0, "quant.calibration_samples: must be > 0")
    for name in ("gptq", "awq"):
        _require(quant[name]["group_size"] > 0, f"quant.{name}.group_size: must be > 0")
    _require(0.0 <= quant["gptq"]["damp_percent"] <= 1.0, "quant.gptq.damp_percent: must be in [0, 1]")
    _require(bool(quant["awq"]["alpha_grid"]), "quant.awq.alpha_grid: must not be empty")
    _require(quant["gguf"]["embedding_dtype"] in {"F16", "F32"}, "quant.gguf.embedding_dtype: must be F16 or F32")
    _require(quant["gguf"]["head_dtype"] in {"Q6_K", "Q4_K"}, "quant.gguf.head_dtype: must be Q6_K or Q4_K")

    serve = cfg["serve"]
    _require(serve["max_new_tokens"] > 0, "serve.max_new_tokens: must be > 0")
    _require(serve["draft_tokens"] > 0, "serve.draft_tokens: must be > 0")
    _require(serve["max_batch"] > 0, "serve.max_batch: must be > 0")
    _require(serve["batch_window_ms"] >= 0, "serve.batch_window_ms: must be >= 0")
    _require(bool(serve["batch_sizes"]) and all(b > 0 for b in serve["batch_sizes"]), "serve.batch_sizes: must be positive")
    _require(serve["requests_per_batch"] > 0, "serve.requests_per_batch: must be > 0")
    _require(serve["gpu_hourly_usd"] >= 0, "serve.gpu_hourly_usd: must be >= 0")
    for comparison in serve["comparison_models"]:
        _require(isinstance(comparison, dict), "serve.comparison_models: entries must be mappings")
        _require(set(comparison) == _COMPARISON_KEYS, "serve.comparison_models: entries need exactly provider/model/input_usd_per_mtok/output_usd_per_mtok/source")
        _require(comparison["output_usd_per_mtok"] > 0, "serve.comparison_models: output price must be > 0")

    ev = cfg["eval"]
    for key in ("task_items", "mmlu_items", "mt_bench_items", "rewardbench_items"):
        _require(ev[key] > 0, f"eval.{key}: must be > 0")
    _require(ev["judge"] == "deterministic-rubric", "eval.judge: only 'deterministic-rubric' is implemented")
    for key in ("guard_threshold", "shield_threshold"):
        _require(0.0 <= cfg["safety"][key] <= 1.0, f"safety.{key}: must be in [0, 1]")


def load_config(path: str | Path) -> "RunConfig":
    source = Path(path)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        raise ConfigError(f"{path}: config file not found")
    try:
        user = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if user is None:
        user = {}
    if not isinstance(user, dict):
        raise ConfigError(f"{path}: top-level YAML value must be a mapping")
    merged = _deep_merge(DEFAULTS, user)
    _validate_node(merged, _schema(), "config")
    _check_ranges(merged)
    out_dir = Path(merged["run"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    return RunConfig(source=source, data=merged, root=PROJECT_ROOT, out_dir=out_dir)


@dataclass(frozen=True)
class RunConfig:
    source: Path
    data: dict[str, Any]
    root: Path
    out_dir: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def with_out(self, out_dir: str | Path) -> "RunConfig":
        path = Path(out_dir)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        return replace(self, out_dir=path)

    def with_seed(self, seed: int) -> "RunConfig":
        data = copy.deepcopy(self.data)
        data["run"]["seed"] = int(seed)
        return replace(self, data=data)

    def canonical_json(self) -> str:
        return json.dumps(self.data, sort_keys=True, separators=(",", ":"))
