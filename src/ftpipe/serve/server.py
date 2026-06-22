"""OpenAI-compatible local HTTP server backed by the batch scheduler."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..model.checkpoint import load_checkpoint
from .engine import ContentPolicyError, Engine
from .scheduler import BatchScheduler


class _Handler(BaseHTTPRequestHandler):
    server_version = "ftpipe/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if self.path == "/metrics":
            engine: Engine = self.server.engine  # type: ignore[attr-defined]
            scheduler: BatchScheduler = self.server.scheduler  # type: ignore[attr-defined]
            with engine.stats.lock:
                stats = engine.stats
                lines = [
                    "# TYPE vllm:num_requests_waiting gauge",
                    f"vllm:num_requests_waiting {stats.waiting}",
                    "# TYPE ftpipe:requests_total counter",
                    f"ftpipe:requests_total {stats.requests}",
                    "# TYPE ftpipe:tokens_generated_total counter",
                    f"ftpipe:tokens_generated_total {stats.completion_tokens}",
                    "# TYPE ftpipe:spec_acceptance_ratio gauge",
                    f"ftpipe:spec_acceptance_ratio {stats.accepted / stats.proposed if stats.proposed else 0.0}",
                ]
            body = ("\n".join(lines) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(404, {"error": {"code": "not_found", "message": self.path}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"code": "not_found", "message": self.path}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"code": "invalid_json", "message": "request body is not JSON"}})
            return
        if payload.get("stream"):
            self._json(400, {"error": {"code": "unsupported", "message": "stream=true is not supported locally"}})
            return
        messages = payload.get("messages") or []
        if not messages or not isinstance(messages, list):
            self._json(400, {"error": {"code": "invalid_request", "message": "messages must be a non-empty list"}})
            return
        prompt = str(messages[-1].get("content", ""))
        engine: Engine = self.server.engine  # type: ignore[attr-defined]
        scheduler: BatchScheduler = self.server.scheduler  # type: ignore[attr-defined]
        max_tokens = int(payload.get("max_tokens", 16))
        started = time.perf_counter()
        try:
            future = scheduler.submit(prompt, max_tokens)
            output = future.result(timeout=120)
        except ContentPolicyError as exc:
            self._json(400, {"error": {"code": "content_policy_violation", "message": str(exc)}})
            return
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._json(500, {"error": {"code": "internal_error", "message": str(exc)}})
            return
        latency_ms = (time.perf_counter() - started) * 1000
        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}-{id(output) % 100000}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", self.server.model_name),  # type: ignore[attr-defined]
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output.text},
                    "finish_reason": output.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "total_tokens": output.prompt_tokens + output.completion_tokens,
            },
        }
        trace_path: Path = self.server.trace_path  # type: ignore[attr-defined]
        trace = {
            "trace_id": response["id"],
            "timestamp_ms": int(time.time() * 1000),
            "model": response["model"],
            "prompt_tokens": output.prompt_tokens,
            "completion_tokens": output.completion_tokens,
            "latency_ms": round(latency_ms, 3),
            "acceptance": output.acceptance,
            "finish_reason": output.finish_reason,
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
        self._json(200, response)


def create_server(
    engine: Engine,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    model_name: str = "ftpipe",
    trace_path: str | Path = "outputs/serving/traces.jsonl",
    max_batch: int = 8,
    window_ms: int = 5,
) -> ThreadingHTTPServer:
    engine.max_batch = max_batch
    scheduler = BatchScheduler(engine, max_batch=max_batch, window_ms=window_ms)
    server = ThreadingHTTPServer((host, port), _Handler)
    server.engine = engine  # type: ignore[attr-defined]
    server.scheduler = scheduler  # type: ignore[attr-defined]
    server.model_name = model_name  # type: ignore[attr-defined]
    trace = Path(trace_path)
    trace.parent.mkdir(parents=True, exist_ok=True)
    server.trace_path = trace  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def serve_checkpoint(cfg: Any, checkpoint: str | Path, port: int = 8000, speculative: bool = True) -> None:
    params, config, tokenizer, _ = load_checkpoint(checkpoint)
    draft_path = Path(cfg.out_dir) / "models" / "draft"
    guard = _load_optional_guard(Path(cfg.out_dir) / "safety" / "guard.json", cfg) if cfg["serve"]["guard"] else None
    shield = _load_optional_shield(Path(cfg.out_dir) / "safety" / "shield.json", cfg) if cfg["serve"]["guard"] else None
    engine = Engine(
        checkpoint,
        draft_path=draft_path,
        guard=guard,
        shield=shield,
        speculative=speculative and bool(cfg["serve"]["speculative"]),
        draft_tokens=int(cfg["serve"]["draft_tokens"]),
    )
    server = create_server(
        engine,
        host="127.0.0.1",
        port=port,
        model_name=str(cfg["project"]["name"]),
        trace_path=Path(cfg.out_dir) / "serving" / "traces.jsonl",
        max_batch=int(cfg["serve"]["max_batch"]),
        window_ms=int(cfg["serve"]["batch_window_ms"]),
    )
    print(f"[serve] listening on http://127.0.0.1:{server.server_address[1]}/v1", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.scheduler.shutdown()  # type: ignore[attr-defined]


def _load_optional_guard(path: Path, cfg: Any) -> Any:
    if not path.exists():
        return None
    from ..safetymodels.guard import Guard

    return Guard.load(path, cfg)


def _load_optional_shield(path: Path, cfg: Any) -> Any:
    if not path.exists():
        return None
    from ..safetymodels.shield import Shield

    return Shield.load(path, cfg)
