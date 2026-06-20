"""Command-line entry point for the local pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RunConfig, load_config


def _config(args: argparse.Namespace) -> RunConfig:
    cfg = load_config(args.config)
    if getattr(args, "out", None):
        cfg = cfg.with_out(args.out)
    if getattr(args, "seed", None) is not None:
        cfg = cfg.with_seed(args.seed)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftpipe", description="End-to-end fine-tuning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline stages")
    run.add_argument("config")
    run.add_argument("--out", default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--stages", default=None, help="comma-separated stage names")
    run.add_argument("--serve", action="store_true", help="serve after artifacts are written")
    run.add_argument("--port", type=int, default=8000)

    inspect = sub.add_parser("inspect", help="print the resolved config")
    inspect.add_argument("config")
    inspect.add_argument("--out", default=None)

    check = sub.add_parser("check-contamination", help="run only the contamination gate")
    check.add_argument("config")
    check.add_argument("--out", default=None)

    serve = sub.add_parser("serve", help="serve a checkpoint over the OpenAI-compatible API")
    serve.add_argument("config")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-speculative", action="store_true")

    bench = sub.add_parser("bench", help="benchmark a checkpoint over local HTTP")
    bench.add_argument("config")
    bench.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        cfg = _config(args)
        print(json.dumps(cfg.data, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        cfg = _config(args)
        from .stages.pipeline import run_pipeline

        stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
        run_pipeline(cfg, stages=stages, serve=args.serve, port=args.port)
        return 0

    if args.command == "check-contamination":
        cfg = _config(args)
        from .data.contamination import check_contamination_command

        return check_contamination_command(cfg)

    if args.command == "serve":
        cfg = _config(args)
        from .serve.server import serve_checkpoint

        serve_checkpoint(
            cfg=cfg,
            checkpoint=Path(args.checkpoint),
            port=args.port,
            speculative=not args.no_speculative,
        )
        return 0

    if args.command == "bench":
        cfg = _config(args)
        from .serve.bench import bench_command

        bench_command(cfg, Path(args.checkpoint))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
