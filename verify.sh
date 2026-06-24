#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0

uv sync --frozen --group dev
uv run --frozen python -m compileall -q src tests
uv run --frozen python -m pytest -q
uv run --frozen python -m ftpipe check-contamination configs/smoke.yaml --out .verify/contamination

bun install --frozen-lockfile
bun test dashboard
./node_modules/.bin/tsc --noEmit
