#!/usr/bin/env bash
# Canonical autoresearch benchmark entrypoint for the Laya MLX runtime.
# Runs the fail-closed decision-parity gate, then times the four deterministic
# fixtures and prints METRIC lines. Exit 0 only when the gate passes AND timing
# completes.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "FATAL: $PY not found" >&2
  exit 2
fi

# LAYA_MODEL / LAYA_BENCH_DTYPE select the configuration under test; LAYA_AGENT
# picks the runtime (fast = LayaFast ANE+MLX router, default; mlx = GPU only).
export LAYA_BENCH_DTYPE="${LAYA_BENCH_DTYPE:-float16}"
exec "$PY" benchmarks/bench_autoresearch.py --model "${LAYA_MODEL:-converted-fp16}" "$@"
