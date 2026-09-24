"""Verify LayaFast: golden gate + interleaved timing vs LayaMLX in one process.

Usage:
    .venv/bin/python benchmarks/gpulock.py -- .venv/bin/python ane/verify_fast.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import benchmark  # type: ignore[import-not-found]  # reason: sibling module resolved via sys.path.insert(benchmarks dir) above
import laya_api
from laya_fast import LayaFast


VerifyGate = TypedDict("VerifyGate", {
    "pass": bool,
    "failures": list[str],
    "max_logit_diff": float,
    "max_prob_diff": float,
})


class TimingRow(TypedDict):
    """Per-fixture interleaved timing row."""

    fast_ms: list[float]
    mlx_ms: list[float]
    fast_p50: float
    mlx_p50: float


class VerifyReport(TypedDict):
    """verify_fast.json payload."""
    load_seconds: dict[str, float]
    gate: VerifyGate
    timing: dict[str, TimingRow]
    rss_mb: NotRequired[int]


def main() -> int:
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)

    t0 = time.perf_counter()
    fast = LayaFast("converted-fp16")
    fast_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    mlx_agent = laya_api.LayaMLX("converted-fp16", dtype="float16", compile=True)
    mlx_load = time.perf_counter() - t0
    print("load: LayaFast %.1f s, LayaMLX %.1f s" % (fast_load, mlx_load), file=sys.stderr)

    # ---- golden gate on LayaFast outputs
    golden = json.load(open(Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "autoresearch_golden.json"))["fixtures"]
    fixtures = benchmark.make_fixtures()
    import bench_autoresearch  # type: ignore[import-not-found]  # reason: sibling module resolved via sys.path.insert(benchmarks dir) above
    current = bench_autoresearch.run_all(fast, fixtures)
    fails, max_logit, max_prob = bench_autoresearch.compare(golden, current)
    print("gate: %s  max|dlogit|=%.3e max|dprob|=%.3e" %
          ("PASS" if not fails else "FAIL", max_logit, max_prob), file=sys.stderr)
    for f in cast(list[Any], fails)[:10]:
        print("  - %s" % f, file=sys.stderr)

    # ---- extra workloads: mixed-length batch + unseen-length singles
    state_s, qs8 = fixtures["batch8_short"]
    state_l, _ = fixtures["batch8_long"]
    qids = list(qs8)
    mixed_qs = {("s_" + k): qs8[k] for k in qids[:4]}
    # long-state questions reuse the same question defs on the long state
    mixed = [("mixed_state", None)]
    # build a mixed fixture: 4 questions on short state + 4 on long state is not
    # expressible in one system_one call (one state per call), so instead use a
    # medium state (~200 tokens) and an 8-question batch on it.
    med_state = (
        "The support ticket queue shows 47 open items. The customer reports intermittent "
        "login failures since the Tuesday deploy. Error rate is 3 percent of sessions. " * 6
    )
    extra = {
        "batch8_medium": (med_state, qs8),
        "single_tiny": ("Ticket #12 is open.", {"q_noul": qs8["q_noul"]}),
        "single_medium": (med_state, {"q_choice4": qs8["q_choice4"]}),
    }
    _, items_med, _ = fast.prepare(med_state, qs8)
    print("medium lens:", [len(i["ids"]) for i in items_med], file=sys.stderr)

    # ---- interleaved timing, 3 rounds
    all_fx = dict(fixtures)
    all_fx.update(extra)
    results: dict[str, dict[str, list[float]]] = {name: {"fast": [], "mlx": []} for name in all_fx}
    for rnd in range(3):
        for name, (state, questions) in all_fx.items():
            for tag, agent in (("fast", fast), ("mlx", mlx_agent)):
                t0 = time.perf_counter()
                agent.system_one(state, questions)
                if agent.last_raw is not None:
                    mx.eval(*agent.last_raw)
                results[name][tag].append((time.perf_counter() - t0) * 1000.0)

    report: VerifyReport = {"load_seconds": {"fast": round(fast_load, 2), "mlx": round(mlx_load, 2)},
              "gate": {"pass": not fails, "failures": fails,
                       "max_logit_diff": max_logit, "max_prob_diff": max_prob},
              "timing": {}}
    for name in all_fx:
        f, m = results[name]["fast"], results[name]["mlx"]
        report["timing"][name] = {
            "fast_ms": [round(t, 1) for t in f], "mlx_ms": [round(t, 1) for t in m],
            "fast_p50": round(float(np.median(f)), 2), "mlx_p50": round(float(np.median(m)), 2),
        }
        print("%-15s fast %7.1f ms  mlx %7.1f ms" %
              (name, report["timing"][name]["fast_p50"], report["timing"][name]["mlx_p50"]),
              file=sys.stderr)

    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                      text=True).strip()) // 1024
    report["rss_mb"] = rss
    print("RSS %d MB" % rss, file=sys.stderr)
    Path(__file__).resolve().parent.joinpath("verify_fast.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote ane/verify_fast.json", file=sys.stderr)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
