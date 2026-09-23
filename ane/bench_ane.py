"""Benchmark + parity for the ANE runtime vs golden fixtures and MLX.

Subcommands:
  runtime   e2e system_one timing per fixture (same protocol as benchmark.py)
  parity    compare answers vs autoresearch_golden.json (labels exact, probs <=2e-2)
  bodytime  isolated Core ML predict() timing per bucket
  split     concurrency test: ANE handles some questions, MLX the rest

Examples:
    .venv/bin/python ane/bench_ane.py runtime --model converted-fp16
    .venv/bin/python ane/bench_ane.py parity --model converted-fp16
    .venv/bin/python ane/bench_ane.py split --model converted-fp16 --mlx-model converted-fp16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark
import laya_api
from ane.ane_runtime import LayaANE


def _stats(times):
    a = np.array(times)
    return {"samples": len(times), "mean_ms": round(float(a.mean()), 3),
            "p50_ms": round(float(np.median(a)), 3), "min_ms": round(float(a.min()), 3),
            "max_ms": round(float(a.max()), 3), "wall_ms": [round(t, 3) for t in times]}


def cmd_runtime(args):
    agent = LayaANE(args.model, buckets=args.buckets)
    fixtures = benchmark.make_fixtures()
    report = {"arm": "ane", "buckets": agent.buckets, "cases": {}}
    for name, (state, questions) in fixtures.items():
        try:
            for _ in range(args.warmups):
                agent.system_one(state, questions)
        except ValueError as e:
            report["cases"][name] = {"skipped": str(e)}
            print("[ane] %s: SKIPPED %s" % (name, e), file=sys.stderr)
            continue
        times = []
        for _ in range(args.samples):
            t0 = time.perf_counter()
            agent.system_one(state, questions)
            times.append((time.perf_counter() - t0) * 1000.0)
        _, _, b = agent.prepare(state, questions)
        report["cases"][name] = {"n_questions": len(questions),
                                 "input_tokens": int(b["n_tokens"]), **_stats(times)}
        print("[ane] %s: p50 %.1f ms  min %.1f  max %.1f" %
              (name, report["cases"][name]["p50_ms"],
               report["cases"][name]["min_ms"], report["cases"][name]["max_ms"]), file=sys.stderr)
    _emit(report, args.output)


def cmd_parity(args):
    golden = json.load(open("autoresearch_golden.json"))["fixtures"]
    agent = LayaANE(args.model, buckets=args.buckets)
    fixtures = benchmark.make_fixtures()
    report = {"cases": {}, "failures": []}
    for name, (state, questions) in fixtures.items():
        if name not in golden:
            continue
        try:
            out = agent.system_one(state, questions)
        except ValueError as e:
            report["cases"][name] = {"skipped": str(e)}
            continue
        g = golden[name]
        diffs = benchmark._answers_agree(g["answers"], out["answers"], args.prob_tol)
        usage_ok = out["usage"] == g["usage"]
        report["cases"][name] = {"answer_diffs": diffs, "usage_equal": usage_ok,
                                 "pass": not diffs and usage_ok}
        if diffs or not usage_ok:
            report["failures"].append(name)
        print("[parity] %s: %s (%d diffs, usage %s)" %
              (name, "PASS" if not diffs and usage_ok else "FAIL", len(diffs), usage_ok),
              file=sys.stderr)
    report["pass"] = not report["failures"]
    _emit(report, args.output)
    return 0 if report["pass"] else 1


def cmd_bodytime(args):
    agent = LayaANE(args.model, buckets=args.buckets)
    report = {"buckets": {}}
    for L in agent.buckets:
        n = min(L, 8)
        ids = np.zeros(n, dtype=np.int64)
        ids[0] = agent.tok.cls_token_id
        item = {"ids": ids, "qtype": 0, "markers": [min(1, n - 1)]}
        inputs = agent.model_inputs(item["ids"], None, 0, item["markers"], L)
        m = agent.models[L]
        for _ in range(5):
            m.predict(inputs)
        times = []
        for _ in range(args.samples):
            t0 = time.perf_counter()
            m.predict(inputs)
            times.append((time.perf_counter() - t0) * 1000.0)
        report["buckets"][L] = _stats(times)
        print("[body] L=%d: p50 %.2f ms  min %.2f" % (L, report["buckets"][L]["p50_ms"],
                                                     report["buckets"][L]["min_ms"]), file=sys.stderr)
    _emit(report, args.output)


def cmd_split(args):
    """Concurrency: split batch8_short questions between ANE (per-question) and
    MLX (batched forward) in two threads; compare wall time vs MLX alone."""
    import threading

    import mlx.core as mx
    mx.set_cache_limit(1 << 30)
    ane = LayaANE(args.model, buckets=args.buckets)
    mlx = laya_api.LayaMLX(args.mlx_model, dtype="float16")
    state, questions = benchmark.make_fixtures()["batch8_short"]
    qids = list(questions.keys())

    def timed(fn, n):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return ts

    # MLX alone, all 8 in one padded batch
    mlx_only = timed(lambda: mlx.system_one(state, questions), args.samples)

    results = {}
    for n_ane in (2, 4):
        qs_ane = {k: questions[k] for k in qids[:n_ane]}
        qs_mlx = {k: questions[k] for k in qids[n_ane:]}

        def split_call():
            err = []
            def run_ane():
                try:
                    results["ane"] = ane.system_one(state, qs_ane)
                except Exception as e:
                    err.append(e)
            def run_mlx():
                try:
                    results["mlx"] = mlx.system_one(state, qs_mlx)
                except Exception as e:
                    err.append(e)
            t1 = threading.Thread(target=run_ane)
            t2 = threading.Thread(target=run_mlx)
            t1.start(); t2.start(); t1.join(); t2.join()
            if err:
                raise err[0]

        ts = timed(split_call, args.samples)
        results["split_%d" % n_ane] = ts
        print("[split] ane=%d mlx=%d: p50 %.1f ms (mlx-only p50 %.1f)" %
              (n_ane, 8 - n_ane, float(np.median(ts)), float(np.median(mlx_only))), file=sys.stderr)

    report = {"mlx_only": _stats(mlx_only),
              "splits": {k: _stats(v) for k, v in results.items() if k.startswith("split_")}}
    _emit(report, args.output)


def _emit(report, output):
    text = json.dumps(report, indent=2)
    if output:
        Path(output).write_text(text + "\n")
        print("wrote %s" % output, file=sys.stderr)
    else:
        print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("runtime", "parity", "bodytime", "split"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="converted-fp16")
        p.add_argument("--mlx-model", default="converted-fp16")
        p.add_argument("--buckets", type=int, nargs="*", default=None)
        p.add_argument("--warmups", type=int, default=3)
        p.add_argument("--samples", type=int, default=15)
        p.add_argument("--prob-tol", type=float, default=2e-2)
        p.add_argument("-o", "--output", default=None)
        p.set_defaults(fn={"runtime": cmd_runtime, "parity": cmd_parity,
                           "bodytime": cmd_bodytime, "split": cmd_split}[name])
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
