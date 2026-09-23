"""Cold-shape latency benchmark for the Laya MLX runtime.

Loads the model once, then issues requests with never-before-seen shapes:
  * N_SINGLE single-question requests with distinct sequence lengths in 20..500
  * N_BATCH batched requests (4-8 questions) with distinct packed token counts

For each request it reports the FIRST-call latency (the cold-shape cost) and
the steady-state latency (median of the repeat calls). With cold dispatch on
(LAYA_COLD_DISPATCH=1, the default) the 2nd call pays the deferred mx.compile
retrace, so it is reported separately.

Before/after: run with the fix flags off to reproduce the old behaviour:
    LAYA_BUCKET=0 LAYA_COLD_DISPATCH=0 LAYA_COMPILE_PACKED=1 \
        .venv/bin/python benchmarks/gpulock.py -- .venv/bin/python benchmarks/bench_cold_shapes.py
vs the new defaults:
    .venv/bin/python benchmarks/gpulock.py -- .venv/bin/python benchmarks/bench_cold_shapes.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_SINGLE = 30
N_BATCH = 10
REPEATS = 3  # calls per request; call 1 = cold, calls 2..N = steady (2nd may compile)

_WORDS = (
    "the customer was charged twice for the same order and wants a refund "
    "the login page rejects valid credentials after the tuesday deploy "
    "shipping shows delivered but the package never arrived at the office "
    "please escalate this ticket to the billing team immediately"
).split()


def _state(n):
    """A state string of ~n words (deterministic, no RNG)."""
    reps = (n // len(_WORDS)) + 2
    return " ".join((_WORDS * reps)[:n])



def _choice_q(i, k=3):
    crit = {f"opt{j}": f"option {j} for case {i}" for j in range(k)}
    return {"type": "choice", "instructions": f"Which option fits case {i}?",
            "criteria": crit}


def _batch_questions(nq, seed):
    qs = {}
    for i in range(nq):
        if i % 3 == 0:
            qs[f"q{i}"] = _choice_q(seed + i, k=3 + (i % 4))
        elif i % 3 == 1:
            qs[f"q{i}"] = {"type": "noul",
                           "instructions": f"Is statement {seed + i} supported?"}
        else:
            qs[f"q{i}"] = {"type": "score",
                           "instructions": f"Rate severity for case {seed + i}.",
                           "criteria": [f"l{j}" for j in range(5)]}
    return qs


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "mean": float(a.mean()), "n": len(vals)}


def main():
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)  # 1 GB freed-buffer cache cap (memory budget)
    import laya_api

    model = os.environ.get("LAYA_MODEL", str(ROOT / "converted-fp16"))
    dtype = os.environ.get("LAYA_BENCH_DTYPE", "float16")
    t0 = time.perf_counter()
    agent = laya_api.LayaMLX(model, dtype=dtype, compile=True)
    load_s = time.perf_counter() - t0
    print(f"[cfg] bucket={agent.bucket} cold_dispatch={agent.cold_dispatch} "
          f"compile_packed={agent.compile_packed} unpad={agent.unpad} "
          f"load={load_s:.2f}s", file=sys.stderr)

    # Warm exactly one shape so one-time costs (weight upload, first Metal
    # pipeline builds) are not attributed to the cold-shape measurements.
    agent.system_one(_state(30), {"w": _choice_q(0)})

    first, second, steady = [], [], []
    shapes = []

    # distinct single-question lengths spread over 20..500 words
    ns = np.linspace(20, 500, N_SINGLE).astype(int)
    ns = sorted(set(int(n) for n in ns))
    for i, n in enumerate(ns):
        s, q = _state(n), {"q": _choice_q(i)}
        L = agent.prepare(s, q)[2]["input_ids"].shape[1]
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            agent.system_one(s, q)
            ts.append((time.perf_counter() - t0) * 1e3)
        first.append(ts[0]); second.append(ts[1]); steady.append(float(np.median(ts[1:])))
        shapes.append(f"single L={L}")

    # distinct batched requests: 4-8 questions, varying state length -> distinct T
    for i in range(N_BATCH):
        nq = 4 + (i % 5)
        s = _state(15 + 37 * i)
        q = _batch_questions(nq, 100 + i * 10)
        b = agent.prepare(s, q)[2]
        B, L = b["input_ids"].shape
        T = int(b["attention_mask"].sum())
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            agent.system_one(s, q)
            ts.append((time.perf_counter() - t0) * 1e3)
        first.append(ts[0]); second.append(ts[1]); steady.append(float(np.median(ts[1:])))
        shapes.append(f"batch B={B} L={L} T={T}")

    f, s2, st = _stats(first), _stats(second), _stats(steady)
    print(f"METRIC n_requests={len(first)}")
    print(f"METRIC first_call_p50_ms={f['p50']:.1f}")
    print(f"METRIC first_call_p90_ms={f['p90']:.1f}")
    print(f"METRIC first_call_mean_ms={f['mean']:.1f}")
    print(f"METRIC second_call_p50_ms={s2['p50']:.1f}")
    print(f"METRIC second_call_p90_ms={s2['p90']:.1f}")
    print(f"METRIC steady_p50_ms={st['p50']:.1f}")
    print(f"METRIC steady_p90_ms={st['p90']:.1f}")
    print(f"METRIC steady_mean_ms={st['mean']:.1f}")
    print(f"METRIC load_seconds={load_s:.2f}")
    mx.clear_cache()
    try:
        import subprocess
        rss_kb = int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())]).strip())
    except Exception:
        rss_kb = -1
    print(f"METRIC rss_mb={rss_kb / 1024:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
