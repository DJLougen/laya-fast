"""Autoresearch benchmark for the Laya MLX runtime.

Two jobs, in this order:

  1. Correctness gate (fail-closed). Every fixture is run through the real
     ``LayaMLX.system_one`` entrypoint and the formatted answers are compared
     against a frozen golden reference (``benchmarks/fixtures/autoresearch_golden.json``) produced
     from the baseline runtime. Choice/score labels must match EXACTLY and every
     probability must land within tolerance. A missing or unreadable golden file
     is a hard failure -- the harness never regenerates it implicitly.

  2. Timing. Same deterministic fixtures as ``benchmark.py`` (imported from it,
     so the workload cannot drift), WARMUPS untimed calls then SAMPLES timed
     calls per fixture, each call followed by an ``mx.eval`` device sync.

Primary metric is ``single_short_p50_ms``; p50 is used rather than mean so a
single thermal spike cannot decide a keep/discard.

Usage:
    python benchmarks/bench_autoresearch.py                 # gate + benchmark, emits METRIC lines
    python benchmarks/bench_autoresearch.py --write-golden  # regenerate the golden reference
    python benchmarks/bench_autoresearch.py --quick         # fewer samples, for harness debugging
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import laya_api


ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

GOLDEN_PATH = BENCH_DIR / "fixtures" / "autoresearch_golden.json"

# Decision-equivalence tolerances. Loose enough that a precision change
# (fp32 -> fp16/bf16 compute) can still pass, tight enough that deleted or
# reordered math shows up. Labels are always compared exactly.
PROB_TOL = 2e-2
SCORE_TOL = 5e-2
ACT_TOL = 5e-2



class FixtureResult(TypedDict):
    """run_all output per fixture; also the golden JSON's per-fixture shape."""

    answers: dict[str, dict[str, Any]]
    usage: "laya_api.Usage"
    logits: list[list[float]]
    act: list[list[float]]

class GoldenDoc(TypedDict):
    """Top-level golden reference document."""

    _note: str
    _dtype: str
    _mlx: str
    fixtures: dict[str, FixtureResult]


class TimingStats(TypedDict):
    """Per-fixture timing statistics."""

    p50_ms: float
    min_ms: float
    p90_ms: float
    mean_ms: float
    std_ms: float
    samples: int

def build_agent(model_dir: str, dtype: str, compile_flag: bool) -> Any:
    import laya_api
    return laya_api.LayaMLX(model_dir, dtype=dtype, compile=compile_flag)


def run_all(agent: Any, fixtures: dict[str, tuple[str, dict[str, Any]]]) -> dict[str, FixtureResult]:
    """{fixture: {"answers": ..., "logits": [[...]], "act": [[...]]}} for gating."""
    import mlx.core as mx
    out: dict[str, FixtureResult] = {}
    for name, (state, questions) in fixtures.items():
        res = agent.system_one(state, questions)
        if agent.last_raw is not None:
            mx.eval(*agent.last_raw)
        logits, act = agent.last_raw
        out[name] = {
            "answers": res["answers"],
            "usage": res["usage"],
            "logits": np.asarray(logits, dtype=np.float32).tolist(),
            "act": np.asarray(act, dtype=np.float32).tolist(),
        }
    return out


def compare(golden: dict[str, FixtureResult], current: dict[str, FixtureResult]) -> tuple[list[str], float, float]:
    """Return (failures, max_abs_logit_diff, max_abs_prob_diff)."""
    fails: list[str] = []
    max_logit = 0.0
    max_prob = 0.0

    if sorted(golden) != sorted(current):
        fails.append("fixture set changed: golden=%s current=%s"
                     % (sorted(golden), sorted(current)))
        return fails, float("nan"), float("nan")

    for fx in sorted(golden):
        g, c = golden[fx], current[fx]

        gl = np.asarray(g["logits"], dtype=np.float64)
        cl = np.asarray(c["logits"], dtype=np.float64)
        if gl.shape != cl.shape:
            fails.append("%s: logit shape %s != golden %s" % (fx, cl.shape, gl.shape))
            continue
        if not np.isfinite(cl).all():
            fails.append("%s: non-finite logits" % fx)
            continue
        max_logit = max(max_logit, float(np.abs(gl - cl).max()))

        ga = np.asarray(g["act"], dtype=np.float64)
        ca = np.asarray(c["act"], dtype=np.float64)
        if ga.shape != ca.shape or not np.isfinite(ca).all():
            fails.append("%s: act logits shape/finite mismatch" % fx)
        else:
            # Gate on the consumed quantity: act_probability = softmax(act)[0].
            # Raw act logits run ~4e3 in magnitude, so an absolute tolerance on
            # them would reject a legitimate precision change while a saturated
            # softmax makes the emitted probability identical.
            def _sm0(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                e = np.exp(x - x.max(-1, keepdims=True))
                return (e / e.sum(-1, keepdims=True))[:, 0]
            d = float(np.abs(_sm0(ga) - _sm0(ca)).max())
            if d > PROB_TOL:
                fails.append("%s: act_probability drift %.4g > %.4g" % (fx, d, PROB_TOL))
        if g["usage"]["input_tokens"] != c["usage"]["input_tokens"]:
            fails.append("%s: input_tokens %s != golden %s"
                         % (fx, c["usage"]["input_tokens"], g["usage"]["input_tokens"]))

        if sorted(g["answers"]) != sorted(c["answers"]):
            fails.append("%s: question set changed" % fx)
            continue

        for qid in sorted(g["answers"]):
            ga_, ca_ = g["answers"][qid], c["answers"][qid]
            if ga_["type"] != ca_["type"]:
                fails.append("%s/%s: type %s != %s" % (fx, qid, ca_["type"], ga_["type"]))
                continue
            t = ga_["type"]
            if t == "choice":
                if ga_["choice"] != ca_["choice"]:
                    fails.append("%s/%s: DECISION CHANGED %r -> %r"
                                 % (fx, qid, ga_["choice"], ca_["choice"]))
                for k, gv in ga_["probabilities"].items():
                    cv = ca_["probabilities"].get(k)
                    if cv is None:
                        fails.append("%s/%s: missing probability %r" % (fx, qid, k))
                        continue
                    d = abs(float(gv) - float(cv))
                    max_prob = max(max_prob, d)
                    if d > PROB_TOL:
                        fails.append("%s/%s: p[%s] %.4f vs golden %.4f (tol %.3g)"
                                     % (fx, qid, k, cv, gv, PROB_TOL))
            elif t == "score":
                d = abs(float(ga_["score"]) - float(ca_["score"]))
                max_prob = max(max_prob, d)
                if d > SCORE_TOL:
                    fails.append("%s/%s: score %.4f vs golden %.4f (tol %.3g)"
                                 % (fx, qid, ca_["score"], ga_["score"], SCORE_TOL))
            else:  # noul
                d = abs(float(ga_["noul"]) - float(ca_["noul"]))
                max_prob = max(max_prob, d)
                if d > PROB_TOL:
                    fails.append("%s/%s: noul %.4f vs golden %.4f (tol %.3g)"
                                 % (fx, qid, ca_["noul"], ga_["noul"], PROB_TOL))
    return fails, max_logit, max_prob


def time_fixtures(agent: Any, fixtures: dict[str, tuple[str, dict[str, Any]]], warmups: int, samples: int) -> dict[str, TimingStats]:
    import mlx.core as mx

    def sync() -> None:
        if agent.last_raw is not None:
            mx.eval(*agent.last_raw)

    stats: dict[str, TimingStats] = {}
    for name, (state, questions) in fixtures.items():
        for _ in range(warmups):
            agent.system_one(state, questions)
            sync()
        times: list[float] = []
        for _ in range(samples):
            t0 = time.perf_counter()
            agent.system_one(state, questions)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times)
        stats[name] = {
            "p50_ms": float(np.median(arr)),
            "min_ms": float(arr.min()),
            "p90_ms": float(np.percentile(arr, 90)),
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "samples": len(times),
        }
        print("[bench] %-13s p50 %7.2f ms  min %7.2f  p90 %7.2f  std %5.2f  (n=%d)"
              % (name, stats[name]["p50_ms"], stats[name]["min_ms"],
                 stats[name]["p90_ms"], stats[name]["std_ms"], len(times)),
              file=sys.stderr)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default=os.environ.get("LAYA_AGENT", "fast"),
                    choices=["mlx", "fast"],
                    help="fast (default): LayaFast ANE+MLX router (laya_fast.py). "
                         "mlx: LayaMLX GPU only. Env LAYA_AGENT overrides the default.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default=os.environ.get("LAYA_BENCH_DTYPE", "float32"))
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="2 warmups / 5 samples")
    ap.add_argument("--write-golden", action="store_true",
                    help="regenerate autoresearch_golden.json from the CURRENT code")
    args = ap.parse_args(argv)

    if args.model is None:
        args.model = str(ROOT / ("converted-fp16" if args.agent == "fast" else "converted"))

    if args.quick:
        args.warmups, args.samples = 2, 5

    if not os.path.isdir(args.model):
        print("FATAL: model dir not found: %s" % args.model, file=sys.stderr)
        return 2

    import mlx.core as mx
    mx.random.seed(0)
    np.random.seed(0)
    import benchmark  # type: ignore[import-not-found]  # reason: sibling module resolved via sys.path.insert(BENCH_DIR); fixtures live here, keeps the workload identical to the parity suite
    fixtures = benchmark.make_fixtures()

    t0 = time.perf_counter()
    if args.agent == "fast":
        from laya_fast import LayaFast
        agent: Any = LayaFast(args.model)
    else:
        agent = build_agent(args.model, args.dtype, not args.no_compile)
    load_s = time.perf_counter() - t0

    if args.write_golden:
        current = run_all(agent, fixtures)
        payload: GoldenDoc = {
            "_note": "Frozen decision reference for verify.sh. Regenerate ONLY "
                     "with an explicit, justified --write-golden run.",
            "_dtype": args.dtype,
            "_mlx": getattr(mx, "__version__", "unknown"),
            "fixtures": current,
        }
        GOLDEN_PATH.write_text(json.dumps(payload, indent=1))
        print("wrote %s" % GOLDEN_PATH, file=sys.stderr)
        return 0

    # ---- correctness gate, fail-closed ------------------------------------
    if not GOLDEN_PATH.exists():
        print("FATAL: golden reference missing: %s (run --write-golden from a known-good "
              "baseline)" % GOLDEN_PATH, file=sys.stderr)
        return 3
    try:
        golden = json.loads(GOLDEN_PATH.read_text())["fixtures"]
    except Exception as e:  # noqa: BLE001
        print("FATAL: cannot read golden reference: %s" % e, file=sys.stderr)
        return 3

    current = run_all(agent, fixtures)
    fails, max_logit, max_prob = compare(golden, current)
    if fails:
        print("FATAL: correctness gate FAILED (%d issue(s)):" % len(fails), file=sys.stderr)
        for f in fails[:25]:
            print("  - %s" % f, file=sys.stderr)
        return 4
    print("[gate] PASS  max|dlogit|=%.3e  max|dprob|=%.3e" % (max_logit, max_prob),
          file=sys.stderr)

    # ---- timing ------------------------------------------------------------
    stats = time_fixtures(agent, fixtures, args.warmups, args.samples)

    print("ASI dtype=%s" % args.dtype)
    print("ASI compiled=%s" % (not args.no_compile))
    print("ASI mlx_version=%s" % getattr(mx, "__version__", "unknown"))
    print("ASI samples=%d" % args.samples)
    print("ASI warmups=%d" % args.warmups)

    print("METRIC single_short_p50_ms=%.3f" % stats["single_short"]["p50_ms"])
    print("METRIC batch8_short_p50_ms=%.3f" % stats["batch8_short"]["p50_ms"])
    print("METRIC single_long_p50_ms=%.3f" % stats["single_long"]["p50_ms"])
    print("METRIC batch8_long_p50_ms=%.3f" % stats["batch8_long"]["p50_ms"])
    print("METRIC single_short_min_ms=%.3f" % stats["single_short"]["min_ms"])
    print("METRIC single_short_std_ms=%.3f" % stats["single_short"]["std_ms"])
    print("METRIC total_p50_ms=%.3f" % sum(s["p50_ms"] for s in stats.values()))
    print("METRIC load_seconds=%.3f" % load_s)
    print("METRIC max_abs_logit_diff=%.6g" % max_logit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
