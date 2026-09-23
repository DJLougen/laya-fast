"""Measure the cost-model inputs for laya_fast.py's router.

ANE side: per-question forward_one cost per bucket (real prepared items).
MLX side: raw_forward + postprocess cost for batches of n questions at padded
length L, sweeping (n, L).

Output: ane/cost_model.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark
import laya_api
from ane.ane_runtime import LayaANE


def _p50(fn, n=15, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def main():
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)

    report = {"ane_per_question_ms": {}, "mlx_batch_ms": {}}

    ane = LayaANE("converted-fp16", buckets=[64, 80, 96, 128])
    # Real items from batch8_short cover lengths 68..104; synthesize one item per
    # bucket by padding a real short item's ids.
    state, qs = benchmark.make_fixtures()["batch8_short"]
    ids, items, b = ane.prepare(state, qs)
    for L in ane.buckets:
        src = items[0]
        n = min(len(src["ids"]), L)
        item = {"ids": src["ids"][:n], "qtype": src["qtype"],
                "markers": [m for m in src["markers"] if m < n] or [0]}
        report["ane_per_question_ms"][L] = _p50(lambda: ane.forward_one(item))
        print("ane L=%d (len %d): %.2f ms" % (L, len(item["ids"]),
                                              report["ane_per_question_ms"][L]), file=sys.stderr)

    mlx = laya_api.LayaMLX("converted-fp16", dtype="float16", compile=True)

    def sync():
        if mlx.last_raw is not None:
            mx.eval(*mlx.last_raw)

    # MLX batch cost over (n, padded L): reuse real items, tile/truncate ids.
    pad_id = mlx.tok.pad_token_id
    base_items = items  # real prepared items, lens 68..104
    for L in (96, 128, 256, 512):
        for n in (1, 2, 4, 8):
            rows = []
            for i in range(n):
                src = base_items[i % len(base_items)]
                seq = list(src["ids"])
                if len(seq) > L:
                    seq = seq[:L]
                rows.append({"ids": seq + [pad_id] * 0, "markers": src["markers"],
                             "qtype": src["qtype"]})
            batch = laya_api.collate_items(
                [{"ids": (r["ids"] + [pad_id] * (L - len(r["ids"])))[:L],
                  "markers": r["markers"], "qtype": r["qtype"]} for r in rows],
                pad_id)
            def call():
                mlx.raw_forward(batch)
                sync()
            report["mlx_batch_ms"]["%d@%d" % (n, L)] = _p50(call)
            print("mlx n=%d L=%d: %.2f ms" % (n, L, report["mlx_batch_ms"]["%d@%d" % (n, L)]),
                  file=sys.stderr)

    Path("ane/cost_model.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote ane/cost_model.json", file=sys.stderr)


if __name__ == "__main__":
    main()
