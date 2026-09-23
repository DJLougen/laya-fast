"""LayaFast: routed runtime — both engines at once.

Holds one LayaMLX (fp16, compiled) and one LayaANE (fixed bucket ladders).
Routing:
  * single question that fits a short ANE bucket (<=128 tokens) -> ANE
    (9.6 ms vs ~14 ms on the GPU). A single long question goes to MLX: the two
    engines tie at 512 tokens, so there is nothing to overlap.
  * multi-question batch -> concurrent split across the two independent engines:
    a worker thread runs k questions sequentially on the ANE while MLX batches
    the rest on the GPU; k minimizes the predicted makespan. This applies to
    LONG questions too — at 512 tokens the ANE and the GPU are equally fast per
    question, so splitting 4/4 on an 8x512 batch measured 216 ms vs 378 ms on
    the GPU alone (1.75x).
  * questions longer than the largest loaded ANE bucket always go to MLX.

Cost model (measured on this M3 Max):
  ANE per-question forward_one: 8.9-9.7 ms for buckets 64-128, 20.4 ms at 256,
  51.7 ms at 512 (see ane/cost_model.json).
  MLX batch ~ 5.5 + 0.088 * (n * L_pad) ms (unpadded fp16 path).

Output schema is identical to LayaMLX.system_one; last_raw is set to
(option_logits [B, K] padded with -1e4, act_logits [B, 2]) as mx arrays so
bench_autoresearch.compare works unchanged.
"""

import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import laya_api
from laya_api import QTYPES, temp_bucket, confidence_from_probs
from ane.ane_runtime import LayaANE

# Measured per-question ANE forward_one cost (ms) by bucket, M3 Max fp16.
# 256/512 are needed for the cross-engine split of long batches (see docstring).
ANE_MS = {64: 8.93, 80: 9.58, 96: 9.62, 128: 9.71, 256: 20.37, 512: 51.69}
# Buckets whose ANE body is worth using for a *single* question. At 256/512 the
# two engines tie, and a single question cannot be overlapped, so the GPU runs it.
SINGLE_MAX_BUCKET = 128
# MLX batch model: ms ~= MLX_INTERCEPT + MLX_PER_TOKEN * n * L_pad.
MLX_INTERCEPT = 5.5
MLX_PER_TOKEN = 0.088

MASK_NEG = -1e4


def _mlx_batch_ms(n, l_pad):
    return MLX_INTERCEPT + MLX_PER_TOKEN * n * l_pad


def _ane_ms(length, buckets):
    for L in buckets:
        if length <= L:
            return ANE_MS.get(L, 9.7)
    return None


class LayaFast:
    def __init__(self, model_dir, ane_buckets=(64, 80, 96, 128, 256, 512), ane_dir=None,
                 compile=True, dtype="float16"):
        # MLX buffer-cache cap is owned by LayaMLX (LAYA_CACHE_LIMIT_MB, 128 MB).
        self.mlx = laya_api.LayaMLX(model_dir, dtype=dtype, compile=compile)
        # Missing compiled bodies are not an error. Those lengths stay on the GPU.
        ane_root = Path(ane_dir) if ane_dir else Path(__file__).resolve().parent / "ane"
        present = []
        for L in ane_buckets:
            folder = ane_root / ("body%d" % L)
            if (folder / "model.mlmodelc").exists() or (folder / "model.mlpackage").exists():
                present.append(L)
        self.ane = LayaANE(model_dir, buckets=present, ane_dir=ane_root) if present else None
        self.tok = self.mlx.tok
        self.cfg = self.mlx.cfg
        self.temperature = self.mlx.temperature
        self.temperature_by_options = self.mlx.temperature_by_options
        self.last_raw = None

    # ------------------------------------------------------------------ prep
    def prepare(self, state, questions):
        return self.mlx.prepare(state, questions)

    # ------------------------------------------------------------------ format
    def _format(self, q, k, logits, act_row):
        """One question's formatted answer from raw logits[k] + act logits."""
        qt = QTYPES[q["t"]]
        z = np.asarray(logits[:k], np.float32) / self.temperature_by_options.get(
            temp_bucket(qt, k), self.temperature[qt])
        p = np.exp(z - z.max())
        p = p / p.sum()
        e = np.exp(act_row - act_row.max())
        act_p = float((e / e.sum())[0])
        ext = {"act_probability": act_p}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {"type": "choice", "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
        if q["t"] == "score":
            return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
        return {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}

    # ------------------------------------------------------------------ paths
    def _ane_rows(self, rows, items, kmax, raw_logits, raw_act, answers, ids):
        for r in rows:
            it = items[r]
            logits, act = self.ane.forward_one(it)
            raw_logits[r, :len(logits)] = logits
            raw_act[r] = act
            answers[ids[r]] = self._format(it["q"], len(it["markers"]), logits, act)

    def _mlx_rows(self, rows, items, pad_id, raw_logits, raw_act, answers, ids):
        sub = [items[r] for r in rows]
        b = laya_api.collate_items(sub, pad_id)
        logits, act = self.mlx.raw_forward(b)
        for j, r in enumerate(rows):
            it = items[r]
            k = len(it["markers"])
            raw_logits[r, :k] = logits[j, :k]
            raw_act[r] = act[j]
            answers[ids[r]] = self._format(it["q"], k, logits[j], act[j])

    def system_one(self, state, questions):
        import mlx.core as mx

        ids, items, b = self.prepare(state, questions)
        n = len(items)
        kmax = b["marker_pos"].shape[1]
        raw_logits = np.full((n, kmax), MASK_NEG, np.float32)
        raw_act = np.zeros((n, 2), np.float32)
        answers = {}
        pad_id = self.tok.pad_token_id

        eligible = []
        if self.ane is not None:
            eligible = [r for r in range(n)
                        if _ane_ms(len(items[r]["ids"]), self.ane.buckets) is not None]

        if n == 1:
            # A lone question cannot be overlapped, so only take the ANE when its
            # body is actually faster (short buckets); long singles stay on MLX.
            if eligible and len(items[0]["ids"]) <= SINGLE_MAX_BUCKET:
                self._ane_rows([0], items, kmax, raw_logits, raw_act, answers, ids)
            else:
                self._mlx_rows([0], items, pad_id, raw_logits, raw_act, answers, ids)
        elif not eligible:
            self._mlx_rows(list(range(n)), items, pad_id, raw_logits, raw_act, answers, ids)
        else:
            # Choose split k minimizing makespan: ANE gets the longest eligible
            # questions first (removing them shrinks MLX's padded L the most).
            by_len = sorted(eligible, key=lambda r: -len(items[r]["ids"]))
            best = None
            for k in range(0, len(by_len) + 1):
                ane_rows = by_len[:k]
                mlx_rows = [r for r in range(n) if r not in set(ane_rows)]
                t_ane = sum(_ane_ms(len(items[r]["ids"]), self.ane.buckets)
                            for r in ane_rows)
                if mlx_rows:
                    l_pad = max(len(items[r]["ids"]) for r in mlx_rows)
                    t_mlx = _mlx_batch_ms(len(mlx_rows), l_pad)
                else:
                    t_mlx = 0.0
                makespan = max(t_ane, t_mlx)
                if best is None or makespan < best[0]:
                    best = (makespan, ane_rows, mlx_rows)
            _, ane_rows, mlx_rows = best
            if not ane_rows:
                self._mlx_rows(mlx_rows, items, pad_id, raw_logits, raw_act, answers, ids)
            elif not mlx_rows:
                self._ane_rows(ane_rows, items, kmax, raw_logits, raw_act, answers, ids)
            else:
                err = []

                def run_ane():
                    try:
                        self._ane_rows(ane_rows, items, kmax, raw_logits, raw_act,
                                       answers, ids)
                    except Exception as e:  # noqa: BLE001
                        err.append(e)

                t = threading.Thread(target=run_ane)
                t.start()
                try:
                    self._mlx_rows(mlx_rows, items, pad_id, raw_logits, raw_act,
                                   answers, ids)
                except Exception as e:  # noqa: BLE001
                    err.append(e)
                t.join()
                if err:
                    raise err[0]

        self.last_raw = (mx.array(raw_logits), mx.array(raw_act))
        return {"model": "rl-agent", "answers": answers,
                "usage": {"input_tokens": b["n_tokens"], "output_tokens": 0}}

    predict = system_one
