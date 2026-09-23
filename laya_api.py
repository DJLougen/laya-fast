"""Jev-compatible inference for the MLX port of the RL Agent (Laya) model.

LayaMLX mirrors source/rl_agent_api.py's RLAgent exactly: same sequence layout,
same calibrated post-processing, same response shape. Token preparation is pure
`tokenizers` + numpy (no torch required); the forward pass goes through
laya_mlx.load_model.

Usage:
    agent = LayaMLX("converted")            # dir produced by convert.py
    out = agent.system_one(state, questions)  # Jev request/response shape
    out = agent.predict(state, questions)     # alias

CLI:
    python laya_api.py --model converted --input request.json [--output out.json]
    python laya_api.py --model converted --state "..." --questions questions.json
    echo '{"state": ..., "questions": {...}}' | python laya_api.py --model converted

Request JSON: {"state": <str|obj>, "questions": {id: {"type": "choice"|"score"|"noul",
"instructions": ..., "criteria": ...}}}. Response is the Jev-shaped dict printed as JSON.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


# ----------------------------------------------------------------------------- tokenizer
class _Tokenizer:
    """Minimal wrapper over tokenizers.Tokenizer matching the AutoTokenizer surface
    that build_sequence uses (special-token ids + encode without special tokens)."""

    def __init__(self, tok_dir):
        from tokenizers import Tokenizer
        self._tk = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        cfg = {}
        cfg_path = os.path.join(tok_dir, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        # ModernBERT defaults; overridden by tokenizer_config.json when present.
        self.mask_token = cfg.get("mask_token", "[MASK]")
        self.mask_token_id = self._id(self.mask_token, 50284)
        self.cls_token_id = self._id(cfg.get("cls_token", "[CLS]"), 50281)
        self.sep_token_id = self._id(cfg.get("sep_token", "[SEP]"), 50282)
        self.pad_token_id = self._id(cfg.get("pad_token", "[PAD]"), 50283)

    def _id(self, token, default):
        tid = self._tk.token_to_id(token)
        return int(tid) if tid is not None else default

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._tk.encode(text, add_special_tokens=add_special_tokens).ids}


# ----------------------------------------------------------------------------- rendering (exact port of rl_common.py)
def serialize_state(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q):
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q, max_len, head_max_len, state_ids=None):
    """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    Returns input_ids and the positions of the per-option [MASK] markers.
    Inference-only port of rl_common.build_sequence (no option_order / truncate_left).
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in range(len(opts)):
        opt_ids.append([tok.mask_token_id] + tok(" " + opts[i].replace(mask_tok, " "),
                                               add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = tok(serialize_state(state).replace(mask_tok, " "),
                        add_special_tokens=False)["input_ids"]
    st = state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def collate_items(items, pad_id):
    """Numpy port of rl_common.collate_items for the inference fields."""
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = it["ids"]
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = it["markers"]
        mmask[i, :k] = True
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
            "n_tokens": int(att.sum())}


def temp_bucket(qtype, k):
    """Key for per-cardinality temperature fitting."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def confidence_from_probs(p, k):
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def _to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": t,
            "ins": qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"]),
            "crit": crit}


# ----------------------------------------------------------------------------- shape bucketing
# mx.compile specializes on input shapes: every distinct (B, L, K) or packed
# (B, L, T) costs one retrace (~50-100 ms padded, ~0.5-1 s packed) on first use.
# Rounding the varying dims to fixed buckets bounds the number of compiled
# variants; pads are masked keys appended at the END so real-token RoPE
# positions and marker indices are unchanged (decision-identical).
_L_BUCKET = 16   # sequence length buckets: multiples of 16 (32 shapes over max_len=512)
_K_BUCKET = 8    # marker-count buckets: multiples of 8
_T_BUCKET = 64   # packed-row buckets: multiples of 64 (one GEMM tile row)


def _round_up(n, m):
    return ((int(n) + m - 1) // m) * m


def _bucket_batch(batch, pad_id):
    """Right-pad input_ids/attention_mask to L_bucket and marker arrays to
    K_bucket. Returns (batch, k_orig); caller slices logits[:, :k_orig]."""
    ids = batch["input_ids"]
    B, L = ids.shape
    Lb = _round_up(L, _L_BUCKET)
    K = batch["marker_pos"].shape[1]
    Kb = _round_up(K, _K_BUCKET)
    if Lb == L and Kb == K:
        return batch, K
    out = dict(batch)
    if Lb != L:
        ids2 = np.full((B, Lb), pad_id, dtype=ids.dtype)
        ids2[:, :L] = ids
        att2 = np.zeros((B, Lb), dtype=batch["attention_mask"].dtype)
        att2[:, :L] = batch["attention_mask"]
        out["input_ids"], out["attention_mask"] = ids2, att2
    if Kb != K:
        mp2 = np.zeros((B, Kb), dtype=batch["marker_pos"].dtype)  # pad -> CLS pos, masked
        mp2[:, :K] = batch["marker_pos"]
        mm2 = np.zeros((B, Kb), dtype=batch["marker_mask"].dtype)
        mm2[:, :K] = batch["marker_mask"]
        out["marker_pos"], out["marker_mask"] = mp2, mm2
    return out, K


def _bucket_pack(pack):
    """Pad packed rows to a T_bucket multiple with dummy rows that alias packed
    row 0 (seq 0's CLS token). Their outputs are computed and discarded; no
    padded slot or marker references them, so results are unchanged."""
    T = int(pack["flat_idx"].shape[0])
    Tb = _round_up(T, _T_BUCKET)
    if Tb == T:
        return pack
    out = dict(pack)
    out["flat_idx"] = np.concatenate(
        [pack["flat_idx"], np.zeros(Tb - T, dtype=np.int32)])
    out["seq_of"] = np.concatenate(
        [pack["seq_of"], np.zeros(Tb - T, dtype=np.int32)])
    return out


# ----------------------------------------------------------------------------- agent
class LayaMLX:
    """MLX twin of rl_agent_api.RLAgent. Same request/response contract as Jev's system_one."""

    def __init__(self, model_dir, dtype="float32", compile=False, unpad=None,
                 bucket=None, cold_dispatch=None, compile_packed=None, warmup=False):
        import mlx.core as mx
        from laya_mlx import load_model, _resolve_model_dir
        # Bound MLX's freed-buffer cache. MLX's default cache ceiling on this
        # 36 GB Mac is 36.7 GB (measured), so variable-length traffic keeps
        # every size of buffer it ever used. Measured (interleaved cache-limit
        # sweep over this repo's benchmark fixtures): a 128 MB cap matches 1 GB
        # latency on all 4 fixtures (single_short 13.88 vs 13.89 ms) and cuts
        # process footprint from 2054 MB to 1154 MB; 0 MB costs ~4% on
        # single_short.
        # LAYA_CACHE_LIMIT_MB=-1 leaves MLX's default untouched.
        _cache_mb = int(os.environ.get("LAYA_CACHE_LIMIT_MB", "128"))
        if _cache_mb >= 0:
            mx.set_cache_limit(_cache_mb << 20)
        model_dir = _resolve_model_dir(model_dir)
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        self.tok = _Tokenizer(os.path.join(model_dir, "tokenizer"))
        self.model = load_model(model_dir, dtype=dtype)
        self.compiled = bool(compile)
        self._forward_fn = mx.compile(self.model) if self.compiled else self.model
        # Unpadded (varlen) forward for batches with padding. Default on;
        # LAYA_UNPAD=0 or unpad=False forces the old padded path (A/B flag).
        self.unpad = (os.environ.get("LAYA_UNPAD", "1") != "0") if unpad is None else bool(unpad)
        # Compiling the packed path buys ~nothing steady-state (measured parity
        # with eager) but each new (B, L, T) costs a ~0.5-1 s retrace, so it is
        # OFF by default; LAYA_COMPILE_PACKED=1 restores the old behaviour.
        self.compile_packed = (os.environ.get("LAYA_COMPILE_PACKED", "0") == "1") \
            if compile_packed is None else bool(compile_packed)
        self._forward_packed_fn = (mx.compile(self.model.forward_packed)
                                   if self.compiled and self.compile_packed
                                   else self.model.forward_packed)
        # Shape bucketing (LAYA_BUCKET=0 restores exact-shape behaviour) and
        # cold dispatch (LAYA_COLD_DISPATCH=0 restores compile-on-first-call).
        self.bucket = (os.environ.get("LAYA_BUCKET", "1") != "0") if bucket is None else bool(bucket)
        self.cold_dispatch = (os.environ.get("LAYA_COLD_DISPATCH", "1") != "0") \
            if cold_dispatch is None else bool(cold_dispatch)
        self._seen_shapes = set()  # bucketed shape keys already sent to mx.compile
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.dtype = dtype
        self.last_raw = None  # (option_logits, act_logits) mx arrays from the last forward
        if warmup:
            self.warmup()

    def warmup(self, l_buckets=None, k_buckets=None, t_buckets=None, batch_size=4):
        """Pre-compile every bucketed shape at load so no live request pays a
        retrace. Off by default; enable via LayaMLX(..., warmup=True).

        Warms the padded path for each (L, K) bucket and, when compile_packed
        is on, the packed path for each (L, T) bucket at ``batch_size``.
        Returns the total warmup time in seconds.
        """
        import time as _time
        import mlx.core as mx
        max_len = int(self.cfg["max_len"])
        if l_buckets is None:
            l_buckets = list(range(_L_BUCKET, max_len + 1, _L_BUCKET))
        if k_buckets is None:
            k_buckets = [_K_BUCKET, 2 * _K_BUCKET, 3 * _K_BUCKET, 4 * _K_BUCKET]
        t0 = _time.perf_counter()
        pad_id = self.tok.pad_token_id
        for L in l_buckets:
            for K in k_buckets:
                b = {"input_ids": np.full((1, L), pad_id, dtype=np.int64),
                     "attention_mask": np.ones((1, L), dtype=np.int64),
                     "marker_pos": np.zeros((1, K), dtype=np.int64),
                     "marker_mask": np.ones((1, K), dtype=bool),
                     "qtype": np.zeros(1, dtype=np.int64)}
                logits, act = self._forward_fn(
                    mx.array(b["input_ids"]), mx.array(b["attention_mask"]),
                    mx.array(b["marker_pos"]), mx.array(b["marker_mask"]),
                    mx.array(b["qtype"]))
                mx.eval(logits, act)
                self._seen_shapes.add(("pad", 1, L, K))
        if self.compiled and self.compile_packed:
            B = int(batch_size)
            if t_buckets is None:
                t_buckets = list(range(_T_BUCKET, B * max_len + 1, _T_BUCKET))
            for L in l_buckets:
                for T in t_buckets:
                    if T > B * L:
                        continue
                    att = np.zeros((B, L), dtype=np.int64)
                    att.reshape(-1)[:T] = 1  # T real tokens in row-major order
                    b = {"input_ids": np.full((B, L), pad_id, dtype=np.int64),
                         "attention_mask": att,
                         "marker_pos": np.zeros((B, _K_BUCKET), dtype=np.int64),
                         "marker_mask": np.ones((B, _K_BUCKET), dtype=bool),
                         "qtype": np.zeros(B, dtype=np.int64)}
                    pack = _pack_maps(b)
                    if pack is None:
                        continue
                    pack = _bucket_pack(pack)
                    logits, act = self._forward_packed_fn(
                        mx.array(b["input_ids"]), mx.array(b["attention_mask"]),
                        mx.array(b["marker_pos"]), mx.array(b["marker_mask"]),
                        mx.array(b["qtype"]),
                        {k: mx.array(v) for k, v in pack.items()})
                    mx.eval(logits, act)
                    self._seen_shapes.add(("pack", B, L, int(pack["flat_idx"].shape[0]),
                                           _K_BUCKET))
        return _time.perf_counter() - t0

    def prepare(self, state, questions):
        """Tokenize + collate a Jev request into the model's numpy batch (no inference).

        The shared state is serialized+tokenized once and reused for every
        question (state_ids is sliced per question inside build_sequence).
        """
        ids, items = list(questions.keys()), []
        state_ids = self.tok(serialize_state(state).replace(self.tok.mask_token, " "),
                             add_special_tokens=False)["input_ids"]
        for qid in ids:
            q = _to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, self.cfg["max_len"],
                                          self.cfg["head_max_len"], state_ids=state_ids)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r: options do not fit in head_max_len=%d tokens"
                                 % (qid, self.cfg["head_max_len"]))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "q": q})
        return ids, items, collate_items(items, self.tok.pad_token_id)

    def raw_forward(self, batch):
        """Run the MLX model on a prepared batch; returns (option_logits, act_logits) numpy arrays."""
        import mlx.core as mx
        pack = _pack_maps(batch) if self.unpad else None
        k_orig = batch["marker_pos"].shape[1]
        if self.bucket:
            batch, k_orig = _bucket_batch(batch, self.tok.pad_token_id)
            if pack is not None:
                # Rebuild pack on the padded layout; if the original batch had
                # no padding (equal lengths), keep it on the padded path so
                # bucketing never changes which path serves a request.
                pack = _pack_maps(batch)
                if pack is not None and self.compile_packed:
                    pack = _bucket_pack(pack)
        if pack is not None:
            key = ("pack", batch["input_ids"].shape[0], batch["input_ids"].shape[1],
                   int(pack["flat_idx"].shape[0]), batch["marker_pos"].shape[1])
            fn = self._forward_packed_fn
            if self.cold_dispatch and self.compiled and self.compile_packed \
                    and key not in self._seen_shapes:
                fn = self.model.forward_packed  # eager: skip the retrace spike
            self._seen_shapes.add(key)
            logits, act = fn(
                mx.array(batch["input_ids"]), mx.array(batch["attention_mask"]),
                mx.array(batch["marker_pos"]), mx.array(batch["marker_mask"]),
                mx.array(batch["qtype"]),
                {k: mx.array(v) for k, v in pack.items()})
        else:
            key = ("pad", batch["input_ids"].shape[0], batch["input_ids"].shape[1],
                   batch["marker_pos"].shape[1])
            fn = self._forward_fn
            if self.cold_dispatch and self.compiled and key not in self._seen_shapes:
                fn = self.model  # eager: skip the retrace spike
            self._seen_shapes.add(key)
            logits, act = fn(mx.array(batch["input_ids"]), mx.array(batch["attention_mask"]),
                             mx.array(batch["marker_pos"]), mx.array(batch["marker_mask"]),
                             mx.array(batch["qtype"]))
        if logits.shape[1] != k_orig:
            logits = logits[:, :k_orig]  # drop bucketed marker pads (all -1e4)
        self.last_raw = (logits, act)
        mx.eval(logits, act)
        return np.asarray(logits, dtype=np.float32), np.asarray(act, dtype=np.float32)


    def system_one(self, state, questions):
        """questions: {id: {"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}} (Jev request shape)."""
        ids, items, b = self.prepare(state, questions)
        logits, act = self.raw_forward(b)
        act = _softmax(act, -1)
        answers, n_tokens = {}, b["n_tokens"]
        for r, qid in enumerate(ids):
            q = items[r]["q"]
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            z = logits[r, :k] / self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": float(act[r, 0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
        return {"model": "rl-agent", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}}

    predict = system_one


def _pack_maps(batch):
    """Index maps for the unpadded (varlen) forward, or None when not eligible.

    Eligible: B > 1 and the batch actually has padding (T < B*L). All arrays are
    int32 numpy; laya_mlx.Model.forward_packed documents their semantics.
    """
    att = np.asarray(batch["attention_mask"])
    B, L = att.shape
    if B <= 1:
        return None
    flat = att.reshape(-1)
    T = int(flat.sum())
    if T >= B * L:
        return None
    flat_idx = np.flatnonzero(flat).astype(np.int32)          # [T]
    slot_src = np.zeros(B * L, dtype=np.int32)
    slot_src[flat_idx] = np.arange(T, dtype=np.int32)         # pads -> row 0
    seq_of = (flat_idx // L).astype(np.int32)                 # [T]
    cls_idx = slot_src[np.arange(B, dtype=np.int64) * L].astype(np.int32)  # CLS is slot 0
    mpos = np.clip(np.asarray(batch["marker_pos"]), 0, L - 1).astype(np.int64)
    marker_idx = slot_src[np.arange(B)[:, None] * L + mpos].astype(np.int32)
    return {"flat_idx": flat_idx, "slot_src": slot_src, "seq_of": seq_of,
            "cls_idx": cls_idx, "marker_idx": marker_idx}


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------------- CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(
        prog="laya_api.py",
        description="Run the MLX RL Agent (Laya) on one Jev-style request and print the JSON response.",
        epilog="Request JSON shape: {\"state\": <str|object>, \"questions\": {id: {\"type\": "
               "\"choice\"|\"score\"|\"noul\", \"instructions\": ..., \"criteria\": ...}}}. "
               "choice criteria may be a {label: description} object or a list of labels; "
               "score criteria is a list of level descriptions; noul criteria may be "
               "{\"false\": ..., \"true\": ...} or omitted.")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "converted-fp16"),
                    help="converted model directory (default: converted-fp16 next to this file)")
    ap.add_argument("--agent", default=os.environ.get("LAYA_AGENT", "mlx"),
                    choices=["fast", "mlx"],
                    help="mlx: GPU-only MLX path, works on a fresh checkout "
                         "with no compiled ANE bodies. fast: LayaFast ANE+MLX "
                         "router for optional acceleration; falls back to MLX "
                         "for lengths without compiled bodies. Override with "
                         "LAYA_AGENT (default: %(default)s).")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16"],
                    help="MLX compute dtype (default: float16; fast requires float16)")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable mx.compile on the MLX path")
    ap.add_argument("--input", default=None,
                    help="request JSON file; omit or pass '-' to read stdin")
    ap.add_argument("--state", default=None, help="state string (use with --questions)")
    ap.add_argument("--questions", default=None, help="questions JSON file (use with --state)")
    ap.add_argument("--output", default=None, help="write response JSON here instead of stdout")
    args = ap.parse_args(argv)

    if args.input and args.input != "-":
        with open(args.input) as f:
            req = json.load(f)
    elif args.state is not None and args.questions:
        with open(args.questions) as f:
            req = {"state": args.state, "questions": json.load(f)}
    else:
        req = json.load(sys.stdin)
    if args.agent == "fast":
        from laya_fast import LayaFast
        agent = LayaFast(args.model, dtype=args.dtype, compile=not args.no_compile)
    else:
        agent = LayaMLX(args.model, dtype=args.dtype, compile=not args.no_compile)
    out = agent.system_one(req["state"], req["questions"])
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
