"""ANE runtime for the English Laya model: host embedding lookup + one Core ML
call per question + fp32 host act head. Same system_one output schema as
LayaMLX.

Usage:
    agent = LayaANE("converted-fp16")                  # all exported buckets
    agent = LayaANE("converted-fp16", buckets=[96])    # only L96
    out = agent.system_one(state, questions)

Buckets are fixed-length mlpackages under ane/body<L>/model.mlpackage. Each
question runs on the smallest bucket that fits its true token length; questions
longer than the largest bucket raise ValueError (never silently truncated).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import laya_api
from laya_api import QTYPES, QTYPE_NAMES, temp_bucket, confidence_from_probs

MAX_MARKERS = 32
MASK_NEG = -1e4


def _package_fingerprint(pkg):
    """Cheap content fingerprint: model spec hash + file sizes (weight.bin is
    ~700 MB, so we hash the small spec and trust size+name for the rest)."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(pkg)).encode())
            h.update(str(f.stat().st_size).encode())
            if f.name == "model.mlmodel":
                h.update(f.read_bytes())
    return h.hexdigest()


def _load_compiled(pkg, units):
    """Load a bucket via a cached .mlmodelc next to the package. Compiles once
    (cold), then loads the compiled bundle directly (warm)."""
    import coremltools as ct

    compiled = pkg.parent / "model.mlmodelc"
    stamp = pkg.parent / "model.mlmodelc.sha256"
    if not pkg.exists():  # shipped compiled-only: load the cached bundle as-is
        return ct.models.CompiledMLModel(str(compiled), compute_units=units)
    fp = _package_fingerprint(pkg)
    if not (compiled.exists() and stamp.exists() and stamp.read_text().strip() == fp):
        ct.utils.compile_model(str(pkg), str(compiled))
        stamp.write_text(fp)
    return ct.models.CompiledMLModel(str(compiled), compute_units=units)


class LayaANE:
    def __init__(self, model_dir, buckets=None, compute_units="cpu_ne", ane_dir=None):
        import coremltools as ct
        from safetensors import safe_open

        self.model_dir = Path(model_dir)
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.encoder_cfg = json.loads((self.model_dir / "encoder" / "config.json").read_text())
        self.tok = laya_api._Tokenizer(str(self.model_dir / "tokenizer"))
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.width = int(self.encoder_cfg["hidden_size"])
        self.window = int(self.encoder_cfg.get("local_attention", 128)) // 2

        ane_dir = Path(ane_dir) if ane_dir else Path(__file__).resolve().parent
        units = {
            "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY,
        }[compute_units]
        if buckets is None:
            buckets = sorted({
                int(p.parent.name[4:])
                for p in ane_dir.glob("body*/model.ml*")
                if p.parent.name[4:].isdigit() and p.suffix in (".mlpackage", ".mlmodelc")
            })
        if not buckets:
            raise ValueError("no exported buckets found under %s" % ane_dir)
        self.models = {}
        for L in buckets:
            pkg = ane_dir / ("body%d" % L) / "model.mlpackage"
            if not pkg.exists() and not (pkg.parent / "model.mlmodelc").exists():
                raise ValueError("missing bucket: %s" % pkg.parent)
            self.models[L] = _load_compiled(pkg, units)
        self.buckets = sorted(self.models)
        self.max_len = self.buckets[-1]

        with safe_open(str(self.model_dir / "model.safetensors"), framework="numpy") as w:
            self.embedding = w.get_tensor("encoder.embeddings.tok_embeddings.weight")
            self.type_embedding = w.get_tensor("type_emb.weight")
            self.action = {
                k: w.get_tensor("act_head." + k).astype(np.float32)
                for k in ("0.weight", "0.bias", "2.weight", "2.bias")
            }
        self._erf = np.frompyfunc(math.erf, 1, 1)
        self._windows = {}
        self.last_raw = None

    # ------------------------------------------------------------------ prep
    def _bucket_for(self, n):
        for L in self.buckets:
            if n <= L:
                return L
        raise ValueError("sequence length %d exceeds largest ANE bucket %d" % (n, self.max_len))

    def _window_mask(self, L):
        w = self._windows.get(L)
        if w is None:
            pos = np.arange(L)
            w = np.abs(pos[:, None] - pos[None, :]) <= self.window
            self._windows[L] = w
        return w

    def model_inputs(self, ids, valid, qtype, marker_pos, L):
        """Build the fixed-shape fp16 inputs for one question on bucket L."""
        ids_pad = np.zeros(L, dtype=np.int64)
        ids_pad[: len(ids)] = ids
        valid_pad = np.zeros(L, dtype=bool)
        valid_pad[: len(ids)] = True
        embeddings = self.embedding[ids_pad].T[None, :, None, :]
        full = np.broadcast_to(valid_pad[None, :], (L, L))
        local = (self._window_mask(L) | ~valid_pad[:, None]) & full
        marker_map = np.zeros((1, L, 1, MAX_MARKERS), np.float16)
        marker_map[0, np.asarray(marker_pos), 0, np.arange(len(marker_pos))] = 1
        return {
            "embeddings": np.ascontiguousarray(embeddings, dtype=np.float16),
            "full_mask": np.where(full.T[:, None, :], 0, MASK_NEG).astype(np.float16)[None],
            "local_mask": np.where(local.T[:, None, :], 0, MASK_NEG).astype(np.float16)[None],
            "type_vectors": np.ascontiguousarray(
                self.type_embedding[qtype][None, :, None, None], dtype=np.float16
            ),
            "marker_map": marker_map,
        }

    def forward_one(self, item):
        """Run one prepared item; returns (option_logits[K], act_logits[2]) fp32."""
        n = len(item["ids"])
        L = self._bucket_for(n)
        out = self.models[L].predict(
            self.model_inputs(item["ids"], None, item["qtype"], item["markers"], L)
        )
        logits = next(v for v in out.values() if v.shape[1] == 1).reshape(-1).astype(np.float32)
        pooled = next(v for v in out.values() if v.shape[1] != 1).reshape(-1).astype(np.float32)
        k = len(item["markers"])
        logits[k:] = MASK_NEG
        p = np.exp(logits - logits.max())
        p /= p.sum()
        kk = max(k, 2)
        entropy = -(p * np.log(np.maximum(p, 1e-9))).sum() / math.log(kk)
        top = np.sort(p)[-2:]
        feats = np.array([top[1], top[1] - top[0], entropy, kk / 255.0], np.float32)
        hidden = np.concatenate([pooled, feats]) @ self.action["0.weight"].T + self.action["0.bias"]
        hidden = hidden * (1 + self._erf(hidden / np.sqrt(2)).astype(np.float32)) / 2
        act = hidden @ self.action["2.weight"].T + self.action["2.bias"]
        return logits[:k], act

    # ------------------------------------------------------------------ API
    def prepare(self, state, questions):
        return laya_api.LayaMLX.prepare(self, state, questions)

    def raw_forward(self, batch):
        """Sequential per-question forward; returns (logits [B,K], act [B,2])."""
        raise NotImplementedError("use system_one; per-question buckets differ")

    def system_one(self, state, questions):
        ids, items, b = self.prepare(state, questions)
        answers = {}
        kmax = b["marker_pos"].shape[1]
        raw_logits = np.full((len(ids), kmax), MASK_NEG, np.float32)
        raw_act = np.zeros((len(ids), 2), np.float32)
        for r, qid in enumerate(ids):
            it = items[r]
            logits, act = self.forward_one(it)
            raw_logits[r, :len(logits)] = logits
            raw_act[r] = act
            act_p = float(laya_api._softmax(act[None], -1)[0, 0])
            q = it["q"]
            k = len(it["markers"])
            qt = QTYPES[q["t"]]
            z = logits / self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": act_p}
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
        self.last_raw = (raw_logits, raw_act)
        return {"model": "rl-agent", "answers": answers,
                "usage": {"input_tokens": b["n_tokens"], "output_tokens": 0}}

    predict = system_one
