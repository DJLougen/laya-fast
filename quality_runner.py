#!/usr/bin/env python3
"""Raw-output quality runner for the Laya-vs-Jev comparison.

Runs the frozen 105-case suite (quality_suite.py, byte-identical copy of
jev-gliner-x-post/suite_large.py, sha256 pinned in quality_protocol.json) against
one of four runtimes and writes raw, ungraded outputs:

  --runtime jev   TypeSafe Jev via POST https://api.typesafe.ai/v1/systemone,
                  model jev-latest (httpx, one connection, bounded 429 retry).
  --runtime mlx   Local converted checkpoint via laya_api.LayaMLX(model_dir, dtype).
  --runtime cpu   Original source checkpoint via source/rl_agent_api.RLAgent on CPU.
  --runtime mps   Same RLAgent on Apple MPS. Fails hard if MPS unavailable;
                  there is NO silent device fallback.

Ground-truth provenance: the suite is synthetic and agent-authored (the original
file's "no model generated GT" comment means no *model* wrote the labels; the
authoring agent did). Labels are used ONLY to build candidate lists for the
entities family (per-candidate noul questions, exactly as the original
run_jev.py did) and for the records family's pre-declared negative probes. No
label text is embedded in any question. This runner never grades.

Calibration gate (fail-closed, all runtimes): a hidden-fair-coin noul must land
in [0.35, 0.65] and a sealed-envelope choice must be nondegenerate; for jev the
returned model name must start with "jev". The actual probe request/response is
saved in the output document.

Caps: at most 110 attempts total (for jev every HTTP POST counts, including 429 retries; for
local runtimes each model call counts once) and at most 100k cumulative input tokens. The
attempt cap is enforced exactly before every call. The token cap is enforced on the consumed
total, and for local runtimes additionally on a per-case projection from tokenized sequence
lengths (prep.seq_lens); for jev no local tokenizer exists, so projection is unavailable and a
single case may carry the total past 100k before the next check stops the run. Cost is null
(no verified price).

Usage:
  python quality_runner.py --runtime jev --output results_jev.json
  python quality_runner.py --runtime mlx --model converted --dtype float32 --output results_mlx.json
  python quality_runner.py --runtime cpu --source source --output results_cpu.json
  python quality_runner.py --runtime mps --source source --output results_mps.json
  python quality_runner.py --dump-requests quality_requests.json   # no inference
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from quality_suite import (TASKS, ACTION_LABELS, COND_LABELS, COND_PLAIN, ASPECT_LABELS,
                           ENT_LABELS, SEV_LEVELS, CONFIRM_RULE)

JEV_API = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_TIMEOUT_S = 180.0
JEV_429_RETRIES = 4          # bounded; only 429 is retried, everything else fails in place
JEV_429_BASE_DELAY_S = 2.0
TORCH_THREADS = 4

MAX_REQUESTS = 110           # attempt cap: every HTTP POST (incl. 429 retries) for jev;
                             # one model call per logical request for local runtimes
MAX_INPUT_TOKENS = 100_000   # cumulative input-token cap; checked BEFORE each request

SUITE_FILE = os.path.join(HERE, "quality_suite.py")

SUITE_PROVENANCE = (
    "Synthetic agent-authored suite: all 105 items and gold labels were hand-written by the "
    "implementing agent on 2026-09-18 (items author = ground-truth author; no model generated "
    "the ground truth, but no independent human annotation exists either). Previously used for "
    "a GLiNER comparison; no Laya-specific tuning. Calibration limits: small correlated sample, "
    "candidate sets for entities/records include gold strings by design (verification, not "
    "extraction), six predeclared ambiguous labels are excluded only at grading time by the "
    "parent - every request is still issued unchanged."
)

# Frozen expected case IDs, in suite order. Self-contained: the runner does NOT read
# quality_expected.json (that file carries labels; this runner must not import them).
EXPECTED_IDS = ([f"route_{i:02d}" for i in range(1, 16)]
                + [f"cond_{i:02d}" for i in range(1, 16)]
                + [f"asp_{i:02d}" for i in range(1, 16)]
                + [f"ent_{i:02d}" for i in range(1, 16)]
                + [f"sev_{i:02d}" for i in range(1, 16)]
                + [f"ver_{i:02d}" for i in range(1, 16)]
                + [f"rec_{i:02d}" for i in range(1, 16)])
EXPECTED_FAMILY_COUNTS = {"route": 15, "conditions": 15, "aspects": 15,
                          "entities": 15, "severity": 15, "verify": 15, "records": 15}


# --------------------------------------------------------------------------- questions
# Ported verbatim from jev-gliner-x-post/run_jev.py build(); question IDs must match
# quality_expected.json exactly (380 decisions across 105 cases).

def noul(instr):
    return {"type": "noul", "instructions": instr}


def choice(instr, criteria):
    return {"type": "choice", "instructions": instr, "criteria": criteria}


def build(item):
    fam, inp = item["family"], item["input"]
    if fam == "route":
        state = inp + " | " + CONFIRM_RULE
        qs = {
            "action": choice("Which single action does the request ask for?",
                             {a: a for a in ACTION_LABELS}),
            "confirm": noul(CONFIRM_RULE),
        }
    elif fam == "conditions":
        state = inp
        qs = {f"cond_{l}": noul(f"Does the message show {COND_PLAIN[l]}?") for l in COND_LABELS}
    elif fam == "aspects":
        state = inp
        qs = {f"aspect_{l}": noul(f"Does the review give an opinion on {l}?") for l in ASPECT_LABELS}
    elif fam == "entities":
        state = inp
        qs = {}
        for l in ENT_LABELS:
            cands = sorted(set(item["gt"][l]) | set(item["distractors"][l]))
            for c in cands:
                qs[f"has_{l}__{c}"] = noul(
                    f'Is "{c}" used in the text as the proper name of a specific {l} '
                    f'(exact wording - not a description, role word, or partial form)?')
    elif fam == "severity":
        state = inp
        qs = {"severity": {"type": "score",
                           "instructions": "How severe is this bug for the business?",
                           "criteria": SEV_LEVELS}}
    elif fam == "verify":
        state = inp  # object state; serialized by the backend
        qs = {"supported": noul("Is the claim supported by ALL the evidence fields?")}
    elif fam == "records":
        state = inp
        qs = {}
        for j, (b, it) in enumerate(item["gt"]["pairs"], 1):
            qs[f"pair_{j}"] = noul(f"Does the text state that {b} bought or acquired {it}?")
        for j, (b, it) in enumerate(item["gt"].get("negative_probes", []), 1):
            qs[f"neg_{j}"] = noul(f"Does the text state that {b} bought or acquired {it}?")
        if not item["gt"]["pairs"]:
            qs["any_purchase"] = noul(
                "Does the text describe any specific person or company buying or acquiring a specific item?")
    else:
        raise ValueError(fam)
    return {"state": state, "model": JEV_MODEL, "questions": qs}


def calibration_probe_body():
    return {
        "state": "A fair coin was flipped and the result is hidden.",
        "model": JEV_MODEL,
        "questions": {
            "heads": noul("Did the hidden coin flip land heads?"),
            "pick": choice("Which of two identical sealed envelopes holds the prize?",
                           {"left": "the left envelope", "right": "the right envelope"}),
        },
    }

# --------------------------------------------------------------------------- suite checks

def suite_sha256():
    h = hashlib.sha256()
    with open(SUITE_FILE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_suite():
    """Fail-closed structural checks against the frozen contract: exact ordered case IDs,
    15 per family, build() succeeds for every item, and every question has a known type.
    Self-contained: reads only quality_suite.py, never the label-bearing expected file."""
    problems = []
    ids = [t["id"] for t in TASKS]
    if ids != EXPECTED_IDS:
        problems.append(f"case IDs differ from frozen expectation "
                        f"(missing={sorted(set(EXPECTED_IDS) - set(ids))}, "
                        f"extra={sorted(set(ids) - set(EXPECTED_IDS))}, "
                        f"ordered_match={ids == EXPECTED_IDS})")
    fams = {}
    for t in TASKS:
        fams[t["family"]] = fams.get(t["family"], 0) + 1
    if fams != EXPECTED_FAMILY_COUNTS:
        problems.append(f"family counts {fams} != {EXPECTED_FAMILY_COUNTS}")

    built = {}
    for t in TASKS:
        try:
            built[t["id"]] = build(t)
        except Exception as e:
            problems.append(f"{t['id']}: build() raised {type(e).__name__}: {e}")
    if problems:
        return problems, built

    for t in TASKS:
        for qid, q in built[t["id"]]["questions"].items():
            if q["type"] not in ("noul", "choice", "score"):
                problems.append(f"{t['id']}/{qid}: unknown type {q['type']}")
    return problems, built


# --------------------------------------------------------------------------- credentials

def load_jev_key():
    """TYPESAFE_API_KEY from env, else a literal `export TYPESAFE_API_KEY=...` line parsed out
    of ~/.zshrc as text (never executed, never printed). No other credential files are read."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key, "env:TYPESAFE_API_KEY"
    zshrc = os.path.expanduser("~/.zshrc")
    try:
        with open(zshrc, "r", errors="replace") as f:
            for line in f:
                m = re.match(r'^\s*export\s+TYPESAFE_API_KEY\s*=\s*(?P<q>["\']?)(?P<v>[^"\'\s#]+)(?P=q)\s*(?:#.*)?$',
                             line)
                if m:
                    return m.group("v"), "file:~/.zshrc(literal export line)"
    except FileNotFoundError:
        pass
    except OSError as e:
        raise SystemExit(f"cannot read ~/.zshrc for TYPESAFE_API_KEY: {e}")
    raise SystemExit("TYPESAFE_API_KEY not found in environment or as a literal export in ~/.zshrc; "
                     "refusing to run jev runtime without credentials")


# --------------------------------------------------------------------------- backends

class JevBackend:
    name = "jev"

    def __init__(self):
        import httpx
        key, self.key_source = load_jev_key()
        self.client = httpx.Client(
            timeout=JEV_TIMEOUT_S,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self.attempts = 0

    def system_one(self, state, questions, budget=None):
        """One logical request. `budget` = remaining HTTP attempts allowed under the cap;
        each POST consumes one. Only 429 is retried, and only while budget remains."""
        body = {"state": state, "model": JEV_MODEL, "questions": questions}
        delay = JEV_429_BASE_DELAY_S
        for attempt in range(JEV_429_RETRIES + 1):
            if budget is not None and budget <= 0:
                return -1, {"error": "attempt budget exhausted"}
            if budget is not None:
                budget -= 1
            self.attempts += 1
            try:
                r = self.client.post(JEV_API, content=json.dumps(body))
            except Exception as e:
                return -1, {"error": f"{type(e).__name__}: {e}"}
            if r.status_code == 429 and attempt < JEV_429_RETRIES and (budget is None or budget > 0):
                time.sleep(delay)
                delay *= 2
                continue
            try:
                payload = r.json()
            except Exception:
                payload = {"body": r.text[:800]}
            if r.status_code != 200:
                return r.status_code, {"http_error": r.status_code,
                                       "body": payload if isinstance(payload, dict) else str(payload)[:800]}
            return r.status_code, payload
        return -1, {"error": "exhausted 429 retries"}

    def prep_info(self, state, questions):
        # Jev tokenization is server-side; no local tokenizer is assumed.
        return None

    def close(self):
        self.client.close()


class MLXBackend:
    name = "mlx"

    def __init__(self, model_dir, dtype):
        from laya_api import LayaMLX
        self.agent = LayaMLX(model_dir, dtype=dtype)
        self.model_dir = model_dir
        self.dtype = dtype
        self._max_len = self._read_max_len(model_dir)
        self.attempts = 0

    @staticmethod
    def _read_max_len(model_dir):
        try:
            with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
                return json.load(f).get("max_len")
        except Exception:
            return None

    def system_one(self, state, questions, budget=None):
        self.attempts += 1
        try:
            return 200, self.agent.system_one(state, questions)
        except Exception as e:
            return -1, {"error": f"{type(e).__name__}: {e}"}

    def prep_info(self, state, questions):
        """Per-question tokenized sequence lengths via LayaMLX.prepare(); flags sequences that
        hit max_len (possible truncation). Best-effort: returns None if prepare is unavailable."""
        prepare = getattr(self.agent, "prepare", None)
        if prepare is None:
            return None
        try:
            batch = prepare(state, questions)
            mask = batch["attention_mask"]
            seq_lens = [int(x) for x in mask.sum(axis=1)]
        except Exception as e:
            return {"prep_error": f"{type(e).__name__}: {e}"}
        qids = list(questions.keys())
        return {"seq_lens": dict(zip(qids, seq_lens)),
                "max_len": self._max_len,
                "possibly_truncated": ([q for q, n in zip(qids, seq_lens)
                                        if self._max_len and n >= self._max_len]
                                       if self._max_len else None)}

    def close(self):
        pass


class TorchBackend:
    """Original RLAgent on an explicit device. No silent fallback: a missing device raises."""

    def __init__(self, source_dir, device):
        # Bound thread pools BEFORE importing torch so OMP/MKL pools are sized at init.
        os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_THREADS))
        import torch
        torch.set_num_threads(TORCH_THREADS)          # intra-op
        torch.set_num_interop_threads(TORCH_THREADS)  # inter-op; raises if already initialized
        if device == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("runtime=mps requested but torch.backends.mps.is_available() is False; "
                             "refusing to fall back to CPU")
        sys.path.insert(0, source_dir)
        from rl_agent_api import RLAgent
        self.agent = RLAgent(source_dir, device=device)
        self.name = f"{device}(RLAgent)"
        self.device = device
        self._max_len = self.agent.cfg.get("max_len")
        self.attempts = 0

    def system_one(self, state, questions, budget=None):
        self.attempts += 1
        try:
            return 200, self.agent.system_one(state, questions)
        except Exception as e:
            return -1, {"error": f"{type(e).__name__}: {e}"}

    def prep_info(self, state, questions):
        """Per-question tokenized lengths via rl_common.build_sequence; flags max_len hits."""
        try:
            from rl_common import build_sequence
            seq_lens, truncated = {}, []
            for qid, qdef in questions.items():
                q = self.agent._to_internal(qdef)
                seq, _markers = build_sequence(self.agent.tok, state, q,
                                               self.agent.cfg["max_len"], self.agent.cfg["head_max_len"])
                seq_lens[qid] = len(seq)
                if self._max_len and len(seq) >= self._max_len:
                    truncated.append(qid)
            return {"seq_lens": seq_lens, "max_len": self._max_len,
                    "possibly_truncated": truncated or None}
        except Exception as e:
            return {"prep_error": f"{type(e).__name__}: {e}"}

    def close(self):
        pass


def make_backend(args):
    if args.runtime == "jev":
        return JevBackend()
    if args.runtime == "mlx":
        return MLXBackend(os.path.abspath(args.model), args.dtype)
    return TorchBackend(os.path.abspath(args.source), args.runtime)


# --------------------------------------------------------------------------- calibration

def check_probe(status, resp, runtime):
    """Fail-closed calibration checks. Returns list of failure strings (empty = pass).
    Every malformed-shape outcome becomes a failure string; nothing raises."""
    fails = []
    if status != 200 or not isinstance(resp, dict):
        return [f"calibration probe failed: HTTP/status {status}: {json.dumps(resp)[:400]}"]
    try:
        ans = resp["answers"]
        heads = ans["heads"].get("noul")
        probs = ans["pick"].get("probabilities", {})
    except Exception as e:
        return [f"calibration probe unparseable: {e}"]
    if not isinstance(heads, (int, float)) or isinstance(heads, bool) or not math.isfinite(heads):
        fails.append(f"probe heads noul missing/non-numeric/non-finite: {heads!r}")
    elif not (0.35 <= heads <= 0.65):
        fails.append(f"hidden fair coin noul={heads} outside [0.35, 0.65]")
    if not isinstance(probs, dict) or set(probs.keys()) != {"left", "right"}:
        fails.append(f"sealed-envelope probabilities malformed: {probs!r}")
    else:
        vals = list(probs.values())
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in vals):
            fails.append(f"sealed-envelope probabilities non-numeric/non-finite: {probs!r}")
        elif not (0.5 <= sum(vals) <= 1.5):
            fails.append(f"sealed-envelope probabilities implausible sum {sum(vals)}: {probs!r}")
        elif max(vals) >= 0.999:
            fails.append(f"sealed-envelope choice degenerate: {probs}")
    if runtime == "jev":
        model = resp.get("model", "")
        if not (isinstance(model, str) and model.startswith("jev")):
            fails.append(f"returned model {model!r} does not begin with 'jev'")
    return fails


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=["jev", "mlx", "cpu", "mps"],
                    help="jev = TypeSafe API; mlx = converted checkpoint via laya_api.LayaMLX; "
                         "cpu/mps = original source checkpoint via RLAgent")
    ap.add_argument("--source", default=os.path.join(HERE, "source"),
                    help="original checkpoint dir (cpu/mps runtimes)")
    ap.add_argument("--model", default=os.path.join(HERE, "converted"),
                    help="converted model dir (mlx runtime)")
    ap.add_argument("--dtype", default="float32", help="mlx runtime dtype")
    ap.add_argument("--output", default=None, help="results JSON path")
    ap.add_argument("--dump-requests", metavar="PATH", default=None,
                    help="write the frozen request bodies for all 105 cases + probe, then exit "
                         "(no inference, no credentials)")
    args = ap.parse_args()

    sha = suite_sha256()
    problems, built = check_suite()
    if problems:
        for p in problems:
            print(f"SUITE CHECK FAILED: {p}", file=sys.stderr)
        raise SystemExit(2)

    if args.dump_requests:
        doc = {"suite_sha256": sha, "suite_provenance": SUITE_PROVENANCE,
               "calibration_probe": calibration_probe_body(),
               "requests": [{"id": t["id"], "family": t["family"], "split": t["split"],
                             "request": built[t["id"]]} for t in TASKS]}
        with open(args.dump_requests, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.dump_requests}: {len(doc['requests'])} case requests + 1 probe")
        return
    if not args.runtime:
        ap.error("--runtime is required unless --dump-requests is given")

    out_path = args.output or os.path.join(HERE, f"results_{args.runtime}.json")

    doc = {
        "model": JEV_MODEL if args.runtime == "jev" else f"rl-agent/{args.runtime}",
        "runtime": args.runtime,
        "runtime_detail": {"source": os.path.abspath(args.source) if args.runtime in ("cpu", "mps") else None,
                           "model_dir": os.path.abspath(args.model) if args.runtime == "mlx" else None,
                           "dtype": args.dtype if args.runtime == "mlx" else None,
                           "torch_threads": TORCH_THREADS if args.runtime in ("cpu", "mps") else None,
                           "key_source": None},
        "suite_sha256": sha,
        "suite_provenance": SUITE_PROVENANCE,
        "caps": {"max_attempts": MAX_REQUESTS, "max_input_tokens": MAX_INPUT_TOKENS,
                 "note": "max_attempts budgets every HTTP POST (429 retries included) for jev; "
                         "for local runtimes it budgets model calls, one per logical request"},
        "results": [],
        "errors": [],
        "calibration_probe": None,
        "usage": {"requests": 0, "attempts": 0, "input_tokens": 0, "output_tokens": 0,
                  "fields": {}, "non_numeric_fields": [], "usage_complete": True,
                  "cost_usd": None},
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    try:
        backend = make_backend(args)
    except BaseException as e:
        doc["errors"].append(f"backend init failed: {type(e).__name__}: {e}")
        save()
        raise
    doc["runtime_detail"]["key_source"] = getattr(backend, "key_source", None)

    def record_usage(resp):
        """Aggregate every numeric usage field the backend returns; mark incomplete when the
        usage object or input_tokens is absent/non-numeric. Non-numeric fields are named, not
        silently zeroed."""
        u = resp.get("usage") if isinstance(resp, dict) else None
        if not isinstance(u, dict):
            doc["usage"]["usage_complete"] = False
            return
        for k, v in u.items():
            if isinstance(v, bool):
                if k not in doc["usage"]["non_numeric_fields"]:
                    doc["usage"]["non_numeric_fields"].append(k)
            elif isinstance(v, (int, float)):
                doc["usage"]["fields"][k] = doc["usage"]["fields"].get(k, 0) + v
            elif k not in doc["usage"]["non_numeric_fields"]:
                doc["usage"]["non_numeric_fields"].append(k)
        tok = u.get("input_tokens")
        if isinstance(tok, (int, float)) and not isinstance(tok, bool):
            doc["usage"]["input_tokens"] += int(tok)
        else:
            doc["usage"]["usage_complete"] = False
        out_tok = u.get("output_tokens")
        if isinstance(out_tok, (int, float)) and not isinstance(out_tok, bool):
            doc["usage"]["output_tokens"] += int(out_tok)

    try:
        # ---- calibration probe (fail-closed) ----
        probe_body = calibration_probe_body()
        t0 = time.time()
        status, probe = backend.system_one(probe_body["state"], probe_body["questions"],
                                           budget=MAX_REQUESTS - backend.attempts)
        probe_lat = int((time.time() - t0) * 1000)
        record_usage(probe)
        doc["calibration_probe"] = {"request": probe_body, "status": status,
                                    "response": probe, "latency_ms": probe_lat}
        fails = check_probe(status, probe, args.runtime)
        if fails:
            doc["errors"].extend(f"calibration: {f}" for f in fails)
            save()
            print("STOP: calibration gate failed:", "; ".join(fails), file=sys.stderr)
            raise SystemExit(3)
        print(f"probe ok: status={status} latency={probe_lat}ms "
              f"model={probe.get('model') if isinstance(probe, dict) else '?'}", flush=True)

        # ---- suite, serial, incremental saves ----
        for item in TASKS:
            if backend.attempts >= MAX_REQUESTS:
                doc["errors"].append(f"CAP: attempt cap {MAX_REQUESTS} reached "
                                     f"({backend.attempts} attempts); stopped before {item['id']}")
                break
            if doc["usage"]["input_tokens"] >= MAX_INPUT_TOKENS:
                doc["errors"].append(f"CAP: input-token cap {MAX_INPUT_TOKENS} reached "
                                     f"({doc['usage']['input_tokens']}); stopped before {item['id']}")
                break
            body = built[item["id"]]
            prep = backend.prep_info(body["state"], body["questions"])
            t0 = time.time()
            status, resp = backend.system_one(body["state"], body["questions"],
                                              budget=MAX_REQUESTS - backend.attempts)
            lat = int((time.time() - t0) * 1000)
            doc["usage"]["requests"] += 1
            if status == 200 and isinstance(resp, dict):
                out = resp
                doc["usage"]["input_tokens"] += usage_of(resp)
                if args.runtime == "jev" and isinstance(resp.get("model"), str):
                    doc["model"] = resp["model"]
            else:
                msg = (f"HTTP {status}: {json.dumps(resp)[:400]}" if status != -1
                       else f"error: {json.dumps(resp)[:400]}")
                out = msg
                doc["errors"].append(f"{item['id']}: {msg}")
            row = {"id": item["id"], "family": item["family"], "split": item["split"],
                   "request": body, "output": out, "latency_ms": lat}
            if prep is not None:
                row["prep"] = prep
            doc["results"].append(row)
            print(f"{item['id']}: status={status} latency={lat}ms "
                  f"cum_in={doc['usage']['input_tokens']} attempts={backend.attempts}", flush=True)
            save()
            if args.runtime == "jev":
                time.sleep(0.15)
    finally:
        doc["usage"]["attempts"] = getattr(backend, "attempts", doc["usage"]["requests"])
        save()
        backend.close()

    print(f"wrote {out_path}: results={len(doc['results'])} errors={len(doc['errors'])} "
          f"requests={doc['usage']['requests']} attempts={doc['usage']['attempts']} "
          f"input_tokens={doc['usage']['input_tokens']} cost_usd=null", flush=True)


if __name__ == "__main__":
    main()
