"""Parity and runtime benchmarks for the MLX port of the RL Agent (Laya) model.

Subcommands
-----------
parity   Compare the MLX port against the original PyTorch RLAgent (source/).
         Checks token-prep equality vs rl_common.collate_items, raw option/act
         logits, calibrated probabilities, and final formatted answers.
         Oracle defaults to CPU fp32; --oracle-device mps --oracle-dtype float16
         and --mlx-dtype float16 give the fp16 parity arms.

runtime  Time end-to-end system_one calls (tokenize + forward + postprocess)
         for ONE arm per process: --arm torch-cpu | torch-mps | mlx.
         3 warmups then >=15 timed samples per case; raw timings + metadata
         emitted as JSON. Arm construction errors are fatal (no fallback).

Fixtures are deterministic and identical across arms/processes: a short state
and a long state (truncated to the 512-token max_len), each with a single
question and an 8-question batch covering every question type and every
temperature bucket in rl_agent_config.json.

Examples
--------
    python benchmarks/benchmark.py parity --source source --model converted
    python benchmarks/benchmark.py parity --source source --model converted-fp16 --mlx-dtype float16
    python benchmarks/benchmark.py parity --source source --model converted --oracle-device mps --oracle-dtype float16
    python benchmarks/benchmark.py runtime --arm mlx --model converted --dtype float32 -o benchmarks/results/mlx_fp32.json
    python benchmarks/benchmark.py runtime --arm torch-cpu --source source -o benchmarks/results/torch_cpu.json
    python benchmarks/benchmark.py runtime --arm torch-mps --source source --dtype float16 -o benchmarks/results/torch_mps_fp16.json
"""
import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, NotRequired, TypedDict, cast

# Bound CPU threading before numpy/torch initialize their pools. --threads is
# pre-scanned so the flag works even though heavy imports happen later.
def _early_threads(default: str = "4") -> str:
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return default


_THREADS = _early_threads()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # laya_api lives at repo root

import laya_api
from laya_api import QTYPES, QTYPE_NAMES, QuestionDef, temp_bucket, Questions


class PrepEntry(TypedDict):
    """Per-key collate comparison inside a parity case."""

    shape_ref: list[int]
    shape_mlx: list[int]
    equal: bool


class PerQ(TypedDict):
    """Per-question raw-logit comparison row."""

    qid: Any
    qtype: str
    k: int
    temp_bucket: str
    logit_max_abs_diff: float
    prob_max_abs_diff: float
    act_max_abs_diff: float
    argmax_ref: int
    argmax_mlx: int
    argmax_agree: bool


class RawCase(TypedDict):
    """Raw-logit summary for one parity case."""

    logit_max_abs_diff: float | None
    act_max_abs_diff: float | None
    per_question: NotRequired[list[PerQ]]


class FormattedCase(TypedDict):
    """Formatted system_one comparison for one parity case."""

    model_field: str | None
    usage_ref: laya_api.Usage | None
    usage_mlx: laya_api.Usage | None
    usage_equal: bool
    answer_diffs: list[dict[str, Any]]


class ParityCase(TypedDict):
    """One fixture's parity report entry (mixed_padding omits several keys)."""

    n_questions: int
    seq_lens: NotRequired[list[int]]
    padded_len: NotRequired[int]
    input_tokens: NotRequired[int]
    prep: NotRequired[dict[str, PrepEntry]]
    prep_equal: NotRequired[bool]
    raw: NotRequired[RawCase]
    formatted: NotRequired[FormattedCase]


ParitySummary = TypedDict("ParitySummary", {
    "max_logit_err": float,
    "max_prob_err": float,
    "max_act_err": float,
    "qtypes": list[str],
    "temp_buckets": list[str],
    "n_cases": int,
    "pass": bool,
})


class ParityReport(TypedDict):
    """Top-level JSON document emitted by cmd_parity."""

    source: str
    model: str
    oracle: dict[str, str]
    mlx: dict[str, str]
    tolerances: dict[str, float]
    threads: int
    cases: dict[str, ParityCase]
    failures: list[str]
    summary: NotRequired[ParitySummary]


class ArmMeta(TypedDict, total=False):
    """Per-arm metadata merged into the runtime report's metadata block."""

    device: str
    torch_threads: int
    mlx_version: str
    compiled: bool


class RuntimeCase(TypedDict):
    """One fixture's timing row in the runtime report."""

    n_questions: int
    input_tokens: int
    samples: int
    wall_ms: list[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    max_ms: float


class RuntimeReport(TypedDict):
    """Top-level JSON document emitted by cmd_runtime."""

    arm: str
    dtype: str
    source: Any
    model: Any
    threads: int
    warmups: int
    samples: int
    load_seconds: float
    cases: dict[str, RuntimeCase]
    metadata: dict[str, Any]

# ----------------------------------------------------------------------------- fixtures
_SHORT_STATE = (
    "The support dashboard shows 47 open tickets, 12 of them older than 72 hours. "
    "The customer on ticket #8814 reports intermittent login failures since the "
    "Tuesday deploy and asks for a status update."
)
_LONG_STATE = (
    "The support ticket queue shows 47 open items. The customer reports intermittent "
    "login failures since the Tuesday deploy. Error rate is 3 percent of sessions. " * 24
)


def _questions8() -> Questions:
    """Eight questions covering all qtypes and every temperature bucket."""
    return {
        "q_choice2": {"type": "choice", "instructions": "Is the customer asking for a refund?",
                      "criteria": {"refund": "wants money back", "no_refund": "no refund requested"}},
        "q_choice4": {"type": "choice", "instructions": "What is the customer's main issue?",
                      "criteria": {"login": "cannot log in", "billing": "billing problem",
                                   "shipping": "shipping delay", "other": "something else"}},
        "q_choice7": {"type": "choice", "instructions": "Which team should own ticket #8814?",
                      "criteria": {"auth": None, "billing": None, "shipping": None, "infra": None,
                                   "frontend": None, "support": None, "sales": None}},
        "q_choice12": {"type": "choice", "instructions": "Pick the best first response step.",
                       "criteria": ["apologize", "ask_logs", "ask_screenshot", "escalate_auth",
                                    "give_status", "offer_refund", "offer_credit", "close_ticket",
                                    "ask_account_id", "check_status_page", "schedule_call", "other"]},
        "q_score5": {"type": "score", "instructions": "Rate the customer's likely frustration.",
                     "criteria": ["calm", "mildly annoyed", "annoyed", "frustrated", "furious"]},
        "q_score8": {"type": "score", "instructions": "Rate urgency of ticket #8814 on 8 levels.",
                     "criteria": ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"]},
        "q_noul": {"type": "noul", "instructions": "The customer cannot access their account."},
        "q_choice3_list": {"type": "choice", "instructions": "How old is the oldest open ticket?",
                           "criteria": ["under_24h", "one_to_three_days", "over_three_days"]},
    }


def make_fixtures() -> dict[str, tuple[str, Questions]]:
    """Deterministic shared fixtures: {name: (state, questions)}. Same in every arm/process."""
    return {
        "single_short": (_SHORT_STATE, {"q_choice4": _questions8()["q_choice4"]}),
        "batch8_short": (_SHORT_STATE, _questions8()),
        "single_long": (_LONG_STATE, {"q_noul": _questions8()["q_noul"]}),
        "batch8_long": (_LONG_STATE, _questions8()),
    }


# ----------------------------------------------------------------------------- torch oracle / arm
def _import_source(source_dir: str) -> tuple[ModuleType, ModuleType]:
    source_dir = str(Path(source_dir).resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    import rl_agent_api  # type: ignore[import-not-found]  # reason: lives in gitignored source/ dir, not tracked  # noqa: F401
    import rl_common  # type: ignore[import-not-found]  # reason: lives in gitignored source/ dir, not tracked
    return rl_agent_api, rl_common


class _AutocastModel:
    """Wrap a DecisionModel so calls run under torch.autocast (used for the MPS fp16 arm,
    since RLAgent only enables autocast on CUDA)."""

    def __init__(self, model: Any, device_type: str, dtype: Any) -> None:
        object.__setattr__(self, "_m", model)
        object.__setattr__(self, "_dev", device_type)
        object.__setattr__(self, "_dt", dtype)

    def __call__(self, *a: Any, **kw: Any) -> Any:
        import torch
        with torch.autocast(device_type=self._dev, dtype=self._dt, enabled=True):
            return self._m(*a, **kw)

    def __getattr__(self, k: str) -> Any:
        return getattr(object.__getattribute__(self, "_m"), k)


def build_torch_agent(source_dir: str, device: str, dtype: str) -> Any:
    """RLAgent on device. dtype float32 -> stock fp32; float16 -> fp16 autocast wrapper."""
    import torch
    rl_agent_api, _ = _import_source(source_dir)
    agent = rl_agent_api.RLAgent(str(source_dir), device=device)
    if dtype == "float16":
        agent.model = _AutocastModel(agent.model, device, torch.float16)
    elif dtype != "float32":
        raise ValueError("unsupported torch dtype %r (use float32 or float16)" % dtype)
    return agent


def torch_raw_forward(agent: Any, batch: laya_api.Batch, device: str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Raw (option_logits, act_logits) from the torch model on a laya_api numpy batch."""
    import torch
    dev = torch.device(device)
    with torch.no_grad():
        logits, act = agent.model(
            torch.from_numpy(batch["input_ids"]).to(dev),
            torch.from_numpy(batch["attention_mask"]).to(dev),
            torch.from_numpy(batch["marker_pos"]).to(dev),
            torch.from_numpy(batch["marker_mask"]).to(dev),
            torch.from_numpy(batch["qtype"]).to(dev))
    return logits.float().cpu().numpy(), act.float().cpu().numpy()


def reference_collate(rl_common: ModuleType, items: list[laya_api.Item], pad_id: int) -> Any:
    """rl_common.collate_items on items shaped the way rl_agent_api.system_one builds them."""
    full = [dict(it, target=[0.0] * len(it["markers"]), label=-1, episode=0, ep_step=0,
                 ep_len=1, src="api") for it in items]
    return rl_common.collate_items([full], pad_id)


def count_input_tokens(source_dir: str, state: str, questions: Questions) -> int:
    """input_tokens for a fixture via the source-side prep (used for torch-arm metadata)."""
    rl_agent_api, rl_common = _import_source(source_dir)
    from transformers import AutoTokenizer
    cfg = json.load(open(os.path.join(str(source_dir), "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(str(source_dir), "tokenizer"))
    n = 0
    for qdef in questions.values():
        q = rl_agent_api.RLAgent._to_internal(qdef)
        seq, _ = rl_common.build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
        n += len(seq)
    return n


# ----------------------------------------------------------------------------- parity
def _default_tols(mlx_dtype: str, oracle_dtype: str) -> dict[str, float]:
    fp16 = "float16" in (mlx_dtype, oracle_dtype)
    if fp16:
        return {"logit": 5e-2, "prob": 2e-2, "act": 5e-2}
    return {"logit": 2e-3, "prob": 5e-4, "act": 2e-3}


def _softmax_row(z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    z = z - z.max()
    e = np.exp(z)
    return cast(npt.NDArray[np.float32], e / e.sum())


def _answers_agree(ref_ans: Mapping[str, Any], mlx_ans: Mapping[str, Any], prob_tol: float) -> list[dict[str, Any]]:
    """Compare formatted answers: labels exact, numeric fields within prob_tol."""
    diffs: list[dict[str, Any]] = []
    for qid, ra in ref_ans.items():
        ma = mlx_ans.get(qid)
        if ma is None:
            diffs.append({"qid": qid, "error": "missing in mlx output"})
            continue
        if ra.get("type") != ma.get("type"):
            diffs.append({"qid": qid, "error": "type mismatch %r vs %r" % (ra.get("type"), ma.get("type"))})
            continue
        for key in ("choice", "score", "noul"):
            if key in ra:
                rv, mv = ra[key], ma.get(key)
                if isinstance(rv, str):
                    if rv != mv:
                        diffs.append({"qid": qid, "field": key, "ref": rv, "mlx": mv})
                elif mv is None or abs(float(rv) - float(mv)) > prob_tol:
                    diffs.append({"qid": qid, "field": key, "ref": rv, "mlx": mv})
        if "confidence" in ra and abs(float(ra["confidence"]) - float(ma.get("confidence", float("nan")))) > prob_tol:
            diffs.append({"qid": qid, "field": "confidence", "ref": ra["confidence"], "mlx": ma.get("confidence")})
        if ra.get("legend") != ma.get("legend"):
            diffs.append({"qid": qid, "field": "legend", "ref": ra.get("legend"), "mlx": ma.get("legend")})
        rp, mp = ra.get("probabilities", {}), ma.get("probabilities", {})
        if set(rp) != set(mp):
            diffs.append({"qid": qid, "field": "probabilities", "error": "key mismatch"})
        else:
            for kk in rp:
                if abs(float(rp[kk]) - float(mp[kk])) > prob_tol:
                    diffs.append({"qid": qid, "field": "probabilities", "key": kk,
                                  "ref": rp[kk], "mlx": mp[kk]})
        ra_act = ra.get("rl_agent", {}).get("act_probability")
        ma_act = ma.get("rl_agent", {}).get("act_probability")
        if ra_act is not None and abs(float(ra_act) - float(ma_act)) > prob_tol:
            diffs.append({"qid": qid, "field": "act_probability", "ref": ra_act, "mlx": ma_act})
    return diffs


def cmd_parity(args: argparse.Namespace) -> int:
    import torch  # noqa: F401  (oracle always needs torch)
    rl_agent_api, rl_common = _import_source(args.source)
    oracle = build_torch_agent(args.source, args.oracle_device, args.oracle_dtype)
    mlx_agent = laya_api.LayaMLX(args.model, dtype=args.mlx_dtype)
    tols = _default_tols(args.mlx_dtype, args.oracle_dtype)
    k: Any
    for k in tols:
        v = getattr(args, "%s_tol" % k)
        if v is not None:
            tols[k] = v

    fixtures = make_fixtures()
    report: ParityReport = {"source": str(args.source), "model": str(args.model),
              "oracle": {"device": args.oracle_device, "dtype": args.oracle_dtype},
              "mlx": {"dtype": args.mlx_dtype}, "tolerances": tols,
              "threads": int(_THREADS), "cases": {}, "failures": []}
    max_errs = {"logit": 0.0, "prob": 0.0, "act": 0.0}
    qtypes_seen: set[int] = set()
    buckets_seen: set[str] = set()
    pad_id = mlx_agent.tok.pad_token_id

    def check(cond: object, msg: str) -> None:
        if not cond:
            report["failures"].append(msg)

    ld: npt.NDArray[np.float32] | float
    for name, (state, questions) in fixtures.items():
        case: ParityCase = {"n_questions": len(questions)}
        ids, items, batch = mlx_agent.prepare(state, questions)
        case["seq_lens"] = [len(it["ids"]) for it in items]
        case["padded_len"] = int(batch["input_ids"].shape[1])
        case["input_tokens"] = int(batch["n_tokens"])
        check(all(len(it["ids"]) > 0 for it in items), "%s: empty prepared sequence" % name)
        check(batch["n_tokens"] > 0, "%s: zero input tokens" % name)

        # ---- prep parity vs rl_common.collate_items
        ref = reference_collate(rl_common, items, pad_id)
        prep: dict[str, PrepEntry] = {}
        for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"):
            rv = ref[key].numpy()
            mv = cast(Mapping[str, Any], batch)[key]
            prep[key] = {"shape_ref": list(rv.shape), "shape_mlx": list(mv.shape),
                         "equal": bool(rv.shape == mv.shape and (rv == mv).all())}
            check(prep[key]["equal"], "%s: prep mismatch in %s" % (name, key))
        case["prep"] = prep

        # ---- raw logits
        t_logits, t_act = torch_raw_forward(oracle, batch, args.oracle_device)
        m_logits, m_act = mlx_agent.raw_forward(batch)
        check(np.isfinite(m_logits).all(), "%s: non-finite mlx option logits" % name)
        check(np.isfinite(m_act).all(), "%s: non-finite mlx act logits" % name)
        shape_ok = t_logits.shape == m_logits.shape and t_act.shape == m_act.shape
        check(t_logits.shape == m_logits.shape, "%s: logits shape %s vs %s"
              % (name, t_logits.shape, m_logits.shape))
        check(t_act.shape == m_act.shape, "%s: act shape %s vs %s" % (name, t_act.shape, m_act.shape))

        per_q: list[PerQ] = []
        for r, qid in enumerate(ids):
            q = laya_api._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            qtypes_seen.add(qt)
            buckets_seen.add(temp_bucket(qt, k))
            tl, ml = t_logits[r, :k], m_logits[r, :k]
            temp = mlx_agent.temperature_by_options.get(temp_bucket(qt, k), mlx_agent.temperature[qt])
            tp, mp = _softmax_row(tl / temp), _softmax_row(ml / temp)
            ld = np.abs(tl - ml)
            pd = np.abs(tp - mp)
            ad = float(np.abs(t_act[r] - m_act[r]).max())
            max_errs["logit"] = max(max_errs["logit"], float(ld.max()))
            max_errs["prob"] = max(max_errs["prob"], float(pd.max()))
            max_errs["act"] = max(max_errs["act"], ad)
            per_q.append({"qid": qid, "qtype": QTYPE_NAMES[qt], "k": k,
                          "temp_bucket": temp_bucket(qt, k),
                          "logit_max_abs_diff": float(ld.max()),
                          "prob_max_abs_diff": float(pd.max()),
                          "act_max_abs_diff": ad,
                          "argmax_ref": int(tp.argmax()), "argmax_mlx": int(mp.argmax()),
                          "argmax_agree": bool(tp.argmax() == mp.argmax())})
            check(bool(tp.argmax() == mp.argmax()), "%s/%s: argmax disagreement" % (name, qid))
        case["raw"] = {"logit_max_abs_diff": float(np.abs(t_logits - m_logits).max()) if shape_ok else None,
                       "act_max_abs_diff": float(np.abs(t_act - m_act).max()) if shape_ok else None,
                       "per_question": per_q}

        # ---- formatted output
        ref_out = oracle.system_one(state, questions)
        mlx_out = mlx_agent.system_one(state, questions)
        case["formatted"] = {
            "model_field": mlx_out.get("model"),
            "usage_ref": ref_out.get("usage"), "usage_mlx": mlx_out.get("usage"),
            "usage_equal": ref_out.get("usage") == mlx_out.get("usage"),
            "answer_diffs": _answers_agree(ref_out["answers"], mlx_out["answers"], tols["prob"])}
        check(case["formatted"]["usage_equal"], "%s: usage mismatch %s vs %s"
              % (name, ref_out.get("usage"), mlx_out.get("usage")))
        check(not case["formatted"]["answer_diffs"],
              "%s: %d formatted-answer diffs" % (name, len(case["formatted"]["answer_diffs"])))
        report["cases"][name] = case

    # ---- mixed-padding raw case: short + long items in one batch
    _, items_s, _ = mlx_agent.prepare(_SHORT_STATE, _questions8())
    _, items_l, _ = mlx_agent.prepare(_LONG_STATE, {"q_noul": _questions8()["q_noul"]})
    mixed_items = items_s + items_l
    mixed = laya_api.collate_items(mixed_items, pad_id)
    ref_mixed = reference_collate(rl_common, mixed_items, pad_id)
    prep_eq = all(ref_mixed[k].numpy().shape == cast(Mapping[str, Any], mixed)[k].shape and (ref_mixed[k].numpy() == cast(Mapping[str, Any], mixed)[k]).all()
                  for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"))
    t_logits, t_act = torch_raw_forward(oracle, mixed, args.oracle_device)
    m_logits, m_act = mlx_agent.raw_forward(mixed)
    shape_ok = t_logits.shape == m_logits.shape and t_act.shape == m_act.shape
    ld = float(np.abs(t_logits - m_logits).max()) if shape_ok else float("nan")
    ad = float(np.abs(t_act - m_act).max()) if shape_ok else float("nan")
    if shape_ok:
        max_errs["logit"] = max(max_errs["logit"], ld)
        max_errs["act"] = max(max_errs["act"], ad)
    report["cases"]["mixed_padding"] = {
        "n_questions": len(mixed_items), "seq_lens": [len(it["ids"]) for it in mixed_items],
        "padded_len": int(mixed["input_ids"].shape[1]),
        "prep_equal": bool(prep_eq),
        "raw": {"logit_max_abs_diff": ld, "act_max_abs_diff": ad}}
    check(prep_eq, "mixed_padding: prep mismatch")
    check(shape_ok, "mixed_padding: output shape mismatch")
    check(np.isfinite(m_logits).all() and np.isfinite(m_act).all(), "mixed_padding: non-finite mlx outputs")

    # ---- coverage + tolerance assertions
    check(qtypes_seen == {0, 1, 2}, "qtype coverage incomplete: %s" % sorted(qtypes_seen))
    expected_buckets = {"choice:2", "choice:3-5", "choice:6-10", "choice:11+",
                        "score:3-5", "score:6-10", "noul:2"}
    check(expected_buckets <= buckets_seen, "temp-bucket coverage incomplete: missing %s"
          % sorted(expected_buckets - buckets_seen))
    check(max_errs["logit"] <= tols["logit"], "logit max err %.6g > tol %.6g" % (max_errs["logit"], tols["logit"]))
    check(max_errs["prob"] <= tols["prob"], "prob max err %.6g > tol %.6g" % (max_errs["prob"], tols["prob"]))
    check(max_errs["act"] <= tols["act"], "act max err %.6g > tol %.6g" % (max_errs["act"], tols["act"]))

    report["summary"] = {"max_logit_err": max_errs["logit"], "max_prob_err": max_errs["prob"],
                         "max_act_err": max_errs["act"],
                         "qtypes": sorted(QTYPE_NAMES[q] for q in qtypes_seen),
                         "temp_buckets": sorted(buckets_seen),
                         "n_cases": len(report["cases"]),
                         "pass": not report["failures"]}
    _emit(report, args.output)
    return 0 if report["summary"]["pass"] else 1


# ----------------------------------------------------------------------------- runtime
def build_arm(args: argparse.Namespace) -> tuple[Any, Callable[[], None], ArmMeta]:
    """Construct one benchmark arm. Errors are fatal — no silent fallback."""
    if args.arm == "torch-cpu":
        if args.dtype != "float32":
            raise SystemExit("arm torch-cpu supports only --dtype float32 (got %r)" % args.dtype)
        if not args.source:
            raise SystemExit("arm torch-cpu requires --source")
        import torch
        torch.set_num_threads(int(args.threads))
        agent = build_torch_agent(args.source, "cpu", "float32")
        return agent, (lambda: None), {"device": "cpu", "torch_threads": torch.get_num_threads()}
    if args.arm == "torch-mps":
        if not args.source:
            raise SystemExit("arm torch-mps requires --source")
        import torch
        if not torch.backends.mps.is_available():
            raise SystemExit("arm torch-mps requested but MPS is not available")
        agent = build_torch_agent(args.source, "mps", args.dtype)
        return agent, torch.mps.synchronize, {"device": "mps"}
    if args.arm == "mlx":
        if not args.model:
            raise SystemExit("arm mlx requires --model (converted directory)")
        import mlx.core as mx
        agent = laya_api.LayaMLX(args.model, dtype=args.dtype, compile=getattr(args, "compile", False))

        def sync() -> None:
            if agent.last_raw is not None:
                mx.eval(*agent.last_raw)
        return agent, sync, {"device": "mlx-gpu", "mlx_version": getattr(mx, "__version__", "unknown"),
                             "compiled": getattr(args, "compile", False)}
    raise SystemExit("unknown arm %r" % args.arm)


def cmd_runtime(args: argparse.Namespace) -> int:
    t_load0 = time.perf_counter()
    agent, sync, arm_meta = build_arm(args)
    load_s = time.perf_counter() - t_load0

    fixtures = make_fixtures()
    report: RuntimeReport = {"arm": args.arm, "dtype": args.dtype, "source": args.source, "model": args.model,
              "threads": int(args.threads), "warmups": args.warmups, "samples": args.samples,
              "load_seconds": round(load_s, 3), "cases": {},
              "metadata": {"platform": sys.platform, "python": sys.version.split()[0],
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "note": "wall time per call includes tokenization + forward + "
                                   "postprocess; CPU thread pools bounded to %d" % int(args.threads)}}
    report["metadata"].update(arm_meta)
    try:
        import torch
        report["metadata"]["torch"] = torch.__version__
    except ImportError:
        pass

    for name, (state, questions) in fixtures.items():
        for _ in range(args.warmups):
            agent.system_one(state, questions)
            sync()
        times: list[float] = []
        for _ in range(args.samples):
            t0 = time.perf_counter()
            agent.system_one(state, questions)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
        if hasattr(agent, "prepare"):
            _, _, batch = agent.prepare(state, questions)
            n_tok = int(batch["n_tokens"])
        else:
            n_tok = count_input_tokens(args.source, state, questions)
        arr = np.array(times)
        report["cases"][name] = {
            "n_questions": len(questions), "input_tokens": n_tok,
            "samples": len(times), "wall_ms": [round(t, 3) for t in times],
            "mean_ms": round(float(arr.mean()), 3), "std_ms": round(float(arr.std()), 3),
            "min_ms": round(float(arr.min()), 3), "p50_ms": round(float(np.median(arr)), 3),
            "max_ms": round(float(arr.max()), 3)}
        print("[%s] %s: mean %.1f ms  p50 %.1f ms  min %.1f  max %.1f  (%d samples, %s tokens)"
              % (args.arm, name, arr.mean(), np.median(arr), arr.min(), arr.max(), len(times), n_tok),
              file=sys.stderr)
    _emit(report, args.output)
    return 0


# ----------------------------------------------------------------------------- CLI
def _emit(report: ParityReport | RuntimeReport, output: str | None) -> None:
    text = json.dumps(report, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(text + "\n")
        print("wrote %s" % output, file=sys.stderr)
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parity", help="compare MLX port vs source PyTorch RLAgent",
                       description="Parity: MLX port vs source RLAgent. Asserts prep equality, "
                                   "finite outputs, max-error tolerances, argmax and formatted-answer "
                                   "agreement; emits a JSON report of raw diffs.")
    p.add_argument("--source", required=True, help="original checkpoint dir (rl_agent_config.json, "
                                                   "model.safetensors, encoder/, tokenizer/, rl_common.py, rl_agent_api.py)")
    p.add_argument("--model", required=True, help="converted MLX model dir (from convert.py)")
    p.add_argument("--mlx-dtype", default="float32", choices=["float32", "float16"],
                   help="dtype of the converted MLX weights (default float32)")
    p.add_argument("--oracle-device", default="cpu", choices=["cpu", "mps"],
                   help="device for the source RLAgent oracle (default cpu)")
    p.add_argument("--oracle-dtype", default="float32", choices=["float32", "float16"],
                   help="oracle compute dtype; float16 wraps the model in autocast (default float32)")
    p.add_argument("--logit-tol", type=float, default=None,
                   help="max abs option-logit diff (default 2e-3 fp32 / 5e-2 fp16)")
    p.add_argument("--prob-tol", type=float, default=None,
                   help="max abs calibrated-probability diff (default 5e-4 fp32 / 2e-2 fp16)")
    p.add_argument("--act-tol", type=float, default=None,
                   help="max abs act-logit diff (default 2e-3 fp32 / 5e-2 fp16)")
    p.add_argument("--threads", default=_THREADS, help="CPU thread bound (default 4)")
    p.add_argument("-o", "--output", default=None, help="write JSON report here (default stdout)")
    p.set_defaults(fn=cmd_parity)

    r = sub.add_parser("runtime", help="time end-to-end system_one for one arm per process",
                       description="Runtime: one arm per process. 3 warmups then >=15 timed "
                                   "system_one calls per case (tokenize+forward+postprocess, with "
                                   "device sync). Emits raw per-call timings + metadata as JSON.")
    r.add_argument("--arm", required=True, choices=["torch-cpu", "torch-mps", "mlx"],
                   help="torch-cpu: source RLAgent fp32 on CPU; torch-mps: source RLAgent on MPS; "
                        "mlx: MLX port via laya_api.LayaMLX")
    r.add_argument("--source", default=None, help="original checkpoint dir (required for torch-* arms)")
    r.add_argument("--model", default=None, help="converted MLX model dir (required for mlx arm)")
    r.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="compute dtype; torch-cpu is fp32-only (default float32)")
    r.add_argument("--compile", action="store_true",
                   help="JIT-compile the MLX forward pass with mx.compile for maximum speed")
    r.add_argument("--warmups", type=int, default=3, help="untimed warmup calls per case (default 3)")
    r.add_argument("--samples", type=int, default=15, help="timed calls per case (default 15)")
    r.add_argument("--threads", default=_THREADS, help="CPU thread bound (default 4)")
    r.add_argument("-o", "--output", default=None, help="write JSON report here (default stdout)")
    r.set_defaults(fn=cmd_runtime)

    args = ap.parse_args(argv)
    if args.cmd == "runtime" and args.samples < 1:
        ap.error("--samples must be >= 1")
    return cast(int, args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
