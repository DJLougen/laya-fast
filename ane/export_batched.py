"""Export batched (B>1) ANE bodies for the short buckets, then verify that each
row's output matches the single-example model.

The router runs its short-batch ANE share as sequential single-question calls;
at L=80 a B=4 call measures 7.68 ms/question vs 9.42 sequential, because the
per-call fixed cost (~2.3 ms) is amortized. Long buckets do not benefit
(L=128: 1.04x), so only the short ladder is exported.

Usage:
    .venv/bin/python ane/export_batched.py --length 80 --batch 4 --output ane/batched/body80
    .venv/bin/python ane/export_batched.py --length 80 --batch 4 --output ane/batched/body80 --verify-only
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ane.ane_model import ConvBody
from ane.export import compute_plan


def build_inputs(body, L, B, seed=0):
    rng = np.random.default_rng(seed)
    width = body.cfg["hidden_size"]
    inputs = {
        "embeddings": torch.from_numpy(rng.standard_normal((B, width, 1, L)).astype(np.float32)),
        "full_mask": torch.zeros(B, L, 1, L),
        "local_mask": torch.zeros(B, L, 1, L),
        "type_vectors": torch.zeros(B, width, 1, 1),
        "marker_map": torch.zeros(B, L, 1, ConvBody.MAX_MARKERS),
    }
    inputs["marker_map"][:, 0] = 1
    return inputs


def export(model_dir, L, B, dest, verify_only=False):
    import coremltools as ct

    torch.set_num_threads(4)
    body = ConvBody(model_dir, L).float().eval()
    inputs = build_inputs(body, L, B)
    target = dest / "model.mlpackage"
    if not verify_only:
        with torch.inference_mode():
            traced = torch.jit.trace(body, tuple(inputs.values()), strict=True, check_trace=False)
        converted = ct.convert(
            traced, source="pytorch", convert_to="mlprogram",
            inputs=[ct.TensorType(name=n, shape=tuple(v.shape), dtype=np.float16)
                    for n, v in inputs.items()],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        dest.mkdir(parents=True, exist_ok=True)
        converted.save(str(target))
        compiled = dest / "model.mlmodelc"
        stamp = dest / "model.mlmodelc.sha256"
        import hashlib
        fp = hashlib.sha256(open(target / "Data/com.apple.CoreML/model.mlmodel", "rb").read()).hexdigest()
        if not compiled.exists() or not stamp.exists() or stamp.read_text().strip() != fp:
            ct.utils.compile_model(str(target), str(compiled))
            stamp.write_text(fp)

    bmodel = ct.models.MLModel(str(target), compute_units=ct.ComputeUnit.CPU_AND_NE)
    data = {n: v.numpy().astype(np.float16) for n, v in inputs.items()}
    out = bmodel.predict(data)
    logits = next(v for v in out.values() if v.shape[1] == 1).reshape(B, -1).astype(np.float32)
    pooled = next(v for v in out.values() if v.shape[1] != 1).reshape(B, -1).astype(np.float32)

    # parity: same example through the B=1 model
    single_dir = dest.parent.parent / ("body%d" % L)
    ones = {n: v[:1] for n, v in data.items()}
    s_out = ct.models.CompiledMLModel(str(single_dir / "model.mlmodelc"),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE).predict(ones)
    s_logits = next(v for v in s_out.values() if v.shape[1] == 1).reshape(-1).astype(np.float32)
    s_pooled = next(v for v in s_out.values() if v.shape[1] != 1).reshape(-1).astype(np.float32)

    d_logits = float(np.abs(logits[0] - s_logits).max())
    d_pooled = float(np.abs(pooled[0] - s_pooled).max())
    # row-to-row consistency: all rows share the same input here (except markers)
    return {"length": L, "batch": B, "logits_absmax": float(np.abs(logits).max()),
            "vs_single_max_abs_logit_diff": d_logits,
            "vs_single_max_abs_pooled_diff": d_pooled}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="converted-fp16")
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    report = export(args.model, args.length, args.batch, args.output, args.verify_only)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (args.output / "report.json").write_text(json.dumps(report, indent=1))
    print("length %d batch %d: max|dlogit| vs single = %.4f, max|dpooled| = %.4f (%.1fs)"
          % (report["length"], report["batch"], report["vs_single_max_abs_logit_diff"],
             report["vs_single_max_abs_pooled_diff"], report["seconds"]))


if __name__ == "__main__":
    main()
