"""Export fixed-shape ANE candidates of the English Laya body to .mlpackage.

Usage:
    .venv/bin/python ane/export.py --length 96 --output ane/body96
    .venv/bin/python ane/export.py --length 96 --output ane/body96 --skip-predict

Writes <output>/model.mlpackage + report.json (compute plan, timing, parity vs
the fp32 MLX model on random embeddings when --skip-predict is not set).
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ane.ane_model import ConvBody


def compute_plan(model, compute_units):
    import coremltools as ct

    units = {
        "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[compute_units]
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=units
    )
    preferred, supported = Counter(), Counter()
    costs = defaultdict(float)
    by_op = defaultdict(Counter)

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            device = type(usage.preferred_compute_device).__name__ if usage else "unknown"
            devices = [type(d).__name__ for d in usage.supported_compute_devices] if usage else []
            preferred[device] += 1
            supported.update(devices)
            by_op[op.operator_name][device] += 1
            if cost:
                costs[device] += cost.weight
            for nested in op.blocks:
                visit(nested)

    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    return {
        "kind": "Core ML anticipated execution plan, not a runtime hardware trace",
        "preferred_operation_counts": dict(preferred),
        "supported_operation_counts": dict(supported),
        "estimated_cost_by_device": dict(costs),
        "non_ane_ops": {k: dict(v) for k, v in by_op.items()
                        if any(d not in ("ANE", "NeuralEngine") for d in v)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="converted-fp16", help="converted model dir")
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compute-units", choices=["cpu_ne", "cpu_gpu", "all", "cpu"], default="cpu_ne")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing output dir")
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit("refusing to reuse existing output: %s (pass --force)" % args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    import coremltools as ct

    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    L = args.length
    body = ConvBody(args.model, L).float()
    body.eval()
    width = body.cfg["hidden_size"]

    rng = np.random.default_rng(0)
    inputs = {
        "embeddings": torch.from_numpy(rng.standard_normal((1, width, 1, L)).astype(np.float32)),
        "full_mask": torch.zeros(1, L, 1, L),
        "local_mask": torch.zeros(1, L, 1, L),
        "type_vectors": torch.zeros(1, width, 1, 1),
        "marker_map": torch.zeros(1, L, 1, ConvBody.MAX_MARKERS),
    }
    inputs["marker_map"][:, 0] = 1

    report = {"length": L, "model": args.model, "compute_units": args.compute_units}
    with torch.inference_mode():
        traced = torch.jit.trace(body, tuple(inputs.values()), strict=True, check_trace=True)
    t0 = time.perf_counter()
    converted = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float16)
            for name, value in inputs.items()
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    target = args.output / "model.mlpackage"
    converted.save(str(target))
    report["conversion_seconds"] = round(time.perf_counter() - t0, 2)

    units = {
        "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]
    t0 = time.perf_counter()
    coreml = ct.models.MLModel(str(target), compute_units=units)
    report["load_compile_seconds"] = round(time.perf_counter() - t0, 2)
    report["compute_plan"] = compute_plan(coreml, args.compute_units)
    print("PLAN", report["compute_plan"]["preferred_operation_counts"],
          "costs", report["compute_plan"]["estimated_cost_by_device"], flush=True)

    if not args.skip_predict:
        arrays = {name: value.numpy().astype(np.float16) for name, value in inputs.items()}
        for _ in range(5):
            coreml.predict(arrays)
        values = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            output = coreml.predict(arrays)
            values.append((time.perf_counter() - t0) * 1e3)
        arr = np.array(values)
        report["timing"] = {
            "samples": len(values), "mean_ms": float(arr.mean()),
            "p50_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()), "max_ms": float(arr.max()),
        }
        report["outputs"] = {
            name: {"shape": list(v.shape), "finite": bool(np.isfinite(v).all())}
            for name, v in output.items()
        }
        print("TIMING p50 %.3f ms  min %.3f  max %.3f" %
              (report["timing"]["p50_ms"], report["timing"]["min_ms"], report["timing"]["max_ms"]),
              flush=True)

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote %s" % (args.output / "report.json"))


if __name__ == "__main__":
    main()
