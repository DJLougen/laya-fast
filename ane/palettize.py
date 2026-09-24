"""W8 palettization screen for an exported ANE body package.

Usage:
    .venv/bin/python ane/palettize.py --package ane/body96/model.mlpackage \
        --bits 8 --mode uniform --group-size 32 --output ane/body96-w8
"""

import argparse
import json
import time
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np


class PalettizeTiming(TypedDict):
    p50_ms: float
    min_ms: float
    max_ms: float


class PalettizeReport(TypedDict):
    """report.json payload for one palettized package."""

    package: str
    bits: int
    mode: str
    group_size: int
    palettize_seconds: float
    size_mb: float
    timing: NotRequired[PalettizeTiming]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--mode", choices=["uniform", "kmeans"], default="uniform")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=30)
    args = ap.parse_args()

    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    if args.output.exists():
        raise SystemExit("refusing to reuse existing output: %s" % args.output)

    t0 = time.perf_counter()
    model = ct.models.MLModel(str(args.package))
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            mode=args.mode, nbits=args.bits, group_size=args.group_size,
            channel_axis=None,
        )
    )
    compressed = palettize_weights(model, config)
    args.output.mkdir(parents=True)
    target = args.output / "model.mlpackage"
    compressed.save(str(target))
    report: PalettizeReport = {"package": str(args.package), "bits": args.bits, "mode": args.mode,
              "group_size": args.group_size,
              "palettize_seconds": round(time.perf_counter() - t0, 2),
              "size_mb": round(sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6, 1)}

    m = ct.models.MLModel(str(target), compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = m.get_spec()
    inputs = {}
    rng = np.random.default_rng(0)
    for feat in spec.description.input:
        shape = tuple(feat.type.multiArrayType.shape)
        inputs[feat.name] = rng.standard_normal(shape).astype(np.float16)
    for _ in range(5):
        m.predict(inputs)
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        m.predict(inputs)
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    report["timing"] = {"p50_ms": float(np.median(a)), "min_ms": float(a.min()),
                        "max_ms": float(a.max())}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
