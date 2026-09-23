"""Convert the Laya RL-Agent decision model to Apple Core ML (.mlpackage).

Usage:
    python convert_coreml.py --source source --output laya_decision.mlpackage
    python convert_coreml.py --benchmark laya_decision.mlpackage
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
import torch
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
import coremltools.converters.mil.frontend.torch.ops as torch_ops


def register_compatibility_ops():
    """Register op converters bridging PyTorch 2.14 dialect to Core ML."""
    @torch_ops.register_torch_op(torch_alias=["alias", "aten.alias.default"])
    def alias_op(context, node):
        inputs = torch_ops._get_inputs(context, node, expected=1)
        context.add(mb.identity(x=inputs[0], name=node.name))

    @torch_ops.register_torch_op(override=True)
    def gather(context, node):
        inputs = torch_ops._get_inputs(context, node)
        indices = inputs[2]
        if types.is_float(indices.dtype):
            indices = mb.cast(x=indices, dtype="int32")
        res = mb.gather_along_axis(x=inputs[0], indices=indices, axis=inputs[1], name=node.name)
        context.add(res)


def convert_model(source_dir, output_path, seq_len=128, max_options=4):
    from rl_agent_api import RLAgent
    register_compatibility_ops()

    print(f"Loading PyTorch model from {source_dir}...")
    agent = RLAgent(source_dir, device="cpu")

    B, L, K = 1, seq_len, max_options
    sample_args = (
        torch.ones((B, L), dtype=torch.long),
        torch.ones((B, L), dtype=torch.long),
        torch.zeros((B, K), dtype=torch.long),
        torch.ones((B, K), dtype=torch.bool),
        torch.zeros((B,), dtype=torch.long),
    )

    print("Exporting torch graph via torch.export...")
    exp = torch.export.export(agent.model, sample_args)
    exp = exp.run_decompositions()

    print(f"Converting to Core ML ({output_path})...")
    mlmodel = ct.convert(
        exp,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(output_path)
    print(f"Saved Core ML package to {output_path}")
    return mlmodel


def benchmark_coreml(model_path, seq_len=128, max_options=4, samples=30):
    print(f"Benchmarking Core ML model at {model_path}...")
    t0 = time.perf_counter()
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
    load_time = time.perf_counter() - t0
    print(f"Load time: {load_time:.3f} s")

    B, L, K = 1, seq_len, max_options
    inputs = {
        "input_ids": np.ones((B, L), dtype=np.int32),
        "attention_mask": np.ones((B, L), dtype=np.int32),
        "marker_pos": np.zeros((B, K), dtype=np.int32),
        "marker_mask": np.ones((B, K), dtype=np.float32),
        "qtype": np.zeros((B,), dtype=np.int32),
    }

    # Warmup
    for _ in range(5):
        _ = model.predict(inputs)

    # Timed runs
    timings = []
    for _ in range(samples):
        t0 = time.perf_counter()
        _ = model.predict(inputs)
        timings.append((time.perf_counter() - t0) * 1000.0)

    print(f"Core ML Latency ({samples} samples, {seq_len} tokens):")
    print(f"  p50:  {np.median(timings):.2f} ms")
    print(f"  mean: {np.mean(timings):.2f} ms")
    print(f"  min:  {np.min(timings):.2f} ms")
    print(f"  max:  {np.max(timings):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Core ML conversion and benchmark tool for Laya")
    parser.add_argument("--source", default="source", help="source directory with original PyTorch checkpoint")
    parser.add_argument("--output", default="laya_decision.mlpackage", help="path to save .mlpackage")
    parser.add_argument("--benchmark", default=None, help="path to .mlpackage to benchmark")
    parser.add_argument("--seq-len", type=int, default=128, help="sequence length for export/benchmark")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_coreml(args.benchmark, seq_len=args.seq_len)
    else:
        convert_model(args.source, args.output, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
