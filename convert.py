"""Convert the original Laya RL-agent checkpoint (PyTorch safetensors) to the MLX layout.

The MLX port uses an identity key mapping — every tensor keeps its original name
and shape — so conversion is a verified dtype cast plus file copy. The script:

  * asserts the source tensor set exactly matches the set implied by
    encoder/config.json + rl_agent_config.json (count, names, shapes);
  * casts floating tensors to the requested dtype (the `temperature` buffer stays
    float32, matching the original where it is a registered buffer);
  * refuses to overwrite: the output directory must not exist; output is staged
    in a sibling temp dir and atomically renamed;
  * writes conversion_metadata.json recording the source revision, source file
    SHA-256, the identity mapping, and per-tensor dtype/shape provenance.

CLI:
    python convert.py --source source --output converted --dtype float32
    python convert.py --source source --output converted-fp16 --dtype float16
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, cast

import mlx.core as mx
from safetensors import safe_open

SOURCE_REPO = "convaiinnovations/laya"
SOURCE_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

# Files the MLX inference path needs (configs + tokenizer). The original
# rl_common.py / rl_agent_api.py are copied too as the semantic reference.
CONFIG_FILES = ["encoder/config.json", "rl_agent_config.json"]
TOKENIZER_FILES = ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
REFERENCE_FILES = ["rl_common.py", "rl_agent_api.py"]

_DTYPES: dict[str, mx.Dtype] = {"float32": mx.float32, "float16": mx.float16}
# Tensors that are buffers, not parameters: keep float32 regardless of --dtype.
BUFFER_KEYS = {"temperature"}


def expected_tensors(encoder_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> dict[str, list[int]]:
    """The exact {name: shape} set implied by the configs (mirrors rl_common.build_model)."""
    d = encoder_cfg["hidden_size"]
    n_layers = encoder_cfg["num_hidden_layers"]
    inter = encoder_cfg["intermediate_size"]
    vocab = encoder_cfg["vocab_size"]
    exp: dict[str, list[int]] = {
        "encoder.embeddings.tok_embeddings.weight": [vocab, d],
        "encoder.embeddings.norm.weight": [d],
        "encoder.final_norm.weight": [d],
        "type_emb.weight": [3, d],
        "temperature": [3],
        "scorer.0.weight": [d], "scorer.0.bias": [d],
        "scorer.1.weight": [d, d], "scorer.1.bias": [d],
        "scorer.3.weight": [1, d], "scorer.3.bias": [1],
        "act_head.0.weight": [256, d + 4], "act_head.0.bias": [256],
        "act_head.2.weight": [len(agent_cfg["act_costs"]) + 1, 256],
        "act_head.2.bias": [len(agent_cfg["act_costs"]) + 1],
    }
    if encoder_cfg.get("norm_bias", False):
        exp["encoder.embeddings.norm.bias"] = [d]
        exp["encoder.final_norm.bias"] = [d]
    for i in range(n_layers):
        p = "encoder.layers.%d." % i
        if i > 0:  # layer 0 has Identity attn_norm in ModernBERT
            exp[p + "attn_norm.weight"] = [d]
            if encoder_cfg.get("norm_bias", False):
                exp[p + "attn_norm.bias"] = [d]
        exp[p + "attn.Wqkv.weight"] = [3 * d, d]
        exp[p + "attn.Wo.weight"] = [d, d]
        exp[p + "mlp_norm.weight"] = [d]
        exp[p + "mlp.Wi.weight"] = [2 * inter, d]
        exp[p + "mlp.Wo.weight"] = [d, inter]
        if encoder_cfg.get("attention_bias", False):
            exp[p + "attn.Wqkv.bias"] = [3 * d]
            exp[p + "attn.Wo.bias"] = [d]
        if encoder_cfg.get("mlp_bias", False):
            exp[p + "mlp.Wi.bias"] = [2 * inter]
            exp[p + "mlp.Wo.bias"] = [d]
        if encoder_cfg.get("norm_bias", False):
            exp[p + "mlp_norm.bias"] = [d]
    for i in range(agent_cfg["head_layers"]):
        p = "head.layers.%d." % i
        exp[p + "self_attn.in_proj_weight"] = [3 * d, d]
        exp[p + "self_attn.in_proj_bias"] = [3 * d]
        exp[p + "self_attn.out_proj.weight"] = [d, d]
        exp[p + "self_attn.out_proj.bias"] = [d]
        exp[p + "linear1.weight"] = [4 * d, d]
        exp[p + "linear1.bias"] = [4 * d]
        exp[p + "linear2.weight"] = [d, 4 * d]
        exp[p + "linear2.bias"] = [d]
        exp[p + "norm1.weight"] = [d]
        exp[p + "norm1.bias"] = [d]
        exp[p + "norm2.weight"] = [d]
        exp[p + "norm2.bias"] = [d]
    return exp


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(source: str, output: str, dtype_name: str) -> None:
    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if os.path.exists(output):
        raise SystemExit("error: output path already exists (overwrite forbidden): %s" % output)
    src_st = os.path.join(source, "model.safetensors")
    for rel in ["model.safetensors"] + CONFIG_FILES + TOKENIZER_FILES:
        if not os.path.isfile(os.path.join(source, rel)):
            raise SystemExit("error: missing required source file: %s" % os.path.join(source, rel))

    with open(os.path.join(source, "encoder", "config.json")) as f:
        encoder_cfg: dict[str, Any] = json.load(f)
    with open(os.path.join(source, "rl_agent_config.json")) as f:
        agent_cfg: dict[str, Any] = json.load(f)
    expected = expected_tensors(encoder_cfg, agent_cfg)

    # Pass 1: metadata-only check of names/shapes (no tensor materialization).
    with safe_open(src_st, framework="numpy") as f:
        actual = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    bad_shape = sorted(k for k in expected if k in actual and actual[k] != expected[k])
    if missing or extra or bad_shape:
        msg = ["error: source checkpoint does not match the expected tensor set"]
        if missing:
            msg.append("  missing: %s" % ", ".join(missing))
        if extra:
            msg.append("  unexpected: %s" % ", ".join(extra))
        if bad_shape:
            msg.append("  wrong shape: %s" % ", ".join(
                "%s expected %s got %s" % (k, expected[k], actual[k]) for k in bad_shape))
        raise SystemExit("\n".join(msg))

    # Pass 2: load, cast, verify.
    dtype = _DTYPES[dtype_name]
    weights = cast(dict[str, mx.array], mx.load(src_st))
    assert set(weights) == set(expected), "mx.load key set drifted from safe_open"
    out: dict[str, mx.array] = {}
    mapping: dict[str, dict[str, Any]] = {}
    for k in sorted(weights):
        v = weights[k]
        assert list(v.shape) == expected[k], "%s shape %s != expected %s" % (k, v.shape, expected[k])
        src_dtype = str(v.dtype)
        if k in BUFFER_KEYS:
            nv = v.astype(mx.float32)
        elif mx.issubdtype(v.dtype, mx.floating):
            nv = v.astype(dtype)
        else:
            nv = v
        out[k] = nv
        mapping[k] = {"source_name": k, "shape": expected[k],
                      "source_dtype": src_dtype, "dtype": str(nv.dtype)}
    mx.eval(*out.values())

    # Stage in a sibling temp dir, then atomically rename (no overwrite path).
    stage = tempfile.mkdtemp(prefix=os.path.basename(output) + ".tmp-",
                             dir=os.path.dirname(output) or ".")
    try:
        mx.save_safetensors(os.path.join(stage, "model.safetensors"), out)
        for rel in CONFIG_FILES + TOKENIZER_FILES + REFERENCE_FILES:
            s = os.path.join(source, rel)
            if not os.path.isfile(s):
                continue
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(s, dst)
        meta = {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_model_safetensors_sha256": sha256_file(src_st),
            "converter": "convert.py (identity key mapping, dtype cast only)",
            "dtype": dtype_name,
            "buffer_keys_kept_float32": sorted(BUFFER_KEYS),
            "tensor_count": len(out),
            "mapping": mapping,
            "created_unix": int(time.time()),
        }
        with open(os.path.join(stage, "conversion_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        if os.path.exists(output):  # raced creation; still refuse
            raise SystemExit("error: output path appeared during conversion: %s" % output)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print("converted %d tensors -> %s (dtype=%s)" % (len(out), output, dtype_name))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert Laya checkpoint to MLX safetensors.")
    ap.add_argument("--source", required=True, help="source model dir (HF snapshot layout)")
    ap.add_argument("--output", required=True, help="output dir (must not exist)")
    ap.add_argument("--dtype", default="float32", choices=sorted(_DTYPES),
                    help="compute dtype for floating tensors (default: float32)")
    args = ap.parse_args(argv)
    convert(args.source, args.output, args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
