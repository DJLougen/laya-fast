# Laya Fast

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Local typed decisions on Apple Silicon. One forward pass, no generated JSON, and — when the optional Neural Engine bodies are compiled — both the Neural Engine and the GPU working at once.

This is an independent runtime for [Convai Innovations' Laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large, 421M, Apache-2.0). It is not an official Convai release, and it is not [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx). That port is MLX-only; this one adds a Neural Engine body and a router that runs the two engines together.

## What it is

Laya answers a constrained question about a piece of text — a choice between labeled options, a rubric score, or a yes/no with a probability. It does not generate tokens. Latency is one bidirectional encoder pass plus a small decision head, and the answer comes back as typed fields, not text you have to parse.

Two runtimes share the same request and response shape:

- **`LayaMLX`** (`--agent mlx`, the CLI default) is the GPU-only path — the first-run route, always available after conversion.
- **`LayaFast`** (`--agent fast`) is the accelerated route: it opts into the ANE + GPU router.
  - One short question (≤128 tokens) runs on the Neural Engine when its compiled body is present.
  - A batch is split: a worker runs some questions on the Neural Engine while the GPU batches the rest. The split comes from a measured cost model (`ane/cost_model.json`).
  - A single long question stays on the GPU — at 512 tokens the two engines tie, so there is nothing to overlap.
  - Anything longer than the largest compiled bucket stays on the GPU.
  - **No Neural Engine bodies? Everything runs on the GPU.** Missing bodies are not an error; `LayaFast` degrades to the same path as `--agent mlx`.

## Why typed local decisions matter

A hosted judge means a network dependency, per-call latency, and text you still have to parse and validate. A local encoder answers offline in milliseconds, with typed probabilities you can threshold, log, and unit-test. That makes it practical to put a decision step — routing, gating, classification — inside a fast local loop instead of behind an API call.

## Measured speed

On the author's M3 Max (30 GPU cores, 36 GB), fp16, median of the project fixtures (2026-09-22, both gates passing):

| Fixture | MLX only | LayaFast |
|---|---:|---:|
| 1 short question (74 tokens) | 13.9 ms | **9.7 ms** |
| 8 short questions | 65 ms | **41 ms** |
| 1 long question (512 tokens) | 54 ms | 54 ms |
| 8×512 | 374 ms | **219 ms** |

Caveats:

- These are single-machine fixture medians, not a benchmark suite. Your numbers depend on chip, memory, and load.
- The 8×512 win is a 4/4 split across two engines, not a faster kernel — it requires the compiled ANE bodies.
- Process footprint was about 1.5 GB with the MLX buffer cache capped at 128 MB.

[summary.md](summary.md) is the detailed technical benchmark history, including approaches that were tried and rejected.

## Speed is not quality

These numbers measure latency, not accuracy. On a 374-decision smoke set with synthetic labels created for this benchmark (not independently human-annotated), the stock English checkpoint scored 69% and the hosted Jev 1.13.0 scored 93%. That gap is specific to this set, not a general quality verdict — but treat this runtime as a fast local decision layer, not a drop-in quality match for a hosted judge, and evaluate it on your own data before trusting its answers.

## Prerequisites

- Apple Silicon Mac, macOS, Python 3.12
- [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (installed with `huggingface_hub` in `requirements.txt`)
- ~2 GB disk for the converted fp16 weights; ~700 MB per Neural Engine bucket if you compile them

## Install and first run

```bash
git clone https://github.com/DJLougen/laya-fast.git
cd laya-fast
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the official English checkpoint and convert it to MLX fp16:

```bash
.venv/bin/hf download convaiinnovations/laya --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir source
.venv/bin/python convert.py --source source --output converted-fp16 --dtype float16
```

Run a request. `examples/questions.json` is a questions-only map, used with `--state`:

```bash
.venv/bin/python laya_api.py \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

`examples/request.json` is a full `{state, questions}` document — pipe it on stdin (or pass it with `--input`):

```bash
.venv/bin/python laya_api.py < examples/request.json
```

The CLI defaults to `--agent mlx` (GPU only). Opt into the accelerated route with `--agent fast` (or `LAYA_AGENT=fast`) — it uses any compiled ANE bodies and falls back to the GPU where none exist:

```bash
.venv/bin/python laya_api.py --agent fast \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

## Python API

```python
from laya_fast import LayaFast

agent = LayaFast("converted-fp16")
print(agent.system_one(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs and outages",
            },
        }
    },
))
```

Question types are `choice`, `score`, and `noul` (P(true)). `choice` criteria can be a `{label: description}` object or a list of labels; `score` criteria is a list of level descriptions; `noul` criteria may be `{"false": ..., "true": ...}` or omitted.

The response mirrors the request: `{"model": "rl-agent", "answers": {id: {...}}, "usage": {...}}`. Each answer carries its typed result — `choice` plus per-label `probabilities`, `score`, or the `noul` probability — plus an `rl_agent` extension with the model's action probability. `choice` and `score` answers also include a `confidence` (1 − normalized entropy).

## What's in the repo — and what isn't

**Included:** the MLX encoder with custom Metal kernels (`laya_mlx.py`), the Neural Engine export/runtime and router (`ane/`, `laya_fast.py`), the Jev-shaped request API and CLI (`laya_api.py`), the converter (`convert.py`), benchmarks, fixtures, and examples.

**Not included — you build them locally:**

- **Model weights.** `hf download` + `convert.py` is the supported path; nothing in git contains weights.
- **Compiled Neural Engine bodies** (`ane/body*/`). Each is a ~700 MB Core ML program produced by `ane/export.py` (see below). Without them, `LayaFast` still works — every request just stays on the GPU.

## Make it your own

Fork the repo. The seams:

| If you want to… | Start here |
|---|---|
| Change routing | `laya_fast.py` — `ANE_MS`, `SINGLE_MAX_BUCKET`, `_mlx_batch_ms` |
| Change the GPU model | `laya_mlx.py` — fused RoPE and GeGLU kernels |
| Change the request API | `laya_api.py` — `LayaMLX.system_one` |
| Export a new Neural Engine bucket | `ane/export.py` |
| Re-measure the cost model | `ane/measure_costs.py`, then edit `ane/cost_model.json` |
| Gate a change against decision parity + speed | `bash autoresearch.sh` |

### Neural Engine bodies (optional acceleration)

Each bucket is a fixed-length Core ML program. Neural Engine support needs the optional dependencies — install them on top of the base requirements (this also enables loading ANE bodies under `--agent fast`):

```bash
.venv/bin/pip install -r requirements-export.txt
```

After `converted-fp16` exists, export the buckets:

```bash
for L in 64 80 96 128 256 512; do
  .venv/bin/python ane/export.py --model converted-fp16 --length $L --output ane/body$L
done
```

Export is slow — minutes per length, and the first compile is the expensive one. `LayaFast` uses whichever of `ane/body{64,80,96,128,256,512}` exist; a missing bucket is not an error, that length just stays on the GPU.

To add a new length: export it, add it to the `ane_buckets` tuple in `LayaFast`, and put a measured milliseconds-per-question entry in `ANE_MS`. Only an in-context A/B through `bash autoresearch.sh` counts — a chain of standalone GEMM timings does not. [summary.md](summary.md) lists the dead ends from the first pass (palettization, batched ANE bodies, finer buckets, tall GEMM outside one projection).

### A different checkpoint

`convert.py` knows the English Laya layout (`convaiinnovations/laya`). A fine-tune with the same tensor names converts the same way. A different encoder (the multilingual mmBERT checkpoint, for example) needs a new body in `ane/ane_model.py` and a new export.

### Voice routing demo

`laya_server.py` and `voice_commands.json` are an optional local HTTP front (default `127.0.0.1:8765`, override with `LAYA_PORT`) that maps a transcript to one action. Edit the catalog to taste; it is not required to use the model.

## Troubleshooting

- **`converted-fp16` not found** — run the `hf download` + `convert.py` steps above, or point `--model` at your converted directory.
- **No speedup from `fast`** — check that `ane/body*/` exists; without bodies the fast agent is the GPU path. First calls also pay `mx.compile` tracing (~50–100 ms per new shape); steady-state numbers are what the fixtures report.
- **dtype** — fp16 is the benchmarked and recommended configuration; the table above does not describe float32.
- **Before changing kernels or routing** — read [CONTRIBUTING.md](CONTRIBUTING.md) for the parity-first gate, and [summary.md](summary.md) for what has already been measured. Bugs and questions: [GitHub issues](https://github.com/DJLougen/laya-fast/issues).

## Layout

```
laya_api.py          Jev-shaped API and CLI
laya_fast.py         ANE + GPU router (--agent fast)
laya_mlx.py          MLX encoder, custom Metal kernels
ane/                 Neural Engine export and runtime
convert.py           PyTorch safetensors -> MLX
benchmark.py         parity and runtime arms
autoresearch.sh      fail-closed speed/parity gate
summary.md           detailed measurement log, including failures
examples/            questions-only and full-request JSON
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Laya's weights stay under Convai's Apache-2.0 license and are downloaded separately.
