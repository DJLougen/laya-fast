# Laya Fast

Local typed decisions on Apple Silicon. One forward pass, no generated JSON, both the Neural Engine and the GPU used at once.

This is an independent runtime for [Convai Innovations' Laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large, 421M, Apache-2.0). It is not an official Convai release, and it is not [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx). That port is MLX-only. This one adds a Neural Engine body and a router that runs the two engines together.

## What it is

Laya answers a constrained question — a choice, a rubric score, or a yes/no with a probability — from a piece of text. It does not write tokens. Latency is one bidirectional encoder pass plus a small decision head.

`LayaFast` is the default:

- One short question (≤128 tokens) runs on the Neural Engine.
- A batch is split. A worker runs some questions on the Neural Engine while the GPU batches the rest. The split is picked from a measured cost model (`ane/cost_model.json`).
- A single long question stays on the GPU. At 512 tokens the two engines tie, so there is nothing to overlap.
- Anything longer than the largest compiled bucket stays on the GPU.

`LayaMLX` is the GPU-only path (`--agent mlx`). Same request and response shape.

## Why it matters

A hosted judge is hundreds of milliseconds and a network hop. A local encoder is tens of milliseconds, and the answer is a typed probability, not text you have to parse.

On the author's M3 Max (30 GPU cores, 36 GB), fp16, median of the project fixtures (2026-09-22, both gates passing):

| Fixture | MLX only | LayaFast |
|---|---:|---:|
| 1 short question (74 tokens) | 13.9 ms | **9.7 ms** |
| 8 short questions | 65 ms | **41 ms** |
| 1 long question (512 tokens) | 54 ms | 54 ms |
| 8×512 | 374 ms | **219 ms** |

The 8×512 win is a 4/4 split across the two engines, not a faster kernel. Process footprint was about 1.5 GB with the MLX buffer cache capped at 128 MB.

These numbers are not a controlled comparison against the public MLX port. That port reports **13.42 ms** p50 for a short English question on a **40-core, 128 GB** M3 Max, and the harness is different (tokenization and formatting included). On the weaker chip here, the MLX-only path landed in the same range. The Neural Engine router is the part that port does not ship. Convai's own published figure is about 33–40 ms p50 on a Tesla T4.

Speed is not accuracy. On a 374-decision smoke set with agent-written labels (not human gold), the stock English checkpoint scored 69% and Jev 1.13.0 scored 93%. Do not treat this runtime as a drop-in quality match for a hosted judge. `summary.md` is the measurement log, including what was tried and rejected.

## How to use it

Apple Silicon, macOS, Python 3.12. The GPU path needs MLX. The fast path also needs the compiled Neural Engine bodies (next section).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
hf download convaiinnovations/laya --local-dir source
.venv/bin/python convert.py --source source --output converted-fp16 --dtype float16
```

```bash
.venv/bin/python laya_api.py \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

`examples/questions.json` is only the questions map. `examples/request.json` is a full `{state, questions}` document — pipe that on stdin:

```bash
.venv/bin/python laya_api.py < examples/request.json
```

GPU only, no Neural Engine:

```bash
.venv/bin/python laya_api.py --agent mlx \
  --state "I was billed twice. Please refund the duplicate." \
  --questions examples/questions.json
```

From Python:

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

Question types are `choice`, `score`, and `noul` (P(true)). `choice` criteria can be a `{label: description}` object or a list of labels.

Weights are not in this repo. `hf download` plus `convert.py` is the supported way to get them. With no compiled Neural Engine bodies, `LayaFast` still runs: every request stays on the GPU.

## How to make your own

Fork this repo. The interesting seams:

| If you want to… | Start here |
|---|---|
| Change routing | `laya_fast.py` — `ANE_MS`, `SINGLE_MAX_BUCKET`, `_mlx_batch_ms` |
| Change the GPU model | `laya_mlx.py` — fused RoPE and GeGLU kernels |
| Change the request API | `laya_api.py` — `LayaMLX.system_one` |
| Export a new Neural Engine bucket | `ane/export.py` |
| Re-measure the cost model | `ane/measure_costs.py`, then edit `ane/cost_model.json` |
| Refuse a change that breaks decisions | `bash autoresearch.sh` |

### Neural Engine bodies

Each bucket is a fixed-length Core ML program, about 700 MB, so they are not in git. After `converted-fp16` exists:

```bash
for L in 64 80 96 128 256 512; do
  .venv/bin/python ane/export.py --model converted-fp16 --length $L --output ane/body$L
done
```

Export is slow (minutes per length, and the first compile is the expensive one). `LayaFast` uses whichever of `ane/body{64,80,96,128,256,512}` is already compiled. A missing bucket is not an error: that length stays on the GPU. With no buckets at all, every request stays on the GPU, which is the same path as `--agent mlx`.

A new length is: export it, add it to the `ane_buckets` tuple in `LayaFast`, and put a measured milliseconds-per-question entry in `ANE_MS`. Do not trust a chain of standalone GEMMs. Only an in-context A/B against `bash autoresearch.sh` counts. `summary.md` lists the dead ends from the first pass (palettization, batched ANE bodies, finer buckets, tall GEMM outside one projection).

### A different checkpoint

`convert.py` knows the English Laya layout (`convaiinnovations/laya`). A fine-tune with the same tensor names converts the same way. A different encoder (the multilingual mmBERT checkpoint, for example) needs a new body in `ane/ane_model.py` and a new export. The public MLX port already runs that smaller checkpoint; this tree has not.

### Voice routing

`laya_server.py` and `voice_commands.json` are an optional local HTTP front (default `127.0.0.1:8765`) that maps a transcript to one action. Edit the catalog. It is not required to use the model.

## Layout

```
laya_api.py          Jev-shaped API and CLI
laya_fast.py         ANE + GPU router (default)
laya_mlx.py          MLX encoder, custom Metal kernels
ane/                 Neural Engine export and runtime
convert.py           PyTorch safetensors -> MLX
benchmark.py         parity and runtime arms
autoresearch.sh      fail-closed speed/parity gate
summary.md           what was measured, including failures
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Laya's weights stay under Convai's Apache-2.0 license and are downloaded separately.
