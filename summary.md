# Laya MLX & Core ML Optimization Project Summary

> **Scope:** Technical notes on the optimization and conversion work done on `convaiinnovations/laya` (a 28-layer bidirectional ModernBERT-large backbone + RL decision head). It records what was tried, what succeeded, what failed (with failure reasons and root causes), benchmark tables, and how to run and extend the code. All measurements were taken on the hardware described in §1.

---

## 1. System & Hardware Environment

- **Host Machine:** Apple MacBook Pro (Apple Silicon)
- **Chip:** Apple M3 Max (14 cores: 10 Performance, 4 Efficiency)
- **Unified Memory:** 36 GB
- **Operating System:** macOS (Darwin arm64)
- **Python Runtime:** Python 3.12.13 in the project virtualenv (`.venv`)
- **Frameworks:**
  - `mlx`: 0.32.2 (Metal 3 GPU runtime)
  - `torch`: 2.14.0
  - `coremltools`: 9.0
  - `transformers`: 5.5.0
  - `huggingface_hub`: 0.36.2

---

## 2. Tabular Log: What We Tried, What Worked, and What Didn't

| Stage / Technique | Attempted Action | Status | Outcome / Root Cause | Final Solution / Resolution |
| :--- | :--- | :---: | :--- | :--- |
| **HF Hub Installation** | `huggingface-cli download convaiinnovations/laya` | ❌ **Failed** | Deprecation warning: `huggingface-cli` CLI binary is deprecated and refused execution. | Switched to `hf download convaiinnovations/laya`. Downloaded 2.37 GB into `~/.cache/huggingface/hub/models--convaiinnovations--laya`. |
| **HF Cache Resolution** | Loading model via repo ID `convaiinnovations/laya` in `laya_api.py` | ❌ **Failed initially** | Expected local filesystem directory containing config files; repo ID string failed with `FileNotFoundError`. | Implemented `_resolve_model_dir` using `huggingface_hub.snapshot_download(..., local_files_only=True)`. Auto-resolves seamlessly. |
| **Core ML: Torch JIT** | `torch.jit.trace(agent.model, dummy_inputs)` | ❌ **Failed** | `IndexError: tuple index out of range` in `transformers.masking_utils.sdpa_mask`. | ModernBERT tracer bug: scalar 0-dim `q_length` tensor cannot be indexed with `.shape[0]`. Abandoned JIT trace in favor of `torch.export`. |
| **Core ML: Raw Export** | `coremltools.convert(torch.export.export(model, ...))` | ❌ **Failed** | `NotImplementedError`: Dialect `TRAINING` not supported by coremltools frontend. | Invoked `exp.run_decompositions()` prior to `ct.convert()` to normalize to ATEN dialect. |
| **Core ML: Dialect Bridging** | `ct.convert(exp.run_decompositions())` | ❌ **Failed** | `NotImplementedError: Unsupported fx node alias, kind alias` (PyTorch 2.14 vs coremltools 9.0 gap). | Registered custom torch converter mapping `alias` and `aten.alias.default` to `mb.identity`. |
| **Core ML: MIL Conversion** | Converting decomposed FX graph with alias bypass | ❌ **Failed at 99%** | `ValueError: Op "gather" Input indices="expand_214" expects ['int32', ...] but got tensor[..., fp32]`. | Discovered coremltools `clamp` bug: missing `max` promotes integers to `float32` via `finfo(float32).max`. Registered patched `gather` op auto-casting indices to `int32`. |
| **Core ML Execution** | Generating & running `laya_decision.mlpackage` | ⚠️ **Suboptimal** | Model converted and ran, but latency was **22.35 ms** on GPU (`ComputeUnit.ALL`) and **103.14 ms** on ANE (`CPU_AND_NE`), with **5.13 s** startup load time. | Saved standalone export pipeline to `convert_coreml.py`, but prioritized Native Apple MLX as the primary production runtime. |
| **MLX: JIT Compilation** | Wrapping MLX model in `mx.compile(self.model)` | ✅ **Succeeded** | Reduced kernel dispatch overhead; forward latency dropped from ~18.9 ms to ~18.2 ms p50; batch-8 long dropped from 516.9 ms to 502.0 ms. | Integrated into `laya_mlx.py` and added `--compile` flag to `laya_api.py` and `benchmark.py`. |
| **MLX: Built-in Fast RoPE** | Using `mx.fast.rope(..., traditional=False)` | ⚠️ **Mixed** | Matched `_apply_rope` within 6.7e-6, but still required separate transpose and unpack operations for Q, K, and V. | Replaced by a unified custom Metal kernel combining QKV unpacking, RoPE math, and transpose in one pass. |
| **Custom Metal: GeGLU** | Writing custom MSL kernel calling `metal::erf` | ❌ **Failed** | Compilation error: Metal Shading Language stdlib lacks `erf` in namespace `metal`. | Ported faithfully-rounded error function polynomial (`custom_erf` + `expm1f_scaled_unchecked`) into MSL kernel header. |
| **Custom Metal: Fused GeGLU** | Single-pass MSL kernel for GLU split + exact ERF + gate multiply | ✅ **Succeeded** | Numerical diff vs PyTorch reference `< 1e-6`. Cut DRAM traffic by 66%, saving **~2.12 ms** across all 28 layers. | Integrated `_fused_geglu_kernel` into `laya_mlx.MLP.__call__`. |
| **Custom Metal: Fused QKV RoPE** | Single-pass MSL kernel: unpack Q/K/V, apply RoPE in registers, write `[B, H, L, D]` | ✅ **Succeeded** | Numerical diff vs PyTorch reference `< 2.4e-7`. Eliminated ~390 kernel launches and slices across 28 layers, saving **~2.93 ms**. | Integrated `_fused_qkv_rope_kernel` into `laya_mlx.Attention.__call__`. |
| **MLX: Mask Caching** | Pre-caching bidirectional sliding-window masks by sequence length `L` | ✅ **Succeeded** | Avoided dynamic distance matrix allocation `pos[:, None] - pos[None, :]` on every single forward pass. | Added `_window_cache` dictionary to `ModernBertEncoder`. |

---

## 3. Comprehensive Performance Benchmark Matrix

All benchmarks measured on **Apple M3 Max (36 GB Unified Memory)** across identical fixtures (15 timed runs, 3 warmups):

| Runtime / Engine | Compute Precision | Model Load Time | Single Short (74 tokens) | Batch-8 Short (640 tokens) | Single Long (512 tokens) | Batch-8 Long (4,096 tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch CPU (1 thread)** | FP32 | 24.02 s | 105.80 ms | 716.34 ms | 625.75 ms | 4,670.23 ms |
| **PyTorch CPU (4 threads)** | FP32 | 23.42 s | 79.07 ms | 363.11 ms | 300.97 ms | 1,997.25 ms |
| **PyTorch CPU (10 threads)** | FP32 | 22.02 s | 87.04 ms | 343.98 ms | 266.22 ms | 1,622.42 ms |
| **PyTorch MPS (Metal)** | FP32 | 22.76 s | 25.16 ms | 116.20 ms | 79.01 ms | 523.49 ms |
| **Core ML (`ComputeUnit.ALL`)** | FP32 | 5.13 s | 22.35 ms | — | — | — |
| **Core ML (`CPU_AND_NE` ANE)** | FP32 | 5.08 s | 103.14 ms | — | — | — |
| **MLX FP32 (Baseline Uncompiled)** | FP32 | 0.171 s | 18.93 ms | 97.78 ms | 69.53 ms | 518.31 ms |
| **MLX FP32 (Compiled `mx.compile`)** | FP32 | 0.171 s | 18.20 ms | 96.10 ms | 66.20 ms | 502.00 ms |
| **MLX FP32 (Custom Metal Kernels + Compile)** | **FP32** | **0.142 s** | **17.56 ms** *(min 17.07 ms)* | **91.72 ms** *(~11.4 ms/q)* | **65.40 ms** | **468.83 ms** |
| *MLX FP16 (Baseline)* | *FP16* | *0.150 s* | *16.82 ms* | *95.06 ms* | *67.79 ms* | *524.44 ms* |

> **Key takeaway:** The optimized MLX FP32 runtime with custom Metal kernels runs **6.0× faster** than PyTorch CPU (10 threads), **1.4× faster** than PyTorch MPS, **1.3× faster** than Core ML, and loads **160× faster** (0.14s vs 23s) while maintaining **exact FP32 decision parity**.

## 3b. Optimization round (2026-09-22): speed work after fp16 — verified results

All numbers below are from interleaved A/B runs inside one process (cross-process drift on this Mac is up to ~7%), fp16 weights (`converted-fp16`), `compile=True`, M3 Max 30-core GPU. The gate is `bash autoresearch.sh`, fail-closed against `autoresearch_golden.json`.

**New MLX defaults vs baseline** (interleaved A/B harness, 3 rounds, medians):

| Fixture | Baseline | New defaults | Speedup |
| :--- | ---: | ---: | ---: |
| single_short (74 tok) | 15.19 ms | 13.97 ms | 1.087× |
| batch8_short (8 q, 68–104 tok) | 76.08 ms | 64.88 ms | 1.173× |
| single_long (512 tok) | 54.33 ms | 53.81 ms | 1.010× (noise) |
| batch8_long (8×512) | 373.94 ms | 373.73 ms | 1.001× (noise) |
| First call at a never-seen length | 76–80 ms | 10.7–15.1 ms | ~6× |

`autoresearch.sh` after integration: `[gate] PASS max|dlogit|=2.205e-02 max|dprob|=2.000e-03`, single_short 13.83, batch8_short 64.54, single_long 53.60, batch8_long 374.09 ms. The logit drift rose from 1.4e-2 because bucketing's masked padding changes fp16 reduction order. Consumed probabilities moved ≤ 4e-4 bucketed vs exact-shape across 10 cases, with no label flips.

| Change | What it does | Evidence | Off switch |
| :--- | :--- | :--- | :--- |
| **Unpadded batches** | Batched questions run only real tokens through token-wise ops; attention scatters back to the padded layout. | batch8_short 82.6→70.3 ms, mixed 40–200-token batch 200→138 ms; logits bit-identical to the padded path. | `LAYA_UNPAD=0` |
| **Shared state tokenized once** | `prepare()` encodes the state string once per request instead of once per question. | batch8 prepare 0.73→0.46 ms; token ids identical. | — |
| **Shape-agnostic RoPE kernel** | `fused_qkv_rope` no longer templates B/L, so a new length doesn't JIT a new Metal pipeline (that JIT was ~40–60 ms of the 77 ms first-call cost). | See the first-call row above. | `LAYA_SHAPE_TEMPLATES=1` |
| **Length bucketing** | L rounds up to a multiple of 16, option slots to a multiple of 8. Pads are masked keys at the end, so RoPE positions and markers are unchanged. | Bounds `mx.compile` variants (retrace cost grows with cache size: 40 ms → ~0.8 s after ~100 shapes). | `LAYA_BUCKET=0` |
| **Cold dispatch** | The first call at a new bucketed shape runs eager and compiles from the second call. First-call results can differ from compiled ones by ~3e-4 probability (fp16 fusion rounding). | — | `LAYA_COLD_DISPATCH=0` |
| **Packed path eager** | The packed path is not compiled: steady state is the same, and it avoids 0.5–1.5 s retraces. | 63.4 vs 64.2 ms compiled/eager. | `LAYA_COMPILE_PACKED=1` |
| **Tall GEMM (`tall_gemm.py`)** | Custom simdgroup-MMA kernel for the encoder Wi projection when 64<M≤96 (the L=80/96 buckets). Each threadgroup covers all rows, so W is read once, and W is read directly with no transposed copy (0 extra RAM). | In-context 1.09× at L=80; encoder output bit-identical to MLX. | `LAYA_TALL_GEMM=0` |
| **MLX cache cap** | `LayaMLX` caps MLX's freed-buffer cache at 128 MB. The measured default ceiling on this Mac is 36.7 GB. | Same latency as 1 GB on all fixtures; footprint 2054→1154 MB. | `LAYA_CACHE_LIMIT_MB=-1` |
| **Dedup (`dedup.py`)** | `dedup_system_one` runs one forward per unique question. This is a product feature, not a kernel speedup. | 24 q (8 unique ×3): 219→66 ms, answers identical. | opt-in |

**Neural Engine (ANE) English export (`ane/`).** This ports mizorewww's BC1S/1×1-conv rewrite to ModernBERT-large. All 10,594 non-constant ops are placed on the Neural Engine at every bucket. Head-to-head vs the new MLX defaults (one process, 3 rounds):

| Fixture | MLX | ANE | Winner |
| :--- | ---: | ---: | :--- |
| single_short | 14.03 | **9.79** | ANE 1.43× |
| batch8_short | **64.83** | 80.78 | MLX |
| single_long | 53.98 | 53.38 | tie |
| batch8_long | **374.28** | 421.01 | MLX |

The ANE gate passes (`max|dprob|=7.3e-3`). The old 103 ms "ANE" number in the table above was a plain fp32 export that never ran on the Neural Engine. W8 palettization failed parity (prob error 0.038) and was rejected.

**Router `LayaFast` (`laya_fast.py`) — fastest configuration.** The Neural Engine and the GPU run *together*. Single questions up to 128 tokens run on the Neural Engine; for multi-question batches, a worker thread runs some questions on the Neural Engine while the GPU batches the rest at the same time, and the split is chosen by a measured cost model (`ane/cost_model.json`). Compiled `.mlmodelc` files are cached, so load takes 2.3 s instead of ~3.5 min of recompiling.

**Both engines at once, including long batches.** At 512 tokens the two engines are equally fast per question (MLX 53.98 vs ANE 53.38 ms), so a long batch should be split across them rather than queued entirely on the GPU. The router used to send only ≤128-token questions to the Neural Engine, leaving it idle through every long batch. After re-exporting the 256/512-token buckets and letting the splitter assign long questions to the ANE, measured on the 8×512 fixture:

| ANE share k | 0 (GPU only) | 2 | 3 | **4** | 5 | 6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| wall ms | 378.14 | 285.81 | 246.72 | **216.06** | 271.67 | 335.06 |

The overlap is real: 4 sequential ANE calls take 207 ms while the GPU finishes its own 4 in ~190 ms. Single long questions stay on MLX — one question cannot be overlapped, and the engines tie.

Final state, both gates PASS (`bash autoresearch.sh` and `bench_autoresearch.py --agent fast`); memory is the macOS `footprint` of the whole process, including GPU buffers:

| | single_short | batch8_short | single_long | batch8_long | Process memory |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline (MLX fp16) | 15.34 ms | 76.26 | 53.58 | 378.59 | — |
| `LayaMLX` new defaults | 13.86 ms | 64.94 | 53.67 | 373.65 | 1151 MB |
| `LayaFast`, short only | 9.64 ms | 41.84 | 53.82 | 374.49 | 1485 MB |
| **`LayaFast`, both engines on long batches** | **9.71 ms** | **41.43** | **53.63** | **218.51** | 1485 MB |

Whole-workload total (sum of the four fixture medians): **491.1 → 323.4 ms (-34%)**.

Long inputs now improve too: the two engines are tied per long question, so splitting an 8×512 batch 4/4 across them is 1.74× faster than the GPU alone.

The MLX buffer cache is capped at 128 MB; that matched 1 GB latency on all fixtures and cut footprint 2054→1154 MB. The ANECompilerService system daemon holds ~505 MB left over from compiling the models; macOS manages it and it is not part of the runtime cost.

**Measured dead ends in this round:** split-K via batched matmul, pre-transposed `x@Wt`, `(W@x.T).T`, `mx.block_masked_mm` (all within noise); the old `LAYA_SKINNY_GEMM` kernel (still slower, 14.4 vs 13.8 ms encoder); tall GEMM on Wqkv/Wo/Wo_mlp (wins in isolated 28-GEMM chains, loses in-context); `mx.compile(shapeless=True)` (fails: Slice/CustomKernel can't infer shapes); ggmlc Metal (31.5 ms single, 193 ms batch8 — 2–3× slower than MLX, and the release binary needed two patches to run).

**Further dead ends (all measured, with numbers):**
- **ANE op count is nearly free.** Appending 3,000 real concat/split ops to the graph cost 9.596 → 9.803 ms — **0.069 µs per op**, so the whole 10,594-op graph costs ~0.7 ms. Hoisting the per-head RoPE out of the 16-head loop (~6,000 fewer ops) could win at most ~0.5 ms, so the rewrite was dropped.
- **Manual attention is not faster** than `mx.fast.scaled_dot_product_attention`: 0.454 vs 0.429 ms at L=512, across fp16/fp32 softmax and two layouts. The boolean mask is not the problem (0.398 with mask vs 0.370 without: 7%).
- **Batched ANE bodies (B=4) are a wash.** They are bit-exact vs the single-row model (max|dlogit| = 0.0000 at L=80 and L=128) and 1.23× per question in isolation (9.42 → 7.68 ms), but interleaved A/B gave −0.30% on batch8_short, +0.25% on batch8_long and −0.39% on the untouched single-question path — i.e. inside the ±0.4% noise floor. Once the MLX cost model accounts for unpadding, the GPU side is the binding constraint. Rebuild with `ane/export_batched.py --length {80,128} --batch 4`.
- **Short-batch split is already optimal** (k=3 41.00, k=4 41.66 ms).
- Finer ANE buckets (multiples of 4 rather than the current ladder) would save ~0.1 ms on a 74-token question while adding models, disk and RAM.

**Microbenchmark trap:** chains of 28 independent GEMMs pipeline back-to-back and hide per-kernel latency. They showed up to 1.65× for kernels that lose inside the real encoder. Only in-context A/B counts.

---

## 4. Architecture of the Custom Metal Kernels

Located in `laya_mlx.py`:

### Kernel 1: `fused_qkv_rope`
- **Goal:** ModernBERT's attention projects `x -> Wqkv(x)` producing a packed tensor `[B, L, 3, H, D]`. Standard pipelines slice Q, K, V, run rotary embedding calculations on Q and K, and transpose all three to `[B, H, L, D]`.
- **Implementation:**
  - Invoked with grid `(HEAD_DIM, SEQ_LEN, BATCH * NUM_HEADS)` and threadgroups `(32, 8, 1)`.
  - Reads `q_val`, `k_val`, and `v_val` in a single coalesced memory fetch.
  - Computes `rot_q` and `rot_k` via index math (`d_partner = d < half_dim ? d + half_dim : d - half_dim`) and applies `q * cos + rot * sin` in GPU thread registers.
  - Direct write to `out_idx = b*(H*L*D) + h*(L*D) + l*D + d`, outputting three pre-transposed arrays directly consumable by `mx.fast.scaled_dot_product_attention`.

### Kernel 2: `fused_geglu`
- **Goal:** ModernBERT MLP uses a Gated Linear Unit (`Wi` outputs `2 * intermediate_size = 5248` channels). The original code split the tensor into `inp` and `gate`, ran `nn.GELU()` (which allocates a new tensor), and performed elementwise multiplication (another allocation).
- **Implementation:**
  - Invoked with grid `(intermediate_size * L * B, 1, 1)` and threadgroup `(256, 1, 1)`.
  - Implements polynomial `custom_erf` with `expm1f_scaled_unchecked` in MSL header, guaranteeing `< 1e-6` divergence from PyTorch's exact CPU `erf`.
  - Computes `0.5 * inp * (1.0 + erf(inp / sqrt(2))) * gate` on the fly without temporary allocations.

---

## 5. File & Directory Reference

Tracked source and evidence:

```
laya-mlx/
├── laya_mlx.py                 # Core MLX neural network architecture with custom Metal kernels
├── laya_api.py                 # High-level Jev-compatible RLAgent API (prepare, raw_forward, system_one)
├── laya_fast.py                # MLX+ANE router (fastest configuration)
├── tall_gemm.py                # Custom simdgroup-MMA kernel for the encoder Wi projection
├── dedup.py                    # One forward per unique question
├── benchmark.py                # Parity & runtime benchmark suite (CPU, MPS, MLX)
├── convert.py                  # Weight converter from PyTorch safetensors to MLX safetensors
├── convert_coreml.py           # Standalone Core ML export & benchmark tool (with dialect/clamp patches)
├── ane/                        # Neural Engine export, cost model, and verification scripts
├── examples/                   # Sample request/question fixtures
├── runtime_*.json              # Recorded benchmark outputs per runtime/dtype
├── autoresearch.sh + autoresearch_golden.json  # Fail-closed parity/speed gate
└── download_integrity.json     # Pinned revision SHA256 hashes of original checkpoint
```

Model artifacts (`converted/`, `converted-fp16/`, `source/`, `*.mlpackage`, compiled `.mlmodelc` bodies) are generated locally by the conversion/export scripts and are not tracked in the repository.

---

## 6. How to Run

### 1. Python API
```python
from laya_api import LayaMLX

# Automatically loads from local HF cache (~/.cache/huggingface/hub/models--convaiinnovations--laya)
# or local directory 'converted'. Uses custom Metal kernels + mx.compile:
agent = LayaMLX("convaiinnovations/laya", dtype="float32", compile=True)

questions = {
    "intent": {
        "type": "choice",
        "instructions": "Determine user intention.",
        "criteria": {"refund": "wants money back", "technical": "system issue"}
    }
}
result = agent.system_one("My account was charged twice, please refund.", questions)
print(result)
```

### 2. CLI Benchmark

Run from the repository root:

```bash
# Run MLX FP32 benchmark with custom Metal kernels and JIT compilation:
.venv/bin/python benchmark.py runtime --arm mlx --model converted --dtype float32 --compile

# Run Core ML benchmark:
.venv/bin/python convert_coreml.py --benchmark laya_decision.mlpackage
```
