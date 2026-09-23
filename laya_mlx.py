"""Native MLX port of the Laya RL-agent decision model (convaiinnovations/laya).

Architecture: ModernBERT-large bidirectional encoder + from-scratch decision head,
ported 1:1 from the original PyTorch implementation (source/rl_common.py
``DecisionModel``) with transformers 5.5.0 ``modeling_modernbert.py`` as the
encoder oracle.

Weight names are identical to the original checkpoint (identity mapping), so
``load_weights(strict=True)`` verifies completeness: every one of the 206 source
tensors is required and no extras are allowed.

Parity notes vs. the PyTorch original:
  * RoPE inverse frequencies and cos/sin are computed in float32, cast to the
    compute dtype, and applied in float32 (q/k upcast, rotate_half, downcast) —
    exactly what HF ``apply_rotary_pos_emb`` does under autocast.
  * Sliding-window layers use the bidirectional window |q-kv| <= local_attention//2
    ANDed with the key padding mask; full layers use the key padding mask only.
    Padded *queries* still attend to valid keys, matching torch SDPA semantics.
  * Attention softmax runs in float32 inside ``mx.fast.scaled_dot_product_attention``,
    matching torch SDPA/eager behaviour.
  * ``nn.LayerNorm`` computes statistics in float32 internally (mx.fast.layer_norm).
  * ``temperature`` is a registered buffer in the original (not a parameter); it is
    kept as a float32 array attribute so strict loading still covers it.
  * ``mx.topk`` returns values in *ascending* order (torch returns descending);
    the act-head feature extraction indexes [-1] / [-2] accordingly.
  * Dropout modules are omitted: every dropout rate is 0.0 in this config and the
    model is inference-only.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _activation(name):
    """ACT2FN equivalent for the activations this architecture can use."""
    if name == "gelu":
        return nn.GELU()  # exact erf GELU, matches ACT2FN["gelu"] and nn.GELU default
    if name == "relu":
        return lambda x: mx.maximum(x, 0)
    raise ValueError("unsupported activation %r (expected 'gelu' or 'relu')" % name)


# ----------------------------------------------------------------------------- encoder
class Embeddings(nn.Module):
    """ModernBertEmbeddings: token embedding -> LayerNorm (no positional embedding)."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["norm_eps"], bias=cfg["norm_bias"])

    def __call__(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


# ----------------------------------------------------------------------------- Custom Metal Kernels
_METAL_AVAILABLE = hasattr(mx.fast, "metal_kernel")

_METAL_HEADER = """
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;
  j = metal::fma(1.442695f, a, 12582912.f);
  j = j - 12582912.0f;
  i = (int)j;
  f = metal::fma(j, -6.93145752e-1f, a);
  s = f * f;
  if (a == 0.0f) s = a;
  r = 1.97350979e-4f;
  r = metal::fma(r, f, 1.39309070e-3f);
  r = metal::fma(r, f, 8.33343994e-3f);
  r = metal::fma(r, f, 4.16668020e-2f);
  r = metal::fma(r, f, 1.66666716e-1f);
  r = metal::fma(r, f, 4.99999970e-1f);
  u = (j == 1) ? (f + 0.5f) : f;
  v = metal::fma(r, s, u);
  s = 0.5f * b;
  t = metal::ldexp(s, i);
  y = t - s;
  x = (t - y) - s;
  r = metal::fma(v, t, x) + y;
  r = r + r;
  if (j == 0) r = v;
  if (j == 1) r = v + v;
  return r;
}

float expm1f(float a) {
  float r = expm1f_scaled_unchecked(a, 1.0f);
  if (metal::abs(a - 1.0f) > 88.0f) {
    r = metal::pow(2.0f, a);
    r = metal::fma(r, r, -1.0f);
  }
  return r;
}

float custom_erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
    u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);
    r = -expm1f(r);
    r = metal::copysign(r, a);
  } else {
    r = -5.96761703e-4f;
    r = metal::fma(r, s, 4.99119423e-3f);
    r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f);
    r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f);
    r = metal::fma(r, a, a);
  }
  return r;
}
"""

_QKV_ROPE_SOURCE = """
    uint d = thread_position_in_grid.x;
    uint l = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;

    // Shape-agnostic: SEQ_LEN / BATCH_SIZE come from the grid so ONE compiled
    // pipeline serves every (B, L); only model constants stay templated.
    const uint SEQ_LEN = threads_per_grid.y;
    const uint BATCH_SIZE = threads_per_grid.z / NUM_HEADS;

    uint b = bh / NUM_HEADS;
    uint h = bh % NUM_HEADS;

    if (d >= HEAD_DIM || b >= BATCH_SIZE) return;

    uint half_dim = HEAD_DIM / 2;

    uint base_l = b * (SEQ_LEN * 3 * NUM_HEADS * HEAD_DIM) + l * (3 * NUM_HEADS * HEAD_DIM) + h * HEAD_DIM;
    uint stride_q = 0;
    uint stride_k = NUM_HEADS * HEAD_DIM;
    uint stride_v = 2 * NUM_HEADS * HEAD_DIM;

    float q_val = float(qkv[base_l + stride_q + d]);
    float k_val = float(qkv[base_l + stride_k + d]);
    float v_val = float(qkv[base_l + stride_v + d]);

    uint d_partner = (d < half_dim) ? (d + half_dim) : (d - half_dim);
    float q_partner = float(qkv[base_l + stride_q + d_partner]);
    float k_partner = float(qkv[base_l + stride_k + d_partner]);

    float rot_q = (d < half_dim) ? -q_partner : q_partner;
    float rot_k = (d < half_dim) ? -k_partner : k_partner;

    uint cs_idx = l * HEAD_DIM + d;
    float c = float(cos[cs_idx]);
    float s = float(sin[cs_idx]);

    float q_rope = q_val * c + rot_q * s;
    float k_rope = k_val * c + rot_k * s;

    uint out_idx = b * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + h * (SEQ_LEN * HEAD_DIM) + l * HEAD_DIM + d;

    q_out[out_idx] = T(q_rope);
    k_out[out_idx] = T(k_rope);
    v_out[out_idx] = T(v_val);
"""

_GEGLU_SOURCE = """
    uint idx = thread_position_in_grid.x;
    uint row = idx / INTERMEDIATE_SIZE;
    uint col = idx % INTERMEDIATE_SIZE;
    uint in_offset = row * (2 * INTERMEDIATE_SIZE);

    float inp = float(x[in_offset + col]);
    float gate = float(x[in_offset + INTERMEDIATE_SIZE + col]);

    float cdf = 0.5f * (1.0f + custom_erf(inp * 0.7071067811865475244f));
    float gelu_out = inp * cdf;
    out[idx] = T(gelu_out * gate);
"""

# Legacy variant: BATCH_SIZE/SEQ_LEN baked as template params -> one Metal
# pipeline per (B, L). Kept only for before/after measurement; enable with
# LAYA_SHAPE_TEMPLATES=1.
_SHAPE_TEMPLATES = os.environ.get("LAYA_SHAPE_TEMPLATES", "0") == "1"

_QKV_ROPE_SOURCE_SHAPED = """
    uint d = thread_position_in_grid.x;
    uint l = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;

    uint b = bh / NUM_HEADS;
    uint h = bh % NUM_HEADS;

    if (d >= HEAD_DIM || l >= SEQ_LEN || b >= BATCH_SIZE) return;

    uint half_dim = HEAD_DIM / 2;

    uint base_l = b * (SEQ_LEN * 3 * NUM_HEADS * HEAD_DIM) + l * (3 * NUM_HEADS * HEAD_DIM) + h * HEAD_DIM;
    uint stride_q = 0;
    uint stride_k = NUM_HEADS * HEAD_DIM;
    uint stride_v = 2 * NUM_HEADS * HEAD_DIM;

    float q_val = float(qkv[base_l + stride_q + d]);
    float k_val = float(qkv[base_l + stride_k + d]);
    float v_val = float(qkv[base_l + stride_v + d]);

    uint d_partner = (d < half_dim) ? (d + half_dim) : (d - half_dim);
    float q_partner = float(qkv[base_l + stride_q + d_partner]);
    float k_partner = float(qkv[base_l + stride_k + d_partner]);

    float rot_q = (d < half_dim) ? -q_partner : q_partner;
    float rot_k = (d < half_dim) ? -k_partner : k_partner;

    uint cs_idx = l * HEAD_DIM + d;
    float c = float(cos[cs_idx]);
    float s = float(sin[cs_idx]);

    float q_rope = q_val * c + rot_q * s;
    float k_rope = k_val * c + rot_k * s;

    uint out_idx = b * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + h * (SEQ_LEN * HEAD_DIM) + l * HEAD_DIM + d;

    q_out[out_idx] = T(q_rope);
    k_out[out_idx] = T(k_rope);
    v_out[out_idx] = T(v_val);
"""

if _METAL_AVAILABLE:
    _fused_qkv_rope_kernel = mx.fast.metal_kernel(
        name="fused_qkv_rope",
        input_names=["qkv", "cos", "sin"],
        output_names=["q_out", "k_out", "v_out"],
        header=_METAL_HEADER,
        source=_QKV_ROPE_SOURCE_SHAPED if _SHAPE_TEMPLATES else _QKV_ROPE_SOURCE,
        compile_options={"math_mode": "fast"},
    )
    _fused_geglu_kernel = mx.fast.metal_kernel(
        name="fused_geglu",
        input_names=["x"],
        output_names=["out"],
        header=_METAL_HEADER,
        source=_GEGLU_SOURCE,
        compile_options={"math_mode": "fast"},
    )
else:
    _fused_qkv_rope_kernel = None
    _fused_geglu_kernel = None

# ----------------------------------------------------------------------------- skinny-M GEMM
# y[M,N] = x[M,K] @ W[N,K]^T for small M (<=128) via simdgroup MMA on a
# pre-transposed Wt[K,N]. Wins on the large-N encoder GEMMs (Wqkv N=3072,
# Wi N=5248): ~1.3x vs MLX at M=74. Loses on N=1024, so gated on N>=2048.
# Enable with LAYA_SKINNY_GEMM=1.
_SKINNY_GEMM = os.environ.get("LAYA_SKINNY_GEMM", "0") == "1"

_SKINNY_MMA_HEADER = """
#include <metal_simdgroup_matrix>
"""

# Threadgroup = 8 simdgroups; sg s owns m-tile (s % MTG) and n-group (s / MTG)
# of NT 8-wide tiles. tg covers NCOLS=(SGS/MTG)*NT*8 columns and MTG*8 rows.
# Per k-tile: cooperative half4 staging of Wt and xp tiles, then MMA from
# threadgroup memory. Fixed config MTG=2 NT=4 KT=64 SGS=8 -> NCOLS=128,
# tgmem 18KB.
_SKINNY_MMA_SOURCE = """
    uint sg = thread_index_in_threadgroup / 32;
    uint tg = threadgroup_position_in_grid.x;
    uint nband = tg % NBANDS;
    uint mband = tg / NBANDS;
    uint mtl = sg % MTG;
    uint ngl = sg / MTG;
    uint mt = mband * MTG + mtl;
    uint nt0 = ngl * NT;
    uint gn0 = nband * NCOLS + nt0 * 8;

    threadgroup half wtile[KT * NCOLS];
    threadgroup half xtile[KT * MTG * 8];

    simdgroup_float8x8 acc[NT];
    for (int nt = 0; nt < NT; nt++)
        acc[nt] = simdgroup_float8x8(0.0f);

    uint tid = thread_index_in_threadgroup;
    uint nbase = nband * NCOLS;
    uint mbase = mband * MTG * 8;
    for (uint kt = 0; kt < K; kt += KT) {
        for (uint i = tid; i < KT * NCOLS / 4; i += TGSIZE) {
            uint kk = i / (NCOLS / 4), cc = i % (NCOLS / 4);
            *(threadgroup half4*)(wtile + kk * NCOLS + cc * 4) =
                *(const device half4*)(Wt + (size_t)(kt + kk) * N + nbase + cc * 4);
        }
        for (uint i = tid; i < KT * MTG * 8; i += TGSIZE) {
            uint kk = i / (MTG * 8), mm = i % (MTG * 8);
            xtile[i] = xp[(size_t)(mbase + mm) * K + kt + kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < KT; kk += 8) {
            simdgroup_half8x8 a;
            simdgroup_load(a, xtile + kk * MTG * 8 + mtl * 8, MTG * 8,
                           ulong2(0, 0), true);
            for (int nt = 0; nt < NT; nt++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wtile + kk * NCOLS + (nt0 + nt) * 8, NCOLS);
                simdgroup_multiply_accumulate(acc[nt], a, b, acc[nt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int nt = 0; nt < NT; nt++) {
        simdgroup_half8x8 h;
        for (int e = 0; e < 2; e++)
            h.thread_elements()[e] = half(acc[nt].thread_elements()[e]);
        simdgroup_store(h, yp + (size_t)mt * 8 * N + gn0 + nt * 8, N);
    }
"""

if _METAL_AVAILABLE:
    _skinny_mma_kernel = mx.fast.metal_kernel(
        name="skinny_gemm_mma_tg",
        input_names=["xp", "Wt"],
        output_names=["yp"],
        header=_SKINNY_MMA_HEADER,
        source=_SKINNY_MMA_SOURCE,
    )
else:
    _skinny_mma_kernel = None

# Fixed tile config (measured best at M=74 on M3 Max).
_SKINNY_MTG = 2
_SKINNY_NT = 4
_SKINNY_KT = 64
_SKINNY_SGS = 8
_SKINNY_NCOLS = (_SKINNY_SGS // _SKINNY_MTG) * _SKINNY_NT * 8  # 128

# id(nn.Linear) -> pre-transposed Wt[K,N], populated lazily.
_skinny_wt_cache = {}


def _skinny_linear(lin, x):
    """x[...,K] @ lin.weight[N,K]^T via the skinny-M MMA kernel.

    Returns None when the shape/dtype is not eligible (caller falls back to
    lin(x)). Eligibility: fp16, M<=128, N>=2048, N%128==0, K%64==0.
    """
    if _skinny_mma_kernel is None or x.dtype != mx.float16:
        return None
    W = lin.weight
    N, K = W.shape
    M = x.size // K
    if M > 128 or N < 2048 or N % _SKINNY_NCOLS or K % _SKINNY_KT:
        return None
    Wt = _skinny_wt_cache.get(id(lin))
    if Wt is None:
        Wt = mx.contiguous(W.T)
        mx.eval(Wt)
        _skinny_wt_cache[id(lin)] = Wt
    x2 = x.reshape(M, K)
    MT = (M + 7) // 8
    MBANDS = (MT + _SKINNY_MTG - 1) // _SKINNY_MTG
    MP = MBANDS * _SKINNY_MTG * 8
    xp = mx.pad(x2, [(0, MP - M), (0, 0)]) if MP != M else x2
    NBANDS = N // _SKINNY_NCOLS
    yp = _skinny_mma_kernel(
        inputs=[xp, Wt],
        template=[("M", M), ("N", N), ("K", K), ("NT", _SKINNY_NT),
                  ("KT", _SKINNY_KT), ("MTG", _SKINNY_MTG),
                  ("NCOLS", _SKINNY_NCOLS), ("NBANDS", NBANDS),
                  ("TGSIZE", _SKINNY_SGS * 32)],
        grid=(NBANDS * MBANDS * _SKINNY_SGS * 32, 1, 1),
        threadgroup=(_SKINNY_SGS * 32, 1, 1),
        output_shapes=[(MP, N)],
        output_dtypes=[x.dtype],
    )[0]
    y = yp[:M].reshape(*x.shape[:-1], N)
    bias = getattr(lin, "bias", None)
    if bias is not None:
        y = y + bias
    return y


# Tall-skinny GEMM (tall_gemm.py) for the encoder Wi projection when
# 64 < M <= 96 (the L=80/96 buckets): MLX pads those to 128 rows, this kernel
# reads each weight once for all rows. Bit-identical outputs, no extra memory
# (scatter mode). Read at trace time; LAYA_TALL_GEMM=0 disables.
_TALL_GEMM = os.environ.get("LAYA_TALL_GEMM", "1") != "0"
try:
    from tall_gemm import tall_linear as _tall_linear
except Exception:  # Metal/kernel unavailable -> MLX matmul only
    _tall_linear = None


def _linear(lin, x):
    """nn.Linear with optional custom fast paths (tall GEMM default on,
    LAYA_SKINNY_GEMM=1 legacy kernel off)."""
    if _TALL_GEMM and _tall_linear is not None:
        y = _tall_linear(lin, x)
        if y is not None:
            return y
    if _SKINNY_GEMM:
        y = _skinny_linear(lin, x)
        if y is not None:
            return y
    return lin(x)


class MLP(nn.Module):
    """ModernBertMLP: Wi -> split(input, gate) -> act(input) * gate -> Wo (GLU)."""

    def __init__(self, cfg):
        super().__init__()
        self.Wi = nn.Linear(cfg["hidden_size"], 2 * cfg["intermediate_size"], bias=cfg["mlp_bias"])
        self.act_name = cfg["hidden_activation"]
        self.act = _activation(self.act_name)
        self.intermediate_size = cfg["intermediate_size"]
        self.Wo = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=cfg["mlp_bias"])

    def __call__(self, x):
        wi = _linear(self.Wi, x)
        if _fused_geglu_kernel is not None and self.act_name == "gelu":
            total_elements = wi.size // 2
            act = _fused_geglu_kernel(
                inputs=[wi],
                template=[("T", wi.dtype), ("INTERMEDIATE_SIZE", self.intermediate_size)],
                grid=(total_elements, 1, 1),
                threadgroup=(min(256, total_elements), 1, 1),
                output_shapes=[wi.shape[:-1] + (self.intermediate_size,)],
                output_dtypes=[wi.dtype],
            )[0]
        else:
            inp, gate = mx.split(wi, 2, axis=-1)
            act = self.act(inp) * gate
        return _linear(self.Wo, act)


class RotaryEmbedding(nn.Module):
    """Per-layer-type RoPE (HF ModernBertRotaryEmbedding, rope_type='default').

    cos/sin are computed in float32 then cast to the model dtype, as in HF.
    Frequencies are derived per call from the stored thetas (plain floats, so they
    are not registered as parameters/buffers).
    """

    def __init__(self, cfg):
        super().__init__()
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.thetas = {lt: float(cfg["rope_parameters"][lt]["rope_theta"])
                       for lt in set(cfg["layer_types"])}

    def __call__(self, seq_len, layer_type, dtype):
        inv_freq = 1.0 / (self.thetas[layer_type]
                          ** (mx.arange(0, self.head_dim, 2).astype(mx.float32) / self.head_dim))
        freqs = mx.arange(seq_len).astype(mx.float32)[:, None] * inv_freq[None, :]  # [L, D/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)                              # [L, D]
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _apply_rope(x, cos, sin):
    """HF apply_rotary_pos_emb on [B, L, H, D] layouts: rotate_half, fp32 math."""
    xf = x.astype(mx.float32)
    cos = cos.astype(x.dtype).astype(mx.float32)[None, :, None, :]
    sin = sin.astype(x.dtype).astype(mx.float32)[None, :, None, :]
    half_dim = x.shape[-1] // 2
    rot = mx.concatenate([-xf[..., half_dim:], xf[..., :half_dim]], axis=-1)
    return (xf * cos + rot * sin).astype(x.dtype)


class Attention(nn.Module):
    """ModernBertAttention: packed Wqkv -> RoPE -> SDPA -> Wo (no biases in this config)."""

    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["hidden_size"] // self.num_heads
        self.scale = self.head_dim ** -0.5
        d = cfg["hidden_size"]
        self.Wqkv = nn.Linear(d, 3 * d, bias=cfg["attention_bias"])
        self.Wo = nn.Linear(d, d, bias=cfg["attention_bias"])

    def _sdpa(self, qkv, cos, sin, mask):
        """qkv [B, L, 3, H, Dh] -> attention out [B, L, H*Dh] (RoPE + SDPA)."""
        B, L = qkv.shape[0], qkv.shape[1]
        if _fused_qkv_rope_kernel is not None:
            template = [("T", qkv.dtype), ("NUM_HEADS", self.num_heads), ("HEAD_DIM", self.head_dim)]
            if _SHAPE_TEMPLATES:
                template += [("BATCH_SIZE", B), ("SEQ_LEN", L)]
            q, k, v = _fused_qkv_rope_kernel(
                inputs=[qkv, cos, sin],
                template=template,
                grid=(self.head_dim, L, B * self.num_heads),
                threadgroup=(min(self.head_dim, 32), min(L, 8), 1),
                output_shapes=[(B, self.num_heads, L, self.head_dim), (B, self.num_heads, L, self.head_dim), (B, self.num_heads, L, self.head_dim)],
                output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype],
            )
        else:
            q = _apply_rope(qkv[:, :, 0], cos, sin).transpose(0, 2, 1, 3)
            k = _apply_rope(qkv[:, :, 1], cos, sin).transpose(0, 2, 1, 3)
            v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return o.transpose(0, 2, 1, 3).reshape(B, L, -1)

    def __call__(self, x, cos, sin, mask):
        B, L, _ = x.shape
        qkv = _linear(self.Wqkv, x).reshape(B, L, 3, self.num_heads, self.head_dim)
        return _linear(self.Wo, self._sdpa(qkv, cos, sin, mask))

    def forward_packed(self, x, cos, sin, mask, pack):
        """x [T, D] packed rows -> [T, D]. QKV runs packed; RoPE/SDPA run on the
        padded [B, L] layout via slot_src (pad slots alias row 0, masked and
        discarded) so per-sequence positions and masks are unchanged."""
        B, L = mask.shape[0], mask.shape[-1]
        qkv = _linear(self.Wqkv, x)                                   # [T, 3*H*Dh]
        qkv5 = qkv[pack["slot_src"]].reshape(B, L, 3, self.num_heads, self.head_dim)
        o = self._sdpa(qkv5, cos, sin, mask)                          # [B, L, D]
        return _linear(self.Wo, o.reshape(B * L, -1)[pack["flat_idx"]])


class EncoderLayer(nn.Module):
    """ModernBertEncoderLayer: pre-norm attention + GLU MLP. Layer 0 skips attn_norm."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn_norm = None if layer_idx == 0 else nn.LayerNorm(
            cfg["hidden_size"], eps=cfg["norm_eps"], bias=cfg["norm_bias"])
        self.attn = Attention(cfg)
        self.mlp_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["norm_eps"], bias=cfg["norm_bias"])
        self.mlp = MLP(cfg)
        self.attention_type = cfg["layer_types"][layer_idx]

    def __call__(self, x, cos, sin, mask):
        n = x if self.attn_norm is None else self.attn_norm(x)
        x = x + self.attn(n, cos, sin, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward_packed(self, x, cos, sin, mask, pack):
        n = x if self.attn_norm is None else self.attn_norm(x)
        x = x + self.attn.forward_packed(n, cos, sin, mask, pack)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ModernBertEncoder(nn.Module):
    """ModernBertModel: embeddings -> N encoder layers -> final_norm."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.layers = [EncoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])]
        self.final_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["norm_eps"], bias=cfg["norm_bias"])
        self.rotary_emb = RotaryEmbedding(cfg)
        # HF: config.sliding_window = local_attention // 2 (half-window, inclusive)
        self.sliding_window = cfg["local_attention"] // 2
        self.layer_types = cfg["layer_types"]
        self._window_cache = {}

    def __call__(self, input_ids, attention_mask):
        L = input_ids.shape[1]
        x = self.embeddings(input_ids)
        # Bool masks, True = attend (same convention as torch SDPA attn_mask).
        key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]          # [B,1,1,L]
        if L not in self._window_cache:
            pos = mx.arange(L)
            self._window_cache[L] = (mx.abs(pos[:, None] - pos[None, :]) <= self.sliding_window)[None, None]
        window = self._window_cache[L]
        masks = {"full_attention": key_mask,
                 "sliding_attention": key_mask & window}
        ropes = {lt: self.rotary_emb(L, lt, x.dtype) for lt in set(self.layer_types)}
        for layer in self.layers:
            cos, sin = ropes[layer.attention_type]
            x = layer(x, cos, sin, masks[layer.attention_type])
        return self.final_norm(x)

    def forward_packed(self, input_ids, attention_mask, pack):
        """Unpadded forward: input_ids [T] (real tokens only) -> hidden [T, D].

        Every token-wise op runs on T rows. Attention still uses the padded
        [B, L] masks/window and per-sequence RoPE positions via pack["slot_src"].
        """
        B, L = attention_mask.shape
        x = self.embeddings(input_ids)                                # [T, D]
        key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]  # [B,1,1,L]
        if L not in self._window_cache:
            pos = mx.arange(L)
            self._window_cache[L] = (mx.abs(pos[:, None] - pos[None, :]) <= self.sliding_window)[None, None]
        window = self._window_cache[L]
        masks = {"full_attention": key_mask,
                 "sliding_attention": key_mask & window}
        ropes = {lt: self.rotary_emb(L, lt, x.dtype) for lt in set(self.layer_types)}
        for layer in self.layers:
            cos, sin = ropes[layer.attention_type]
            x = layer.forward_packed(x, cos, sin, masks[layer.attention_type], pack)
        return self.final_norm(x)


# ----------------------------------------------------------------------------- decision head
class HeadSelfAttention(nn.Module):
    """torch.nn.MultiheadAttention(batch_first=True) equivalent.

    in_proj packs [q; k; v] along dim 0 exactly like torch's in_proj_weight,
    so the original checkpoint maps 1:1.
    """

    def __init__(self, d, nhead):
        super().__init__()
        self.num_heads = nhead
        self.head_dim = d // nhead
        self.scale = self.head_dim ** -0.5
        self.in_proj_weight = mx.zeros((3 * d, d))
        self.in_proj_bias = mx.zeros((3 * d,))
        self.out_proj = nn.Linear(d, d)

    def _sdpa(self, qkv, key_mask):
        """qkv [B, L, 3, H, Dh] -> attention out [B, L, D] (no RoPE in the head)."""
        B, L = qkv.shape[0], qkv.shape[1]
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=key_mask)
        return o.transpose(0, 2, 1, 3).reshape(B, L, -1)

    def __call__(self, x, key_mask):
        B, L, D = x.shape
        qkv = (x @ self.in_proj_weight.T + self.in_proj_bias).reshape(
            B, L, 3, self.num_heads, self.head_dim)
        return _linear(self.out_proj, self._sdpa(qkv, key_mask))

    def forward_packed(self, x, key_mask, pack):
        """x [T, D] packed rows -> [T, D]; SDPA on the padded layout via slot_src."""
        B, L = key_mask.shape[0], key_mask.shape[-1]
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias           # [T, 3D]
        qkv5 = qkv[pack["slot_src"]].reshape(B, L, 3, self.num_heads, self.head_dim)
        o = self._sdpa(qkv5, key_mask)                                # [B, L, D]
        return _linear(self.out_proj, o.reshape(B * L, -1)[pack["flat_idx"]])

class HeadLayer(nn.Module):
    """torch.nn.TransformerEncoderLayer(norm_first=True, batch_first=True).

    FFN activation is ReLU -- the PyTorch default for TransformerEncoderLayer.
    (Only scorer/act_head use GELU; do not "fix" this.)
    """

    def __init__(self, d, nhead, dim_ff):
        super().__init__()
        self.self_attn = HeadSelfAttention(d, nhead)
        self.linear1 = nn.Linear(d, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def __call__(self, x, key_mask):
        x = x + self.self_attn(self.norm1(x), key_mask)
        x = x + _linear(self.linear2, mx.maximum(_linear(self.linear1, self.norm2(x)), 0))
        return x

    def forward_packed(self, x, key_mask, pack):
        x = x + self.self_attn.forward_packed(self.norm1(x), key_mask, pack)
        x = x + _linear(self.linear2, mx.maximum(_linear(self.linear1, self.norm2(x)), 0))
        return x


class TransformerHead(nn.Module):
    """torch.nn.TransformerEncoder container (enable_nested_tensor=False)."""

    def __init__(self, d, nhead, dim_ff, n_layers):
        super().__init__()
        self.layers = [HeadLayer(d, nhead, dim_ff) for _ in range(n_layers)]

    def __call__(self, x, key_mask):
        for layer in self.layers:
            x = layer(x, key_mask)
        return x

    def forward_packed(self, x, key_mask, pack):
        for layer in self.layers:
            x = layer.forward_packed(x, key_mask, pack)
        return x


# ----------------------------------------------------------------------------- model
class Model(nn.Module):
    """MLX twin of rl_common.DecisionModel.

    __call__(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        -> (option_logits [B, K] float32, act_logits [B, n_act])
    """

    def __init__(self, encoder_config, agent_config):
        super().__init__()
        self.encoder = ModernBertEncoder(encoder_config)
        d = encoder_config["hidden_size"]
        nhead = max(1, d // 64)
        head_layers = agent_config["head_layers"]
        self.head = TransformerHead(d, nhead, 4 * d, head_layers) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        # List attribute -> parameters named scorer.0.* / scorer.3.*, matching the
        # original nn.Sequential(LayerNorm, Linear, GELU, Linear) numbering.
        self.scorer = [nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)]
        n_act = len(agent_config["act_costs"]) + 1
        self.act_head = [nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act)]
        self.temperature = mx.ones((3,))  # buffer in the original; kept float32

    def _score_tail(self, m, pooled, marker_mask):
        """Shared tail: marker rows m [B, K, D] + pooled CLS [B, D] -> logits, act."""
        for mod in self.scorer:
            m = mod(m)
        logits = m[..., 0].astype(mx.float32)
        keep = marker_mask.astype(mx.bool_)
        logits = mx.where(keep, logits, mx.array(-1e4, dtype=mx.float32))
        # act head sees pooled CLS + detached summary of the answer distribution
        p = mx.softmax(logits, axis=-1)                   # stop_gradient: no-op in inference
        k = mx.maximum(keep.sum(-1), 2).astype(mx.float32)
        ent = -(p * mx.log(mx.maximum(p, 1e-9))).sum(-1) / mx.log(k)
        top2 = mx.topk(p, 2, axis=-1)                     # ascending order in MLX
        feats = mx.stack([top2[:, -1], top2[:, -1] - top2[:, -2], ent, k / 255.0], axis=-1)
        pooled = pooled.astype(mx.float32)
        act_in = mx.concatenate([pooled, feats], axis=-1).astype(self.act_head[0].weight.dtype)
        for mod in self.act_head:
            act_in = mod(act_in)
        return logits, act_in

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad_key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]  # True = attend
            h = self.head(h, pad_key_mask)
        B = h.shape[0]
        pos = mx.maximum(marker_pos, 0).astype(mx.int32)
        m = h[mx.arange(B)[:, None], pos]                 # gather marker positions -> [B, K, D]
        return self._score_tail(m, h[:, 0], marker_mask)

    def forward_packed(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, pack):
        """Unpadded (varlen) forward for padded batches.

        input_ids [B, L] is gathered to packed ids [T] via pack["flat_idx"];
        every token-wise op runs on T rows. pack (numpy int arrays, built in
        laya_api.raw_forward):
          flat_idx   [T]    flat (b*L+l) indices of real tokens, row-major
          slot_src   [B*L]  packed row backing each padded slot (pads -> row 0,
                            masked in attention and discarded on gather-back)
          seq_of     [T]    sequence index of each packed row (type embedding)
          cls_idx    [B]    packed row of each sequence's CLS token
          marker_idx [B, K] packed row of each marker position
        """
        ids = input_ids.reshape(-1)[pack["flat_idx"]]                 # [T]
        h = self.encoder.forward_packed(ids, attention_mask, pack)    # [T, D]
        h = h + self.type_emb(qtype)[pack["seq_of"]]
        if self.head is not None:
            pad_key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]  # True = attend
            h = self.head.forward_packed(h, pad_key_mask, pack)
        m = h[pack["marker_idx"]]                                     # [B, K, D]
        return self._score_tail(m, h[pack["cls_idx"]], marker_mask)


def _resolve_model_dir(model_dir):
    if os.path.isdir(model_dir):
        return model_dir
    try:
        import huggingface_hub
        return huggingface_hub.snapshot_download(model_dir, local_files_only=True)
    except Exception:
        return model_dir


def load_model(model_dir, dtype="float32"):
    """Build the MLX model and strictly load converted weights from ``model_dir``.

    ``dtype`` is the compute dtype: every floating parameter is cast to it at load
    (the ``temperature`` buffer stays float32). Use "float32" for tight parity with
    the PyTorch reference, "float16" for the performance arm.
    """
    model_dir = _resolve_model_dir(model_dir)
    with open(os.path.join(model_dir, "encoder", "config.json")) as f:
        encoder_config = json.load(f)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        agent_config = json.load(f)
    if isinstance(dtype, str):
        if dtype not in _DTYPES:
            raise ValueError("dtype must be one of %s, got %r" % (sorted(_DTYPES), dtype))
        dtype = _DTYPES[dtype]
    model = Model(encoder_config, agent_config)
    weights = mx.load(os.path.join(model_dir, "model.safetensors"))
    weights = {k: (v.astype(dtype) if k != "temperature" and mx.issubdtype(v.dtype, mx.floating)
                   else v.astype(mx.float32))
               for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model
