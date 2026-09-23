"""Tall-skinny fp16 GEMM for the Laya encoder's small-M linears.

Computes y[M,N] = x[M,K] @ W[N,K]^T (+ optional bias) for nn.Linear layers,
fp16 in/out with fp32 accumulation, via a custom Metal simdgroup-MMA kernel.

Why this exists: MLX's steel GEMM pads M to 64-row tiles, so M=74 costs the
same as M=128 (two 64-row tiles, each threadgroup re-streaming its W panel).
Here every threadgroup covers ALL M rows -- padded only to a multiple of 8
inside the kernel (M=74 -> 80 rows = 10 simdgroup m-tiles; no host-side
mx.pad, the x-tile staging zero-fills rows >= M and the epilogue store is
row-guarded so the output is exactly [M,N]) -- and owns a BN-wide column band
of W, so each W element is fetched from DRAM exactly once per GEMM and M=74
pays for 80 rows of MMA work instead of 128.

`tall_linear(lin, x)` mirrors `laya_mlx._linear` semantics for nn.Linear with
optional bias. It returns None when the shape/dtype is not eligible; the
caller must fall back to `lin(x)`.

Eligibility:
  * x.dtype == mx.float16 and lin.weight is [N, K] fp16
  * 64 < M <= 96 where M = x.size // K (x may be 2-D or 3-D [..., K]) --
    the band where MLX pads M to two 64-row tiles but this kernel needs
    only ceil(M/8)*8 rows of MMA work. Outside it, MLX wins in-context.
  * K % KT == 0 for the k-tile depth KT (64)
  * (N, K) present in _CFG_TABLE and N % BN == 0. Only (5248, 1024) -- the
    Wi GEMM -- is enabled: measured in-context win ~1.07-1.11x on the
    encoder at L=74..96 on M3 Max. Other shapes win in isolated chains but
    lose or tie in-context (dependent-op latency dominates).

Tile config: threadgroup = MT = ceil(M/8) simdgroups (MT*32 threads); simdgroup
s owns m-tile s and loops over NT = BN/8 n-tiles. Per k-tile the threadgroup
cooperatively stages a W tile and x[0..MP, k..k+KT] into threadgroup memory
with half4 vectorized loads; each W element is read from DRAM exactly once
per GEMM. BN is chosen per N to keep the grid (N/BN threadgroups) large
enough to fill 30 GPU cores.

Three W-staging modes (TALL_MODE env, default "scatter"):
  * "scatter": stage W[n0..n0+BN, kt..kt+KT] rows with half4 loads along k,
    scatter-store transposed into wtile[KT][BN]; plain b-frag loads. No extra
    memory. Measured in-context ~1.09x at M=80 on M3 Max.
  * "tload": stage W rows into wtile[BN][KT], b-frag via simdgroup_load
    transpose=true. No extra memory. ~1.07x at M=80.
  * "wt": stage a cached Wt[K,N] transpose (lin._tall_wt, +10.5 MB per Wi
    layer / ~294 MB total); plain b-frag loads. Fastest (~1.11x at M=80) but
    costs RAM. Select with TALL_MODE=wt or TALL_WT=1; requires prewarm()
    before mx.compile so the transpose is not traced into the graph.
"""

import os

import mlx.core as mx

_METAL_HEADER = """
#include <metal_simdgroup_matrix>
"""

# Threadgroup = MT simdgroups; sg s owns m-tile s (rows s*8..s*8+7). The tg
# covers all M rows and BN columns of W.
#
# Per k-tile: all threads cooperatively copy Wt[kt..kt+KT, n0..n0+BN] into
# wtile (row-major [KT][BN], half4 along n) and x[0..MP, kt..kt+KT] into
# xtile ([MP][KT], half4 along k; rows >= M are zero-filled so no host-side
# pad is needed). Then each sg does KT/8 k-steps of NT MMAs: a-frag from
# xtile, b-frag from wtile -- both plain (non-transposed) loads.
#
# Epilogue: guarded scalar store. For an 8x8 simdgroup fragment, lane t holds
# elements (fm, fn) and (fm, fn+1) where qid=t/4, fm=(qid&4)+((t/2)%4),
# fn=(qid&2)*2+(t%2)*2 (see steel BaseMMAFrag::get_coord). Rows >= M are
# skipped so y is exactly [M,N] -- no slice, no padded output buffer.
_TALL_SOURCE = """
    uint sg = thread_index_in_threadgroup / 32;
    uint lane = thread_index_in_threadgroup % 32;
    uint tid = thread_index_in_threadgroup;
    uint n0 = threadgroup_position_in_grid.x * BN;

    threadgroup half wtile[KT * BN];
    threadgroup half xtile[MP * KT];

    simdgroup_float8x8 acc[MPSG][NT];
    for (int i = 0; i < MPSG; i++)
        for (int j = 0; j < NT; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kt = 0; kt < K; kt += KT) {
        // stage Wt tile: KT rows x BN cols, half4 along n
        for (uint i = tid; i < KT * BN / 4; i += TGSIZE) {
            uint kk = i / (BN / 4), nn = i % (BN / 4);
            *(threadgroup half4*)(wtile + kk * BN + nn * 4) =
                *(const device half4*)(Wt + (size_t)(kt + kk) * N + n0 + nn * 4);
        }
        // stage x tile: MP rows x KT cols, half4 along k; zero-fill rows >= M
        for (uint i = tid; i < MP * KT / 4; i += TGSIZE) {
            uint mm = i / (KT / 4), kk = i % (KT / 4);
            half4 v = (mm < uint(M))
                ? *(const device half4*)(xp + (size_t)mm * K + kt + kk * 4)
                : half4(0.0h);
            *(threadgroup half4*)(xtile + mm * KT + kk * 4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < KT; kk += 8) {
            simdgroup_half8x8 a[MPSG];
            for (int i = 0; i < MPSG; i++)
                simdgroup_load(a[i], xtile + (sg * MPSG + i) * 8 * KT + kk,
                               KT, ulong2(0, 0), false);
            for (int nt = 0; nt < NT; nt++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wtile + kk * BN + nt * 8, BN,
                               ulong2(0, 0), false);
                for (int i = 0; i < MPSG; i++)
                    simdgroup_multiply_accumulate(acc[i][nt], a[i], b,
                                                  acc[i][nt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // guarded scalar epilogue: lane -> (fm, fn) per steel get_coord
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int i = 0; i < MPSG; i++) {
        uint row = (sg * MPSG + i) * 8 + fm;
        if (row < uint(M)) {
            device half* orow = yp + (size_t)row * N + n0 + fn;
            for (int nt = 0; nt < NT; nt++) {
                orow[nt * 8]     = half(acc[i][nt].thread_elements()[0]);
                orow[nt * 8 + 1] = half(acc[i][nt].thread_elements()[1]);
            }
        }
    }
"""

try:
    _tall_kernel = mx.fast.metal_kernel(
        name="tall_gemm_mma",
        input_names=["xp", "Wt"],
        output_names=["yp"],
        header=_METAL_HEADER,
        source=_TALL_SOURCE,
    )
except Exception:  # Metal unavailable
    _tall_kernel = None

# ---------------------------------------------------------------- direct-W variants (no Wt copy)
# Same tiling as _TALL_SOURCE but read W[N,K] directly -- no cached
# transpose, no extra 294 MB. Two stagings:
#   _TALL_TLOAD_SOURCE: stage W rows into wtile[BN][KT] with half4 loads
#     along k (coalesced), then b-frag via simdgroup_load transpose=true.
#   _TALL_SCATTER_SOURCE: stage W rows but scatter-store transposed into
#     wtile[KT][BN] (scalar writes), then plain b-frag loads.
_TALL_TLOAD_SOURCE = """
    uint sg = thread_index_in_threadgroup / 32;
    uint lane = thread_index_in_threadgroup % 32;
    uint tid = thread_index_in_threadgroup;
    uint n0 = threadgroup_position_in_grid.x * BN;

    threadgroup half wtile[BN * KT];
    threadgroup half xtile[MP * KT];

    simdgroup_float8x8 acc[MPSG][NT];
    for (int i = 0; i < MPSG; i++)
        for (int j = 0; j < NT; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kt = 0; kt < K; kt += KT) {
        // stage W tile: BN rows x KT cols, half4 along k
        for (uint i = tid; i < BN * KT / 4; i += TGSIZE) {
            uint nn = i / (KT / 4), kk = i % (KT / 4);
            *(threadgroup half4*)(wtile + nn * KT + kk * 4) =
                *(const device half4*)(W + (size_t)(n0 + nn) * K + kt + kk * 4);
        }
        for (uint i = tid; i < MP * KT / 4; i += TGSIZE) {
            uint mm = i / (KT / 4), kk = i % (KT / 4);
            half4 v = (mm < uint(M))
                ? *(const device half4*)(xp + (size_t)mm * K + kt + kk * 4)
                : half4(0.0h);
            *(threadgroup half4*)(xtile + mm * KT + kk * 4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < KT; kk += 8) {
            simdgroup_half8x8 a[MPSG];
            for (int i = 0; i < MPSG; i++)
                simdgroup_load(a[i], xtile + (sg * MPSG + i) * 8 * KT + kk,
                               KT, ulong2(0, 0), false);
            for (int nt = 0; nt < NT; nt++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wtile + (nt * 8) * KT + kk, KT,
                               ulong2(0, 0), true);
                for (int i = 0; i < MPSG; i++)
                    simdgroup_multiply_accumulate(acc[i][nt], a[i], b,
                                                  acc[i][nt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int i = 0; i < MPSG; i++) {
        uint row = (sg * MPSG + i) * 8 + fm;
        if (row < uint(M)) {
            device half* orow = yp + (size_t)row * N + n0 + fn;
            for (int nt = 0; nt < NT; nt++) {
                orow[nt * 8]     = half(acc[i][nt].thread_elements()[0]);
                orow[nt * 8 + 1] = half(acc[i][nt].thread_elements()[1]);
            }
        }
    }
"""

_TALL_SCATTER_SOURCE = """
    uint sg = thread_index_in_threadgroup / 32;
    uint lane = thread_index_in_threadgroup % 32;
    uint tid = thread_index_in_threadgroup;
    uint n0 = threadgroup_position_in_grid.x * BN;

    threadgroup half wtile[KT * BN];
    threadgroup half xtile[MP * KT];

    simdgroup_float8x8 acc[MPSG][NT];
    for (int i = 0; i < MPSG; i++)
        for (int j = 0; j < NT; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kt = 0; kt < K; kt += KT) {
        // stage W tile transposed: read half4 along k, scatter 4 scalars
        for (uint i = tid; i < BN * KT / 4; i += TGSIZE) {
            uint nn = i / (KT / 4), kk = i % (KT / 4);
            half4 v = *(const device half4*)(W + (size_t)(n0 + nn) * K
                                             + kt + kk * 4);
            wtile[(kk * 4 + 0) * BN + nn] = v[0];
            wtile[(kk * 4 + 1) * BN + nn] = v[1];
            wtile[(kk * 4 + 2) * BN + nn] = v[2];
            wtile[(kk * 4 + 3) * BN + nn] = v[3];
        }
        for (uint i = tid; i < MP * KT / 4; i += TGSIZE) {
            uint mm = i / (KT / 4), kk = i % (KT / 4);
            half4 v = (mm < uint(M))
                ? *(const device half4*)(xp + (size_t)mm * K + kt + kk * 4)
                : half4(0.0h);
            *(threadgroup half4*)(xtile + mm * KT + kk * 4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < KT; kk += 8) {
            simdgroup_half8x8 a[MPSG];
            for (int i = 0; i < MPSG; i++)
                simdgroup_load(a[i], xtile + (sg * MPSG + i) * 8 * KT + kk,
                               KT, ulong2(0, 0), false);
            for (int nt = 0; nt < NT; nt++) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wtile + kk * BN + nt * 8, BN,
                               ulong2(0, 0), false);
                for (int i = 0; i < MPSG; i++)
                    simdgroup_multiply_accumulate(acc[i][nt], a[i], b,
                                                  acc[i][nt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int i = 0; i < MPSG; i++) {
        uint row = (sg * MPSG + i) * 8 + fm;
        if (row < uint(M)) {
            device half* orow = yp + (size_t)row * N + n0 + fn;
            for (int nt = 0; nt < NT; nt++) {
                orow[nt * 8]     = half(acc[i][nt].thread_elements()[0]);
                orow[nt * 8 + 1] = half(acc[i][nt].thread_elements()[1]);
            }
        }
    }
"""

try:
    _tall_tload_kernel = mx.fast.metal_kernel(
        name="tall_gemm_mma_tload",
        input_names=["xp", "W"],
        output_names=["yp"],
        header=_METAL_HEADER,
        source=_TALL_TLOAD_SOURCE,
    )
    _tall_scatter_kernel = mx.fast.metal_kernel(
        name="tall_gemm_mma_scatter",
        input_names=["xp", "W"],
        output_names=["yp"],
        header=_METAL_HEADER,
        source=_TALL_SCATTER_SOURCE,
    )
except Exception:  # Metal unavailable
    _tall_tload_kernel = None
    _tall_scatter_kernel = None

# Column-band width per N: smaller BN -> more threadgroups -> better occupancy
# on 30 cores for small-N shapes. All encoder N (1024, 3072, 5248) divide all
# of these; the table is tuned by an in-context BN sweep.
_KT = int(os.environ.get("TALL_KT", "64"))
# (N, K) -> (BN, MPSG); tuned in-context on M3 Max (see module docstring).
# Only the Wi GEMM (N=5248, K=1024) wins in the real encoder: it is the
# largest weight (10.5 MB fp16) so the read-W-once structure pays off, and
# its N gives 82 threadgroups at BN=64 -- enough to fill 30 cores. The
# N=1024 shapes win in isolated chains but lose in-context (dependent-op
# latency, not throughput, dominates there); Wqkv is neutral-to-negative.
_CFG_TABLE = {
    (5248, 1024): (64, 1),
}
_CFG_DEFAULT = None
_MPSG = int(os.environ.get("TALL_MPSG", "0"))  # 0 = use table

_BN_OVERRIDE = os.environ.get("TALL_BN")  # "N:BN,N:BN" sweep hook
# "wt" (cached transpose, +294 MB) | "tload" | "scatter" (direct W, no copy)
_MODE = os.environ.get("TALL_MODE", "scatter")
if os.environ.get("TALL_WT") == "1":
    _MODE = "wt"


def _cfg_for(N, K):
    if _BN_OVERRIDE:
        for kv in _BN_OVERRIDE.split(","):
            k, v = kv.split(":")
            if int(k) == N:
                v = int(v)
                return (v if N % v == 0 else None,
                        _MPSG if _MPSG else 1)
        return None, 1
    cfg = _CFG_TABLE.get((N, K), _CFG_DEFAULT)
    if cfg is None:
        return None, 1
    bn, mpsg = cfg
    if _MPSG:
        mpsg = _MPSG
    if N % bn:
        return None, mpsg
    return bn, mpsg


def tall_linear(lin, x):
    """x[...,K] @ lin.weight[N,K]^T (+bias) via the tall-skinny MMA kernel.

    Returns None when ineligible (caller falls back to lin(x)). See module
    docstring for the eligibility rules.
    """
    if _tall_kernel is None or x.dtype != mx.float16:
        return None
    W = lin.weight
    if W.dtype != mx.float16 or W.ndim != 2:
        return None
    N, K = W.shape
    if x.shape[-1] != K:
        return None
    M = x.size // K
    # In-context win zone: M in (64, 96] -- where MLX pads to two 64-row
    # tiles but this kernel needs only <=96 rows of MMA work. M<=64 and
    # M>96 measured slower in-context on M3 Max.
    if M <= 64 or M > 96 or K % _KT:
        return None
    bn, mpsg = _cfg_for(N, K)
    if bn is None:
        return None
    # Kernel selection: "wt" uses a cached Wt[K,N] transpose (lin._tall_wt,
    # +10.5 MB per Wi layer); "tload"/"scatter" read W[N,K] directly with no
    # extra memory. TALL_MODE env overrides; default is the fastest measured.
    mode = _MODE
    kern = {"wt": _tall_kernel, "tload": _tall_tload_kernel,
            "scatter": _tall_scatter_kernel}.get(mode)
    if kern is None:
        return None
    if mode == "wt":
        # Cached on the Linear object itself: id()-keyed dicts are unsafe
        # because a GC'd Linear's id can be reused by a new one.
        Wt = getattr(lin, "_tall_wt", None)
        if Wt is None:
            Wt = mx.contiguous(W.T)
            mx.eval(Wt)
            lin._tall_wt = Wt
        warg = Wt
    else:
        warg = W
    MT = (M + 7) // 8
    MP = MT * 8
    while MT % mpsg:
        mpsg -= 1
    sgs = MT // mpsg
    x2 = x.reshape(M, K)
    nt = bn // 8
    nbands = N // bn
    y = kern(
        inputs=[x2, warg],
        template=[("M", M), ("N", N), ("K", K), ("BN", bn), ("NT", nt),
                  ("KT", _KT), ("MP", MP), ("MPSG", mpsg),
                  ("TGSIZE", sgs * 32)],
        grid=(nbands * sgs * 32, 1, 1),
        threadgroup=(sgs * 32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
    y = y.reshape(*x.shape[:-1], N)
    bias = getattr(lin, "bias", None)
    if bias is not None:
        y = y + bias
    return y


def _iter_linears(module):
    """Yield every nn.Linear in a Module tree (uses MLX's named_modules)."""
    import mlx.nn as nn
    for _, m in module.named_modules():
        if isinstance(m, nn.Linear):
            yield m


def prewarm(module):
    """Build the cached Wt[K,N] transpose on every eligible nn.Linear under
    ``module`` (e.g. ``model.encoder``); returns the count warmed.

    Only needed for TALL_MODE=wt. Call it BEFORE ``mx.compile`` traces the
    model: ``tall_linear`` builds Wt lazily, and if that happens inside a
    compiled trace the transpose becomes a graph node recomputed on every
    call (~5 ms of wasted work in this encoder). One uncompiled forward pass
    with ``laya_mlx._linear`` patched to route through ``tall_linear``
    achieves the same thing. No-op in the default "scatter"/"tload" modes.
    """
    if _MODE != "wt":
        return 0
    wts = []
    for lin in _iter_linears(module):
        W = lin.weight
        if W.ndim != 2 or W.dtype != mx.float16:
            continue
        N, K = W.shape
        bn, _ = _cfg_for(N, K)
        if bn is None or K % _KT:
            continue
        if getattr(lin, "_tall_wt", None) is None:
            lin._tall_wt = mx.contiguous(W.T)
        wts.append(lin._tall_wt)
    if wts:
        mx.eval(*wts)
    return len(wts)
