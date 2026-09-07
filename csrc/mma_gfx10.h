#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#else
#error "mma_gfx10.h is only intended for ROCm/HIP."
#endif

#include <cstdint>
#include <type_traits>

// ============================================================================
// gfx10 (RDNA2) attention kernel: SIMD dot-product backend.
//
// RDNA2 consumer (gfx1035 / Rembrandt) has NO MFMA tensor core and NO WMMA.
// The only dense-math building blocks are the per-lane SIMD dot instructions:
//   - int8 QK : v_dot4c_i32_i8  (__builtin_amdgcn_sdot4)  -> 4x int8 dot per inst
//   - fp16 PV : v_dot2c_f32_f16 (__builtin_amdgcn_fdot2)  -> 2x fp16 dot per inst
// These are lane-independent: each lane accumulates into its own output
// register(s), there is no cross-lane matrix-tile sharing.
//
// Design (per-lane-row + LDS sharing, chosen to keep register pressure low and
// avoid the LLVM 'Branch Probability Basic Block Placement' codegen segfault
// documented in gfx1035.md §0.10):
//   Block = 128 threads (4 warps). Each block processes BM=32 Q rows.
//   The 32 rows x 4 column-groups tile is laid out so that thread `tid` owns
//   row `tid>>2` and column group `tid&3`, holding HD/4 output columns.
//   - QK: the 4 threads of a row split the BN scores (stride-4 over columns).
//   - softmax: fully per-row in LDS (s_tile[row][*]); each thread reads the
//     whole row -> no cross-lane shuffle needed.
//   - PV: V_T tile staged to LDS as [D][N]; each thread does V_DOT2 over
//     adjacent kv-pairs for its HD/4 output columns.
// All heavy loops use #pragma unroll 1 to keep basic blocks small (the 
// fully-unrolled dot4 chain was the trigger for the RDNA2 LLVM codegen crash).
//
// Compiled only for __GFX10__ (guarded in the body); on __GFX11__ the body is
// empty and the host dispatch selects by runtime device gfxMajor, so the
// gfx1103 WMMA path is untouched.
// ============================================================================
namespace sageattn_gfx10 {

typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));

// v_dot4c_i32_i8: c += a.4xI8 * b.4xI8 (signed), accumulate int32
__device__ __forceinline__ int sdot4_i32_i8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}
// v_dot2c_f32_f16: c += a.2xf16 * b.2xf16 (fp32 accumulate)
__device__ __forceinline__ float fdot2_f32_f16(unsigned a, unsigned b, float c) {
    return __builtin_amdgcn_fdot2(
        *reinterpret_cast<const v2h*>(&a), *reinterpret_cast<const v2h*>(&b), c, false);
}
// load 4 contiguous int8 as 32-bit
__device__ __forceinline__ int load_i8_quad(const int8_t* p) {
    return *reinterpret_cast<const int*>(p);
}
// load 2 contiguous fp16 as 32-bit
__device__ __forceinline__ unsigned load_h2_quad(const __half* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

// ---------------------------------------------------------------------------
// INQ (in-kernel Q int8 quant): mirror quant_qk_int8 rounding exactly.
//   round-half-away + truncate-toward-zero + saturate [-128,127] (matches
//   float_to_int8's v_cvt_i32_f32 path on gfx10).
__device__ __forceinline__ int8_t gfx10_q8_round(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int i = static_cast<int>(x);
    if (i > 127) i = 127;
    if (i < -128) i = -128;
    return static_cast<int8_t>(i);
}
// pack 4 int8 (LE order = load_i8_quad byte order) into one dword
__device__ __forceinline__ int gfx10_q8_pack(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (static_cast<int>(a) & 0xff) | ((static_cast<int>(b) & 0xff) << 8) |
           ((static_cast<int>(c) & 0xff) << 16) | ((static_cast<int>(d) & 0xff) << 24);
}
// raw 2-byte element (fp16 or bf16 bit pattern) -> float
__device__ __forceinline__ float gfx10_q8_2f(unsigned raw_half, bool bf16) {
    if (bf16) {
        __hip_bfloat16 b = *reinterpret_cast<const __hip_bfloat16*>(&raw_half);
        return __bfloat162float(b);
    }
    __half h = *reinterpret_cast<const __half*>(&raw_half);
    return __half2float(h);
}

// output dtype conversion
template <typename ODT> __device__ __forceinline__ ODT gfx10_out_convert(float v);
template <> __device__ __forceinline__ __half gfx10_out_convert<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __hip_bfloat16 gfx10_out_convert<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

// convert input dtype (fp16/bf16) to fp16
template <typename QDT> __device__ __forceinline__ __half gfx10_to_half(QDT v);
template <> __device__ __forceinline__ __half gfx10_to_half<__half>(__half v) { return v; }
template <> __device__ __forceinline__ __half gfx10_to_half<__hip_bfloat16>(__hip_bfloat16 v) { return __float2half(__bfloat162float(v)); }
// bit pattern of QDT value as fp16 (upper 16 bits zero)
template <typename QDT> __device__ __forceinline__ unsigned gfx10_half_bits(QDT v) {
    return static_cast<unsigned>(__half_as_ushort(gfx10_to_half<QDT>(v)));
}

// ============================================================================
// gfx10 v3: TRUE TILE LAYOUT - 1 warp cooperates on QK reduction across D
// ============================================================================
//
// Design (gfx1035 RDNA2, no WMMA, V_DOT4 / V_DOT2 only):
//   - 1 block = 4 warps × 32 lanes = 128 threads
//   - BM = 64 (4 warps × 16 lanes-per-row)
//   - Each warp = 16 rows: lane 0..15 own row 0..15 of that warp's tile
//   - 16 lanes collaborate on D-dim QK reduction (1 V_DOT4 per lane, then
//     5-step permlanex16 tree reduce = 1 QK value)
//   - PV: 16 lanes hold 1 row × D cols (D=64: 4 cols/lane × 16 lanes; D=128: 8/lane)
//
// Compared to v2 (per-lane-row 16-elem dot4 chain):
//   - QK: 1 dot4 per lane (1 cycle) + 5 permlanex reduce = 6 cycle per QK value
//     vs v2's 16 cycle chain × 1 row × amortized 2 rows = 8 cycle/row amortized
//   - PV: 1 fdot2 per lane × 16 lanes parallel = 16 fdot2/cycle vs v2's 16 fdot2/cycle
//     (1 row × CL=16 cols) — but v2 has 2 rows in parallel = 32 fdot2/cycle per thread pair
//   - Net: v3 has half the rows per warp (16 vs 32) but full D-dim parallel per QK
//
// Block size = 4 warps allows up to 4 blocks per CU in parallel (1 wave on 4 SIMDs).
// ============================================================================

// 32-lane full warp reduction (for designs where each lane holds 1 row of QUADS_PER_LANE int32 chunks
// and the whole warp cooperates on a QK reduction).


// ============================================================================
// v8: multi-head shared-K block.
//   Same one-row-per-thread flash layout as v7p (scr/row_m/row_l private), but
//   each block serves HG head-rows for the SAME m-block, staging the K tile
//   ONCE (shared across heads) and the V tile per head with 16B (int4) loads.
//   Rationale: on gfx1035 the v7p H-dispatch re-read the same K for every head
//   block (37/42 ms of the H=8 time was K/V LDS staging alone); v8 cuts that
//   HG-fold for K and vectorizes V staging 4x.
//
//   Template args: HD, ISC, BN (16), BM (Q rows per block == threads), HG
//   (heads per block), ODT.
// ============================================================================
template <int HD, bool ISC, int BN, int BM, int HG, typename ODT, bool IPV = false, bool DBUF = true>
__global__ __attribute__((amdgpu_num_vgpr(224))) __launch_bounds__(BM, 1) void attn_kernel_gfx10_i8_v8_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t q_stride_n_dir_unused,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h,
    int64_t ks_stride_b, int64_t ks_stride_h,
    int tensor_layout, int diag, int v_native,
    const float* __restrict__ v_scale, int64_t vs_stride_b, int64_t vs_stride_h) {
#if defined(__GFX10__)
    const int diag_qk = diag & 1;
    const int diag_sm = (diag >> 1) & 1;
    const int diag_pv = (diag >> 2) & 1;
    const int diag_st = (diag >> 3) & 1;
    const int diag_wb = (diag >> 4) & 1;
    // IPV: int8 PV (P quantized to [0,127], V staged from a pre-quantized int8 tensor
    // recomputed per 32-key tile). V bytes in LDS are halved and the softmax multiply
    // becomes 4-MAC sdot4 instead of fdot2 (2-MAC). v_scale[] holds the per-tile fp16
    // V scale used to rescale each tile contribution into acc[].
    // v_native=1: v is the natural [B,H,N,D] layout (d-contiguous, n-stride = v_stride_h
    // halves); staging reads full 64B sectors and transposes into the [D][N] LDS tiles.
    // v_native=0: v is pre-transposed V_T [B,H,D,N] (n-contiguous), as in v7p.
    constexpr int QUADS = HD / 4;
    constexpr int NTHREAD = BM;
    constexpr int K_STRIDE = HD;
    constexpr int V_STRIDE = BN;
    constexpr int KS_TOTAL = BN * K_STRIDE;
    constexpr int VS_TOTAL = HD * V_STRIDE;
    // IPV rows: 16 int8 d's per staging slot (int4 load); fp16: 8 halves = 16 bytes.
    constexpr int VDSW = IPV ? 16 : 8;
    static_assert((HD * BN / VDSW) % NTHREAD == 0, "V int4 staging must divide evenly");
    static_assert(HG * KS_TOTAL + (HG * VS_TOTAL * (int)sizeof(std::conditional_t<IPV, int8_t, __half>)) * (DBUF ? 2 : 1) <= 49152,
                  "v8 LDS double-buffer exceeds 48KiB budget");

    __shared__ __attribute__((aligned(32))) int8_t k_buf[HG][(DBUF ? 2 : 1)][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) std::conditional_t<IPV, int8_t, __half> v_buf[HG][(DBUF ? 2 : 1)][VS_TOTAL];
    const int8_t* v8p = reinterpret_cast<const int8_t*>(v);

    const int tid = threadIdx.x;
    const int64_t m = blockIdx.x * BM + tid;
    const int64_t b = blockIdx.z;
    const int64_t hg0 = blockIdx.y * HG;

    const bool valid = (m < qo_len);
    (void)q_stride_n_dir_unused;
    (void)tensor_layout;

    // ---- per-head Q rows + scales, resident in regs ----
    int q_reg[HG][QUADS];
    float qs[HG];
    bool hok[HG];
    #pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
        const int64_t h = hg0 + hh;
        hok[hh] = (h < q_heads);
        const float qsv = valid && hok[hh]
            ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)] : 0.0f;
        qs[hh] = qsv;
        if (valid && hok[hh]) {
            const int64_t qb = b * q_stride_b + h * q_stride_h;
            const int8_t* qrow = q + qb + m * q_stride_n;
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq)
                q_reg[hh][dq] = load_i8_quad(qrow + dq * 4);
        } else {
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) q_reg[hh][dq] = 0;
        }
    }

    float acc[HG][HD];
    float row_m[HG], row_l[HG];
    #pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
        #pragma unroll
        for (int c = 0; c < HD; ++c) acc[hh][c] = 0.0f;
        row_m[hh] = -3.0e38f;
        row_l[hh] = 0.0f;
    }

    // ---- per-thread staging lambda for V (native or transposed) ----
    auto stage_tile = [&](int dst, int64_t kb0, int hh, int is_k) {
        const int64_t h = hg0 + hh;
        const int64_t kvh = h / (q_heads / kv_heads);
        if (is_k) {
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb0 + r;
                reinterpret_cast<int*>(&k_buf[hh][dst][r * K_STRIDE + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(k + b * k_stride_b + kvh * k_stride_h + n * k_stride_n + ck * 4) : 0;
            }
        } else if (v_native) {
            // natural [B,H,N,D]: tile [HD][BN]. fp16: int4 (8 halves) per slot;
            // IPV: int4 = 16 int8 d's per slot. Warp covers n rows of coalesced d-sectors.
            #pragma unroll 1
            for (int u = 0; u < (HD * BN / VDSW) / NTHREAD; ++u) {
                const int slot = tid + u * NTHREAD;
                const int n_local = slot / (HD / VDSW);  // 0..BN-1
                const int dg = slot % (HD / VDSW);       // 0..(HD/VDSW)-1
                const int64_t n = kb0 + n_local;
                if (n < kv_len && (dg * VDSW) < HD) {
                    if (IPV) {
                        const int8_t* src = v8p + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + dg * VDSW;
                        int4 val = *reinterpret_cast<const int4*>(src);
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[hh][dst][(dg * VDSW + jj) * V_STRIDE + n_local] = reinterpret_cast<const int8_t*>(&val)[jj];
                    } else {
                        const __half* src = v + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + dg * VDSW;
                        int4 val = *reinterpret_cast<const int4*>(src);
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[hh][dst][(dg * VDSW + jj) * V_STRIDE + n_local] =
                                reinterpret_cast<__half*>(&val)[jj];
                    }
                }
            }
        } else {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            #pragma unroll 1
            for (int i = tid; i < HD * BN / 8; i += NTHREAD) {
                const int e = i * 8;
                const int d = e / BN, cj8 = e % BN;
                const __half* src = v + vbase + d * v_stride_h + (kb0 + cj8);
                int4 val = make_int4(0, 0, 0, 0);
                #pragma unroll
                for (int kk = 0; kk < 8; ++kk)
                    if (kb0 + cj8 + kk < kv_len)
                        reinterpret_cast<__half*>(&val)[kk] = src[kk];
                *reinterpret_cast<int4*>(&v_buf[hh][dst][d * V_STRIDE + cj8]) = val;
            }
        }
    };

    // ---- prologue: stage tile 0 ----
    {
        #pragma unroll
        for (int hh = 0; hh < HG; ++hh)
            stage_tile(0, 0, hh, 1);   // K
        #pragma unroll
        for (int hh = 0; hh < HG; ++hh)
            stage_tile(0, 0, hh, 0);   // V
    }
    __syncthreads();

    constexpr int K_ELEMS = BN * QUADS;
    constexpr int V_V4 = (HD * BN / VDSW) / NTHREAD;   // int4 per head per thread
    constexpr int NK_STG = K_ELEMS / NTHREAD;

    #pragma unroll 1
    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        const int buf = DBUF ? static_cast<int>((kb / BN) & 1) : 0;
        const int ob = DBUF ? (buf ^ 1) : 0;
        const int64_t nb = kb + BN;

        // ---- prefetch tile t+1 (transposed register path) / direct stage (native) ----
        unsigned k_stg[HG][NK_STG];
        int4 v_stg[HG][V_V4];
        if (!v_native) {
            if ((nb < kv_len) && !diag_st) {
                #pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    const int64_t h = hg0 + hh;
                    const int64_t kvh = h / (q_heads / kv_heads);
                    const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
                    #pragma unroll
                    for (int u = 0; u < NK_STG; ++u) {
                        const int i = tid + u * NTHREAD;
                        const int r = i / QUADS, ck = i % QUADS;
                        const int64_t n = nb + r;
                        const int64_t kvhh = ((hg0 + hh) / (q_heads / kv_heads));
                        k_stg[hh][u] = (n < kv_len)
                            ? static_cast<unsigned>(load_i8_quad(k + b * k_stride_b + kvhh * k_stride_h + n * k_stride_n + ck * 4)) : 0u;
                    }
                    #pragma unroll
                    for (int u4 = 0; u4 < V_V4; ++u4) {
                        const int e = (tid + u4 * NTHREAD) * 8;
                        const int d = e / BN, cj8 = e % BN;
                        const int64_t n = nb + cj8;
                        if (n + 8 <= kv_len) {
                            v_stg[hh][u4] = *reinterpret_cast<const int4*>(v + vbase + d * v_stride_h + n);
                        } else {
                            int4 val = make_int4(0, 0, 0, 0);
                            #pragma unroll
                            for (int kk = 0; kk < 8; ++kk)
                                if (n + kk < kv_len)
                                    reinterpret_cast<__half*>(&val)[kk] = *(v + vbase + d * v_stride_h + n + kk);
                            v_stg[hh][u4] = val;
                        }
                    }
                }
            }
        } else if ((nb < kv_len) && !diag_st) {
            // native path: preload next tile into registers now; LDS store deferred until
            // after compute (global-load latency overlaps the current tile's QK/PV).
            #pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                const int64_t h = hg0 + hh;
                const int64_t kvh = h / (q_heads / kv_heads);
                #pragma unroll
                for (int u = 0; u < NK_STG; ++u) {
                    const int i = tid + u * NTHREAD;
                    const int r = i / QUADS, ck = i % QUADS;
                    const int64_t n = nb + r;
                    k_stg[hh][u] = (n < kv_len)
                        ? static_cast<unsigned>(load_i8_quad(k + b * k_stride_b + kvh * k_stride_h + n * k_stride_n + ck * 4)) : 0u;
                }
                #pragma unroll
                for (int u = 0; u < V_V4; ++u) {
                    const int slot = tid + u * NTHREAD;
                    const int n_local = slot / (HD / VDSW);
                    const int d8 = slot % (HD / VDSW);
                    const int64_t n = nb + n_local;
                    int4 val = make_int4(0, 0, 0, 0);
                    if (n < kv_len && (d8 * VDSW) < HD) {
                        if (IPV) {
                            val = *reinterpret_cast<const int4*>(v8p + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + d8 * VDSW);
                        } else {
                            val = *reinterpret_cast<const int4*>(v + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + d8 * VDSW);
                        }
                    }
                    v_stg[hh][u] = val;
                }
            }
        }

        // ---- per-head compute against shared K tile ----
        #pragma unroll 1
        for (int hh = 0; hh < HG; ++hh) {
            const int64_t h = hg0 + hh;
            const bool hokh = hok[hh];
            if (!hokh) continue;
            const int64_t kvh = h / (q_heads / kv_heads);

            // QK
            float scr[BN];
            {
                #pragma unroll
                for (int j0 = 0; j0 < BN; j0 += 4) {
                    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    if (!diag_qk)
                    #pragma unroll
                    for (int dq = 0; dq < QUADS; dq += 4) {
                        const int4 k0 = *reinterpret_cast<const int4*>(&k_buf[hh][buf][(j0 + 0) * K_STRIDE + dq * 4]);
                        const int4 k1 = *reinterpret_cast<const int4*>(&k_buf[hh][buf][(j0 + 1) * K_STRIDE + dq * 4]);
                        const int4 k2 = *reinterpret_cast<const int4*>(&k_buf[hh][buf][(j0 + 2) * K_STRIDE + dq * 4]);
                        const int4 k3 = *reinterpret_cast<const int4*>(&k_buf[hh][buf][(j0 + 3) * K_STRIDE + dq * 4]);
                        s0 = sdot4_i32_i8(q_reg[hh][dq + 0], k0.x, s0); s0 = sdot4_i32_i8(q_reg[hh][dq + 1], k0.y, s0);
                        s0 = sdot4_i32_i8(q_reg[hh][dq + 2], k0.z, s0); s0 = sdot4_i32_i8(q_reg[hh][dq + 3], k0.w, s0);
                        s1 = sdot4_i32_i8(q_reg[hh][dq + 0], k1.x, s1); s1 = sdot4_i32_i8(q_reg[hh][dq + 1], k1.y, s1);
                        s1 = sdot4_i32_i8(q_reg[hh][dq + 2], k1.z, s1); s1 = sdot4_i32_i8(q_reg[hh][dq + 3], k1.w, s1);
                        s2 = sdot4_i32_i8(q_reg[hh][dq + 0], k2.x, s2); s2 = sdot4_i32_i8(q_reg[hh][dq + 1], k2.y, s2);
                        s2 = sdot4_i32_i8(q_reg[hh][dq + 2], k2.z, s2); s2 = sdot4_i32_i8(q_reg[hh][dq + 3], k2.w, s2);
                        s3 = sdot4_i32_i8(q_reg[hh][dq + 0], k3.x, s3); s3 = sdot4_i32_i8(q_reg[hh][dq + 1], k3.y, s3);
                        s3 = sdot4_i32_i8(q_reg[hh][dq + 2], k3.z, s3); s3 = sdot4_i32_i8(q_reg[hh][dq + 3], k3.w, s3);
                    }
                    for (int tj = 0; tj < 4; ++tj) {
                        int s = (tj == 0) ? s0 : (tj == 1) ? s1 : (tj == 2) ? s2 : s3;
                        int64_t n = kb + j0 + tj;
                        const int kscale_idx = static_cast<int>((kb + j0 + tj) / MIN_BLK_K);
                        const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
                        float sc = static_cast<float>(s) * (qs[hh] * ksj);
                        if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
                        scr[j0 + tj] = sc;
                    }
                }
            }

            // softmax (row-local, registers)
            {
                if (diag_sm) row_l[hh] = 1.0f;
                if (!diag_sm) {
                    float lm = scr[0];
                    #pragma unroll
                    for (int j = 1; j < BN; ++j) lm = fmaxf(lm, scr[j]);
                    float gm = fmaxf(row_m[hh], lm);
                    float alpha = (row_l[hh] > 0.0f) ? exp2f(row_m[hh] - gm) : 0.0f;
                    row_m[hh] = gm;
                    row_l[hh] *= alpha;
                    #pragma unroll
                    for (int c = 0; c < HD; ++c) acc[hh][c] *= alpha;
                    float ps = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < BN; ++j) {
                        float p = exp2f(scr[j] - row_m[hh]);
                        scr[j] = p;
                        ps += p;
                    }
                    row_l[hh] += ps;
                }
            }

            // PV
            {
                if (!diag_pv) {
                    if (IPV) {
                        // int8 PV: P -> [0,127] quads, V int8 quads, sdot4 (4 MAC/inst).
                        const float vs = (v_scale
                            ? v_scale[b * vs_stride_b + kvh * vs_stride_h + static_cast<int>(kb / BN)] : 127.0f * 127.0f)
                            * (1.0f / (127.0f * 127.0f));
                        unsigned p8[BN / 4];
                        #pragma unroll
                        for (int j = 0; j < BN; j += 4) {
                            unsigned w = 0;
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                int pi = static_cast<int>(scr[j + jj] * 127.0f + 0.5f);
                                if (pi < 0) pi = 0;
                                if (pi > 127) pi = 127;
                                w |= static_cast<unsigned>(pi) << (jj * 8);
                            }
                            p8[j / 4] = w;
                        }
                        #pragma unroll 4
                        for (int c0 = 0; c0 < HD; ++c0) {
                            const int4 vi0 = *reinterpret_cast<const int4*>(&v_buf[hh][buf][c0 * V_STRIDE + 0]);
                            const int4 vi1 = *reinterpret_cast<const int4*>(&v_buf[hh][buf][c0 * V_STRIDE + 16]);
                            int a = sdot4_i32_i8(static_cast<int>(p8[0]), vi0.x, 0);
                            a = sdot4_i32_i8(static_cast<int>(p8[1]), vi0.y, a);
                            a = sdot4_i32_i8(static_cast<int>(p8[2]), vi0.z, a);
                            a = sdot4_i32_i8(static_cast<int>(p8[3]), vi0.w, a);
                            int b = sdot4_i32_i8(static_cast<int>(p8[4]), vi1.x, 0);
                            b = sdot4_i32_i8(static_cast<int>(p8[5]), vi1.y, b);
                            b = sdot4_i32_i8(static_cast<int>(p8[6]), vi1.z, b);
                            b = sdot4_i32_i8(static_cast<int>(p8[7]), vi1.w, b);
                            if ((diag & 32) && kb == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                                tid < 8 && c0 == 0) {
                                for (int j = 0; j < 8; ++j) g_ipv_dbg[tid * 256 + j] = p8[j];
                                for (int c = 0; c < 4; ++c)
                                    for (int j = 0; j < 32; ++j)
                                        g_ipv_dbg[tid * 256 + 8 + c * 32 + j] = static_cast<unsigned>(
                                            static_cast<int8_t>(v_buf[hh][buf][c * V_STRIDE + j]));
                                g_ipv_dbg[tid * 256 + 136] = *reinterpret_cast<const unsigned*>(&vs);
                                g_ipv_dbg[tid * 256 + 137] = static_cast<unsigned>(kvh);
                            }
                            acc[hh][c0] += static_cast<float>(a + b) * vs;
                        }
                    } else {
                    unsigned p2[BN / 2];
                    #pragma unroll
                    for (int j = 0; j < BN; j += 2)
                        p2[j / 2] = __half_as_ushort(__float2half(scr[j])) |
                                    (static_cast<unsigned>(__half_as_ushort(__float2half(scr[j + 1]))) << 16);
                    #pragma unroll 4
                    for (int c0 = 0; c0 < HD; c0 += 2) {
                        #pragma unroll
                        for (int jj = 0; jj < BN; jj += 8) {
                            const int4 va = *reinterpret_cast<const int4*>(&v_buf[hh][buf][(c0 + 0) * V_STRIDE + jj]);
                            const int4 vb = *reinterpret_cast<const int4*>(&v_buf[hh][buf][(c0 + 1) * V_STRIDE + jj]);
                            acc[hh][c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 0], va.x, acc[hh][c0 + 0]);
                            acc[hh][c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 1], va.y, acc[hh][c0 + 0]);
                            acc[hh][c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 2], va.z, acc[hh][c0 + 0]);
                            acc[hh][c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 3], va.w, acc[hh][c0 + 0]);
                            acc[hh][c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 0], vb.x, acc[hh][c0 + 1]);
                            acc[hh][c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 1], vb.y, acc[hh][c0 + 1]);
                            acc[hh][c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 2], vb.z, acc[hh][c0 + 1]);
                            acc[hh][c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 3], vb.w, acc[hh][c0 + 1]);
                        }
                    }
                    }
                }
            }
        }

        // ---- store staged tile t+1 -> LDS (preloaded in regs above) ----
        if (!DBUF) __syncthreads();  // single buffer: all threads must finish reading before overwrite
        if ((nb < kv_len) && !diag_st) {
            const int ob = DBUF ? (buf ^ 1) : buf;
            #pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                #pragma unroll
                for (int u = 0; u < NK_STG; ++u) {
                    const int i = tid + u * NTHREAD;
                    const int r = i / QUADS, ck = i % QUADS;
                    reinterpret_cast<int*>(&k_buf[hh][ob][r * K_STRIDE + ck * 4])[0] = static_cast<int>(k_stg[hh][u]);
                }
                if (v_native) {
                    if (IPV) {
                        #pragma unroll
                        for (int u = 0; u < V_V4; ++u) {
                            const int slot = tid + u * NTHREAD;
                            const int n_local = slot / (HD / VDSW);
                            const int d8 = slot % (HD / VDSW);
                            #pragma unroll
                            for (int jj = 0; jj < VDSW; ++jj)
                                v_buf[hh][ob][(d8 * VDSW + jj) * V_STRIDE + n_local] =
                                    reinterpret_cast<const int8_t*>(&v_stg[hh][u])[jj];
                        }
                    } else {
                    #pragma unroll
                    for (int u = 0; u < V_V4; ++u) {
                        const int slot = tid + u * NTHREAD;
                        const int n_local = slot / (HD / 8);
                        const int d8 = slot % (HD / 8);
                        #pragma unroll
                        for (int jj = 0; jj < 8; ++jj)
                            v_buf[hh][ob][(d8 * 8 + jj) * V_STRIDE + n_local] =
                                reinterpret_cast<__half*>(&v_stg[hh][u])[jj];
                    }
                    }
                } else {
                    #pragma unroll
                    for (int u4 = 0; u4 < V_V4; ++u4) {
                        const int e = (tid + u4 * NTHREAD) * 8;
                        const int d = e / BN, cj8 = e % BN;
                        *reinterpret_cast<int4*>(&v_buf[hh][ob][d * V_STRIDE + cj8]) = v_stg[hh][u4];
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- write back each head ----
    if (!diag_wb) {
        #pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
            const int64_t h = hg0 + hh;
            if (valid && hok[hh]) {
                float inv = 1.0f / row_l[hh];
                int64_t base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
                #pragma unroll
                for (int c = 0; c < HD; ++c)
                    out[base + c] = gfx10_out_convert<ODT>(acc[hh][c] * inv);
            }
        }
    }
#endif
}  // attn_kernel_gfx10_i8_v8_t

// =============================================================================
// v10: register-tiled PV (Triton-style tt.dot clone). Same QK/softmax as v9,
// but P (fp16) is staged to LDS and PV is a cross-lane tiled GEMM: each lane
// owns a 2-row x (HD/2)-dim output microtile, keys are paired into fdot2
// operands, and V values LDS-read once are broadcast/reused across the
// wavefront's row groups (TM independent accumulator chains per lane -> ILP to
// hide LDS latency). P is stored transposed [BN][BM] so PV loads are
// bank-conflict-free.
// =============================================================================
template <int HD, bool C, int BN, int BM, typename ODT, int TM = 2, bool IPV = false, bool INQ = false>
__global__ __attribute__((amdgpu_num_vgpr(224))) __launch_bounds__(BM, 1) void attn_kernel_gfx10_i10_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t q_stride_n_dir_unused,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h,
    int64_t ks_stride_b, int64_t ks_stride_h,
    int tensor_layout, int diag, int v_native,
    const float* __restrict__ v_scale, int64_t vs_stride_b, int64_t vs_stride_h,
    const void* __restrict__ q_fp, int q_src_bf16, float sm_scale_log2e) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;       // int8 dwords per row
    constexpr int NTHREAD = BM;
    constexpr int K_STRIDE = HD;
    constexpr int V_STRIDE = BN;
    constexpr int KS_TOTAL = BN * K_STRIDE;
    constexpr int VS_TOTAL = HD * V_STRIDE;   // [HD][BN] transposed V tile
    constexpr int VDSW = IPV ? 16 : 8;        // int8 / fp16 staging slots
    constexpr int RSTR = BN + 4;              // p8_buf padded row stride (mult of 4, odd dwords/row)
    // PV register tiling: TM rows x TD dims per lane.
    constexpr int TD = HD / TM;               // dims per lane (dgo groups)
    constexpr int NROWG = BM / TM;            // row groups (PV lanes along rows)
    constexpr int NDMG = HD / TD;             // dim groups (== NTHREAD / NROWG)
    static_assert(NROWG * NDMG == NTHREAD, "v10 lane tiling must fill the block");
    static_assert(NROWG * NDMG == BM, "v10 lane tiling must fill the block");
    static_assert(BN % 2 == 0, "v10 requires even BN");
    static_assert(KS_TOTAL * 2 + VS_TOTAL * 2 * (int)sizeof(std::conditional_t<IPV, int8_t, __half>)
                  + (IPV ? BM * RSTR : BM * BN * (int)sizeof(__half))
                  + BM * 2 * (int)sizeof(float) <= 49152, "v10 LDS budget");

    __shared__ __attribute__((aligned(32))) int8_t k_buf[2][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) std::conditional_t<IPV, int8_t, __half> v_buf[2][VS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half p_buf[BN * BM];   // transposed P [k][m] (fp16 path)
    __shared__ __attribute__((aligned(32))) int8_t p8_buf[BM * RSTR];// row-major P [m][n] (int8-PV path)
    __shared__ __attribute__((aligned(32))) float alpha_buf[BM];     // per-row rescale per tile
    __shared__ __attribute__((aligned(32))) float l_buf[BM];         // final row sums

    const int8_t* v8p = IPV ? reinterpret_cast<const int8_t*>(v) : nullptr;
    (void)v8p;

    const int tid = threadIdx.x;
    const int64_t m = blockIdx.x * BM + tid;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len) && (h < q_heads);

    const int diag_qk = diag & 1;
    const int diag_sm = (diag >> 1) & 1;
    const int diag_pv = (diag >> 2) & 1;
    const int diag_st = (diag >> 3) & 1;
    const int diag_wb = (diag >> 4) & 1;
    (void)q_stride_n_dir_unused; (void)tensor_layout; (void)v_native;
    (void)v_scale; (void)vs_stride_b; (void)vs_stride_h;
    (void)q_fp; (void)q_src_bf16; (void)sm_scale_log2e;

    // ---- Q row resident in registers (QK row owned by lane == tid) ----
    // INQ: quantize fp16/bf16 Q in-kernel, skipping the q8 global prepass
    // round-trip (big win when kv is small and the prepass dominates end-to-end
    // time). Scale math mirrors quant_qk_int8: per MIN_BLK_Q=32-row subgroup,
    // amax over the wave's 32 rows (lane==row => one wave == one subgroup),
    // q8) = 127*q/amax, qsv = amax*sm_scale_log2e/127 stored in the kernel.
    float qsv;
    int q_reg[QUADS];
    if constexpr (INQ) {
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        float row_amax = 1e-7f;
        if (valid) {
            const char* qrow = reinterpret_cast<const char*>(q_fp) + (qb + m * q_stride_n) * 2;
            constexpr int NW = HD / 2;   // dwords (2 halves each)
            #pragma unroll
            for (int x = 0; x < NW; ++x) {
                const unsigned u = reinterpret_cast<const unsigned*>(qrow)[x];
                const float f0 = gfx10_q8_2f(u & 0xffffu, q_src_bf16 != 0);
                const float f1 = gfx10_q8_2f(u >> 16, q_src_bf16 != 0);
                row_amax = fmaxf(row_amax, fabsf(f0));
                row_amax = fmaxf(row_amax, fabsf(f1));
            }
        }
        // full-wave butterfly (32 lanes == one MIN_BLK_Q=32 row group)
        #pragma unroll
        for (int sh = 1; sh < 32; sh <<= 1)
            row_amax = fmaxf(row_amax, __shfl_xor(row_amax, sh));
        qsv = valid ? (row_amax * (1.0f / 127.0f) * sm_scale_log2e) : 0.0f;
        if (valid) {
            const char* qrow = reinterpret_cast<const char*>(q_fp) + (qb + m * q_stride_n) * 2;
            constexpr int NW = HD / 2;
            const float iscale = 127.0f / row_amax;
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) {
                const unsigned a = reinterpret_cast<const unsigned*>(qrow)[2 * dq];
                const unsigned b_ = reinterpret_cast<const unsigned*>(qrow)[2 * dq + 1];
                const float f0 = gfx10_q8_2f(a & 0xffffu, q_src_bf16 != 0);
                const float f1 = gfx10_q8_2f(a >> 16, q_src_bf16 != 0);
                const float f2 = gfx10_q8_2f(b_ & 0xffffu, q_src_bf16 != 0);
                const float f3 = gfx10_q8_2f(b_ >> 16, q_src_bf16 != 0);
                q_reg[dq] = gfx10_q8_pack(
                    gfx10_q8_round(f0 * iscale), gfx10_q8_round(f1 * iscale),
                    gfx10_q8_round(f2 * iscale), gfx10_q8_round(f3 * iscale));
            }
        } else {
            qsv = 0.0f;
        }
    } else {
        qsv = valid
            ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)] : 0.0f;
        if (valid) {
            const int64_t qb = b * q_stride_b + h * q_stride_h;
            const int8_t* qrow = q + qb + m * q_stride_n;
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = load_i8_quad(qrow + dq * 4);
        } else {
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
        }
    }

    float acc[TM][TD];
    float row_m = -3.0e38f, row_l = 0.0f;
    #pragma unroll
    for (int u = 0; u < TM; ++u)
        #pragma unroll
        for (int dd = 0; dd < TD; ++dd) acc[u][dd] = 0.0f;

    auto stage_kv = [&](int dst, int64_t kb0) {
        #pragma unroll 1
        for (int i = tid; i < BN * QUADS; i += NTHREAD) {
            const int r = i / QUADS, ck = i % QUADS;
            const int64_t n = kb0 + r;
            reinterpret_cast<int*>(&k_buf[dst][r * K_STRIDE + ck * 4])[0] =
                (n < kv_len) ? load_i8_quad(k + b * k_stride_b + kvh * k_stride_h + n * k_stride_n + ck * 4) : 0;
        }
        #pragma unroll 1
        for (int u = 0; u < (HD * BN / VDSW) / NTHREAD; ++u) {
            const int slot = tid + u * NTHREAD;
            if (slot < HD * BN / VDSW) {
                // slot = dg*BN + n_local: consecutive threads write consecutive
                // n_local (LDS banks, stride 1) instead of strided dg (bank alias).
                const int n_local = slot % BN;
                const int dg = slot / BN;
                const int64_t n = kb0 + n_local;
                if ((dg * VDSW) < HD) {
                    if (n < kv_len) {
                        if constexpr (IPV) {
                            const int8_t* src = v8p + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + dg * VDSW;
                            int4 val = *reinterpret_cast<const int4*>(src);
                            #pragma unroll
                            for (int jj = 0; jj < VDSW; ++jj)
                                v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] = reinterpret_cast<const int8_t*>(&val)[jj];
                        } else {
                            const __half* src = v + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + dg * VDSW;
                            int4 val = *reinterpret_cast<const int4*>(src);
                            #pragma unroll
                            for (int jj = 0; jj < VDSW; ++jj)
                                v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] = reinterpret_cast<__half*>(&val)[jj];
                        }
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] =
                                std::conditional_t<IPV, int8_t, __half>{0};
                    }
                }
            }
        }
    };

    // prologue: stage tile 0
    stage_kv(0, 0);
    __syncthreads();

    // Causal: rows m in [m0, m0+BM) attend only n <= m, so kv tiles with
    // kb >= m0+BM are entirely above the diagonal (every n > every m in this
    // block) and are masked to -inf anyway. Clamp the tile loop to the block
    // diagonal -> ~half the QK+PV work for causal self, ~half the K/V staging.
    const int64_t kb_lim = C ? min(kv_len, min(qo_len, blockIdx.x * static_cast<int64_t>(BM) + BM)) : kv_len;

    #pragma unroll 1
    for (int64_t kb = 0; kb < kb_lim; kb += BN) {
        const int buf = static_cast<int>((kb / BN) & 1);
        const int64_t nb = kb + BN;

        if ((nb < kb_lim) && !diag_st) {
            stage_kv(buf ^ 1, nb);
        }

        // ---- QK: BN key scores for this lane's row ----
        float scr[BN];
        {
            #pragma unroll
            for (int j0 = 0; j0 < BN; j0 += 4) {
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                if (!diag_qk)
                #pragma unroll
                for (int dq = 0; dq < QUADS; dq += 4) {
                    const int4 k0 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 0) * K_STRIDE + dq * 4]);
                    const int4 k1 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 1) * K_STRIDE + dq * 4]);
                    const int4 k2 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 2) * K_STRIDE + dq * 4]);
                    const int4 k3 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 3) * K_STRIDE + dq * 4]);
                    s0 = sdot4_i32_i8(q_reg[dq + 0], k0.x, s0); s0 = sdot4_i32_i8(q_reg[dq + 1], k0.y, s0);
                    s0 = sdot4_i32_i8(q_reg[dq + 2], k0.z, s0); s0 = sdot4_i32_i8(q_reg[dq + 3], k0.w, s0);
                    s1 = sdot4_i32_i8(q_reg[dq + 0], k1.x, s1); s1 = sdot4_i32_i8(q_reg[dq + 1], k1.y, s1);
                    s1 = sdot4_i32_i8(q_reg[dq + 2], k1.z, s1); s1 = sdot4_i32_i8(q_reg[dq + 3], k1.w, s1);
                    s2 = sdot4_i32_i8(q_reg[dq + 0], k2.x, s2); s2 = sdot4_i32_i8(q_reg[dq + 1], k2.y, s2);
                    s2 = sdot4_i32_i8(q_reg[dq + 2], k2.z, s2); s2 = sdot4_i32_i8(q_reg[dq + 3], k2.w, s2);
                    s3 = sdot4_i32_i8(q_reg[dq + 0], k3.x, s3); s3 = sdot4_i32_i8(q_reg[dq + 1], k3.y, s3);
                    s3 = sdot4_i32_i8(q_reg[dq + 2], k3.z, s3); s3 = sdot4_i32_i8(q_reg[dq + 3], k3.w, s3);
                }
                for (int tj = 0; tj < 4; ++tj) {
                    int s = (tj == 0) ? s0 : (tj == 1) ? s1 : (tj == 2) ? s2 : s3;
                    const int64_t n = kb + j0 + tj;
                    const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + static_cast<int>(n / MIN_BLK_K)];
                    float sc = static_cast<float>(s) * (qsv * ksj);
                    if ((!valid) || (C && n > m) || (n >= kv_len)) sc = -3.0e38f;
                    scr[j0 + tj] = sc;
                }
            }
        }

        // ---- softmax (row-local) + P/alpha staging to LDS ----
        float P_al = 1.0f;
        {
            if (!diag_sm) {
                float lm = scr[0];
                #pragma unroll
                for (int j = 1; j < BN; ++j) lm = fmaxf(lm, scr[j]);
                float gm = fmaxf(row_m, lm);
                float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
                P_al = alpha;
                row_m = gm;
                row_l *= alpha;
                float ps = 0.0f;
                #pragma unroll
                for (int j = 0; j < BN; ++j) { float p = exp2f(scr[j] - row_m); ps += p; }
                row_l += ps;
                if constexpr (IPV) {
                    const int rowr = static_cast<int>(m % BM);
                    #pragma unroll
                    for (int j = 0; j < BN; j += 4) {
                        unsigned w = 0;
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            float p = exp2f(scr[j + jj] - row_m);
                            int pi = static_cast<int>(p * 127.0f + 0.5f);
                            if (pi < 0) pi = 0;
                            if (pi > 127) pi = 127;
                            w |= static_cast<unsigned>(pi) << (jj * 8);
                        }
                        *reinterpret_cast<unsigned*>(&p8_buf[rowr * RSTR + j]) = w;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < BN; ++j) p_buf[j * BM + (int)(m % BM)] = __float2half(exp2f(scr[j] - row_m));
                }
            } else {
                #pragma unroll
                for (int j = 0; j < BN; ++j) p_buf[j * BM + (int)(m % BM)] = __float2half(scr[j]);
                if constexpr (IPV) {
                    const int rowr = static_cast<int>(m % BM);
                    #pragma unroll
                    for (int j = 0; j < BN; j += 4) {
                        unsigned w = 0;
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            float p = exp2f(scr[j + jj] - row_m);
                            int pi = static_cast<int>(p * 127.0f + 0.5f);
                            if (pi < 0) pi = 0;
                            if (pi > 127) pi = 127;
                            w |= static_cast<unsigned>(pi) << (jj * 8);
                        }
                        *reinterpret_cast<unsigned*>(&p8_buf[rowr * RSTR + j]) = w;
                    }
                }
            }
            alpha_buf[m % BM] = P_al;
            l_buf[m % BM] = row_l;
        }

        // PV lanes must see this tile's P/alpha before the tiled GEMM
        __syncthreads();

        // ---- PV: register-tiled GEMM ----
        if (!diag_pv) {
            const int rg = tid % NROWG;
            const int dgo = tid / NROWG;
            const int r0 = rg * TM;
            #pragma unroll
            for (int u = 0; u < TM; ++u) {
                const float al = alpha_buf[r0 + u];
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd) acc[u][dd] *= al;
            }
            if constexpr (IPV) {
                // int8-PV: 4 keys per 32-bit P (row-major padded p8_buf) and V operand,
                // sdot4 = 4 MAC/inst, V LDS bytes halved vs fp16. Per-32-key-tile v_scale[]
                // folds in on each sdot4 (int32 -> float, fma with vs).
                const float vs = (v_scale
                    ? v_scale[b * vs_stride_b + kvh * vs_stride_h + static_cast<int>(kb / 32)] : 127.0f * 127.0f)
                    * (1.0f / (127.0f * 127.0f));
                int iacc[TM][TD];
                #pragma unroll
                for (int u = 0; u < TM; ++u)
                    #pragma unroll
                    for (int dd = 0; dd < TD; ++dd) iacc[u][dd] = 0;
                #pragma unroll
                for (int kp = 0; kp < BN / 4; ++kp) {
                    const int k = kp * 4;
                    unsigned pp[TM];
                    #pragma unroll
                    for (int u = 0; u < TM; ++u)
                        pp[u] = *reinterpret_cast<const unsigned*>(&p8_buf[(r0 + u) * RSTR + k]);
                    #pragma unroll
                    for (int dd = 0; dd < TD; ++dd) {
                        const int d = dgo * TD + dd;
                        const int vv = *reinterpret_cast<const int*>(&v_buf[buf][d * V_STRIDE + k]);
                        #pragma unroll
                        for (int u = 0; u < TM; ++u)
                            iacc[u][dd] = sdot4_i32_i8(static_cast<int>(pp[u]), vv, iacc[u][dd]);
                    }
                }
                #pragma unroll
                for (int u = 0; u < TM; ++u)
                    #pragma unroll
                    for (int dd = 0; dd < TD; ++dd) acc[u][dd] += static_cast<float>(iacc[u][dd]) * vs;
            } else {
                #pragma unroll
                for (int kp = 0; kp < BN / 2; ++kp) {
                    const int k = kp * 2;
                    unsigned pp[TM];
                    #pragma unroll
                    for (int up = 0; up < TM / 2; ++up) {
                        // 2 halfs at p_buf[k*BM + r0+up*2] = rows {r0+2up, r0+2up+1} of key k
                        const unsigned pk0 = *reinterpret_cast<const unsigned*>(&p_buf[k * BM + r0 + up * 2]);
                        const unsigned pk1 = *reinterpret_cast<const unsigned*>(&p_buf[(k + 1) * BM + r0 + up * 2]);
                        pp[up * 2]     = (pk0 & 0xffffu) | ((pk1 & 0xffffu) << 16);   // (k,r0+2up),(k+1,r0+2up)
                        pp[up * 2 + 1] = (pk0 >> 16)      | (pk1 & 0xffff0000u);      // (k,r0+2up+1),(k+1,r0+2up+1)
                    }
                    #pragma unroll
                    for (int dd = 0; dd < TD; ++dd) {
                        const int d = dgo * TD + dd;
                        const unsigned vv = *reinterpret_cast<const unsigned*>(&v_buf[buf][d * V_STRIDE + k]);
                        #pragma unroll
                        for (int u = 0; u < TM; ++u)
                            acc[u][dd] = fdot2_f32_f16(pp[u], vv, acc[u][dd]);
                    }
                }
            }
        }

        // All threads must finish staging the next tile (and reading the current one)
        // before the double-buffer flips.
        __syncthreads();
    }

    // ---- writeback: lane writes its 2-row x TD-dim slices ----
    if (!diag_wb) {
        const int rg = tid % NROWG;
        const int dgo = tid / NROWG;
        const int r0 = rg * TM;
        #pragma unroll
        for (int u = 0; u < TM; ++u) {
            const int64_t row = blockIdx.x * BM + r0 + u;
            if ((row < qo_len) && (h < q_heads)) {
                const float inv = 1.0f / l_buf[r0 + u];
                const int64_t base = b * o_stride_b + row * o_stride_n + h * o_stride_h;
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd)
                    out[base + dgo * TD + dd] = gfx10_out_convert<ODT>(acc[u][dd] * inv);
            }
        }
    }
#endif
}  // attn_kernel_gfx10_i10_t


}  // namespace sageattn_gfx10
