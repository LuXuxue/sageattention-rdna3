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
// gfx10 int8 QK attention kernel (quantized path).
//
// Template args:
//   HD  - head dim (64 or 128)
//   ISC - causal
//   BN  - kv tile width (16 / 32)
//   ODT - output dtype (__half / __hip_bfloat16)
// ============================================================================
template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_i8_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h,
    int64_t ks_stride_b, int64_t ks_stride_h,
    int tensor_layout) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;   // int8 dwords per row
    constexpr int BM = 32;          // Q rows per block
    constexpr int NW = 4;           // col groups (threads per row) = 4
    constexpr int CL = HD / NW;     // output cols per thread (16 for HD64, 32 for HD128)
    constexpr int NTHREAD = BM * NW; // 128 threads

    const int tid = threadIdx.x;
    const int row = tid >> 2;        // 0..31 Q row within block
    const int ct = tid & 3;          // 0..3 column group
    const int64_t m = blockIdx.x * BM + row;   // global Q row

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);       // GQA: q head h -> kv head h/groups

    // ---- LDS tiles ----
    // LDS padding (same scheme as v3/v4) to break bank conflicts:
    //   k_tile: +4 bytes/row  breaks even/odd j alias (2-way)
    //   v_tile: +1 half/row   breaks ct alias (4-way)
    //   s_tile: +1 float/row  breaks row-alias (8-way)
    //   p_tile: +1 half/row   breaks row-alias (4-way)
    constexpr int K_STRIDE = HD + 4;
    constexpr int V_STRIDE = BN + 1;
    constexpr int S_STRIDE = BN + 1;
    constexpr int P_STRIDE = BN + 1;
    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * K_STRIDE];
    __shared__ __half v_tile[HD * V_STRIDE];      // V_T tile [D][N], row=d, col=n
    __shared__ float s_tile[BM * S_STRIDE];       // raw QK scores
    __shared__ __half p_tile[BM * P_STRIDE];      // softmax probs

    const bool valid = (m < qo_len);

    // ---- stage Q tile (BM x HD int8) cooperatively ----
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    // (K/V tiles are staged inside the kv-loop below.)

    const float qs = valid
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)]
        : 0.0f;

    // ---- PV accumulators (per-thread CL columns) ----
    float acc[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) acc[c] = 0.0f;
    float row_m = -3.0e38f, row_l = 0.0f;

    // ---- loop over kv tiles ----
    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile (BN x HD int8) from global K
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * K_STRIDE + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        // stage v_tile [HD][BN] from global V_T [B,H,D,N]
        {
            // V_T[b,h,d,n] address = v + b*vsb + h*vsh + d*vsd + n
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n; // H stride = stride(1), V_T has kv_heads rows
            const int64_t vsd = v_stride_h;                        // D-dim stride = stride(2)
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * V_STRIDE + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK: thread computes its j-slice of the row's BN scores ----
        const int out_base_col = ct * CL;
        // j in [0,BN) with j%4==ct. Inner dot4 chain unrolled x8 for ILP across the
        // D-reduction (bounded basic block; full unroll would re-trigger the RDNA2
        // LLVM codegen crash from gfx1035.md §0.10).
        #pragma unroll 2
        for (int j = ct; j < BN; j += NW) {
            int s = 0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i)
                s = sdot4_i32_i8(
                    load_i8_quad(&q_tile[row * HD + i * 4]),
                    load_i8_quad(&k_tile[j * K_STRIDE + i * 4]), s);
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc = static_cast<float>(s) * (qs * ksj);
            if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
            s_tile[row * S_STRIDE + j] = sc;
        }
        __syncthreads();

        // ---- online softmax (per-row, read from LDS s_tile) ----
        float lm = -3.0e38f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * S_STRIDE + j]);
        float gm = fmaxf(row_m, lm);
        // score ALREADY includes sm_scale*log2e (folded into q_scale by quant),
        // so softmax uses exp2 directly (NOT exp2(*log2e)) to match the gfx11 kernel.
        float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
        row_m = gm;
        row_l *= alpha;
        #pragma unroll
        for (int c = 0; c < CL; ++c) acc[c] *= alpha;
        // pass 2: exp -> p_tile, accumulate sum
        float ps = 0.0f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) {
            float p = exp2f(s_tile[row * S_STRIDE + j] - row_m);
            p_tile[row * P_STRIDE + j] = __float2half(p);
            ps += p;
        }
        row_l += ps;

        // ---- PV: for each kv-pair, accumulate into all CL output columns ----
        // Hoist the p_pair load (reused across all columns) and fully unroll the
        // column loop so the CL independent acc chains run in parallel (ILP).
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2 = load_h2_quad(&p_tile[row * P_STRIDE + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc[c] = fdot2_f32_f16(p2, load_h2_quad(&v_tile[d * V_STRIDE + jj]), acc[c]);
            }
        }
        __syncthreads();
    }

    // ---- normalize + write back ----
    if (valid) {
        float inv = 1.0f / row_l;
        int64_t base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_t

// ============================================================================
// gfx10 int8 QK attention kernel v2: 2 rows per thread for ILP.
//
// Motivation: per-lane-row v1 (BM=32, NW=4) has 1 thread holding 1 row's
//   16-element dot4 chain. Even with 8-way unroll, the chain takes 16 cycles
//   per j, and 8 j iterations = 128 cycles per QK per thread. With 2 rows
//   per thread, the 2 dot4 chains are independent and the compiler can
//   interleave them, halving the per-row QK time.
//
// Layout:
//   Block = 128 threads, BM=64, NW=4 (col groups = 4 per row).
//   Thread t -> row_pair = t>>3 (0..15), within-pair row = (t>>2)&1 (0..1),
//                 col_group = t&3 (0..3).
//   Each thread holds 2 rows × 4 col-groups = 8 acc, 2 row-m, 2 row-l.
//   QK: 2 independent dot4 chains (per row), interleaved for ILP.
//   PV: 8 acc (2 rows × 4 col groups), interleaved dot2.
// ============================================================================
template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_i8_v2_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t q_stride_n_dir_unused,  // padding to match v1's signature
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h,
    int64_t ks_stride_b, int64_t ks_stride_h,
    int tensor_layout) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;     // int8 dwords per row
    constexpr int BM = 64;            // Q rows per block (2x v1)
    constexpr int NW = 4;             // col groups per row
    constexpr int NR = 2;             // rows per thread
    constexpr int CL = HD / NW;       // output cols per thread (16 for HD64)
    constexpr int NTHREAD = (BM / NR) * NW;  // 32 * 4 = 128 threads
    static_assert(NR * (BM / NR) == BM, "BM must be multiple of NR");
    static_assert(NTHREAD == 128, "v2 layout expects 128 threads");

    const int tid = threadIdx.x;
    const int rp = tid >> 2;          // row-pair index 0..31 (BM/NR=32)
    const int ri = tid & 2 ? 1 : 0;   // within-pair row index 0/1 (we use bit 1)
    // Use (t & 0x4) for row select, lower 2 bits for col group
    const int row_in_pair = (tid >> 2) & 1;  // 0 or 1
    const int ct = tid & 3;            // 0..3 column group
    const int row0 = rp * 2 + 0;       // first row owned
    const int row1 = rp * 2 + 1;       // second row owned
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    // ---- LDS tiles ----
    constexpr int K_STRIDE = HD + 4;
    constexpr int V_STRIDE = BN + 1;
    constexpr int S_STRIDE = BN + 1;
    constexpr int P_STRIDE = BN + 1;
    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * K_STRIDE];
    __shared__ __half v_tile[HD * V_STRIDE];
    __shared__ float s_tile[BM * S_STRIDE];
    __shared__ __half p_tile[BM * P_STRIDE];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    // ---- stage Q tile (BM x HD int8) cooperatively ----
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    __syncthreads();  // ensure all Q writes are visible before pre-load to regs

    // Pre-load this thread's 2 rows of Q into registers (2 rows * QUADS i32)
    int q_reg0[QUADS], q_reg1[QUADS];
    #pragma unroll
    for (int i = 0; i < QUADS; ++i) {
        q_reg0[i] = *reinterpret_cast<const int*>(&q_tile[row0 * HD + i * 4]);
        q_reg1[i] = *reinterpret_cast<const int*>(&q_tile[row1 * HD + i * 4]);
    }

    const float qs0 = valid0
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m0 / MIN_BLK_Q)]
        : 0.0f;
    const float qs1 = valid1
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m1 / MIN_BLK_Q)]
        : 0.0f;

    // ---- PV accumulators: 2 rows × CL cols ----
    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile (BN x HD int8)
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * K_STRIDE + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        // stage v_tile [HD][BN] from V_T
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * V_STRIDE + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK: 2 independent dot4 chains per j (one per row) ----
        const int out_base_col = ct * CL;
        #pragma unroll 4
        for (int j = ct; j < BN; j += NW) {
            int s0 = 0, s1 = 0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i) {
                int k_i = load_i8_quad(&k_tile[j * K_STRIDE + i * 4]);
                s0 = sdot4_i32_i8(q_reg0[i], k_i, s0);
                s1 = sdot4_i32_i8(q_reg1[i], k_i, s1);
            }
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc0 = static_cast<float>(s0) * (qs0 * ksj);
            float sc1 = static_cast<float>(s1) * (qs1 * ksj);
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * S_STRIDE + j] = sc0;
            s_tile[row1 * S_STRIDE + j] = sc1;
        }
        __syncthreads();

        // ---- online softmax (per row, 2 rows) ----
        // Row 0
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row0 * S_STRIDE + j]);
            float gm = fmaxf(row_m0, lm);
            float alpha = (row_l0 > 0.0f) ? exp2f(row_m0 - gm) : 0.0f;
            row_m0 = gm; row_l0 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc0[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row0 * S_STRIDE + j] - row_m0);
                p_tile[row0 * P_STRIDE + j] = __float2half(p);
                ps += p;
            }
            row_l0 += ps;
        }
        // Row 1
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row1 * S_STRIDE + j]);
            float gm = fmaxf(row_m1, lm);
            float alpha = (row_l1 > 0.0f) ? exp2f(row_m1 - gm) : 0.0f;
            row_m1 = gm; row_l1 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc1[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row1 * S_STRIDE + j] - row_m1);
                p_tile[row1 * P_STRIDE + j] = __float2half(p);
                ps += p;
            }
            row_l1 += ps;
        }

        // ---- PV: 2 rows × CL cols, interleaved dot2 ----
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2_0 = load_h2_quad(&p_tile[row0 * P_STRIDE + jj]);
            unsigned p2_1 = load_h2_quad(&p_tile[row1 * P_STRIDE + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc0[c] = fdot2_f32_f16(p2_0, load_h2_quad(&v_tile[d * V_STRIDE + jj]), acc0[c]);
                acc1[c] = fdot2_f32_f16(p2_1, load_h2_quad(&v_tile[d * V_STRIDE + jj]), acc1[c]);
            }
        }
        __syncthreads();
    }

    // ---- normalize + write back ----
    if (valid0) {
        float inv = 1.0f / row_l0;
        int64_t base = b * o_stride_b + m0 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc0[c] * inv);
    }
    if (valid1) {
        float inv = 1.0f / row_l1;
        int64_t base = b * o_stride_b + m1 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc1[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_v2_t

// ============================================================================
// EXPERIMENT v2x: v2 with FULLY fp32 probability path.
//   - s_tile computed then p stored/held in fp32
//   - PV accumulation in fp32 (NO fp16 conversion of p; V read as fp16->fp32)
// Diagnoses whether BN=32 精度崩坏 comes from fp16 rounding in PV. Slower than
// v2/v2f (no fdot2) but should be maximally accurate.
// ============================================================================
template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_i8_v2x_t(
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
    int tensor_layout) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;
    constexpr int BM = 64;
    constexpr int NW = 4;
    constexpr int NR = 2;
    constexpr int CL = HD / NW;
    constexpr int NTHREAD = (BM / NR) * NW;

    const int tid = threadIdx.x;
    const int rp = tid >> 2;
    const int ct = tid & 3;
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * HD];
    __shared__ __half v_tile[HD * BN];
    __shared__ float s_tile[BM * BN];
    __shared__ float p_tilef[BM * BN];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    __syncthreads();

    int q_reg0[QUADS], q_reg1[QUADS];
    #pragma unroll
    for (int i = 0; i < QUADS; ++i) {
        q_reg0[i] = *reinterpret_cast<const int*>(&q_tile[row0 * HD + i * 4]);
        q_reg1[i] = *reinterpret_cast<const int*>(&q_tile[row1 * HD + i * 4]);
    }

    const float qs0 = valid0
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m0 / MIN_BLK_Q)]
        : 0.0f;
    const float qs1 = valid1
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m1 / MIN_BLK_Q)]
        : 0.0f;

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * HD + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * BN + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        const int out_base_col = ct * CL;
        #pragma unroll 4
        for (int j = ct; j < BN; j += NW) {
            int s0 = 0, s1 = 0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i) {
                int k_i = load_i8_quad(&k_tile[j * HD + i * 4]);
                s0 = sdot4_i32_i8(q_reg0[i], k_i, s0);
                s1 = sdot4_i32_i8(q_reg1[i], k_i, s1);
            }
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc0 = static_cast<float>(s0) * (qs0 * ksj);
            float sc1 = static_cast<float>(s1) * (qs1 * ksj);
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * BN + j] = sc0;
            s_tile[row1 * BN + j] = sc1;
        }
        __syncthreads();

        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row0 * BN + j]);
            float gm = fmaxf(row_m0, lm);
            float alpha = (row_l0 > 0.0f) ? exp2f(row_m0 - gm) : 0.0f;
            row_m0 = gm; row_l0 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc0[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row0 * BN + j] - row_m0);
                p_tilef[row0 * BN + j] = p;
                ps += p;
            }
            row_l0 += ps;
        }
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row1 * BN + j]);
            float gm = fmaxf(row_m1, lm);
            float alpha = (row_l1 > 0.0f) ? exp2f(row_m1 - gm) : 0.0f;
            row_m1 = gm; row_l1 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc1[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row1 * BN + j] - row_m1);
                p_tilef[row1 * BN + j] = p;
                ps += p;
            }
            row_l1 += ps;
        }

        // PV in fp32 (no fp16 rounding of p)
        for (int j = 0; j < BN; ++j) {
            float p0 = p_tilef[row0 * BN + j];
            float p1 = p_tilef[row1 * BN + j];
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                float vv = __half2float(v_tile[d * BN + j]);
                acc0[c] += p0 * vv;
                acc1[c] += p1 * vv;
            }
        }
        __syncthreads();
    }

    if (valid0) {
        float inv = 1.0f / row_l0;
        int64_t base = b * o_stride_b + m0 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc0[c] * inv);
    }
    if (valid1) {
        float inv = 1.0f / row_l1;
        int64_t base = b * o_stride_b + m1 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc1[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_v2x_t


// ============================================================================
// EXPERIMENT: v2 with fp32 (instead of fp16) p_tile.
// Hypothesis: D=128 int8 v2/BN=32 online-softmax 精度崩坏 (kv>2048) 主因是
// fp16 p_tile 逐 tile 存概率量化 + acc*=alpha 重标定累积误差。改用 fp32 p_tile
// 应能恢复精度同时保留 BN=32 的速度。PV 每次读 p 需从 fp32 转回 2xfp16 供 fdot2。
// ============================================================================
template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_i8_v2f_t(
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
    int tensor_layout) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;
    constexpr int BM = 64;
    constexpr int NW = 4;
    constexpr int NR = 2;
    constexpr int CL = HD / NW;
    constexpr int NTHREAD = (BM / NR) * NW;

    const int tid = threadIdx.x;
    const int rp = tid >> 2;
    const int ct = tid & 3;
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * HD];
    __shared__ __half v_tile[HD * BN];
    __shared__ float s_tile[BM * BN];
    __shared__ float p_tilef[BM * BN];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    __syncthreads();

    int q_reg0[QUADS], q_reg1[QUADS];
    #pragma unroll
    for (int i = 0; i < QUADS; ++i) {
        q_reg0[i] = *reinterpret_cast<const int*>(&q_tile[row0 * HD + i * 4]);
        q_reg1[i] = *reinterpret_cast<const int*>(&q_tile[row1 * HD + i * 4]);
    }

    const float qs0 = valid0
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m0 / MIN_BLK_Q)]
        : 0.0f;
    const float qs1 = valid1
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m1 / MIN_BLK_Q)]
        : 0.0f;

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * HD + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * BN + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        const int out_base_col = ct * CL;
        #pragma unroll 4
        for (int j = ct; j < BN; j += NW) {
            int s0 = 0, s1 = 0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i) {
                int k_i = load_i8_quad(&k_tile[j * HD + i * 4]);
                s0 = sdot4_i32_i8(q_reg0[i], k_i, s0);
                s1 = sdot4_i32_i8(q_reg1[i], k_i, s1);
            }
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc0 = static_cast<float>(s0) * (qs0 * ksj);
            float sc1 = static_cast<float>(s1) * (qs1 * ksj);
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * BN + j] = sc0;
            s_tile[row1 * BN + j] = sc1;
        }
        __syncthreads();

        // Row 0
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row0 * BN + j]);
            float gm = fmaxf(row_m0, lm);
            float alpha = (row_l0 > 0.0f) ? exp2f(row_m0 - gm) : 0.0f;
            row_m0 = gm; row_l0 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc0[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row0 * BN + j] - row_m0);
                p_tilef[row0 * BN + j] = p;
                ps += p;
            }
            row_l0 += ps;
        }
        // Row 1
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row1 * BN + j]);
            float gm = fmaxf(row_m1, lm);
            float alpha = (row_l1 > 0.0f) ? exp2f(row_m1 - gm) : 0.0f;
            row_m1 = gm; row_l1 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc1[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row1 * BN + j] - row_m1);
                p_tilef[row1 * BN + j] = p;
                ps += p;
            }
            row_l1 += ps;
        }

        // PV: pack 2 fp32 p into fp16x2 for fdot2
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2_0 =
                (static_cast<unsigned>(__half_as_ushort(__float2half(p_tilef[row0 * BN + jj + 1]))) << 16) |
                static_cast<unsigned>(__half_as_ushort(__float2half(p_tilef[row0 * BN + jj])));
            unsigned p2_1 =
                (static_cast<unsigned>(__half_as_ushort(__float2half(p_tilef[row1 * BN + jj + 1]))) << 16) |
                static_cast<unsigned>(__half_as_ushort(__float2half(p_tilef[row1 * BN + jj])));
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc0[c] = fdot2_f32_f16(p2_0, load_h2_quad(&v_tile[d * BN + jj]), acc0[c]);
                acc1[c] = fdot2_f32_f16(p2_1, load_h2_quad(&v_tile[d * BN + jj]), acc1[c]);
            }
        }
        __syncthreads();
    }

    if (valid0) {
        float inv = 1.0f / row_l0;
        int64_t base = b * o_stride_b + m0 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc0[c] * inv);
    }
    if (valid1) {
        float inv = 1.0f / row_l1;
        int64_t base = b * o_stride_b + m1 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc1[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_v2f_t


// ============================================================================
// gfx10 int8 QK attention kernel v2.2: same as v2 but PV reads V_T directly
// from global (bypasses LDS v_tile).
//
// Trade-off:
//   + Skip v_tile staging (1 __syncthreads + LDS write per kv-tile)
//   + Less LDS pressure (saves 4 KiB / block for D=64)
//   - PV reads V_T from L1/L2 each iteration (cache-friendly; 32B per row × 16 rows
//     per thread = 512B per thread per kv-tile, 16KB per block per kv-tile, fits
//     in 32KB L1)
//
// Dispatched only for self-attn long sequences where the saved sync outweighs
// the extra global reads. Set SAGEATTN_GFX10_VT_GLOBAL=1 to force, =2 to auto.
// ============================================================================
template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_i8_v2_2_t(
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
    int tensor_layout) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;
    constexpr int BM = 64, NW = 4, NR = 2;
    constexpr int CL = HD / NW;
    constexpr int NTHREAD = (BM / NR) * NW;

    const int tid = threadIdx.x;
    const int rp = tid >> 2;
    const int ct = tid & 3;
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * HD];
    __shared__ float s_tile[BM * BN];
    __shared__ __half p_tile[BM * BN];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    // stage Q
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    __syncthreads();

    int q_reg0[QUADS], q_reg1[QUADS];
    #pragma unroll
    for (int i = 0; i < QUADS; ++i) {
        q_reg0[i] = *reinterpret_cast<const int*>(&q_tile[row0 * HD + i * 4]);
        q_reg1[i] = *reinterpret_cast<const int*>(&q_tile[row1 * HD + i * 4]);
    }

    const float qs0 = valid0
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m0 / MIN_BLK_Q)]
        : 0.0f;
    const float qs1 = valid1
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m1 / MIN_BLK_Q)]
        : 0.0f;

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
    const int64_t vsd = v_stride_h;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile only (no v_tile!)
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * HD + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        __syncthreads();

        // ---- QK (same as v2) ----
        const int out_base_col = ct * CL;
        #pragma unroll 2
        for (int j = ct; j < BN; j += NW) {
            int s0 = 0, s1 = 0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i) {
                int k_i = load_i8_quad(&k_tile[j * HD + i * 4]);
                s0 = sdot4_i32_i8(q_reg0[i], k_i, s0);
                s1 = sdot4_i32_i8(q_reg1[i], k_i, s1);
            }
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc0 = static_cast<float>(s0) * (qs0 * ksj);
            float sc1 = static_cast<float>(s1) * (qs1 * ksj);
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * BN + j] = sc0;
            s_tile[row1 * BN + j] = sc1;
        }
        __syncthreads();

        // ---- online softmax (2 rows) ----
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row0 * BN + j]);
            float gm = fmaxf(row_m0, lm);
            float alpha = (row_l0 > 0.0f) ? exp2f(row_m0 - gm) : 0.0f;
            row_m0 = gm; row_l0 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc0[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row0 * BN + j] - row_m0);
                p_tile[row0 * BN + j] = __float2half(p);
                ps += p;
            }
            row_l0 += ps;
        }
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row1 * BN + j]);
            float gm = fmaxf(row_m1, lm);
            float alpha = (row_l1 > 0.0f) ? exp2f(row_m1 - gm) : 0.0f;
            row_m1 = gm; row_l1 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc1[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row1 * BN + j] - row_m1);
                p_tile[row1 * BN + j] = __float2half(p);
                ps += p;
            }
            row_l1 += ps;
        }

        // ---- PV: read V_T from global (L1/L2 cache) ----
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2_0 = load_h2_quad(&p_tile[row0 * BN + jj]);
            unsigned p2_1 = load_h2_quad(&p_tile[row1 * BN + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                int64_t n = kb + jj;
                if (n < kv_len) {
                    unsigned v2 = *reinterpret_cast<const unsigned*>(v + vbase + d * vsd + n);
                    acc0[c] = fdot2_f32_f16(p2_0, v2, acc0[c]);
                    acc1[c] = fdot2_f32_f16(p2_1, v2, acc1[c]);
                }
            }
        }
        __syncthreads();
    }

    if (valid0) {
        float inv = 1.0f / row_l0;
        int64_t base = b * o_stride_b + m0 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc0[c] * inv);
    }
    if (valid1) {
        float inv = 1.0f / row_l1;
        int64_t base = b * o_stride_b + m1 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc1[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_v2_2_t

// ============================================================================
// gfx10 fp16/bf16 direct attention kernel (no quantization).
//
// Q/K are read as QDT (__half or __hip_bfloat16) and converted to fp16 when
// staged into LDS, so the fp16 V_DOT2 is always used for QK. V is V_T (fp16).
// PV identical to the int8 kernel (V_DOT2 over kv-pairs). Output written as ODT.
// Same per-lane-row layout as the int8 kernel: 128 threads, 32 rows, 4 col-groups.
// ============================================================================
template <int HD, bool ISC, int BN, typename QDT, typename ODT>
__global__ void attn_kernel_gfx10_direct_t(
    const QDT* __restrict__ q, const QDT* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    float sm_scale_log2e, int tensor_layout) {
#if defined(__GFX10__)
    constexpr int BM = 32, NW = 4, CL = HD / NW, NTHREAD = BM * NW;

    const int tid = threadIdx.x;
    const int row = tid >> 2;
    const int ct = tid & 3;
    const int64_t m = blockIdx.x * BM + row;
    const int64_t b = blockIdx.z, h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);       // GQA: q head h -> kv head h/groups

    // LDS padding to break bank conflicts (fp16 rows have 1:1 hbank mapping)
    constexpr int Q_STRIDE = HD + 1;  // +1 half
    constexpr int K_STRIDE = HD + 1;  // +1 half
    constexpr int V_STRIDE = BN + 1;  // +1 half
    constexpr int S_STRIDE = BN + 1;  // +1 float
    constexpr int P_STRIDE = BN + 1;  // +1 half
    __shared__ __half q_tile[BM * Q_STRIDE];
    __shared__ __half k_tile[BN * K_STRIDE];
    __shared__ __half v_tile[HD * V_STRIDE];
    __shared__ float s_tile[BM * S_STRIDE];
    __shared__ __half p_tile[BM * P_STRIDE];

    const bool valid = (m < qo_len);

    // stage Q tile (BM x HD) as fp16
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * HD; i += NTHREAD) {
            int r = i / HD, cq = i % HD;
            int64_t gr = blockIdx.x * BM + r;
            q_tile[r * Q_STRIDE + cq] = (gr < qo_len) ? gfx10_to_half<QDT>(q[qb + gr * q_stride_n + cq]) : __float2half(0.0f);
        }
    }

    float acc[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) acc[c] = 0.0f;
    float row_m = -3.0e38f, row_l = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile as fp16
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * HD; i += NTHREAD) {
                int r = i / HD, ck = i % HD;
                int64_t n = kb + r;
                k_tile[r * K_STRIDE + ck] = (n < kv_len) ? gfx10_to_half<QDT>(k[kqb + n * k_stride_n + ck]) : __float2half(0.0f);
            }
        }
        // stage v_tile [HD][BN] from V_T
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n; // V_T has kv_heads rows
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                v_tile[d * V_STRIDE + cj] = (n < kv_len) ? v[vbase + d * vsd + n] : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK via fp16 V_DOT2 over d-pairs (4 independent accumulators for ILP) ----
        const int out_base_col = ct * CL;
        #pragma unroll 1
        for (int j = ct; j < BN; j += NW) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const unsigned* qr = reinterpret_cast<const unsigned*>(&q_tile[row * Q_STRIDE]);
            const unsigned* kr = reinterpret_cast<const unsigned*>(&k_tile[j * K_STRIDE]);
            #pragma unroll 1
            for (int dq = 0; dq < HD; dq += 8) {
                s0 = fdot2_f32_f16(qr[dq / 2 + 0], kr[dq / 2 + 0], s0);
                s1 = fdot2_f32_f16(qr[dq / 2 + 1], kr[dq / 2 + 1], s1);
                s2 = fdot2_f32_f16(qr[dq / 2 + 2], kr[dq / 2 + 2], s2);
                s3 = fdot2_f32_f16(qr[dq / 2 + 3], kr[dq / 2 + 3], s3);
            }
            float s = (s0 + s1) + (s2 + s3);
            int64_t n = kb + j;
            float sc = s * sm_scale_log2e;
            if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
            s_tile[row * S_STRIDE + j] = sc;
        }
        __syncthreads();

        // ---- online softmax (score already has sm_scale*log2e) ----
        float lm = -3.0e38f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * S_STRIDE + j]);
        float gm = fmaxf(row_m, lm);
        float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
        row_m = gm;
        row_l *= alpha;
        #pragma unroll
        for (int c = 0; c < CL; ++c) acc[c] *= alpha;
        float ps = 0.0f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) {
            float p = exp2f(s_tile[row * S_STRIDE + j] - row_m);
            p_tile[row * P_STRIDE + j] = __float2half(p);
            ps += p;
        }
        row_l += ps;

        // ---- PV via V_DOT2 over kv-pairs (hoist p load, unroll column loop) ----
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2 = load_h2_quad(&p_tile[row * P_STRIDE + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc[c] = fdot2_f32_f16(p2, load_h2_quad(&v_tile[d * V_STRIDE + jj]), acc[c]);
            }
        }
        __syncthreads();
    }

    if (valid) {
        float inv = 1.0f / row_l;
        int64_t base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_direct_t

// ============================================================================
// gfx10 fp16/bf16 direct attention kernel v2: 2 rows per thread for ILP.
//
// Same structure as attn_kernel_gfx10_i8_v2_t, but with fp16 V_DOT2 for QK
// and explicit bf16->fp16 conversion on Q/K stage.
// ============================================================================
template <int HD, bool ISC, int BN, typename QDT, typename ODT>
__global__ void attn_kernel_gfx10_direct_v2_t(
    const QDT* __restrict__ q, const QDT* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t q_stride_n_dir_unused,  // padding to match v1
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    float sm_scale_log2e, int tensor_layout) {
#if defined(__GFX10__)
    constexpr int BM = 64, NW = 4, NR = 2;
    constexpr int CL = HD / NW;       // 16 for D=64, 32 for D=128
    constexpr int NTHREAD = (BM / NR) * NW;  // 32 * 4 = 128

    const int tid = threadIdx.x;
    const int rp = tid >> 2;          // 0..31 (BM/NR row-pairs)
    const int ct = tid & 3;            // 0..3 column group
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    __shared__ __half q_tile[BM * HD];
    __shared__ __half k_tile[BN * HD];
    __shared__ __half v_tile[HD * BN];
    __shared__ float s_tile[BM * BN];
    __shared__ __half p_tile[BM * BN];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    // stage Q tile (BM x HD) as fp16
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * HD; i += NTHREAD) {
            int r = i / HD, cq = i % HD;
            int64_t gr = blockIdx.x * BM + r;
            q_tile[r * HD + cq] = (gr < qo_len)
                ? gfx10_to_half<QDT>(q[qb + gr * q_stride_n + cq])
                : __float2half(0.0f);
        }
    }
    __syncthreads();  // ensure all Q writes visible before pre-load to regs

    // Pre-load this thread's 2 rows of Q into registers (2 rows * HD/2 u32)
    constexpr int H2 = HD / 2;
    unsigned q_reg0[H2], q_reg1[H2];
    #pragma unroll
    for (int i = 0; i < H2; ++i) {
        q_reg0[i] = *reinterpret_cast<const unsigned*>(&q_tile[row0 * HD + i * 2]);
        q_reg1[i] = *reinterpret_cast<const unsigned*>(&q_tile[row1 * HD + i * 2]);
    }

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile as fp16
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * HD; i += NTHREAD) {
                int r = i / HD, ck = i % HD;
                int64_t n = kb + r;
                k_tile[r * HD + ck] = (n < kv_len)
                    ? gfx10_to_half<QDT>(k[kqb + n * k_stride_n + ck])
                    : __float2half(0.0f);
            }
        }
        // stage v_tile [HD][BN] from V_T
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                v_tile[d * BN + cj] = (n < kv_len)
                    ? v[vbase + d * vsd + n]
                    : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK: 2 independent fdot2 chains per j (one per row) ----
        const int out_base_col = ct * CL;
        #pragma unroll 2
        for (int j = ct; j < BN; j += NW) {
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll 8
            for (int dq = 0; dq < HD; dq += 2) {
                unsigned k_dq = load_h2_quad(&k_tile[j * HD + dq]);
                s0 = fdot2_f32_f16(q_reg0[dq / 2], k_dq, s0);
                s1 = fdot2_f32_f16(q_reg1[dq / 2], k_dq, s1);
            }
            int64_t n = kb + j;
            float sc0 = s0 * sm_scale_log2e;
            float sc1 = s1 * sm_scale_log2e;
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * BN + j] = sc0;
            s_tile[row1 * BN + j] = sc1;
        }
        __syncthreads();

        // ---- online softmax (per row) ----
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row0 * BN + j]);
            float gm = fmaxf(row_m0, lm);
            float alpha = (row_l0 > 0.0f) ? exp2f(row_m0 - gm) : 0.0f;
            row_m0 = gm; row_l0 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc0[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row0 * BN + j] - row_m0);
                p_tile[row0 * BN + j] = __float2half(p);
                ps += p;
            }
            row_l0 += ps;
        }
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row1 * BN + j]);
            float gm = fmaxf(row_m1, lm);
            float alpha = (row_l1 > 0.0f) ? exp2f(row_m1 - gm) : 0.0f;
            row_m1 = gm; row_l1 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc1[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row1 * BN + j] - row_m1);
                p_tile[row1 * BN + j] = __float2half(p);
                ps += p;
            }
            row_l1 += ps;
        }

        // ---- PV: 2 rows × CL cols, interleaved dot2 ----
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2_0 = load_h2_quad(&p_tile[row0 * BN + jj]);
            unsigned p2_1 = load_h2_quad(&p_tile[row1 * BN + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc0[c] = fdot2_f32_f16(p2_0, load_h2_quad(&v_tile[d * BN + jj]), acc0[c]);
                acc1[c] = fdot2_f32_f16(p2_1, load_h2_quad(&v_tile[d * BN + jj]), acc1[c]);
            }
        }
        __syncthreads();
    }

    if (valid0) {
        float inv = 1.0f / row_l0;
        int64_t base = b * o_stride_b + m0 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc0[c] * inv);
    }
    if (valid1) {
        float inv = 1.0f / row_l1;
        int64_t base = b * o_stride_b + m1 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc1[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_direct_v2_t

// ============================================================================
// gfx10 fp16/bf16 direct kernel v2.2: same as direct_v2 but PV reads V_T from
// global (bypasses LDS v_tile). See attn_kernel_gfx10_i8_v2_2_t for rationale.
// ============================================================================
template <int HD, bool ISC, int BN, typename QDT, typename ODT>
__global__ void attn_kernel_gfx10_direct_v2_2_t(
    const QDT* __restrict__ q, const QDT* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t q_stride_n_dir_unused,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    float sm_scale_log2e, int tensor_layout) {
#if defined(__GFX10__)
    constexpr int BM = 64, NW = 4, NR = 2;
    constexpr int CL = HD / NW;
    constexpr int NTHREAD = (BM / NR) * NW;

    const int tid = threadIdx.x;
    const int rp = tid >> 2;
    const int ct = tid & 3;
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    __shared__ __half q_tile[BM * HD];
    __shared__ __half k_tile[BN * HD];
    __shared__ float s_tile[BM * BN];
    __shared__ __half p_tile[BM * BN];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    // stage Q
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * HD; i += NTHREAD) {
            int r = i / HD, cq = i % HD;
            int64_t gr = blockIdx.x * BM + r;
            q_tile[r * HD + cq] = (gr < qo_len)
                ? gfx10_to_half<QDT>(q[qb + gr * q_stride_n + cq])
                : __float2half(0.0f);
        }
    }
    __syncthreads();

    constexpr int H2 = HD / 2;
    unsigned q_reg0[H2], q_reg1[H2];
    #pragma unroll
    for (int i = 0; i < H2; ++i) {
        q_reg0[i] = *reinterpret_cast<const unsigned*>(&q_tile[row0 * HD + i * 2]);
        q_reg1[i] = *reinterpret_cast<const unsigned*>(&q_tile[row1 * HD + i * 2]);
    }

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
    const int64_t vsd = v_stride_h;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile only (no v_tile)
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * HD; i += NTHREAD) {
                int r = i / HD, ck = i % HD;
                int64_t n = kb + r;
                k_tile[r * HD + ck] = (n < kv_len)
                    ? gfx10_to_half<QDT>(k[kqb + n * k_stride_n + ck])
                    : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK (same as direct_v2) ----
        const int out_base_col = ct * CL;
        #pragma unroll 2
        for (int j = ct; j < BN; j += NW) {
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll 8
            for (int dq = 0; dq < HD; dq += 2) {
                unsigned k_dq = load_h2_quad(&k_tile[j * HD + dq]);
                s0 = fdot2_f32_f16(q_reg0[dq / 2], k_dq, s0);
                s1 = fdot2_f32_f16(q_reg1[dq / 2], k_dq, s1);
            }
            int64_t n = kb + j;
            float sc0 = s0 * sm_scale_log2e;
            float sc1 = s1 * sm_scale_log2e;
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * BN + j] = sc0;
            s_tile[row1 * BN + j] = sc1;
        }
        __syncthreads();

        // ---- softmax (2 rows) ----
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row0 * BN + j]);
            float gm = fmaxf(row_m0, lm);
            float alpha = (row_l0 > 0.0f) ? exp2f(row_m0 - gm) : 0.0f;
            row_m0 = gm; row_l0 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc0[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row0 * BN + j] - row_m0);
                p_tile[row0 * BN + j] = __float2half(p);
                ps += p;
            }
            row_l0 += ps;
        }
        {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row1 * BN + j]);
            float gm = fmaxf(row_m1, lm);
            float alpha = (row_l1 > 0.0f) ? exp2f(row_m1 - gm) : 0.0f;
            row_m1 = gm; row_l1 *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc1[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row1 * BN + j] - row_m1);
                p_tile[row1 * BN + j] = __float2half(p);
                ps += p;
            }
            row_l1 += ps;
        }

        // ---- PV: read V_T from global (L1/L2 cache) ----
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2_0 = load_h2_quad(&p_tile[row0 * BN + jj]);
            unsigned p2_1 = load_h2_quad(&p_tile[row1 * BN + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                int64_t n = kb + jj;
                if (n < kv_len) {
                    unsigned v2 = *reinterpret_cast<const unsigned*>(v + vbase + d * vsd + n);
                    acc0[c] = fdot2_f32_f16(p2_0, v2, acc0[c]);
                    acc1[c] = fdot2_f32_f16(p2_1, v2, acc1[c]);
                }
            }
        }
        __syncthreads();
    }

    if (valid0) {
        float inv = 1.0f / row_l0;
        int64_t base = b * o_stride_b + m0 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc0[c] * inv);
    }
    if (valid1) {
        float inv = 1.0f / row_l1;
        int64_t base = b * o_stride_b + m1 * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc1[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_direct_v2_2_t

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
__device__ __forceinline__ int warp_reduce_sdot4(int partial) {
    int other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 8);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 4);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 2);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 1);
    partial += other;
    return partial;
}

// 32-lane full warp reduction (for designs where each lane holds 1 row of QUADS_PER_LANE int32 chunks
// and the whole warp cooperates on a QK reduction).
__device__ __forceinline__ int warp_reduce_full_sdot4(int partial) {
    int other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 16);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 8);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 4);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 2);
    partial += other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, partial, 1);
    partial += other;
    return partial;
}

__device__ __forceinline__ float warp_reduce_fadd(float partial) {
    int other;
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, __float_as_int(partial), 8);
    partial = __int_as_float(__float_as_int(partial) + other);
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, __float_as_int(partial), 4);
    partial = __int_as_float(__float_as_int(partial) + other);
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, __float_as_int(partial), 2);
    partial = __int_as_float(__float_as_int(partial) + other);
    other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, __float_as_int(partial), 1);
    partial = __int_as_float(__float_as_int(partial) + other);
    return partial;
}

template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_i8_v3_t(
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
    int tensor_layout) {
#if defined(__GFX10__)
    // v3: BM=64, NR=2, NW=4 — same as v2 (proven layout, BN=16 to match triton).
    constexpr int QUADS = HD / 4;
    constexpr int BM = 64;
    constexpr int NW = 4;
    constexpr int NR = 2;
    constexpr int CL = HD / NW;
    constexpr int NTHREAD = (BM / NR) * NW;
    static_assert(NR * (BM / NR) == BM, "BM must be multiple of NR");
    static_assert(NTHREAD == 128, "v3 layout expects 128 threads");

    const int tid = threadIdx.x;
    const int rp = tid >> 2;
    const int ct = tid & 3;
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    // LDS padding to eliminate bank conflicts:
    //   k_tile: +4 bytes/row breaks 2-way conflict (HD=64) / 4-way (HD=128)
    //   v_tile: +1 half/row breaks 4-way conflict (4 ct groups alias)
    //   s_tile: +1 float/row breaks 8-way conflict (row stride = 32 banks ≡ 0)
    //   p_tile: +1 half/row breaks 4-way conflict (even/odd rp alias)
    constexpr int K_STRIDE = HD + 4;   // k_tile row stride (int8 dwords)
    constexpr int V_STRIDE = BN + 1;   // v_tile column stride (halfs)
    constexpr int S_STRIDE = BN + 1;   // s_tile column stride (floats)
    constexpr int P_STRIDE = BN + 1;   // p_tile column stride (halfs)
    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * K_STRIDE];
    __shared__ __half v_tile[HD * V_STRIDE];  // [D][N] layout, padded
    __shared__ float s_tile[BM * S_STRIDE];
    __shared__ __half p_tile[BM * P_STRIDE];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    __syncthreads();

    int q_reg0[QUADS], q_reg1[QUADS];
    #pragma unroll
    for (int i = 0; i < QUADS; ++i) {
        q_reg0[i] = *reinterpret_cast<const int*>(&q_tile[row0 * HD + i * 4]);
        q_reg1[i] = *reinterpret_cast<const int*>(&q_tile[row1 * HD + i * 4]);
    }

    const float qs0 = valid0 ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m0 / MIN_BLK_Q)] : 0.0f;
    const float qs1 = valid1 ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m1 / MIN_BLK_Q)] : 0.0f;

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c]=0; acc1[c]=0; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * K_STRIDE + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * V_STRIDE + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        const int out_base_col = ct * CL;
        #pragma unroll 4
        for (int j = ct; j < BN; j += NW) {
            int s0=0, s1=0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i) {
                int k_i = load_i8_quad(&k_tile[j * K_STRIDE + i * 4]);
                s0 = sdot4_i32_i8(q_reg0[i], k_i, s0);
                s1 = sdot4_i32_i8(q_reg1[i], k_i, s1);
            }
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc0 = static_cast<float>(s0) * (qs0 * ksj);
            float sc1 = static_cast<float>(s1) * (qs1 * ksj);
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * S_STRIDE + j] = sc0;
            s_tile[row1 * S_STRIDE + j] = sc1;
        }
        __syncthreads();

        auto softmax_one = [&](float& row_m, float& row_l, float acc[], int row, bool valid) {
            if (!valid) return;
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * S_STRIDE + j]);
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            row_m = gm;
            row_l *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row * S_STRIDE + j] - row_m);
                p_tile[row * P_STRIDE + j] = __float2half(p);
                ps += p;
            }
            row_l += ps;
        };
        softmax_one(row_m0, row_l0, acc0, row0, valid0);
        softmax_one(row_m1, row_l1, acc1, row1, valid1);
        __syncthreads();

        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p0 = valid0 ? load_h2_quad(&p_tile[row0 * P_STRIDE + jj]) : 0u;
            unsigned p1 = valid1 ? load_h2_quad(&p_tile[row1 * P_STRIDE + jj]) : 0u;
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                unsigned vq = load_h2_quad(&v_tile[d * V_STRIDE + jj]);
                if (valid0) acc0[c] = fdot2_f32_f16(p0, vq, acc0[c]);
                if (valid1) acc1[c] = fdot2_f32_f16(p1, vq, acc1[c]);
            }
        }
        __syncthreads();
    }

    auto write_one = [&](float row_l, float acc[], int row, bool valid) {
        if (!valid) return;
        float inv = 1.0f / row_l;
        int64_t base = b * o_stride_b + blockIdx.x * BM * o_stride_n + row * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    };
    write_one(row_l0, acc0, row0, valid0);
    write_one(row_l1, acc1, row1, valid1);
#endif
}  // attn_kernel_gfx10_i8_v3_t

// ============================================================================
// gfx10 fp16/bf16 direct kernel v3.2: 1 warp = 2 rows, 16 lanes cooperate
// ============================================================================
template <int HD, bool ISC, int BN, typename QDT, typename ODT>
__global__ void attn_kernel_gfx10_direct_v3_t(
    // direct v3 padding marker
    const QDT* __restrict__ q, const QDT* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t q_stride_n_dir_unused,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    float sm_scale_log2e, int tensor_layout) {
#if defined(__GFX10__)
    constexpr int BM = 128;
    constexpr int NW = 1;
    constexpr int CL = HD / 16;  // D=64: 4 cols/lane, D=128: 8 cols/lane
    constexpr int NTHREAD = 128;
    constexpr int ROWS_PER_HALF = 16;
    constexpr int ROWS_PER_WARP = 32;
    constexpr int H2 = HD / 2;          // fp16 pairs per row
    constexpr int H2_PER_LANE = H2 / 16;  // D=64: 2, D=128: 4
    static_assert(H2_PER_LANE * 16 == H2, "H2 must be multiple of 16");
    static_assert(NTHREAD == 128, "v3 expects 128 threads (4 warps)");

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int half_warp = lane >> 4;
    const int lane_in_half = lane & 15;
    const int row = warp * ROWS_PER_WARP + half_warp * ROWS_PER_HALF + lane_in_half;
    const int64_t m = blockIdx.x * BM + row;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len);

    // LDS padding to break bank conflicts
    constexpr int K_STRIDE = HD + 1;  // +1 half
    constexpr int V_STRIDE = BN + 1;  // +1 half
    constexpr int S_STRIDE = BN + 1;  // +1 float
    constexpr int P_STRIDE = BN + 1;  // +1 half
    __shared__ __half q_tile[BM * HD];
    __shared__ __half k_tile[BN * K_STRIDE];
    __shared__ __half v_tile[HD * V_STRIDE];
    __shared__ float s_tile[BM * S_STRIDE];
    __shared__ __half p_tile[BM * P_STRIDE];

    // Pre-load Q into registers (lane_in_half = which D-dim chunk)
    unsigned q_reg[H2_PER_LANE];
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        const QDT* qrow = q + qb + m * q_stride_n;
        #pragma unroll
        for (int i = 0; i < H2_PER_LANE; ++i) {
            int d_chunk = lane_in_half * H2_PER_LANE + i;
            q_reg[i] = valid
                ? *reinterpret_cast<const unsigned*>(&qrow[d_chunk * 2])
                : 0u;
        }
    }

    float acc[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) acc[c] = 0.0f;
    float row_m = -3.0e38f, row_l = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // stage k_tile (BN × HD)
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            constexpr int K_CHUNKS = (BN * HD) / 2;  // half-pairs
            constexpr int CHUNKS_PER_THREAD = (K_CHUNKS + NTHREAD - 1) / NTHREAD;
            #pragma unroll 1
            for (int it = 0; it < CHUNKS_PER_THREAD; ++it) {
                int idx = tid + it * NTHREAD;
                if (idx < K_CHUNKS) {
                    int r = idx / (HD / 2);
                    int cp = idx % (HD / 2);
                    int64_t n = kb + r;
                    const QDT* krow = k + kqb + n * k_stride_n;
                    *reinterpret_cast<unsigned*>(&k_tile[r * K_STRIDE + cp * 2]) =
                        (n < kv_len)
                            ? *reinterpret_cast<const unsigned*>(&krow[cp * 2])
                            : 0u;
                }
            }
        }
        // stage v_tile [HD][BN]
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * V_STRIDE + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK: 16 lanes cooperate, fdot2 chain per chunk ----
        if (valid) {
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < H2_PER_LANE; ++i) {
                    int d_chunk = lane_in_half * H2_PER_LANE + i;
                    unsigned k_chunk = load_h2_quad(&k_tile[j * K_STRIDE + d_chunk * 2]);
                    partial = fdot2_f32_f16(q_reg[i], k_chunk, partial);
                }
                partial = warp_reduce_fadd(partial);
                float sc = partial * sm_scale_log2e;
                int64_t n = kb + j;
                if (ISC && n > m) sc = -3.0e38f;
                if (n >= kv_len) sc = -3.0e38f;
                if (lane_in_half == 0) {
                    s_tile[row * S_STRIDE + j] = sc;
                }
            }
        }
        __syncthreads();

        // ---- softmax: all 16 lanes of half-warp (each has own row_m/row_l) ----
        if (valid) {
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * S_STRIDE + j]);
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            row_m = gm; row_l *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row * S_STRIDE + j] - row_m);
                p_tile[row * P_STRIDE + j] = __float2half(p);
                ps += p;
            }
            row_l += ps;
        }
        __syncthreads();

        // ---- PV: 16 lanes cooperate, each lane has its own row's acc and p_tile ----
        if (valid) {
            const int col_base = lane_in_half * CL;
            #pragma unroll 2
            for (int jj = 0; jj < BN; jj += 2) {
                unsigned p2 = load_h2_quad(&p_tile[row * P_STRIDE + jj]);
                #pragma unroll
                for (int c_local = 0; c_local < CL; ++c_local) {
                    int c = col_base + c_local;
                    acc[c_local] = fdot2_f32_f16(p2, load_h2_quad(&v_tile[c * V_STRIDE + jj]), acc[c_local]);
                }
            }
        }
        __syncthreads();
    }

    // ---- write back: 16 lanes of half-warp collectively write 1 row × 64 cols ----
    if (valid) {
        const float inv = 1.0f / row_l;
        // owned row = (warp*2 + half_warp) * ROWS_PER_HALF (the first row of this half-warp)
        const int64_t out_base = b * o_stride_b +
            ((warp * 2 + half_warp) * ROWS_PER_HALF) * o_stride_n + h * o_stride_h;
        const int col_base = lane_in_half * CL;
        #pragma unroll
        for (int c_local = 0; c_local < CL; ++c_local) {
            int c = col_base + c_local;
            out[out_base + c] = gfx10_out_convert<ODT>(acc[c_local] * inv);
        }
    }
#endif
}  // attn_kernel_gfx10_direct_v3_t


// ============================================================================
// gfx10 int8 attention kernel v4: BM=128, 2 syncs/tile (from 4).
//
// Optimization over v3:
//   1. BM=128 (doubles Q rows per block → halves grid blocks → 2x fewer syncs)
//   2. Remove 2 unnecessary __syncthreads between QK→softmax and softmax→PV:
//      each thread reads only its own rows' s_tile/p_tile entries, so no
//      cross-thread data sharing exists in those phases.
//   Combined: 4x reduction in total sync overhead vs v3 (halved blocks × 2x
//   fewer syncs per block).
//
// Template args: HD (head dim), ISC (causal), BN (kv tile width), BM (Q rows).
// ============================================================================
template <int HD, bool ISC, int BN, int BM, typename ODT>
__global__ void attn_kernel_gfx10_i8_v4_t(
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
    int tensor_layout) {
#if defined(__GFX10__)
    constexpr int QUADS = HD / 4;
    constexpr int NW = 4;
    constexpr int NR = 2;
    constexpr int CL = HD / NW;
    constexpr int NTHREAD = (BM / NR) * NW;
    static_assert(NR * (BM / NR) == BM, "BM must be multiple of NR");
    static_assert(NTHREAD == (BM / NR) * 4, "v4 expects NW=4");

    const int tid = threadIdx.x;
    const int rp = tid >> 2;
    const int ct = tid & 3;
    const int row0 = rp * 2 + 0;
    const int row1 = rp * 2 + 1;
    const int64_t m0 = blockIdx.x * BM + row0;
    const int64_t m1 = blockIdx.x * BM + row1;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);

    // LDS padding to eliminate bank conflicts (same as v3)
    constexpr int K_STRIDE = HD + 4;
    constexpr int V_STRIDE = BN + 1;
    constexpr int S_STRIDE = BN + 1;
    constexpr int P_STRIDE = BN + 1;
    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * K_STRIDE];
    __shared__ __half v_tile[HD * V_STRIDE];
    __shared__ float s_tile[BM * S_STRIDE];
    __shared__ __half p_tile[BM * P_STRIDE];

    const bool valid0 = (m0 < qo_len);
    const bool valid1 = (m1 < qo_len);

    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            int r = i / QUADS, cq = i % QUADS;
            int64_t gr = blockIdx.x * BM + r;
            const int8_t* qrow = q + qb + gr * q_stride_n;
            reinterpret_cast<int*>(&q_tile[r * HD + cq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(qrow + cq * 4) : 0;
        }
    }
    __syncthreads();

    int q_reg0[QUADS], q_reg1[QUADS];
    #pragma unroll
    for (int i = 0; i < QUADS; ++i) {
        q_reg0[i] = *reinterpret_cast<const int*>(&q_tile[row0 * HD + i * 4]);
        q_reg1[i] = *reinterpret_cast<const int*>(&q_tile[row1 * HD + i * 4]);
    }

    const float qs0 = valid0 ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m0 / MIN_BLK_Q)] : 0.0f;
    const float qs1 = valid1 ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m1 / MIN_BLK_Q)] : 0.0f;

    float acc0[CL], acc1[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) { acc0[c]=0; acc1[c]=0; }
    float row_m0 = -3.0e38f, row_l0 = 0.0f;
    float row_m1 = -3.0e38f, row_l1 = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // --- Load K, V tiles into LDS ---
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                reinterpret_cast<int*>(&k_tile[r * K_STRIDE + ck * 4])[0] =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        {
            const int64_t vbase = b * v_stride_b + kvh * v_stride_n;
            const int64_t vsd = v_stride_h;
            #pragma unroll 1
            for (int i = tid; i < HD * BN; i += NTHREAD) {
                int d = i / BN, cj = i % BN;
                int64_t n = kb + cj;
                const __half* vp = v + vbase + d * vsd + n;
                v_tile[d * V_STRIDE + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();  // sync #1: K/V visible for QK

        // --- QK scores: write s_tile (no sync needed before softmax,
        //     since each thread reads only its own rows) ---
        const int out_base_col = ct * CL;
        #pragma unroll 4
        for (int j = ct; j < BN; j += NW) {
            int s0=0, s1=0;
            #pragma unroll 8
            for (int i = 0; i < QUADS; ++i) {
                int k_i = load_i8_quad(&k_tile[j * K_STRIDE + i * 4]);
                s0 = sdot4_i32_i8(q_reg0[i], k_i, s0);
                s1 = sdot4_i32_i8(q_reg1[i], k_i, s1);
            }
            int64_t n = kb + j;
            const int kscale_idx = static_cast<int>((kb + j) / MIN_BLK_K);
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sc0 = static_cast<float>(s0) * (qs0 * ksj);
            float sc1 = static_cast<float>(s1) * (qs1 * ksj);
            if ((!valid0) || (ISC && n > m0) || (n >= kv_len)) sc0 = -3.0e38f;
            if ((!valid1) || (ISC && n > m1) || (n >= kv_len)) sc1 = -3.0e38f;
            s_tile[row0 * S_STRIDE + j] = sc0;
            s_tile[row1 * S_STRIDE + j] = sc1;
        }
        // NO sync here — each thread reads its own rows' s_tile

        // --- Softmax: reads s_tile, writes p_tile (same rows, no cross-thread) ---
        auto softmax_one = [&](float& row_m, float& row_l, float acc[], int row, bool valid) {
            if (!valid) return;
            float lm = -3.0e38f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * S_STRIDE + j]);
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            row_m = gm;
            row_l *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s_tile[row * S_STRIDE + j] - row_m);
                p_tile[row * P_STRIDE + j] = __float2half(p);
                ps += p;
            }
            row_l += ps;
        };
        softmax_one(row_m0, row_l0, acc0, row0, valid0);
        softmax_one(row_m1, row_l1, acc1, row1, valid1);
        // NO sync here — each thread reads only its own rows' p_tile

        // --- PV: read p_tile + v_tile, accumulate ---
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p0 = valid0 ? load_h2_quad(&p_tile[row0 * P_STRIDE + jj]) : 0u;
            unsigned p1 = valid1 ? load_h2_quad(&p_tile[row1 * P_STRIDE + jj]) : 0u;
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                unsigned vq = load_h2_quad(&v_tile[d * V_STRIDE + jj]);
                if (valid0) acc0[c] = fdot2_f32_f16(p0, vq, acc0[c]);
                if (valid1) acc1[c] = fdot2_f32_f16(p1, vq, acc1[c]);
            }
        }
        __syncthreads();  // sync #2: PV done, safe to overwrite K/V for next tile
    }

    auto write_one = [&](float row_l, float acc[], int row, bool valid) {
        if (!valid) return;
        float inv = 1.0f / row_l;
        int64_t base = b * o_stride_b + blockIdx.x * BM * o_stride_n + row * o_stride_n + h * o_stride_h;
        const int out_base_col = ct * CL;
        #pragma unroll
        for (int c = 0; c < CL; ++c)
            out[base + out_base_col + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    };
    write_one(row_l0, acc0, row0, valid0);
    write_one(row_l1, acc1, row1, valid1);
#endif
}  // attn_kernel_gfx10_i8_v4_t


// ============================================================================
// gfx10 int8 attention kernel v7p: Triton-style 2-stage software pipelined.
//
// Mirrors the Triton int8 kernel that hits the gfx1035 hardware wall on the
// SDXL D=64 benchmarks (~12ms vs v2's 128ms). The three structural wins over
// v2:
//   1. BM = NTHREAD Q rows per block with ONE row per thread. Q, the BN scores
//      and the P vector live entirely in registers, so softmax needs NO LDS
//      and NO cross-thread reduction (each thread owns its row fully).
//   2. K (BN x HD i8) and V (HD x BN f16) tiles are DOUBLE-BUFFERED in LDS.
//      Global loads for tile t+1 are issued at the top of iteration t and the
//      resulting register data is written to LDS after computing tile t, so
//      global latency is hidden behind QK/PV (Triton num_stages=2 scheme).
//      Exactly ONE __syncthreads per tile (Triton emits one s_barrier too).
//   3. All K/V LDS reads are per-warp broadcasts (every thread reads the same
//      tile elements), so bank conflicts do not arise even without padding;
//      16B vector reads (ds_read_b128) keep the LDS instruction count down.
//
// Softmax uses v2's -3.0e38 sentinel scheme (valid max always dominates, so
// the early all-masked causal tiles reduce to exp(-inf)=0 after the first
// valid tile; see v2 notes).
//
// Template args: HD, ISC (causal), BN (kv tile width, use 16), BM (Q rows per
// block == #threads). D=64 -> BM=128; D=128 -> BM=64 (register-bound).
// ============================================================================
template <int HD, bool ISC, int BN, int BM, typename ODT>
__global__ __attribute__((amdgpu_num_vgpr(128))) __launch_bounds__(BM, 2) void attn_kernel_gfx10_i8_v7p_t(
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
    int tensor_layout, int diag) {
#if defined(__GFX10__)
    const int diag_qk = diag & 1;   // skip QK compute (scr := 0)
    const int diag_sm = (diag >> 1) & 1;   // skip softmax (no max/exp)
    const int diag_pv = (diag >> 2) & 1;   // skip PV (no dots/acc update)
    const int diag_st = (diag >> 3) & 1;   // skip staging (prefetch + LDS store)
    const int diag_wb = (diag >> 4) & 1;   // skip writeback (epilogue)
    constexpr int QUADS = HD / 4;             // i8 dwords per row
    constexpr int NTHREAD = BM;               // one Q row per thread
    constexpr int K_STRIDE = HD;              // rows are 64B/128B -> 16B aligned
    constexpr int V_STRIDE = BN;              // 16 halves = 32B rows
    constexpr int KS_TOTAL = BN * K_STRIDE;   // K tile bytes per buffer
    constexpr int VS_TOTAL = HD * V_STRIDE;   // V tile halves per buffer
    static_assert(KS_TOTAL * 2 + VS_TOTAL * 2 * (int)sizeof(__half) <= 49152,
                  "v7p LDS double-buffer exceeds 48KiB budget");

    // 16B-aligned LDS buffers (ds_read_b128 / ds_write_b128).
    __shared__ __attribute__((aligned(32))) int8_t k_buf[2][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half   v_buf[2][VS_TOTAL];

    const int tid = threadIdx.x;              // 0..NTHREAD-1 == Q row in block
    const int64_t m = blockIdx.x * BM + tid;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len);
    (void)q_stride_n_dir_unused;
    (void)tensor_layout;

    // ---- Q row lives in registers for the whole block ----
    int q_reg[QUADS];
    if (valid) {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        const int8_t* qrow = q + qb + m * q_stride_n;
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq)
            q_reg[dq] = load_i8_quad(qrow + dq * 4);
    } else {
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
    }
    const float qs = valid
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)]
        : 0.0f;

    float acc[HD];
    #pragma unroll
    for (int c = 0; c < HD; ++c) acc[c] = 0.0f;
    float row_m = -3.0e38f, row_l = 0.0f;

    const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
    const int64_t vbase = b * v_stride_b + kvh * v_stride_n;

    // ---- helper: stage K tile (BN*QUADS i32) from global V into LDS buffer ----
    //   note V_T is [B,H,D,N] (n-contiguous), matching the [d][j] LDS layout.
    // ---- stage one K/V tile into LDS buffer 'dst' (global offset kb) ----
    auto stage_to_lds = [&](int dst, int64_t kb0) {
        const int64_t vsd = v_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BN * QUADS; i += NTHREAD) {
            int r = i / QUADS, ck = i % QUADS;
            int64_t n = kb0 + r;
            reinterpret_cast<int*>(&k_buf[dst][r * K_STRIDE + ck * 4])[0] =
                (n < kv_len) ? load_i8_quad(k + kqb + n * k_stride_n + ck * 4) : 0;
        }
        #pragma unroll 1
        for (int i = tid; i < HD * BN; i += NTHREAD) {
            int d = i / BN, cj = i % BN;
            int64_t n = kb0 + cj;
            v_buf[dst][d * V_STRIDE + cj] =
                (n < kv_len) ? *(v + vbase + d * vsd + n) : __float2half(0.0f);
        }
    };

    stage_to_lds(0, 0);
    __syncthreads();

    // Per-thread register staging volumes for tile t+1. Trip counts are uniform
    // across threads and compile-time constant (BN*QUADS % NTHREAD == 0 and
    // (HD*BN/2) % NTHREAD == 0 for HD in {64,128}, BN=16), so the staging loops
    // below fully unroll and k_stg/v_stg stay in registers.
    constexpr int K_ELEMS = BN * QUADS;             // u32 (4 i8) per K tile
    constexpr int V_U32 = (HD * BN) / 2;            // u32 (2 halves) per V tile
    constexpr int NK_STG = K_ELEMS / NTHREAD;       // K u32 this thread stages
    constexpr int NV_STG4 = V_U32 / NTHREAD;        // V u32 this thread stages

    #pragma unroll 1
    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        const int buf = static_cast<int>((kb / BN) & 1);
        const int64_t nb = kb + BN;

        // ---- prefetch tile t+1 global -> registers (latency hidden under compute t) ----
        unsigned k_stg[NK_STG];
        unsigned v_stg[NV_STG4];
        if ((nb < kv_len) && !diag_st) {
            #pragma unroll
            for (int u = 0; u < NK_STG; ++u) {
                const int i = tid + u * NTHREAD;    // element index in [0, K_ELEMS)
                const int r = i / QUADS, ck = i % QUADS;
                const int64_t n = nb + r;
                k_stg[u] = (n < kv_len)
                    ? static_cast<unsigned>(load_i8_quad(k + kqb + n * k_stride_n + ck * 4)) : 0u;
            }
            #pragma unroll
            for (int u = 0; u < NV_STG4; ++u) {
                const int i = tid + u * NTHREAD;    // u32 index in [0, V_U32)
                const int d = (2 * i) / BN, cj = (2 * i) % BN;   // cj even (BN even)
                const int64_t n = nb + cj;
                const __half h0 = (n < kv_len)
                    ? *(v + vbase + d * v_stride_h + n) : __float2half(0.0f);
                const __half h1 = ((n + 1) < kv_len)
                    ? *(v + vbase + d * v_stride_h + n + 1) : __float2half(0.0f);
                v_stg[u] = static_cast<unsigned>(__half_as_ushort(h0))
                    | (static_cast<unsigned>(__half_as_ushort(h1)) << 16);
            }
        }

        // ---- compute tile t from LDS buffer buf ----
        // QK: 4 independent dot4 chains (j, j+1, j+2, j+3), 16B K reads.
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
                    int64_t n = kb + j0 + tj;
                    const int kscale_idx = static_cast<int>((kb + j0 + tj) / MIN_BLK_K);
                    const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
                    float sc = static_cast<float>(s) * (qs * ksj);
                    if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
                    scr[j0 + tj] = sc;
                }
            }
        }

        // ---- softmax entirely in registers (this thread's row only) ----
        {
            if (diag_sm) row_l = 1.0f;
            if (!diag_sm) {
                float lm = scr[0];
                #pragma unroll
                for (int j = 1; j < BN; ++j) lm = fmaxf(lm, scr[j]);
                float gm = fmaxf(row_m, lm);
                float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
                row_m = gm;
                row_l *= alpha;
                #pragma unroll
                for (int c = 0; c < HD; ++c) acc[c] *= alpha;
                float ps = 0.0f;
                #pragma unroll
                for (int j = 0; j < BN; ++j) {
                    float p = exp2f(scr[j] - row_m);
                    scr[j] = p;
                    ps += p;
                }
                row_l += ps;
            }
        }

        // ---- PV: 16B V reads, 4 dot2 per column per b128 ----
        {
            if (!diag_pv) {
                unsigned p2[BN / 2];
            #pragma unroll
            for (int j = 0; j < BN; j += 2)
                p2[j / 2] = __half_as_ushort(__float2half(scr[j])) |
                            (static_cast<unsigned>(__half_as_ushort(__float2half(scr[j + 1]))) << 16);
            #pragma unroll 4
            for (int c0 = 0; c0 < HD; c0 += 2) {
                #pragma unroll
                for (int jj = 0; jj < BN; jj += 8) {
                    const int4 va = *reinterpret_cast<const int4*>(&v_buf[buf][(c0 + 0) * V_STRIDE + jj]);
                    const int4 vb = *reinterpret_cast<const int4*>(&v_buf[buf][(c0 + 1) * V_STRIDE + jj]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 0], va.x, acc[c0 + 0]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 1], va.y, acc[c0 + 0]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 2], va.z, acc[c0 + 0]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 3], va.w, acc[c0 + 0]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 0], vb.x, acc[c0 + 1]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 1], vb.y, acc[c0 + 1]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 2], vb.z, acc[c0 + 1]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 3], vb.w, acc[c0 + 1]);
                }
            }
            }
        }

        // ---- store staged tile t+1 regs -> LDS buffer buf^1 (same unrolled mapping) ----
        if ((nb < kv_len) && !diag_st) {
            const int ob = buf ^ 1;
            #pragma unroll
            for (int u = 0; u < NK_STG; ++u) {
                const int i = tid + u * NTHREAD;    // element index in [0, K_ELEMS)
                const int r = i / QUADS, ck = i % QUADS;
                reinterpret_cast<int*>(&k_buf[ob][r * K_STRIDE + ck * 4])[0] = static_cast<int>(k_stg[u]);
            }
            #pragma unroll
            for (int u = 0; u < NV_STG4; ++u) {
                const int i = tid + u * NTHREAD;    // u32 index in [0, V_U32)
                const int d = (2 * i) / BN, cj = (2 * i) % BN;   // cj even -> 4B aligned
                *reinterpret_cast<unsigned*>(&v_buf[ob][d * V_STRIDE + cj]) = v_stg[u];
            }
        }
        __syncthreads();
    }

    // ---- normalize + write back (single row) ----
    if (valid && !diag_wb) {
        float inv = 1.0f / row_l;
        int64_t base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
        #pragma unroll
        for (int c = 0; c < HD; ++c)
            out[base + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_v7p_t


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


// ============================================================================
// v8r-v3: row-per-thread PV (HD=64, BN=16, BM rows/block, NTHREAD=BM).
//
// The fragment mappings (v1 4-key/b64, v2/v2.1 8-key/b128 with 2 key-slices)
// were all CORRECT but their PV section never went below ~69ms @ s=4096, while
// v8's identical-count b128 PV costs only ~11ms. Diffing the PV loops revealed
// no instruction difference, so the fragment mapping itself (2 threads per row,
// shfl reduction, doubled V LDS reads) is the serialization source.
// v3 goes back to v8's structure: each thread OWNS one full row (all 16 keys),
// so softmax is purely local (no butterfly), writeback is purely local, and the
// PV is v8's verbatim loop (p2[8], c0+=2 unroll4, jj in {0,8}, paired int4).
//   row = tid; BM=128, grid.x=(qo_len+127)/128 (matches v8's block count).
// QK stays zero-redundant: 16 keys x 16 sdot4 per thread = 256 sdot4/tile.
// ============================================================================

template <int HD, bool ISC, int BN, int BM, typename ODT>
__global__ __attribute__((amdgpu_num_vgpr(224))) __launch_bounds__(BM, 1) void attn_kernel_gfx10_i8_v8r_t(
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
    int tensor_layout, int diag, int v_native) {
#if defined(__GFX10__)
    static_assert(HD == 64 && BN == 16, "v8r requires HD=64, BN=16");
    constexpr int QUADS = HD / 4;          // 16 int8 dwords per row
    constexpr int NTHREAD = BM;            // 128 for BM=128
    constexpr int K_STRIDE = HD;           // 64 int8
    constexpr int V_STRIDE = BN;           // 16 half
    constexpr int KS_TOTAL = BN * K_STRIDE;
    constexpr int VS_TOTAL = HD * V_STRIDE;
    static_assert(KS_TOTAL * 2 + VS_TOTAL * 2 * (int)sizeof(__half) + BM * HD <= 49152,
                  "v8r LDS budget");

    __shared__ __attribute__((aligned(32))) int8_t k_buf[2][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half   v_buf[2][VS_TOTAL];
    __shared__ __attribute__((aligned(32))) int8_t   q_buf[BM * HD];

    const int tid = threadIdx.x;
    const int row = tid;                        // row within block (0..BM-1)
    const int64_t m = blockIdx.x * BM + row;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;            // HG=1
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len) && (h < q_heads);

    const float qsv = valid
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)] : 0.0f;

    const int diag_qk = diag & 1;
    const int diag_sm = (diag >> 1) & 1;
    const int diag_pv = (diag >> 2) & 1;
    const int diag_st = (diag >> 3) & 1;
    const int diag_wb = (diag >> 4) & 1;
    (void)q_stride_n_dir_unused; (void)tensor_layout; (void)v_native;

    // ---- stage Q tile once (BM x HD int8) ----
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * QUADS; i += NTHREAD) {
            const int r = i / QUADS, dq = i % QUADS;
            const int64_t gr = blockIdx.x * BM + r;
            reinterpret_cast<int*>(&q_buf[r * HD + dq * 4])[0] =
                (gr < qo_len) ? load_i8_quad(q + qb + gr * q_stride_n + dq * 4) : 0;
        }
    }

    // full Q row resident (lane reads its own row once)
    int q_reg[QUADS];
    __syncthreads();   // q_buf ready
    {
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq)
            q_reg[dq] = reinterpret_cast<const int*>(&q_buf[row * HD + dq * 4])[0];
    }

    float acc[HD];
    float row_l = 0.0f, row_m = -3.0e38f;
    #pragma unroll
    for (int c = 0; c < HD; ++c) acc[c] = 0.0f;

    // ---- double-buffered K/V stage (direct stage of the other buffer, like v8) ----
    auto stage_kv = [&](int dst, int64_t kb0) {
        #pragma unroll 1
        for (int i = tid; i < BN * QUADS; i += NTHREAD) {
            const int r = i / QUADS, ck = i % QUADS;
            const int64_t n = kb0 + r;
            reinterpret_cast<int*>(&k_buf[dst][r * K_STRIDE + ck * 4])[0] =
                (n < kv_len) ? load_i8_quad(k + b * k_stride_b + kvh * k_stride_h + n * k_stride_n + ck * 4) : 0;
        }
        #pragma unroll 1
        for (int i = tid; i < HD * BN / 8; i += NTHREAD) {
            if (i < HD * BN / 8) {
                const int n_local = i / (HD / 8);
                const int d8 = i % (HD / 8);
                const int64_t n = kb0 + n_local;
                if (n < kv_len) {
                    const __half* src = v + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + d8 * 8;
                    int4 val = *reinterpret_cast<const int4*>(src);
                    #pragma unroll
                    for (int jj = 0; jj < 8; ++jj)
                        v_buf[dst][(d8 * 8 + jj) * V_STRIDE + n_local] = reinterpret_cast<__half*>(&val)[jj];
                }
            }
        }
    };

    stage_kv(0, 0);
    __syncthreads();

    #pragma unroll 1
    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        const int buf = static_cast<int>((kb / BN) & 1);
        const int64_t nb = kb + BN;

        if ((nb < kv_len) && !diag_st) {
            stage_kv(buf ^ 1, nb);
        }

        // ---- QK: this lane's full 16-key row ----
        float s[16];
        #pragma unroll
        for (int j = 0; j < BN; ++j) {
            int sc = 0;
            if (!diag_qk)
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) {
                const int kk = reinterpret_cast<const int*>(&k_buf[buf][j * K_STRIDE + dq * 4])[0];
                sc = sdot4_i32_i8(q_reg[dq], kk, sc);
            }
            const int kn = static_cast<int>(kb + j);
            const int kscale_idx = kn / MIN_BLK_K;
            const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + kscale_idx];
            float sv = static_cast<float>(sc) * (qsv * ksj);
            if (!valid || (ISC && (kn > m)) || (kn >= kv_len)) sv = -3.0e38f;
            s[j] = sv;
        }

        // ---- softmax: fully local over this row's 16 keys ----
        if (diag_sm) {
            row_l = 1.0f;
        } else {
            float lm = s[0];
            #pragma unroll
            for (int j = 1; j < BN; ++j) lm = fmaxf(lm, s[j]);
            const float gm = fmaxf(row_m, lm);
            const float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            row_m = gm;
            row_l *= alpha;
            if (!diag_pv) {
                #pragma unroll
                for (int c = 0; c < HD; ++c) acc[c] *= alpha;
            }
            float ps = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float p = exp2f(s[j] - row_m);
                s[j] = p;
                ps += p;
            }
            row_l += ps;
        }

        // ---- PV: v8's verbatim loop (full 16-key row, paired adjacent cols) ----
        if (!diag_pv) {
            unsigned p2[BN / 2];
            #pragma unroll
            for (int j = 0; j < BN; j += 2)
                p2[j / 2] = __half_as_ushort(__float2half(s[j])) |
                            (static_cast<unsigned>(__half_as_ushort(__float2half(s[j + 1]))) << 16);
            #pragma unroll 4
            for (int c0 = 0; c0 < HD; c0 += 2) {
                #pragma unroll
                for (int jj = 0; jj < BN; jj += 8) {
                    const int4 va = *reinterpret_cast<const int4*>(&v_buf[buf][(c0 + 0) * V_STRIDE + jj]);
                    const int4 vb = *reinterpret_cast<const int4*>(&v_buf[buf][(c0 + 1) * V_STRIDE + jj]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 0], static_cast<unsigned>(va.x), acc[c0 + 0]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 1], static_cast<unsigned>(va.y), acc[c0 + 0]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 2], static_cast<unsigned>(va.z), acc[c0 + 0]);
                    acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 3], static_cast<unsigned>(va.w), acc[c0 + 0]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 0], static_cast<unsigned>(vb.x), acc[c0 + 1]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 1], static_cast<unsigned>(vb.y), acc[c0 + 1]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 2], static_cast<unsigned>(vb.z), acc[c0 + 1]);
                    acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 3], static_cast<unsigned>(vb.w), acc[c0 + 1]);
                }
            }
        }

        __syncthreads();
    }

    // ---- writeback: fully local ----
    if (valid && !diag_wb) {
        const float inv = 1.0f / row_l;
        const int64_t base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
        #pragma unroll
        for (int c = 0; c < HD; ++c)
            out[base + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i8_v8r_t


template <int HD, bool ISC, int BN, int BM, typename ODT, bool IPV = false>
__global__ __attribute__((amdgpu_num_vgpr(160))) __launch_bounds__(BM, 2) void attn_kernel_gfx10_i9_t(
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
    constexpr int QUADS = HD / 4;       // int8 dwords per row
    constexpr int NTHREAD = BM;         // 1 row per lane
    constexpr int K_STRIDE = HD;
    constexpr int V_STRIDE = BN;
    constexpr int KS_TOTAL = BN * K_STRIDE;
    constexpr int VS_TOTAL = HD * V_STRIDE;   // [HD][BN] transposed V tile
    constexpr int VDSW = IPV ? 16 : 8;        // int8s per staging slot
    static_assert(KS_TOTAL * 2 + VS_TOTAL * 2 * (int)sizeof(std::conditional_t<IPV, int8_t, __half>) <= 49152,
                  "v9 LDS double-buffer budget");

    __shared__ __attribute__((aligned(32))) int8_t k_buf[2][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) std::conditional_t<IPV, int8_t, __half> v_buf[2][VS_TOTAL];
    const int8_t* v8p = reinterpret_cast<const int8_t*>(v);

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

    // ---- Q row resident in registers ----
    const float qsv = valid
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)] : 0.0f;
    int q_reg[QUADS];
    if (valid) {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        const int8_t* qrow = q + qb + m * q_stride_n;
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = load_i8_quad(qrow + dq * 4);
    } else {
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
    }

    float acc[HD];
    float row_m = -3.0e38f, row_l = 0.0f;
    #pragma unroll
    for (int c = 0; c < HD; ++c) acc[c] = 0.0f;

    // ---- stage one K+V tile (global -> LDS) ----
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
                const int n_local = slot / (HD / VDSW);
                const int dg = slot % (HD / VDSW);
                const int64_t n = kb0 + n_local;
                if (n < kv_len && (dg * VDSW) < HD) {
                    if (IPV) {
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
                }
            }
        }
    };

    // prologue: stage tile 0
    stage_kv(0, 0);
    __syncthreads();

    #pragma unroll 1
    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        const int buf = static_cast<int>((kb / BN) & 1);
        const int64_t nb = kb + BN;

        // issue next tile's global loads (overlaps current tile compute)
        if ((nb < kv_len) && !diag_st) {
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
                    if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
                    scr[j0 + tj] = sc;
                }
            }
        }

        // ---- softmax (row-local) ----
        if (!diag_sm) {
            float lm = scr[0];
            #pragma unroll
            for (int j = 1; j < BN; ++j) lm = fmaxf(lm, scr[j]);
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            row_m = gm;
            row_l *= alpha;
            #pragma unroll
            for (int c = 0; c < HD; ++c) acc[c] *= alpha;
            float ps = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) { float p = exp2f(scr[j] - row_m); scr[j] = p; ps += p; }
            row_l += ps;
        }

        // ---- PV ----
        if (!diag_pv) {
            if (IPV) {
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
                    int ai = 0;
                    #pragma unroll
                    for (int kk = 0; kk < BN / 16; ++kk) {
                        const int4 vi = *reinterpret_cast<const int4*>(&v_buf[buf][c0 * V_STRIDE + kk * 16]);
                        ai = sdot4_i32_i8(static_cast<int>(p8[kk * 4 + 0]), vi.x, ai);
                        ai = sdot4_i32_i8(static_cast<int>(p8[kk * 4 + 1]), vi.y, ai);
                        ai = sdot4_i32_i8(static_cast<int>(p8[kk * 4 + 2]), vi.z, ai);
                        ai = sdot4_i32_i8(static_cast<int>(p8[kk * 4 + 3]), vi.w, ai);
                    }
                    acc[c0] += static_cast<float>(ai) * vs;
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
                        const int4 va = *reinterpret_cast<const int4*>(&v_buf[buf][(c0 + 0) * V_STRIDE + jj]);
                        const int4 vb = *reinterpret_cast<const int4*>(&v_buf[buf][(c0 + 1) * V_STRIDE + jj]);
                        acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 0], va.x, acc[c0 + 0]);
                        acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 1], va.y, acc[c0 + 0]);
                        acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 2], va.z, acc[c0 + 0]);
                        acc[c0 + 0] = fdot2_f32_f16(p2[(jj >> 1) + 3], va.w, acc[c0 + 0]);
                        acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 0], vb.x, acc[c0 + 1]);
                        acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 1], vb.y, acc[c0 + 1]);
                        acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 2], vb.z, acc[c0 + 1]);
                        acc[c0 + 1] = fdot2_f32_f16(p2[(jj >> 1) + 3], vb.w, acc[c0 + 1]);
                    }
                }
            }
        }
        // All threads must finish staging the next tile (and reading the current one)
        // before the double-buffer flips; otherwise we race the LDS write/read.
        __syncthreads();
    }

    // ---- writeback ----
    if (valid && !diag_wb) {
        const float inv = 1.0f / row_l;
        const int64_t base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
        #pragma unroll
        for (int c = 0; c < HD; ++c)
            out[base + c] = gfx10_out_convert<ODT>(acc[c] * inv);
    }
#endif
}  // attn_kernel_gfx10_i9_t

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
