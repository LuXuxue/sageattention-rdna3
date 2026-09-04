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


}  // namespace sageattn_gfx10
