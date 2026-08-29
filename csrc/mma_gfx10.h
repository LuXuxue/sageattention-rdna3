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
    __shared__ int8_t q_tile[BM * HD];
    __shared__ int8_t k_tile[BN * HD];
    __shared__ __half v_tile[HD * BN];      // V_T tile [D][N], row=d, col=n
    __shared__ float s_tile[BM * BN];       // raw QK scores
    __shared__ __half p_tile[BM * BN];      // softmax probs

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
                reinterpret_cast<int*>(&k_tile[r * HD + ck * 4])[0] =
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
                v_tile[d * BN + cj] = (n < kv_len) ? *vp : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK: thread computes its j-slice of the row's BN scores ----
        const float ks = k_scale[b * ks_stride_b + kvh * ks_stride_h + static_cast<int>(kb / MIN_BLK_K)];
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
                    load_i8_quad(&k_tile[j * HD + i * 4]), s);
            int64_t n = kb + j;
            float sc = static_cast<float>(s) * (qs * ks);
            if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
            s_tile[row * BN + j] = sc;
        }
        __syncthreads();

        // ---- online softmax (per-row, read from LDS s_tile) ----
        float lm = -3.0e38f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * BN + j]);
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
            float p = exp2f(s_tile[row * BN + j] - row_m);
            p_tile[row * BN + j] = __float2half(p);
            ps += p;
        }
        row_l += ps;

        // ---- PV: for each kv-pair, accumulate into all CL output columns ----
        // Hoist the p_pair load (reused across all columns) and fully unroll the
        // column loop so the CL independent acc chains run in parallel (ILP).
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2 = load_h2_quad(&p_tile[row * BN + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc[c] = fdot2_f32_f16(p2, load_h2_quad(&v_tile[d * BN + jj]), acc[c]);
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

    __shared__ __half q_tile[BM * HD];
    __shared__ __half k_tile[BN * HD];
    __shared__ __half v_tile[HD * BN];
    __shared__ float s_tile[BM * BN];
    __shared__ __half p_tile[BM * BN];

    const bool valid = (m < qo_len);

    // stage Q tile (BM x HD) as fp16
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        #pragma unroll 1
        for (int i = tid; i < BM * HD; i += NTHREAD) {
            int r = i / HD, cq = i % HD;
            int64_t gr = blockIdx.x * BM + r;
            q_tile[r * HD + cq] = (gr < qo_len) ? gfx10_to_half<QDT>(q[qb + gr * q_stride_n + cq]) : __float2half(0.0f);
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
                k_tile[r * HD + ck] = (n < kv_len) ? gfx10_to_half<QDT>(k[kqb + n * k_stride_n + ck]) : __float2half(0.0f);
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
                v_tile[d * BN + cj] = (n < kv_len) ? v[vbase + d * vsd + n] : __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- QK via fp16 V_DOT2 over d-pairs ----
        const int out_base_col = ct * CL;
        #pragma unroll 1
        for (int j = ct; j < BN; j += NW) {
            float s = 0.0f;
            #pragma unroll 1
            for (int dq = 0; dq < HD; dq += 2)
                s = fdot2_f32_f16(
                    load_h2_quad(&q_tile[row * HD + dq]),
                    load_h2_quad(&k_tile[j * HD + dq]), s);
            int64_t n = kb + j;
            float sc = s * sm_scale_log2e;
            if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
            s_tile[row * BN + j] = sc;
        }
        __syncthreads();

        // ---- online softmax (score already has sm_scale*log2e) ----
        float lm = -3.0e38f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * BN + j]);
        float gm = fmaxf(row_m, lm);
        float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
        row_m = gm;
        row_l *= alpha;
        #pragma unroll
        for (int c = 0; c < CL; ++c) acc[c] *= alpha;
        float ps = 0.0f;
        #pragma unroll 1
        for (int j = 0; j < BN; ++j) {
            float p = exp2f(s_tile[row * BN + j] - row_m);
            p_tile[row * BN + j] = __float2half(p);
            ps += p;
        }
        row_l += ps;

        // ---- PV via V_DOT2 over kv-pairs (hoist p load, unroll column loop) ----
        #pragma unroll 2
        for (int jj = 0; jj < BN; jj += 2) {
            unsigned p2 = load_h2_quad(&p_tile[row * BN + jj]);
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                int d = out_base_col + c;
                acc[c] = fdot2_f32_f16(p2, load_h2_quad(&v_tile[d * BN + jj]), acc[c]);
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


}  // namespace sageattn_gfx10
