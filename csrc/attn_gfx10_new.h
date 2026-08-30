// Independent gfx10 kernel source for v3.4 optimization.
// Designed to coexist with attn_gfx11.hip — uses same kernel name pattern but
// is independent of mma_gfx11.h, mma_gfx10.h attn_kernel_*_t implementations.
//
// Layout: BM=128, NW=1 (16 lanes cooperate per row), NR=8 (8 rows/thread)
// - Block: 128 threads = 4 warps × 32 lanes = 8 half-warps × 16 lanes
// - Each half-warp owns 1 Q row, lane L holds D-chunk L (lane 0..15 = row0 of warp)
// - 8 rows per block (4 warps × 2 half-warps each owning 1 row)
// - QK: each lane computes 1 sdot4 partial, warp_reduce_sdot4 xor 8/4/2/1 fuses 16
//       chunks into 1 score, with lane 0 writing to s_tile (race-free)
// - softmax: all 16 lanes execute (redundant but race-free), produce per-row
//       row_m/row_l + write p_tile (lane 0 only)
// - PV: lane 0..15 own CL=HD/16 cols each of the shared row's output
//       (same value across lane L vs L^16 after xor 16 swap)
// - writeback: lane 0..15 writes; lane 16..31 redundant (after xor 16 swap, identical)

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace sageattn_gfx10_new {

constexpr int MIN_BLK_Q = 32;
constexpr int MIN_BLK_K = 16;

typedef _Float16 v2h __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int load_i8_quad(const int8_t* p) {
    return *reinterpret_cast<const int*>(p);
}

__device__ __forceinline__ unsigned load_h2_quad(const __half* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

// 16-lane xor-shuffle reduction (for designs where lane 0..15 hold D-chunks).
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

__device__ __forceinline__ int sdot4_i32_i8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

__device__ __forceinline__ float fdot2_f32_f16(unsigned a, unsigned b, float c) {
    return __builtin_amdgcn_fdot2(
        *reinterpret_cast<const v2h*>(&a), *reinterpret_cast<const v2h*>(&b), c, false);
}

template <typename ODT> __device__ __forceinline__ ODT gfx10_out_convert(float v);
template <> __device__ __forceinline__ __half gfx10_out_convert<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __hip_bfloat16 gfx10_out_convert<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

// v3.5: 16-lane cooperative QK/PV, BM=128 (8 rows/block), BN=16/32.
// Each half-warp (16 lanes) owns 1 Q row; all 16 lanes share the same m.
template <int HD, bool ISC, int BN, typename ODT>
__global__ void attn_kernel_gfx10_new_v35_t(
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
    // BM=128: 8 rows/block (4 warps × 2 half-warps × 1 row/half-warp).
    // BN is template param.
    constexpr int BM = 8;
    constexpr int NTHREAD = 128;
    constexpr int QUADS = HD / 4;        // int8 dwords per row (D=64: 16; D=128: 32)
    constexpr int CL = HD / 16;          // output cols per lane (D=64: 4; D=128: 8)
    constexpr int ROWS_PER_HALF = 1;     // one row per half-warp
    static_assert(CL * 16 == HD, "HD must be multiple of 16");
    static_assert(NTHREAD == 128, "v3.5 expects 128 threads");

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int half_warp = lane >> 4;          // 0 or 1
    const int lane_in_half = lane & 15;       // 0..15 within half-warp
    const int row = warp * 2 + half_warp;    // 0..7
    const int64_t m = blockIdx.x * BM + row;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len);

    // ---- LDS tiles ----
    __shared__ int8_t k_tile[BN * HD];
    __shared__ __half v_tile[HD * BN];
    __shared__ float s_tile[BM * BN];       // [8][BN] scores
    __shared__ __half p_tile[BM * BN];       // [8][BN] softmax probs

    // Pre-load Q row into registers (lane_in_half holds D-chunk lane_in_half).
    // QUADS = 16 (D=64) or 32 (D=128); lane holds chunks lane_in_half*QUADS/16 .. +1.
    int q_reg;
    {
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        const int8_t* qrow = q + qb + m * q_stride_n;
        const int d_chunk = lane_in_half * (QUADS / 16);  // D=64: lane*1; D=128: lane*2
        q_reg = valid ? load_i8_quad(qrow + d_chunk * 4) : 0;
    }

    const float qs = valid
        ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)]
        : 0.0f;

    float acc[CL];
    #pragma unroll
    for (int c = 0; c < CL; ++c) acc[c] = 0.0f;
    float row_m = -3.0e38f, row_l = 0.0f;

    for (int64_t kb = 0; kb < kv_len; kb += BN) {
        // Stage K tile (BN × HD int8)
        {
            const int64_t kqb = b * k_stride_b + kvh * k_stride_h;
            #pragma unroll 1
            for (int i = tid; i < BN * QUADS; i += NTHREAD) {
                int r = i / QUADS, ck = i % QUADS;
                int64_t n = kb + r;
                const int8_t* krow = k + kqb + n * k_stride_n;
                *reinterpret_cast<int*>(&k_tile[r * HD + ck * 4]) =
                    (n < kv_len) ? load_i8_quad(krow + ck * 4) : 0;
            }
        }
        // Stage V tile [D][N] from V_T
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

        // QK: lane_in_half in 0..15 holds 1 D-chunk partial for shared row `row`.
        // Reduce 16 partials via xor 8/4/2/1 (warp_reduce_sdot4, designed for 16 lanes).
        // Lane 0 (lane_in_half == 0) writes s_tile (race-free: only one writer per slot).
        // But wait: lane 16..31 has half_warp=1, lane_in_half=0..15. They're in a different
        // half-warp and own row=1. So lane 0..15 (half_warp 0) write s_tile[row0*BN+j], lane 16..31 write
        // s_tile[row1*BN+j]. That's race-free.
        if (valid) {
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                int partial = sdot4_i32_i8(
                    q_reg,
                    *reinterpret_cast<const int*>(&k_tile[j * HD + lane_in_half * (QUADS / 16) * 4]),
                    0);
                int reduced = warp_reduce_sdot4(partial);  // 16-lane reduction within half-warp
                float sc = static_cast<float>(reduced) * (qs *
                    k_scale[b * ks_stride_b + kvh * ks_stride_h +
                            static_cast<int>((kb + j) / MIN_BLK_K)]);
                int64_t n = kb + j;
                if ((ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
                s_tile[row * BN + j] = sc;
            }
        }
        __syncthreads();

        // Softmax: per-row (all 16 lanes compute the same values, race-free).
        // Lane 0 writes p_tile (only one writer per row).
        if (valid) {
            float lm = -3.0e38f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) lm = fmaxf(lm, s_tile[row * BN + j]);
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            row_m = gm;
            row_l *= alpha;
            #pragma unroll
            for (int c = 0; c < CL; ++c) acc[c] *= alpha;
            float ps = 0.0f;
            if (lane_in_half == 0) {
                #pragma unroll
                for (int j = 0; j < BN; ++j) {
                    float p = exp2f(s_tile[row * BN + j] - row_m);
                    p_tile[row * BN + j] = __float2half(p);
                    ps += p;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < BN; ++j) {
                    ps += exp2f(s_tile[row * BN + j] - row_m);
                }
            }
            row_l += ps;
        }
        __syncthreads();

        // PV: lane_in_half in 0..15 owns CL=HD/16 cols of the shared row's output.
        // v_tile[d*BN+jj] loaded for col d = lane_in_half*CL..(lane_in_half+1)*CL.
        if (valid) {
            const int col_base = lane_in_half * CL;
            #pragma unroll 2
            for (int jj = 0; jj < BN; jj += 2) {
                unsigned p2 = load_h2_quad(&p_tile[row * BN + jj]);
                #pragma unroll
                for (int c = 0; c < CL; ++c) {
                    int d = col_base + c;
                    acc[c] = fdot2_f32_f16(p2, load_h2_quad(&v_tile[d * BN + jj]), acc[c]);
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        float inv = 1.0f / row_l;
        const int col_base = lane_in_half * CL;
        int64_t out_base = b * o_stride_b + m * o_stride_n + h * o_stride_h;
        #pragma unroll
        for (int c = 0; c < CL; ++c) {
            int d = col_base + c;
            out[out_base + d] = gfx10_out_convert<ODT>(acc[c] * inv);
        }
    }
#endif
}

}  // namespace sageattn_gfx10_new