#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/tensor_inl.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#error "attn_gfx11.cu is only intended for ROCm/HIP."
#endif

#include "reduction_utils.cuh"
#include "mma_gfx11.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

using torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

namespace {

constexpr int kNHD = 0;
constexpr int kHND = 1;
constexpr float kLog2e = 1.4426950408889634f;

constexpr int RM = 16;
constexpr int BK = 16;

constexpr int MIN_BLK_Q = 32;
constexpr int MIN_BLK_K = 16;

constexpr int LDS_PAD = 16;

Tensor new_empty_like(const Tensor& like, std::initializer_list<int64_t> sizes, ScalarType dtype) {
    return torch::stable::new_empty(like, std::vector<int64_t>(sizes), std::make_optional(dtype));
}

hipStream_t current_hip_stream(const Tensor& tensor) {
    int32_t device_index = tensor.get_device_index();
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream));
    return reinterpret_cast<hipStream_t>(stream);
}

__device__ __forceinline__ float to_float(const __half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(const __hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __half from_float_f16(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __hip_bfloat16 from_float_bf16(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ int8_t float_to_int8(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int32_t rounded;
    asm volatile("v_cvt_i32_f32 %[dst], %[src]" : [dst] "=v"(rounded) : [src] "v"(x));
    rounded = rounded > 127 ? 127 : rounded;
    rounded = rounded < -128 ? -128 : rounded;
    return static_cast<int8_t>(rounded);
}

__device__ __forceinline__ float max3_f32(float a, float b, float c) {
    float result;
    asm volatile("v_max3_f32 %[dst], %[src0], %[src1], %[src2]"
                 : [dst] "=v"(result)
                 : [src0] "v"(a), [src1] "v"(b), [src2] "v"(c));
    return result;
}

__device__ __forceinline__ float fast_shfl_xor(float val, int mask) {
    return __shfl_xor(val, mask, 32);
}

template <typename T>
__global__ void mean_hnd_kernel(
    const T* __restrict__ input,
    T* __restrict__ mean_out,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t head_dim,
    const int64_t in_stride_b,
    const int64_t in_stride_n,
    const int64_t in_stride_h) {
    constexpr int TileD = 16;
    __shared__ float partial_sum[256];

    const int tid = threadIdx.x;
    const int d_local = tid & (TileD - 1);
    const int s_lane = tid >> 4;
    const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
    const int64_t h = blockIdx.y;
    const int64_t b = blockIdx.z;
    const int64_t d = d_base + d_local;

    float local_sum = 0.0f;
    if (d < head_dim) {
        for (int64_t s = s_lane; s < seq_len; s += 16) {
            const int64_t offset = b * in_stride_b + s * in_stride_n + h * in_stride_h + d;
            local_sum += to_float(input[offset]);
        }
    }
    partial_sum[tid] = local_sum;
    __syncthreads();

    if (tid < TileD) {
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            sum += partial_sum[i * TileD + tid];
        }
        const int64_t mean_d = d_base + tid;
        if (mean_d < head_dim) {
            const float value = sum / static_cast<float>(seq_len);
            if constexpr (std::is_same<T, __half>::value) {
                mean_out[(b * heads + h) * head_dim + mean_d] = from_float_f16(value);
            } else {
                mean_out[(b * heads + h) * head_dim + mean_d] = from_float_bf16(value);
            }
        }
    }
}

template <typename T, int HeadDim>
__global__ void quant_qk_int8_hnd_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ key_mean,
    int8_t* __restrict__ query_out,
    int8_t* __restrict__ key_out,
    float* __restrict__ query_scale,
    float* __restrict__ key_scale,
    const int64_t batch,
    const int64_t q_heads,
    const int64_t kv_heads,
    const int64_t q_len,
    const int64_t kv_len,
    const int q_groups,
    const int k_groups,
    const float sm_scale_log2e,
    const int64_t q_in_stride_b,
    const int64_t q_in_stride_n,
    const int64_t q_in_stride_h,
    const int64_t k_in_stride_b,
    const int64_t k_in_stride_n,
    const int64_t k_in_stride_h) {
    constexpr int Threads = 256;
    __shared__ float shared_amax;

    const int group = blockIdx.x;
    const int head = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const bool is_q = group < q_groups;
    const int local_group = is_q ? group : group - q_groups;
    const int rows_per_group = is_q ? MIN_BLK_Q : MIN_BLK_K;
    const int64_t seq_len = is_q ? q_len : kv_len;
    const int64_t base_row = static_cast<int64_t>(local_group) * rows_per_group;
    const int active_heads = is_q ? static_cast<int>(q_heads) : static_cast<int>(kv_heads);
    if (b >= batch || head >= active_heads || base_row >= seq_len) return;

    const T* in = is_q ? query : key;
    int8_t* out = is_q ? query_out : key_out;
    float* scale_out = is_q ? query_scale : key_scale;
    const int64_t heads = is_q ? q_heads : kv_heads;
    const int scale_groups = is_q ? q_groups : k_groups;
    const int64_t in_stride_b = is_q ? q_in_stride_b : k_in_stride_b;
    const int64_t in_stride_n = is_q ? q_in_stride_n : k_in_stride_n;
    const int64_t in_stride_h = is_q ? q_in_stride_h : k_in_stride_h;
    constexpr int PackElems = 8;
    const int packs = (rows_per_group * HeadDim) / PackElems;

    const float pass1_scale = is_q ? sm_scale_log2e : 1.0f;
    float local_amax = 1e-7f;
    for (int pack = tid; pack < packs; pack += Threads) {
        const int elem_base = pack * PackElems;
        const int row = elem_base / HeadDim;
        const int d = elem_base - row * HeadDim;
        const int64_t seq = base_row + row;
        if (seq < seq_len) {
            const int64_t in_off = static_cast<int64_t>(b) * in_stride_b + seq * in_stride_n + head * in_stride_h + d;
            const uint4 raw = *reinterpret_cast<const uint4*>(in + in_off);
            const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
            for (int i = 0; i < PackElems; ++i) {
                float v = to_float(values[i]);
                if (!is_q && key_mean != nullptr) {
                    v -= to_float(key_mean[(b * heads + head) * HeadDim + d + i]);
                }
                local_amax = fmaxf(local_amax, fabsf(v * pass1_scale));
            }
        }
    }
    const float block_amax = vllm::blockReduceMax(local_amax);
    if (tid == 0) {
        shared_amax = block_amax;
        scale_out[(static_cast<int64_t>(b) * active_heads + head) * scale_groups + local_group] =
            shared_amax / 127.0f;
    }
    __syncthreads();
    const float inv_scale = 127.0f / shared_amax;

    for (int pack = tid; pack < packs; pack += Threads) {
        const int elem_base = pack * PackElems;
        const int row = elem_base / HeadDim;
        const int d = elem_base - row * HeadDim;
        const int64_t seq = base_row + row;
        if (seq < seq_len) {
            const int64_t in_off = static_cast<int64_t>(b) * in_stride_b + seq * in_stride_n + head * in_stride_h + d;
            const int64_t out_off = (static_cast<int64_t>(b) * active_heads + head) * seq_len * HeadDim + seq * HeadDim + d;
            const uint4 raw = *reinterpret_cast<const uint4*>(in + in_off);
            const T* values = reinterpret_cast<const T*>(&raw);
            char4 out0, out1;
            float v0 = to_float(values[0]), v1 = to_float(values[1]);
            float v2 = to_float(values[2]), v3 = to_float(values[3]);
            float v4 = to_float(values[4]), v5 = to_float(values[5]);
            float v6 = to_float(values[6]), v7 = to_float(values[7]);
            if (!is_q && key_mean != nullptr) {
                const int64_t mean_base = (b * heads + head) * HeadDim + d;
                v0 -= to_float(key_mean[mean_base + 0]);
                v1 -= to_float(key_mean[mean_base + 1]);
                v2 -= to_float(key_mean[mean_base + 2]);
                v3 -= to_float(key_mean[mean_base + 3]);
                v4 -= to_float(key_mean[mean_base + 4]);
                v5 -= to_float(key_mean[mean_base + 5]);
                v6 -= to_float(key_mean[mean_base + 6]);
                v7 -= to_float(key_mean[mean_base + 7]);
            }
            const float extra_scale = is_q ? sm_scale_log2e : 1.0f;
            out0.x = float_to_int8(v0 * inv_scale * extra_scale);
            out0.y = float_to_int8(v1 * inv_scale * extra_scale);
            out0.z = float_to_int8(v2 * inv_scale * extra_scale);
            out0.w = float_to_int8(v3 * inv_scale * extra_scale);
            out1.x = float_to_int8(v4 * inv_scale * extra_scale);
            out1.y = float_to_int8(v5 * inv_scale * extra_scale);
            out1.z = float_to_int8(v6 * inv_scale * extra_scale);
            out1.w = float_to_int8(v7 * inv_scale * extra_scale);
            *reinterpret_cast<char4*>(out + out_off) = out0;
            *reinterpret_cast<char4*>(out + out_off + 4) = out1;
        }
    }
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE = __half>
__device__ __forceinline__ void attn_kernel_impl(
    const int8_t* __restrict__ q,
    const int8_t* __restrict__ k,
    const V_DTYPE* __restrict__ v,
    void* __restrict__ output,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const int tensor_layout) {

    constexpr int WARPS = BLOCK_M / RM;
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / BK;
    constexpr int ColTiles = BLOCK_N / BK;
    constexpr int KStride = HeadDim + LDS_PAD;
    constexpr int VStride = HeadDim + LDS_PAD;

    constexpr int PStride = BLOCK_N + 8;
    __shared__ int8_t k_tile[BLOCK_N * KStride];
    __shared__ __half v_tile[BLOCK_N * VStride];
    __shared__ __half p_lds[WARPS * 16 * PStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) return;

    const int64_t hkv = hq / (num_qo_heads / num_kv_heads);
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * RM;

    using namespace sageattn_gfx11;

    int32_t_v4 q_frag[DTiles];
    const int q_row = lane & 15;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_idx = q_start + q_row;
        if (q_idx < qo_len) {
            const int d_base = dt * BK;
            const int64_t q_off = (tensor_layout == kHND) ?
                (b * q_stride_b + hq * q_stride_h + q_idx * q_stride_n + d_base) :
                (b * q_stride_b + q_idx * q_stride_n + hq * q_stride_h + d_base);
            q_frag[dt] = *reinterpret_cast<const int32_t_v4*>(q + q_off);
        } else {
            q_frag[dt] = int32_t_v4{0, 0, 0, 0};
        }
    }

    const int q_scale_idx = static_cast<int>(q_start / MIN_BLK_Q);
    const float qs = q_scale[b * qs_stride_b + hq * qs_stride_h + q_scale_idx];

    v8f out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    float row_m[8], row_l[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row_m[e] = -FLT_MAX * 0.5f;
        row_l[e] = 0.0f;
    }

    const int64_t kv_limit = IsCausal ? min(q_base + BLOCK_M, kv_len) : kv_len;
    constexpr int KVecsPerRow = HeadDim / 16;
    constexpr int VVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int VVecsTotal = BLOCK_N * VVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;
    constexpr int VPrefetchPerThread = (VVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    uint4 v_prefetch[VPrefetchPerThread];

    for (int i = tid; i < KVecsTotal; i += THREADS) {
        const int n = i / KVecsPerRow;
        const int d = (i - n * KVecsPerRow) * 16;
        const int64_t k_idx = n;
        if (k_idx < kv_len) {
            const int64_t k_off = (tensor_layout == kHND) ?
                (b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d) :
                (b * k_stride_b + k_idx * k_stride_n + hkv * k_stride_h + d);
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) =
                *reinterpret_cast<const uint4*>(k + k_off);
        } else {
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = make_uint4(0, 0, 0, 0);
        }
    }
    for (int i = tid; i < VVecsTotal; i += THREADS) {
        const int n = i / VVecsPerRow;
        const int d = (i - n * VVecsPerRow) * 8;
        const int64_t v_idx = n;
        if (v_idx < kv_len) {
            const int64_t v_off = (tensor_layout == kHND) ?
                (b * v_stride_b + hkv * v_stride_h + v_idx * v_stride_n + d) :
                (b * v_stride_b + v_idx * v_stride_n + hkv * v_stride_h + d);
            if constexpr (std::is_same<V_DTYPE, __half>::value) {
                *reinterpret_cast<uint4*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2) =
                    *reinterpret_cast<const uint4*>(v + v_off);
            } else {
                const V_DTYPE* vsrc = v + v_off;
                __half* vdst = reinterpret_cast<__half*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    vdst[j] = __float2half_rn(__bfloat162float(vsrc[j]));
                }
            }
        } else {
            *reinterpret_cast<uint4*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BLOCK_N) {
        const int64_t next_base = kb_base + BLOCK_N;
        const bool has_next = (next_base < kv_limit);

        if (has_next) {
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    const int64_t k_idx = next_base + n;
                    if (k_idx < kv_len) {
                        const int64_t k_off = (tensor_layout == kHND) ?
                            (b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d) :
                            (b * k_stride_b + k_idx * k_stride_n + hkv * k_stride_h + d);
                        k_prefetch[i] = *reinterpret_cast<const uint4*>(k + k_off);
                    } else {
                        k_prefetch[i] = make_uint4(0, 0, 0, 0);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < VPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < VVecsTotal) {
                    const int n = vec / VVecsPerRow;
                    const int d = (vec - n * VVecsPerRow) * 8;
                    const int64_t v_idx = next_base + n;
                    if (v_idx < kv_len) {
                        const int64_t v_off = (tensor_layout == kHND) ?
                            (b * v_stride_b + hkv * v_stride_h + v_idx * v_stride_n + d) :
                            (b * v_stride_b + v_idx * v_stride_n + hkv * v_stride_h + d);
                        v_prefetch[i] = *reinterpret_cast<const uint4*>(v + v_off);
                    } else {
                        v_prefetch[i] = make_uint4(0, 0, 0, 0);
                    }
                }
            }
        }

        float score_cache[ColTiles][8];

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8i score_acc = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int k_row = lane & 15;
                const int32_t_v4 k_frag = load_i8_frag(
                    k_tile + ct * BK * KStride, k_row, dt * BK, KStride);
                score_acc = wmma_i32_iu8(q_frag[dt], k_frag, score_acc);
            }

            const int64_t k_col_start = kb_base + ct * BK;
            const int k_scale_idx = static_cast<int>(k_col_start / MIN_BLK_K);
            const float ks_val = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
            const float score_scale = qs * ks_val;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = static_cast<float>(score_acc[e]) * score_scale;
                if constexpr (IsCausal) {
                    const int q_row_idx = static_cast<int>(q_start) + acc_row(lane, e);
                    const int k_col_idx = static_cast<int>(k_col_start) + (lane & 15);
                    if (k_col_idx > q_row_idx) s = -FLT_MAX * 0.5f;
                }
                if (k_col_start + (lane & 15) >= kv_len) {
                    s = -FLT_MAX * 0.5f;
                }
                score_cache[ct][e] = s;
            }
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float local_mx = score_cache[0][e];
#pragma unroll
            for (int ct = 1; ct < ColTiles; ++ct)
                local_mx = fmaxf(local_mx, score_cache[ct][e]);
            float gm = fmaxf(row_m[e], local_mx);
            gm = fmaxf(gm, fast_shfl_xor(gm, 8));
            gm = fmaxf(gm, fast_shfl_xor(gm, 4));
            gm = fmaxf(gm, fast_shfl_xor(gm, 2));
            gm = fmaxf(gm, fast_shfl_xor(gm, 1));
            const float alpha = (row_l[e] == 0.0f) ? 0.0f : fast_exp2(row_m[e] - gm);
            row_m[e] = gm;
            row_l[e] *= alpha;
            for (int dt = 0; dt < DTiles; ++dt)
                out_acc[dt][e] *= alpha;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct)
                score_cache[ct][e] = fast_exp2(score_cache[ct][e] - gm);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float partial_sm = 0.0f;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct)
                partial_sm += score_cache[ct][e];
            partial_sm += fast_shfl_xor(partial_sm, 8);
            partial_sm += fast_shfl_xor(partial_sm, 4);
            partial_sm += fast_shfl_xor(partial_sm, 2);
            partial_sm += fast_shfl_xor(partial_sm, 1);
            row_l[e] += partial_sm;
        }

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            __half* p_base = p_lds + wave * 16 * PStride;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int p_row = 2 * e + (lane >> 4);
                const int p_col = ct * 16 + (lane & 15);
                p_base[p_row * PStride + p_col] = __float2half_rn(score_cache[ct][e]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            const int my_row = lane & 15;
            const __half* p_row_base = p_lds + wave * 16 * PStride + my_row * PStride + ct * 16;
            const unsigned int* p_u32 = reinterpret_cast<const unsigned int*>(p_row_base);
            v16h p_frag;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned int word = p_u32[k];
                p_frag[2 * k] = reinterpret_cast<__half&>(word);
                unsigned int hi = word >> 16;
                p_frag[2 * k + 1] = reinterpret_cast<__half&>(hi);
            }

#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_col = dt * BK + acc_col(lane);
                const char* v_base = reinterpret_cast<const char*>(v_tile) + ct * BK * VStride * 2;
                const v16h v_frag = sageattn_gfx11::load_fp16_col_frag(v_base, d_col, VStride * 2);
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(p_frag, v_frag, out_acc[dt]);
            }
        }
        __syncthreads();

        if (has_next) {
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
            for (int i = 0; i < VPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < VVecsTotal) {
                    const int n = vec / VVecsPerRow;
                    const int d = (vec - n * VVecsPerRow) * 8;
                    if constexpr (std::is_same<V_DTYPE, __half>::value) {
                        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2) = v_prefetch[i];
                    } else {
                        const V_DTYPE* vsrc = reinterpret_cast<const V_DTYPE*>(&v_prefetch[i]);
                        __half* vdst = reinterpret_cast<__half*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            vdst[j] = __float2half_rn(__bfloat162float(vsrc[j]));
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    const int col = acc_col(lane);
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int d = dt * BK + col;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row_offset = acc_row(lane, e);
            const int64_t q_idx = q_start + row_offset;
            if (q_idx < qo_len) {
                const float inv_l = (row_l[e] > 0.0f) ? (1.0f / row_l[e]) : 0.0f;
                const int64_t o_off = (tensor_layout == kHND) ?
                    (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                    (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
                if constexpr (std::is_same<V_DTYPE, __half>::value) {
                    reinterpret_cast<__half*>(output)[o_off] = __float2half_rn(out_acc[dt][e] * inv_l);
                } else {
                    reinterpret_cast<__hip_bfloat16*>(output)[o_off] =
                        from_float_bf16(out_acc[dt][e] * inv_l);
                }
            }
        }
    }
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void attn_kernel_wpe1(
    const int8_t* __restrict__ q,
    const int8_t* __restrict__ k,
    const V_DTYPE* __restrict__ v,
    void* __restrict__ output,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const int tensor_layout) {
    attn_kernel_impl<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void attn_kernel_wpe2(
    const int8_t* __restrict__ q,
    const int8_t* __restrict__ k,
    const V_DTYPE* __restrict__ v,
    void* __restrict__ output,
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const int64_t qs_stride_b,
    const int64_t qs_stride_h,
    const int64_t ks_stride_b,
    const int64_t ks_stride_h,
    const int tensor_layout) {
    attn_kernel_impl<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void fp16_attn_kernel_impl(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const float sm_scale_log2e,
    const int tensor_layout) {

    constexpr int WARPS = BLOCK_M / RM;
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / BK;
    constexpr int ColTiles = BLOCK_N / BK;
    constexpr int KStride = HeadDim + LDS_PAD;
    constexpr int VStride = HeadDim + LDS_PAD;
    constexpr int PStride = BLOCK_N + 8;

    __shared__ __half k_tile[BLOCK_N * KStride];
    __shared__ __half v_tile[BLOCK_N * VStride];
    __shared__ __half p_lds[WARPS * 16 * PStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) return;

    const int64_t hkv = hq / (num_qo_heads / num_kv_heads);
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * RM;

    using namespace sageattn_gfx11;

    v16h q_frag[DTiles];
    const int q_row = lane & 15;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_idx = q_start + q_row;
        if (q_idx < qo_len) {
            const int d_base = dt * BK;
            const int64_t q_off = (tensor_layout == kHND) ?
                (b * q_stride_b + hq * q_stride_h + q_idx * q_stride_n + d_base) :
                (b * q_stride_b + q_idx * q_stride_n + hq * q_stride_h + d_base);
            const __half* src = q + q_off;
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = static_cast<_Float16>(src[i]);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = static_cast<_Float16>(0.0f);
        }
    }

    v8f out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) out_acc[dt] = v8f{0,0,0,0,0,0,0,0};
    float row_m[8], row_l[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row_m[e] = -FLT_MAX * 0.5f;
        row_l[e] = 0.0f;
    }

    const int64_t kv_limit = IsCausal ? min(q_base + BLOCK_M, kv_len) : kv_len;
    constexpr int KVecsPerRow = HeadDim / 8;
    constexpr int VVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int VVecsTotal = BLOCK_N * VVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;
    constexpr int VPrefetchPerThread = (VVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    uint4 v_prefetch[VPrefetchPerThread];

    for (int i = tid; i < KVecsTotal; i += THREADS) {
        const int n = i / KVecsPerRow;
        const int d = (i - n * KVecsPerRow) * 8;
        const int64_t k_idx = n;
        if (k_idx < kv_len) {
            const int64_t k_off = (tensor_layout == kHND) ?
                (b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d) :
                (b * k_stride_b + k_idx * k_stride_n + hkv * k_stride_h + d);
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) =
                *reinterpret_cast<const uint4*>(k + k_off);
        } else {
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = make_uint4(0,0,0,0);
        }
    }
    for (int i = tid; i < VVecsTotal; i += THREADS) {
        const int n = i / VVecsPerRow;
        const int d = (i - n * VVecsPerRow) * 8;
        const int64_t v_idx = n;
        if (v_idx < kv_len) {
            const int64_t v_off = (tensor_layout == kHND) ?
                (b * v_stride_b + hkv * v_stride_h + v_idx * v_stride_n + d) :
                (b * v_stride_b + v_idx * v_stride_n + hkv * v_stride_h + d);
            *reinterpret_cast<uint4*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2) =
                *reinterpret_cast<const uint4*>(v + v_off);
        } else {
            *reinterpret_cast<uint4*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2) = make_uint4(0,0,0,0);
        }
    }
    __syncthreads();

    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BLOCK_N) {
        const int64_t next_base = kb_base + BLOCK_N;
        const bool has_next = (next_base < kv_limit);

        if (has_next) {
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 8;
                    const int64_t k_idx = next_base + n;
                    if (k_idx < kv_len) {
                        const int64_t k_off = (tensor_layout == kHND) ?
                            (b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d) :
                            (b * k_stride_b + k_idx * k_stride_n + hkv * k_stride_h + d);
                        k_prefetch[i] = *reinterpret_cast<const uint4*>(k + k_off);
                    } else { k_prefetch[i] = make_uint4(0,0,0,0); }
                }
            }
#pragma unroll
            for (int i = 0; i < VPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < VVecsTotal) {
                    const int n = vec / VVecsPerRow;
                    const int d = (vec - n * VVecsPerRow) * 8;
                    const int64_t v_idx = next_base + n;
                    if (v_idx < kv_len) {
                        const int64_t v_off = (tensor_layout == kHND) ?
                            (b * v_stride_b + hkv * v_stride_h + v_idx * v_stride_n + d) :
                            (b * v_stride_b + v_idx * v_stride_n + hkv * v_stride_h + d);
                        v_prefetch[i] = *reinterpret_cast<const uint4*>(v + v_off);
                    } else { v_prefetch[i] = make_uint4(0,0,0,0); }
                }
            }
        }

        float score_cache[ColTiles][8];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8f score_acc = v8f{0,0,0,0,0,0,0,0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int k_row = lane & 15;
                const __half* k_ptr = &k_tile[(ct * BK + k_row) * KStride + dt * BK];
                v16h k_frag;
#pragma unroll
                for (int i = 0; i < 16; ++i) k_frag[i] = static_cast<_Float16>(k_ptr[i]);
                score_acc = wmma_f32_f16(q_frag[dt], k_frag, score_acc);
            }
            const int64_t k_col_start = kb_base + ct * BK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = score_acc[e] * sm_scale_log2e;
                if constexpr (IsCausal) {
                    const int q_row_idx = static_cast<int>(q_start) + acc_row(lane, e);
                    const int k_col_idx = static_cast<int>(k_col_start) + (lane & 15);
                    if (k_col_idx > q_row_idx) s = -FLT_MAX * 0.5f;
                }
                if (k_col_start + (lane & 15) >= kv_len) s = -FLT_MAX * 0.5f;
                score_cache[ct][e] = s;
            }
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float local_mx = score_cache[0][e];
#pragma unroll
            for (int ct = 1; ct < ColTiles; ++ct)
                local_mx = fmaxf(local_mx, score_cache[ct][e]);
            float gm = fmaxf(row_m[e], local_mx);
            gm = fmaxf(gm, fast_shfl_xor(gm, 8));
            gm = fmaxf(gm, fast_shfl_xor(gm, 4));
            gm = fmaxf(gm, fast_shfl_xor(gm, 2));
            gm = fmaxf(gm, fast_shfl_xor(gm, 1));
            const float alpha = (row_l[e] == 0.0f) ? 0.0f : fast_exp2(row_m[e] - gm);
            row_m[e] = gm;
            row_l[e] *= alpha;
            for (int dt = 0; dt < DTiles; ++dt)
                out_acc[dt][e] *= alpha;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct)
                score_cache[ct][e] = fast_exp2(score_cache[ct][e] - gm);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float partial_sm = 0.0f;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct)
                partial_sm += score_cache[ct][e];
            partial_sm += fast_shfl_xor(partial_sm, 8);
            partial_sm += fast_shfl_xor(partial_sm, 4);
            partial_sm += fast_shfl_xor(partial_sm, 2);
            partial_sm += fast_shfl_xor(partial_sm, 1);
            row_l[e] += partial_sm;
        }

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            __half* p_base = p_lds + wave * 16 * PStride;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int p_row = 2 * e + (lane >> 4);
                const int p_col = ct * 16 + (lane & 15);
                p_base[p_row * PStride + p_col] = __float2half_rn(score_cache[ct][e]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            const int my_row = lane & 15;
            const __half* p_row_base = p_lds + wave * 16 * PStride + my_row * PStride + ct * 16;
            const unsigned int* p_u32 = reinterpret_cast<const unsigned int*>(p_row_base);
            v16h p_frag;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned int word = p_u32[k];
                p_frag[2 * k] = reinterpret_cast<__half&>(word);
                unsigned int hi = word >> 16;
                p_frag[2 * k + 1] = reinterpret_cast<__half&>(hi);
            }
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_col = dt * BK + acc_col(lane);
                const char* v_base = reinterpret_cast<const char*>(v_tile) + ct * BK * VStride * 2;
                const v16h v_frag = load_fp16_col_frag(v_base, d_col, VStride * 2);
                out_acc[dt] = wmma_f32_f16(p_frag, v_frag, out_acc[dt]);
            }
        }
        __syncthreads();

        if (has_next) {
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 8;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
            for (int i = 0; i < VPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < VVecsTotal) {
                    const int n = vec / VVecsPerRow;
                    const int d = (vec - n * VVecsPerRow) * 8;
                    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(v_tile) + (n * VStride + d) * 2) = v_prefetch[i];
                }
            }
            __syncthreads();
        }
    }

    const int col = acc_col(lane);
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int d = dt * BK + col;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int64_t q_idx = q_start + acc_row(lane, e);
            if (q_idx < qo_len) {
                const float inv_l = (row_l[e] > 0.0f) ? (1.0f / row_l[e]) : 0.0f;
                const int64_t o_off = (tensor_layout == kHND) ?
                    (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                    (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
                output[o_off] = from_float_f16(out_acc[dt][e] * inv_l);
            }
        }
    }
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void fp16_attn_kernel_wpe1(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ output,
    const int64_t batch_size, const int64_t qo_len, const int64_t kv_len,
    const int64_t num_qo_heads, const int64_t num_kv_heads,
    const int64_t q_stride_b, const int64_t q_stride_n, const int64_t q_stride_h,
    const int64_t k_stride_b, const int64_t k_stride_n, const int64_t k_stride_h,
    const int64_t v_stride_b, const int64_t v_stride_n, const int64_t v_stride_h,
    const int64_t o_stride_b, const int64_t o_stride_n, const int64_t o_stride_h,
    const float sm_scale_log2e, const int tensor_layout) {
    fp16_attn_kernel_impl<HeadDim, IsCausal, BLOCK_M, BLOCK_N>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h,
        sm_scale_log2e, tensor_layout);
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void fp16_attn_kernel_wpe2(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ output,
    const int64_t batch_size, const int64_t qo_len, const int64_t kv_len,
    const int64_t num_qo_heads, const int64_t num_kv_heads,
    const int64_t q_stride_b, const int64_t q_stride_n, const int64_t q_stride_h,
    const int64_t k_stride_b, const int64_t k_stride_n, const int64_t k_stride_h,
    const int64_t v_stride_b, const int64_t v_stride_n, const int64_t v_stride_h,
    const int64_t o_stride_b, const int64_t o_stride_n, const int64_t o_stride_h,
    const float sm_scale_log2e, const int tensor_layout) {
    fp16_attn_kernel_impl<HeadDim, IsCausal, BLOCK_M, BLOCK_N>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h,
        sm_scale_log2e, tensor_layout);
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void bf16_attn_kernel_impl(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v,
    unsigned short* __restrict__ output,
    const int64_t batch_size,
    const int64_t qo_len,
    const int64_t kv_len,
    const int64_t num_qo_heads,
    const int64_t num_kv_heads,
    const int64_t q_stride_b,
    const int64_t q_stride_n,
    const int64_t q_stride_h,
    const int64_t k_stride_b,
    const int64_t k_stride_n,
    const int64_t k_stride_h,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int64_t o_stride_b,
    const int64_t o_stride_n,
    const int64_t o_stride_h,
    const float sm_scale_log2e,
    const int tensor_layout) {

    constexpr int WARPS = BLOCK_M / RM;
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / BK;
    constexpr int ColTiles = BLOCK_N / BK;
    constexpr int KStride = HeadDim + LDS_PAD;
    constexpr int VStride = HeadDim + LDS_PAD;
    constexpr int PStride = BLOCK_N + 8;

    __shared__ __hip_bfloat16 k_tile[BLOCK_N * KStride];
    __shared__ __half v_tile[BLOCK_N * VStride];
    __shared__ __half p_lds[WARPS * 16 * PStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) return;

    const int64_t hkv = hq / (num_qo_heads / num_kv_heads);
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * RM;

    using namespace sageattn_gfx11;

    v16bf q_frag[DTiles];
    const int q_row = lane & 15;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_idx = q_start + q_row;
        if (q_idx < qo_len) {
            const int d_base = dt * BK;
            const int64_t q_off = (tensor_layout == kHND) ?
                (b * q_stride_b + hq * q_stride_h + q_idx * q_stride_n + d_base) :
                (b * q_stride_b + q_idx * q_stride_n + hq * q_stride_h + d_base);
            const __hip_bfloat16* src = q + q_off;
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = src[i];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = static_cast<__bf16>(0.0f);
        }
    }

    v8f out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) out_acc[dt] = v8f{0,0,0,0,0,0,0,0};
    float row_m[8], row_l[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row_m[e] = -FLT_MAX * 0.5f;
        row_l[e] = 0.0f;
    }

    const int64_t kv_limit = IsCausal ? min(q_base + BLOCK_M, kv_len) : kv_len;
    constexpr int KVecsPerRow = HeadDim / 8;
    constexpr int VVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int VVecsTotal = BLOCK_N * VVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;
    constexpr int VPrefetchPerThread = (VVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    uint4 v_prefetch[VPrefetchPerThread];

    for (int i = tid; i < KVecsTotal; i += THREADS) {
        const int n = i / KVecsPerRow;
        const int d = (i - n * KVecsPerRow) * 8;
        const int64_t k_idx = n;
        if (k_idx < kv_len) {
            const int64_t k_off = (tensor_layout == kHND) ?
                (b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d) :
                (b * k_stride_b + k_idx * k_stride_n + hkv * k_stride_h + d);
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) =
                *reinterpret_cast<const uint4*>(k + k_off);
        } else {
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = make_uint4(0,0,0,0);
        }
    }
    for (int i = tid; i < VVecsTotal; i += THREADS) {
        const int n = i / VVecsPerRow;
        const int d = (i - n * VVecsPerRow) * 8;
        const int64_t v_idx = n;
        if (v_idx < kv_len) {
            const int64_t v_off = (tensor_layout == kHND) ?
                (b * v_stride_b + hkv * v_stride_h + v_idx * v_stride_n + d) :
                (b * v_stride_b + v_idx * v_stride_n + hkv * v_stride_h + d);
            const __hip_bfloat16* vsrc = v + v_off;
            __half* vdst = &v_tile[n * VStride + d];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                vdst[j] = __float2half_rn(__bfloat162float(vsrc[j]));
            }
        } else {
            *reinterpret_cast<uint4*>(&v_tile[n * VStride + d]) = make_uint4(0,0,0,0);
        }
    }
    __syncthreads();

    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BLOCK_N) {
        const int64_t next_base = kb_base + BLOCK_N;
        const bool has_next = (next_base < kv_limit);

        if (has_next) {
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 8;
                    const int64_t k_idx = next_base + n;
                    if (k_idx < kv_len) {
                        const int64_t k_off = (tensor_layout == kHND) ?
                            (b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d) :
                            (b * k_stride_b + k_idx * k_stride_n + hkv * k_stride_h + d);
                        k_prefetch[i] = *reinterpret_cast<const uint4*>(k + k_off);
                    } else { k_prefetch[i] = make_uint4(0,0,0,0); }
                }
            }
#pragma unroll
            for (int i = 0; i < VPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < VVecsTotal) {
                    const int n = vec / VVecsPerRow;
                    const int d = (vec - n * VVecsPerRow) * 8;
                    const int64_t v_idx = next_base + n;
                    if (v_idx < kv_len) {
                        const int64_t v_off = (tensor_layout == kHND) ?
                            (b * v_stride_b + hkv * v_stride_h + v_idx * v_stride_n + d) :
                            (b * v_stride_b + v_idx * v_stride_n + hkv * v_stride_h + d);
                        v_prefetch[i] = *reinterpret_cast<const uint4*>(v + v_off);
                    } else { v_prefetch[i] = make_uint4(0,0,0,0); }
                }
            }
        }

        float score_cache[ColTiles][8];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8f score_acc = v8f{0,0,0,0,0,0,0,0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int k_row = lane & 15;
                const __hip_bfloat16* k_ptr = &k_tile[(ct * BK + k_row) * KStride + dt * BK];
                v16bf k_frag;
#pragma unroll
                for (int i = 0; i < 16; ++i) k_frag[i] = k_ptr[i];
                score_acc = wmma_f32_bf16(q_frag[dt], k_frag, score_acc);
            }
            const int64_t k_col_start = kb_base + ct * BK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = score_acc[e] * sm_scale_log2e;
                if constexpr (IsCausal) {
                    const int q_row_idx = static_cast<int>(q_start) + acc_row(lane, e);
                    const int k_col_idx = static_cast<int>(k_col_start) + (lane & 15);
                    if (k_col_idx > q_row_idx) s = -FLT_MAX * 0.5f;
                }
                if (k_col_start + (lane & 15) >= kv_len) s = -FLT_MAX * 0.5f;
                score_cache[ct][e] = s;
            }
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float local_mx = score_cache[0][e];
#pragma unroll
            for (int ct = 1; ct < ColTiles; ++ct)
                local_mx = fmaxf(local_mx, score_cache[ct][e]);
            float gm = fmaxf(row_m[e], local_mx);
            gm = fmaxf(gm, fast_shfl_xor(gm, 8));
            gm = fmaxf(gm, fast_shfl_xor(gm, 4));
            gm = fmaxf(gm, fast_shfl_xor(gm, 2));
            gm = fmaxf(gm, fast_shfl_xor(gm, 1));
            const float alpha = (row_l[e] == 0.0f) ? 0.0f : fast_exp2(row_m[e] - gm);
            row_m[e] = gm;
            row_l[e] *= alpha;
            for (int dt = 0; dt < DTiles; ++dt)
                out_acc[dt][e] *= alpha;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct)
                score_cache[ct][e] = fast_exp2(score_cache[ct][e] - gm);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float partial_sm = 0.0f;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct)
                partial_sm += score_cache[ct][e];
            partial_sm += fast_shfl_xor(partial_sm, 8);
            partial_sm += fast_shfl_xor(partial_sm, 4);
            partial_sm += fast_shfl_xor(partial_sm, 2);
            partial_sm += fast_shfl_xor(partial_sm, 1);
            row_l[e] += partial_sm;
        }

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            __half* p_base = p_lds + wave * 16 * PStride;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int p_row = 2 * e + (lane >> 4);
                const int p_col = ct * 16 + (lane & 15);
                p_base[p_row * PStride + p_col] = __float2half_rn(score_cache[ct][e]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            const int my_row = lane & 15;
            const __half* p_row_base = p_lds + wave * 16 * PStride + my_row * PStride + ct * 16;
            const unsigned int* p_u32 = reinterpret_cast<const unsigned int*>(p_row_base);
            v16h p_frag;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned int word = p_u32[k];
                p_frag[2 * k] = reinterpret_cast<__half&>(word);
                unsigned int hi = word >> 16;
                p_frag[2 * k + 1] = reinterpret_cast<__half&>(hi);
            }
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_col = dt * BK + acc_col(lane);
                const char* v_base = reinterpret_cast<const char*>(v_tile) + ct * BK * VStride * 2;
                const v16h v_frag = load_fp16_col_frag(v_base, d_col, VStride * 2);
                out_acc[dt] = wmma_f32_f16(p_frag, v_frag, out_acc[dt]);
            }
        }
        __syncthreads();

        if (has_next) {
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 8;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
            for (int i = 0; i < VPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < VVecsTotal) {
                    const int n = vec / VVecsPerRow;
                    const int d = (vec - n * VVecsPerRow) * 8;
                    const __hip_bfloat16* vsrc = reinterpret_cast<const __hip_bfloat16*>(&v_prefetch[i]);
                    __half* vdst = &v_tile[n * VStride + d];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        vdst[j] = __float2half_rn(__bfloat162float(vsrc[j]));
                    }
                }
            }
            __syncthreads();
        }
    }

    const int col = acc_col(lane);
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int d = dt * BK + col;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int64_t q_idx = q_start + acc_row(lane, e);
            if (q_idx < qo_len) {
                const float inv_l = (row_l[e] > 0.0f) ? (1.0f / row_l[e]) : 0.0f;
                const int64_t o_off = (tensor_layout == kHND) ?
                    (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                    (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
                output[o_off] = (unsigned short)(__float_as_uint(out_acc[dt][e] * inv_l) >> 16);
            }
        }
    }
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void bf16_attn_kernel_wpe1(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v, unsigned short* __restrict__ output,
    const int64_t batch_size, const int64_t qo_len, const int64_t kv_len,
    const int64_t num_qo_heads, const int64_t num_kv_heads,
    const int64_t q_stride_b, const int64_t q_stride_n, const int64_t q_stride_h,
    const int64_t k_stride_b, const int64_t k_stride_n, const int64_t k_stride_h,
    const int64_t v_stride_b, const int64_t v_stride_n, const int64_t v_stride_h,
    const int64_t o_stride_b, const int64_t o_stride_n, const int64_t o_stride_h,
    const float sm_scale_log2e, const int tensor_layout) {
    bf16_attn_kernel_impl<HeadDim, IsCausal, BLOCK_M, BLOCK_N>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h,
        sm_scale_log2e, tensor_layout);
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void bf16_attn_kernel_wpe2(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v, unsigned short* __restrict__ output,
    const int64_t batch_size, const int64_t qo_len, const int64_t kv_len,
    const int64_t num_qo_heads, const int64_t num_kv_heads,
    const int64_t q_stride_b, const int64_t q_stride_n, const int64_t q_stride_h,
    const int64_t k_stride_b, const int64_t k_stride_n, const int64_t k_stride_h,
    const int64_t v_stride_b, const int64_t v_stride_n, const int64_t v_stride_h,
    const int64_t o_stride_b, const int64_t o_stride_n, const int64_t o_stride_h,
    const float sm_scale_log2e, const int tensor_layout) {
    bf16_attn_kernel_impl<HeadDim, IsCausal, BLOCK_M, BLOCK_N>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h,
        sm_scale_log2e, tensor_layout);
}

}

Tensor mean_seq_gfx11(Tensor input, int64_t tensor_layout) {
    const int64_t batch = input.size(0);
    const int64_t heads = (tensor_layout == kHND) ? input.size(1) : input.size(2);
    const int64_t seq_len = (tensor_layout == kHND) ? input.size(2) : input.size(1);
    const int64_t head_dim = input.size(3);

    const int64_t in_stride_b = input.stride(0);
    const int64_t in_stride_n = (tensor_layout == kHND) ? input.stride(2) : input.stride(1);
    const int64_t in_stride_h = (tensor_layout == kHND) ? input.stride(1) : input.stride(2);

    Tensor output = new_empty_like(input, {batch, heads, head_dim}, input.scalar_type());
    const hipStream_t stream = current_hip_stream(input);
    dim3 block(256);
    dim3 grid((head_dim + 15) / 16, heads, batch);

    if (input.scalar_type() == ScalarType::Half) {
        mean_hnd_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            seq_len, heads, head_dim,
            in_stride_b, in_stride_n, in_stride_h);
    } else {
        mean_hnd_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
            seq_len, heads, head_dim,
            in_stride_b, in_stride_n, in_stride_h);
    }
    return output;
}

std::vector<Tensor> quant_qk_int8_gfx11(
    Tensor query, Tensor key, Tensor key_mean,
    int64_t tensor_layout, double sm_scale) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t q_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    Tensor q_int8 = new_empty_like(query, {batch, q_heads, q_len, head_dim}, ScalarType::Char);
    Tensor k_int8 = new_empty_like(key, {batch, kv_heads, kv_len, head_dim}, ScalarType::Char);

    const int q_groups = (q_len + MIN_BLK_Q - 1) / MIN_BLK_Q;
    const int k_groups = (kv_len + MIN_BLK_K - 1) / MIN_BLK_K;

    Tensor q_scale = new_empty_like(query, {batch, q_heads, q_groups}, ScalarType::Float);
    Tensor k_scale = new_empty_like(key, {batch, kv_heads, k_groups}, ScalarType::Float);

    const hipStream_t stream = current_hip_stream(query);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;
    const bool has_mean = key_mean.numel() > 0;

    const int64_t q_sb = query.stride(0);
    const int64_t q_sn = (tensor_layout == kHND) ? query.stride(2) : query.stride(1);
    const int64_t q_sh = (tensor_layout == kHND) ? query.stride(1) : query.stride(2);
    const int64_t k_sb = key.stride(0);
    const int64_t k_sn = (tensor_layout == kHND) ? key.stride(2) : key.stride(1);
    const int64_t k_sh = (tensor_layout == kHND) ? key.stride(1) : key.stride(2);

    dim3 block(256);
    dim3 grid(q_groups + k_groups, max(q_heads, kv_heads), batch);

    if (query.scalar_type() == ScalarType::Half) {
        if (head_dim == 64) {
            quant_qk_int8_hnd_kernel<__half, 64><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(query.data_ptr()),
                reinterpret_cast<const __half*>(key.data_ptr()),
                has_mean ? reinterpret_cast<const __half*>(key_mean.data_ptr()) : nullptr,
                reinterpret_cast<int8_t*>(q_int8.data_ptr()),
                reinterpret_cast<int8_t*>(k_int8.data_ptr()),
                reinterpret_cast<float*>(q_scale.data_ptr()),
                reinterpret_cast<float*>(k_scale.data_ptr()),
                batch, q_heads, kv_heads, q_len, kv_len,
                q_groups, k_groups, sm_scale_log2e,
                q_sb, q_sn, q_sh, k_sb, k_sn, k_sh);
        } else {
            quant_qk_int8_hnd_kernel<__half, 128><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(query.data_ptr()),
                reinterpret_cast<const __half*>(key.data_ptr()),
                has_mean ? reinterpret_cast<const __half*>(key_mean.data_ptr()) : nullptr,
                reinterpret_cast<int8_t*>(q_int8.data_ptr()),
                reinterpret_cast<int8_t*>(k_int8.data_ptr()),
                reinterpret_cast<float*>(q_scale.data_ptr()),
                reinterpret_cast<float*>(k_scale.data_ptr()),
                batch, q_heads, kv_heads, q_len, kv_len,
                q_groups, k_groups, sm_scale_log2e,
                q_sb, q_sn, q_sh, k_sb, k_sn, k_sh);
        }
    } else {
        if (head_dim == 64) {
            quant_qk_int8_hnd_kernel<__hip_bfloat16, 64><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()),
                reinterpret_cast<const __hip_bfloat16*>(key.data_ptr()),
                has_mean ? reinterpret_cast<const __hip_bfloat16*>(key_mean.data_ptr()) : nullptr,
                reinterpret_cast<int8_t*>(q_int8.data_ptr()),
                reinterpret_cast<int8_t*>(k_int8.data_ptr()),
                reinterpret_cast<float*>(q_scale.data_ptr()),
                reinterpret_cast<float*>(k_scale.data_ptr()),
                batch, q_heads, kv_heads, q_len, kv_len,
                q_groups, k_groups, sm_scale_log2e,
                q_sb, q_sn, q_sh, k_sb, k_sn, k_sh);
        } else {
            quant_qk_int8_hnd_kernel<__hip_bfloat16, 128><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()),
                reinterpret_cast<const __hip_bfloat16*>(key.data_ptr()),
                has_mean ? reinterpret_cast<const __hip_bfloat16*>(key_mean.data_ptr()) : nullptr,
                reinterpret_cast<int8_t*>(q_int8.data_ptr()),
                reinterpret_cast<int8_t*>(k_int8.data_ptr()),
                reinterpret_cast<float*>(q_scale.data_ptr()),
                reinterpret_cast<float*>(k_scale.data_ptr()),
                batch, q_heads, kv_heads, q_len, kv_len,
                q_groups, k_groups, sm_scale_log2e,
                q_sb, q_sn, q_sh, k_sb, k_sn, k_sh);
        }
    }
    return {q_int8, q_scale, k_int8, k_scale};
}

Tensor qk_int8_sv_bf16_attn_gfx11(
    Tensor query, Tensor key, Tensor value, Tensor output,
    Tensor q_scale, Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, double sm_scale) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = query.size(1);
    const int64_t kv_heads = key.size(1);
    const int64_t qo_len = query.size(2);
    const int64_t kv_len = key.size(2);
    const int64_t head_dim = query.size(3);

    const hipStream_t stream = current_hip_stream(query);

    const int64_t q_stride_b = query.stride(0);
    const int64_t q_stride_n = query.stride(2);
    const int64_t q_stride_h = query.stride(1);
    const int64_t k_stride_b = key.stride(0);
    const int64_t k_stride_n = key.stride(2);
    const int64_t k_stride_h = key.stride(1);
    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = (tensor_layout == kHND) ? value.stride(2) : value.stride(1);
    const int64_t v_stride_h = (tensor_layout == kHND) ? value.stride(1) : value.stride(2);
    const int64_t o_stride_b = output.stride(0);
    const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
    const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);
    const int64_t qs_stride_b = q_scale.stride(0);
    const int64_t qs_stride_h = q_scale.stride(1);
    const int64_t ks_stride_b = k_scale.stride(0);
    const int64_t ks_stride_h = k_scale.stride(1);

    #define LAUNCH_ATTN(HD, CAUSAL, BM, BN, VTYPE) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            attn_kernel_wpe1<HD, CAUSAL, BM, BN, VTYPE><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const int8_t*>(query.data_ptr()), \
                reinterpret_cast<const int8_t*>(key.data_ptr()), \
                reinterpret_cast<const VTYPE*>(value.data_ptr()), \
                output.data_ptr(), \
                reinterpret_cast<const float*>(q_scale.data_ptr()), \
                reinterpret_cast<const float*>(k_scale.data_ptr()), \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                static_cast<int>(tensor_layout)); \
        } while(0)

    #define LAUNCH_ATTN_WPE2(HD, CAUSAL, BM, BN, VTYPE) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            attn_kernel_wpe2<HD, CAUSAL, BM, BN, VTYPE><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const int8_t*>(query.data_ptr()), \
                reinterpret_cast<const int8_t*>(key.data_ptr()), \
                reinterpret_cast<const VTYPE*>(value.data_ptr()), \
                output.data_ptr(), \
                reinterpret_cast<const float*>(q_scale.data_ptr()), \
                reinterpret_cast<const float*>(k_scale.data_ptr()), \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                static_cast<int>(tensor_layout)); \
        } while(0)

    if (value.scalar_type() == ScalarType::BFloat16) {
        #define VT __hip_bfloat16
    } else {
        #define VT __half
    }

    if (head_dim == 64) {
        const bool is_self_attn = (qo_len == kv_len);
        if (is_self_attn) {
            if (is_causal) { LAUNCH_ATTN(64, true, 64, 64, VT); }
            else { LAUNCH_ATTN(64, false, 64, 64, VT); }
        } else if (kv_len <= 77) {
            if (is_causal) { LAUNCH_ATTN(64, true, 64, 16, VT); }
            else { LAUNCH_ATTN(64, false, 64, 16, VT); }
        } else {
            if (is_causal) { LAUNCH_ATTN(64, true, 64, 32, VT); }
            else { LAUNCH_ATTN(64, false, 64, 32, VT); }
        }
    } else {
        if (is_causal) { LAUNCH_ATTN_WPE2(128, true, 64, 32, VT); }
        else { LAUNCH_ATTN_WPE2(128, false, 64, 32, VT); }
    }
    #undef VT
    #undef LAUNCH_ATTN
    #undef LAUNCH_ATTN_WPE2

    return output;
}

Tensor fp16_attn_gfx11(
    Tensor query, Tensor key, Tensor value, Tensor output,
    int64_t tensor_layout, int64_t is_causal, double sm_scale) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t qo_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    const hipStream_t stream = current_hip_stream(query);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;

    const int64_t q_stride_b = query.stride(0);
    const int64_t q_stride_n = (tensor_layout == kHND) ? query.stride(2) : query.stride(1);
    const int64_t q_stride_h = (tensor_layout == kHND) ? query.stride(1) : query.stride(2);
    const int64_t k_stride_b = key.stride(0);
    const int64_t k_stride_n = (tensor_layout == kHND) ? key.stride(2) : key.stride(1);
    const int64_t k_stride_h = (tensor_layout == kHND) ? key.stride(1) : key.stride(2);
    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = (tensor_layout == kHND) ? value.stride(2) : value.stride(1);
    const int64_t v_stride_h = (tensor_layout == kHND) ? value.stride(1) : value.stride(2);
    const int64_t o_stride_b = output.stride(0);
    const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
    const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);

    #define LAUNCH_FP16(HD, CAUSAL, BM, BN) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            fp16_attn_kernel_wpe2<HD, CAUSAL, BM, BN><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const __half*>(query.data_ptr()), \
                reinterpret_cast<const __half*>(key.data_ptr()), \
                reinterpret_cast<const __half*>(value.data_ptr()), \
                reinterpret_cast<__half*>(output.data_ptr()), \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                sm_scale_log2e, static_cast<int>(tensor_layout)); \
        } while(0)

    if (head_dim == 64) {
        if (kv_len <= 128) {
            if (is_causal) { LAUNCH_FP16(64, true, 64, 16); }
            else { LAUNCH_FP16(64, false, 64, 16); }
        } else {
            const bool is_self_attn = (qo_len == kv_len);
            if (is_self_attn) {
                if (is_causal) { LAUNCH_FP16(64, true, 64, 64); }
                else { LAUNCH_FP16(64, false, 64, 64); }
            } else {
                if (is_causal) { LAUNCH_FP16(64, true, 64, 32); }
                else { LAUNCH_FP16(64, false, 64, 32); }
            }
        }
    } else {
        if (is_causal) { LAUNCH_FP16(128, true, 64, 32); }
        else { LAUNCH_FP16(128, false, 64, 32); }
    }
    #undef LAUNCH_FP16

    return output;
}

Tensor bf16_attn_gfx11(
    Tensor query, Tensor key, Tensor value, Tensor output,
    int64_t tensor_layout, int64_t is_causal, double sm_scale) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t qo_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    const hipStream_t stream = current_hip_stream(query);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;

    const int64_t q_stride_b = query.stride(0);
    const int64_t q_stride_n = (tensor_layout == kHND) ? query.stride(2) : query.stride(1);
    const int64_t q_stride_h = (tensor_layout == kHND) ? query.stride(1) : query.stride(2);
    const int64_t k_stride_b = key.stride(0);
    const int64_t k_stride_n = (tensor_layout == kHND) ? key.stride(2) : key.stride(1);
    const int64_t k_stride_h = (tensor_layout == kHND) ? key.stride(1) : key.stride(2);
    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = (tensor_layout == kHND) ? value.stride(2) : value.stride(1);
    const int64_t v_stride_h = (tensor_layout == kHND) ? value.stride(1) : value.stride(2);
    const int64_t o_stride_b = output.stride(0);
    const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
    const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);

    #define LAUNCH_BF16(HD, CAUSAL, BM, BN) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            bf16_attn_kernel_wpe2<HD, CAUSAL, BM, BN><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()), \
                reinterpret_cast<const __hip_bfloat16*>(key.data_ptr()), \
                reinterpret_cast<const __hip_bfloat16*>(value.data_ptr()), \
                reinterpret_cast<unsigned short*>(output.data_ptr()), \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                sm_scale_log2e, static_cast<int>(tensor_layout)); \
        } while(0)

    if (head_dim == 64) {
        if (kv_len <= 128) {
            if (is_causal) { LAUNCH_BF16(64, true, 64, 16); }
            else { LAUNCH_BF16(64, false, 64, 16); }
        } else {
            const bool is_self_attn_bf = (qo_len == kv_len);
            if (is_self_attn_bf) {
                if (is_causal) { LAUNCH_BF16(64, true, 64, 64); }
                else { LAUNCH_BF16(64, false, 64, 64); }
            } else {
                if (is_causal) { LAUNCH_BF16(64, true, 64, 32); }
                else { LAUNCH_BF16(64, false, 64, 32); }
            }
        }
    } else {
        if (is_causal) { LAUNCH_BF16(128, true, 64, 32); }
        else { LAUNCH_BF16(128, false, 64, 32); }
    }
    #undef LAUNCH_BF16

    return output;
}
