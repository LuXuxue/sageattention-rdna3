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
#include <cstdlib>
#include <optional>
#include <type_traits>
#include <vector>

using torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

namespace {

constexpr int kNHD = 0;
constexpr int kHND = 1;
constexpr float kLog2e = 1.4426950408889634f;

// 实验: V 全局转置存储 (V_T [B,H,D,N]) + PV 改 out = P @ V (B operand 行读)
// 注意: 本宏只应在 host 代码 (dispatch) 使用 #if; 模板 __device__ 函数体内
//       用 #if 会触发 hipcc 解析 bug (wpe1 undeclared), 故 PV 段为无条件代码
#ifndef SAGEATTN_VT_GLOBAL
#define SAGEATTN_VT_GLOBAL 0
#endif

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
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ __half from_float_f16(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __hip_bfloat16 from_float_bf16(float v) { return __float2bfloat16(v); }

// QK 向量元素类型转换 (v16h 元素为 _Float16, v16bf 元素为 __bf16)
__device__ __forceinline__ _Float16 to_qk_elem(const __half v) {
    return static_cast<_Float16>(__half2float(v));
}
__device__ __forceinline__ __bf16 to_qk_elem(const __hip_bfloat16 v) {
    return static_cast<__bf16>(__bfloat162float(v));
}

template <typename QK_DTYPE>
__device__ __forceinline__ auto qk_zero() {
    if constexpr (std::is_same<QK_DTYPE, __half>::value) {
        return static_cast<_Float16>(0.0f);
    } else {
        return static_cast<__bf16>(0.0f);
    }
}

__device__ __forceinline__ int8_t float_to_int8(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int32_t rounded;
    asm volatile("v_cvt_i32_f32 %[dst], %[src]" : [dst] "=v"(rounded) : [src] "v"(x));
    rounded = rounded > 127 ? 127 : rounded;
    rounded = rounded < -128 ? -128 : rounded;
    return static_cast<int8_t>(rounded);
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
            } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
                mean_out[(b * heads + h) * head_dim + mean_d] = from_float_bf16(value);
            } else {
                mean_out[(b * heads + h) * head_dim + mean_d] = value;
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

// V [B, N, H, D] (NHD) / [B, H, N, D] (HND) -> V_T [B, H, D, N] (contiguous, n 连续)
__global__ void v_transpose_kernel(
    const __half* __restrict__ v,
    __half* __restrict__ v_t,
    const int64_t batch_size,
    const int64_t seq_len,
    const int64_t num_heads,
    const int64_t head_dim,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int tensor_layout) {
    const int64_t total8 = batch_size * num_heads * seq_len * (head_dim / 8);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total8; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t d8 = i % (head_dim / 8);
        const int64_t n = (i / (head_dim / 8)) % seq_len;
        const int64_t h = (i / (seq_len * head_dim / 8)) % num_heads;
        const int64_t b = i / (seq_len * head_dim / 8 * num_heads);
        const int64_t d0 = d8 * 8;
        const int64_t v_off = (tensor_layout == kHND) ?
            (b * v_stride_b + h * v_stride_h + n * v_stride_n + d0) :
            (b * v_stride_b + n * v_stride_n + h * v_stride_h + d0);
        const int64_t vt_off = ((b * num_heads + h) * head_dim + d0) * seq_len + n;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v_t[vt_off + j * seq_len] = v[v_off + j];
        }
    }
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE = __half,
          typename OUT_DTYPE = V_DTYPE>
__device__ __forceinline__ void attn_kernel_impl_t(
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

    __shared__ int8_t k_tile[BLOCK_N * KStride];
    __shared__ __half v_tile[BLOCK_N * VStride];

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

    // q fragment (转置 QK 的 B operand = q^T): lane L 持有 q 行 (L&15) 的 16 i8
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

    // 最后一个 block 的越界 wave (q_start >= qo_len) 不产生输出, 守卫避免 q_scale 越界读
    const float qs = (q_start < qo_len)
        ? q_scale[b * qs_stride_b + hq * qs_stride_h +
                  static_cast<int>(q_start / MIN_BLK_Q)]
        : 0.0f;

    // out_acc (转置 PV 输出 = out^T): lane L 持有 out 行 (L&15) 的 8 D 列 (偶/奇)
    v8f out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    // per-row 状态: 每 lane 1 行 (行 = L&15), lane L 与 L^16 冗余但一致
    float row_m = -FLT_MAX * 0.5f;
    float row_l = 0.0f;

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
    if (!SAGEATTN_VT_GLOBAL) {
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
    }
    __syncthreads();

    const int hw = lane >> 4;
    const int m_row = lane & 15;

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
    if (!SAGEATTN_VT_GLOBAL) {
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
        }

        float score_cache[ColTiles][8];

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8i score_acc = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                // 转置 QK: A = k (行 = BLOCK_N 维), B = q^T
                const int32_t_v4 k_frag = load_i8_frag(
                    k_tile + ct * BK * KStride, m_row, dt * BK, KStride);
                score_acc = wmma_i32_iu8(k_frag, q_frag[dt], score_acc);
            }

            const int64_t k_col_start = kb_base + ct * BK;
            const int k_scale_idx = static_cast<int>(k_col_start / MIN_BLK_K);
            const float ks_val = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
            const float score_scale = qs * ks_val;
            const int q_row_idx = static_cast<int>(q_start) + m_row;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = static_cast<int>(k_col_start) + 2 * e + hw;
                float s = static_cast<float>(score_acc[e]) * score_scale;
                if constexpr (IsCausal) {
                    if (col > q_row_idx) s = -FLT_MAX * 0.5f;
                }
                if (col >= kv_len) s = -FLT_MAX * 0.5f;
                score_cache[ct][e] = s;
            }
        }

        // ---- per-row max: 局部归约 (同行偶/奇列) -> permlanex16 合并 ----
        float local_mx = score_cache[0][0];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                local_mx = fmaxf(local_mx, score_cache[ct][e]);
            }
        }
        float gm = fmaxf(row_m, local_mx);
        gm = fmaxf(gm, permlanex16(gm));
        const float alpha = (row_l == 0.0f) ? 0.0f : fast_exp2(row_m - gm);
        row_m = gm;
        row_l *= alpha;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) out_acc[dt] *= alpha;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                score_cache[ct][e] = fast_exp2(score_cache[ct][e] - gm);
            }
        }

        // ---- per-row sum: 局部累加 -> permlanex16 合并 ----
        float partial_sm = 0.0f;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial_sm += score_cache[ct][e];
            }
        }
        row_l += partial_sm + permlanex16(partial_sm);

        // ---- P fragment 寄存器组装 + PV ----
        // SAGEATTN_VT_GLOBAL=1: V 已转置为 V_T [B,H,D,N], 用 out = P @ V (B operand 行读 b128)
        // SAGEATTN_VT_GLOBAL=0: 原转置 PV (out^T = V^T @ P^T, LDS v_frag 列读)
        // 注意: 用运行时 if (编译器 DCE), 不能用 #if/if constexpr (模板体内会触发 hipcc 解析 bug)
        if (SAGEATTN_VT_GLOBAL) {
        // out^T = V^T @ P^T: A = V^T (V_T 行 L&15 的 16 连续 n = 行读 b128), B = P^T (p_frag)
        // C = out^T: lane L 持 out^T[2e+hw][L&15] = out[L&15][2e+hw] (转置解释, 匹配写回)
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int64_t vt_off =
                    ((b * num_kv_heads + hkv) * HeadDim + (dt * BK + m_row)) * kv_len + (kb_base + ct * BK);
                const v16h v_frag_t = *reinterpret_cast<const v16h*>(v + vt_off);
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(v_frag_t, p_frag, out_acc[dt]);
            }
        }
        } else {
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_col = dt * BK + m_row;
                const char* v_base = reinterpret_cast<const char*>(v_tile) + ct * BK * VStride * 2;
                const v16h v_frag = sageattn_gfx11::load_fp16_col_frag(v_base, d_col, VStride * 2);
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(v_frag, p_frag, out_acc[dt]);
            }
        }
        }

        if (has_next) {
            // 写前 barrier: 确保所有 warp 完成当前 tile 的 QK/PV 读,
            // 否则快 warp 覆盖慢 warp 正在读的 k_tile/v_tile (race)
            __syncthreads();
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
    if (!SAGEATTN_VT_GLOBAL) {
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
    }
            __syncthreads();
        }
    }

    // ---- 写回: out 行 (L&15) 的 D 列 {dt*16 + 2e + hw} ----
    const int64_t q_idx = q_start + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = dt * BK + 2 * e + hw;
                const int64_t o_off = (tensor_layout == kHND) ?
                    (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                    (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
                // 输出 dtype 与 V 输入 dtype 可分离 (V 可预转 fp16, 输出直接写 bf16)
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    reinterpret_cast<__half*>(output)[o_off] = __float2half_rn(out_acc[dt][e] * inv_l);
                } else {
                    reinterpret_cast<__hip_bfloat16*>(output)[o_off] =
                        from_float_bf16(out_acc[dt][e] * inv_l);
                }
            }
        }
    }
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE = __half, typename OUT_DTYPE = V_DTYPE>
__device__ __forceinline__ void attn_kernel_impl_32_t(
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

    constexpr int WARPS = BLOCK_M / (2 * RM);  // 每 warp 32 行
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / BK;
    constexpr int ColTiles = BLOCK_N / BK;
    constexpr int KStride = HeadDim + LDS_PAD;
    constexpr int VStride = HeadDim + LDS_PAD;

    __shared__ int8_t k_tile[BLOCK_N * KStride];
    __shared__ __half v_tile[BLOCK_N * VStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    if (b >= batch_size || hq >= num_qo_heads || q_base >= qo_len) return;

    const int64_t hkv = hq / (num_qo_heads / num_kv_heads);
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * (2 * RM);  // 每 warp 2 子块

    using namespace sageattn_gfx11;

    // q fragment x2 (转置 QK 的 B operand = q^T): 每 warp 2 个子块, lane L 持 q 行 (L&15) 的 16 i8
    int32_t_v4 q_frag[2][DTiles];
    const int q_row = lane & 15;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            const int64_t q_idx = q_start + s * RM + q_row;
            if (q_idx < qo_len) {
                const int d_base = dt * BK;
                const int64_t q_off = (tensor_layout == kHND) ?
                    (b * q_stride_b + hq * q_stride_h + q_idx * q_stride_n + d_base) :
                    (b * q_stride_b + q_idx * q_stride_n + hq * q_stride_h + d_base);
                q_frag[s][dt] = *reinterpret_cast<const int32_t_v4*>(q + q_off);
            } else {
                q_frag[s][dt] = int32_t_v4{0, 0, 0, 0};
            }
        }
    }

    // 最后一个 block 的越界 wave (q_start >= qo_len) 不产生输出, 守卫避免 q_scale 越界读
    float qs[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int64_t qs_idx = q_start + s * RM;
        qs[s] = (qs_idx < qo_len)
            ? q_scale[b * qs_stride_b + hq * qs_stride_h +
                      static_cast<int>(qs_idx / MIN_BLK_Q)]
            : 0.0f;
    }

    // out_acc x2 (转置 PV 输出 = out^T): lane L 持有 out 行 (L&15) 的 8 D 列 (偶/奇)
    v8f out_acc[2][DTiles];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            out_acc[s][dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    // per-row 状态 x2: 每 lane 1 行 (行 = L&15), lane L 与 L^16 冗余但一致
    float row_m[2] = {-FLT_MAX * 0.5f, -FLT_MAX * 0.5f};
    float row_l[2] = {0.0f, 0.0f};

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
    if (!SAGEATTN_VT_GLOBAL) {
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
    }
    __syncthreads();

    const int hw = lane >> 4;
    const int m_row = lane & 15;

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
    if (!SAGEATTN_VT_GLOBAL) {
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
        }

        float score_cache[2][ColTiles][8];

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            // 2 子块共享 k_frag: 每迭代 2 个独立 WMMA (ILP 提升)
            v8i score_acc0 = v8i{0, 0, 0, 0, 0, 0, 0, 0};
            v8i score_acc1 = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                // 转置 QK: A = k (行 = BLOCK_N 维), B = q^T
                const int32_t_v4 k_frag = load_i8_frag(
                    k_tile + ct * BK * KStride, m_row, dt * BK, KStride);
                score_acc0 = wmma_i32_iu8(k_frag, q_frag[0][dt], score_acc0);
                score_acc1 = wmma_i32_iu8(k_frag, q_frag[1][dt], score_acc1);
            }

            const int64_t k_col_start = kb_base + ct * BK;
            const int k_scale_idx = static_cast<int>(k_col_start / MIN_BLK_K);
            const float ks_val = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];

#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const v8i score_acc = (s == 0) ? score_acc0 : score_acc1;
                const float score_scale = qs[s] * ks_val;
                const int q_row_idx = static_cast<int>(q_start) + s * RM + m_row;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = static_cast<int>(k_col_start) + 2 * e + hw;
                    float sv = static_cast<float>(score_acc[e]) * score_scale;
                    if constexpr (IsCausal) {
                        if (col > q_row_idx) sv = -FLT_MAX * 0.5f;
                    }
                    if (col >= kv_len) sv = -FLT_MAX * 0.5f;
                    score_cache[s][ct][e] = sv;
                }
            }
        }

        // ---- per-row max/exp/sum (online softmax) x2 子块 ----
#pragma unroll
        for (int s = 0; s < 2; ++s) {
        float local_mx = score_cache[s][0][0];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                local_mx = fmaxf(local_mx, score_cache[s][ct][e]);
            }
        }
        float gm = fmaxf(row_m[s], local_mx);
        gm = fmaxf(gm, permlanex16(gm));
        const float alpha = (row_l[s] == 0.0f) ? 0.0f : fast_exp2(row_m[s] - gm);
        row_m[s] = gm;
        row_l[s] *= alpha;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) out_acc[s][dt] *= alpha;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                score_cache[s][ct][e] = fast_exp2(score_cache[s][ct][e] - gm);
            }
        }

        // ---- per-row sum: 局部累加 -> permlanex16 合并 ----
        float partial_sm = 0.0f;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial_sm += score_cache[s][ct][e];
            }
        }
        row_l[s] += partial_sm + permlanex16(partial_sm);
        }

        // ---- P fragment 寄存器组装 + PV ----
        // SAGEATTN_VT_GLOBAL=1: V 已转置为 V_T [B,H,D,N], 用 out = P @ V (B operand 行读 b128)
        // SAGEATTN_VT_GLOBAL=0: 原转置 PV (out^T = V^T @ P^T, LDS v_frag 列读)
        // 注意: 用运行时 if (编译器 DCE), 不能用 #if/if constexpr (模板体内会触发 hipcc 解析 bug)
        // ---- PV x2 子块 (2 个独立 WMMA, ILP 提升) ----
        if (SAGEATTN_VT_GLOBAL) {
        // out^T = V^T @ P^T: A = V^T (V_T 行读), B = P^T (p_frag)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
                float p_vals[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[s][ct][e];
                const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                    const int64_t vt_off =
                        ((b * num_kv_heads + hkv) * HeadDim + (dt * BK + m_row)) * kv_len + (kb_base + ct * BK);
                    const v16h v_frag_t = *reinterpret_cast<const v16h*>(v + vt_off);
                    out_acc[s][dt] = sageattn_gfx11::wmma_f32_f16(v_frag_t, p_frag, out_acc[s][dt]);
                }
            }
        }
        } else {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
                float p_vals[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[s][ct][e];
                const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                    const int d_col = dt * BK + m_row;
                    const char* v_base = reinterpret_cast<const char*>(v_tile) + ct * BK * VStride * 2;
                    const v16h v_frag = sageattn_gfx11::load_fp16_col_frag(v_base, d_col, VStride * 2);
                    out_acc[s][dt] = sageattn_gfx11::wmma_f32_f16(v_frag, p_frag, out_acc[s][dt]);
                }
            }
        }
        }

        if (has_next) {
            // 写前 barrier: 确保所有 warp 完成当前 tile 的 QK/PV 读,
            // 否则快 warp 覆盖慢 warp 正在读的 k_tile/v_tile (race)
            __syncthreads();
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
    if (!SAGEATTN_VT_GLOBAL) {
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
    }
            __syncthreads();
        }
    }

    // ---- 写回 x2 子块: out 行 (L&15) 的 D 列 {dt*16 + 2e + hw} ----
#pragma unroll
    for (int s = 0; s < 2; ++s) {
    const int64_t q_idx = q_start + s * RM + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l[s] > 0.0f) ? (1.0f / row_l[s]) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = dt * BK + 2 * e + hw;
                const int64_t o_off = (tensor_layout == kHND) ?
                    (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                    (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
                // 输出 dtype 与 V 输入 dtype 可分离 (V 可预转 fp16, 输出直接写 bf16)
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    reinterpret_cast<__half*>(output)[o_off] = __float2half_rn(out_acc[s][dt][e] * inv_l);
                } else {
                    reinterpret_cast<__hip_bfloat16*>(output)[o_off] =
                        from_float_bf16(out_acc[s][dt][e] * inv_l);
                }
            }
        }
    }
    }
    }

// ===== 每 warp 32 行版本 (BM=128, 4 warps; 2 子块共享 k_frag, ILP 提升) =====
// launch wrapper (wpe1)
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE, typename OUT_DTYPE = V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 32 * 32, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void attn_kernel_wpe1_32_t(
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
    attn_kernel_impl_32_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE, OUT_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

// launch wrapper (wpe4 高 occupancy)
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE, typename OUT_DTYPE = V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 32 * 32, 4)
__attribute__((amdgpu_waves_per_eu(4, 4)))
void attn_kernel_wpe4_32_t(
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
    attn_kernel_impl_32_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE, OUT_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE, typename OUT_DTYPE = V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void attn_kernel_wpe1_t(
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
    attn_kernel_impl_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE, OUT_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

// 转置布局 kernel 的 launch wrapper (wpe2)
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE, typename OUT_DTYPE = V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void attn_kernel_wpe2_t(
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
    attn_kernel_impl_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE, OUT_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

// 转置布局 kernel 的 launch wrapper (wpe4: 高 occupancy 实验)
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N, typename V_DTYPE, typename OUT_DTYPE = V_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 4)))
void attn_kernel_wpe4_t(
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
    attn_kernel_impl_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, V_DTYPE, OUT_DTYPE>(
        q, k, v, output, q_scale, k_scale,
        batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h,
        k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h,
        o_stride_b, o_stride_n, o_stride_h,
        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,
        tensor_layout);
}

// fp16/bf16 direct 转置 kernel (Triton 风格): qk^T = k @ q^T, out^T = V^T @ P^T
// 与 int8 转置 kernel 相同策略: permlanex16 归约 + 寄存器 P fragment
// QK_DTYPE: __half (fp16 WMMA) 或 __hip_bfloat16 (bf16 WMMA); V 总是转 fp16 计算
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N,
          typename QK_DTYPE, typename V_DTYPE, typename OUT_DTYPE>
__device__ __forceinline__ void direct_attn_kernel_impl_t(
    const QK_DTYPE* __restrict__ q,
    const QK_DTYPE* __restrict__ k,
    const V_DTYPE* __restrict__ v,
    OUT_DTYPE* __restrict__ output,
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

    __shared__ QK_DTYPE k_tile[BLOCK_N * KStride];
    __shared__ __half v_tile[BLOCK_N * VStride];

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
    using QK_VEC = std::conditional_t<std::is_same<QK_DTYPE, __half>::value, v16h, v16bf>;

    // q fragment (转置 QK 的 B operand): lane L 持 Q 行 (L&15) 的 16 个值
    QK_VEC q_frag[DTiles];
    const int q_row = lane & 15;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        const int64_t q_idx = q_start + q_row;
        if (q_idx < qo_len) {
            const int d_base = dt * BK;
            const int64_t q_off = (tensor_layout == kHND) ?
                (b * q_stride_b + hq * q_stride_h + q_idx * q_stride_n + d_base) :
                (b * q_stride_b + q_idx * q_stride_n + hq * q_stride_h + d_base);
            const QK_DTYPE* src = q + q_off;
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = to_qk_elem(src[i]);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = qk_zero<QK_DTYPE>();
        }
    }

    // out_acc (转置 PV 输出 = out^T): lane L 持 out 行 (L&15) 的 8 D 列 (偶/奇)
    v8f out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    float row_m = -FLT_MAX * 0.5f;
    float row_l = 0.0f;

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

    const int hw = lane >> 4;
    const int m_row = lane & 15;

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
            v8f score_acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const QK_DTYPE* k_ptr = &k_tile[(ct * BK + m_row) * KStride + dt * BK];
                QK_VEC k_frag;
#pragma unroll
                for (int i = 0; i < 16; ++i) k_frag[i] = to_qk_elem(k_ptr[i]);
                if constexpr (std::is_same<QK_DTYPE, __half>::value) {
                    score_acc = wmma_f32_f16(k_frag, q_frag[dt], score_acc);
                } else {
                    score_acc = wmma_f32_bf16(k_frag, q_frag[dt], score_acc);
                }
            }

            const int64_t k_col_start = kb_base + ct * BK;
            const int q_row_idx = static_cast<int>(q_start) + m_row;

#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = static_cast<int>(k_col_start) + 2 * e + hw;
                float s = static_cast<float>(score_acc[e]) * sm_scale_log2e;
                if constexpr (IsCausal) {
                    if (col > q_row_idx) s = -FLT_MAX * 0.5f;
                }
                if (col >= kv_len) s = -FLT_MAX * 0.5f;
                score_cache[ct][e] = s;
            }
        }

        // ---- per-row max: 局部归约 -> permlanex16 合并 ----
        float local_mx = score_cache[0][0];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                local_mx = fmaxf(local_mx, score_cache[ct][e]);
            }
        }
        float gm = fmaxf(row_m, local_mx);
        gm = fmaxf(gm, permlanex16(gm));
        const float alpha = (row_l == 0.0f) ? 0.0f : fast_exp2(row_m - gm);
        row_m = gm;
        row_l *= alpha;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) out_acc[dt] *= alpha;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                score_cache[ct][e] = fast_exp2(score_cache[ct][e] - gm);
            }
        }

        // ---- per-row sum: 局部累加 -> permlanex16 合并 ----
        float partial_sm = 0.0f;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial_sm += score_cache[ct][e];
            }
        }
        row_l += partial_sm + permlanex16(partial_sm);

        // ---- P fragment 寄存器组装 + PV ----
        // SAGEATTN_VT_GLOBAL=1: V 已转置为 V_T, 用 out = P @ V (B operand 行读 b128)
        if (SAGEATTN_VT_GLOBAL) {
        // out^T = V^T @ P^T: A = V^T (V_T 行读), B = P^T (p_frag), C = out^T (匹配写回)
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int64_t vt_off =
                    ((b * num_kv_heads + hkv) * HeadDim + (dt * BK + m_row)) * kv_len + (kb_base + ct * BK);
                const v16h v_frag_t = *reinterpret_cast<const v16h*>(v + vt_off);
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(v_frag_t, p_frag, out_acc[dt]);
            }
        }
        } else {
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_col = dt * BK + m_row;
                const char* v_base = reinterpret_cast<const char*>(v_tile) + ct * BK * VStride * 2;
                const v16h v_frag = sageattn_gfx11::load_fp16_col_frag(v_base, d_col, VStride * 2);
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(v_frag, p_frag, out_acc[dt]);
            }
        }
        }

        if (has_next) {
            // 写前 barrier: 防止快 warp 覆盖慢 warp 正在读的 k_tile/v_tile
            __syncthreads();
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

    // ---- 写回: out 行 (L&15) 的 D 列 {dt*16 + 2e + hw} ----
    const int64_t q_idx = q_start + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = dt * BK + 2 * e + hw;
                const int64_t o_off = (tensor_layout == kHND) ?
                    (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                    (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    output[o_off] = from_float_f16(out_acc[dt][e] * inv_l);
                } else {
                    output[o_off] = from_float_bf16(out_acc[dt][e] * inv_l);
                }
            }
        }
    }
}

// fp16 direct 转置 kernel 的 launch wrapper (wpe2, 与旧 fp16 kernel 一致)
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void fp16_attn_kernel_wpe2_t(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ output,
    const int64_t batch_size, const int64_t qo_len, const int64_t kv_len,
    const int64_t num_qo_heads, const int64_t num_kv_heads,
    const int64_t q_stride_b, const int64_t q_stride_n, const int64_t q_stride_h,
    const int64_t k_stride_b, const int64_t k_stride_n, const int64_t k_stride_h,
    const int64_t v_stride_b, const int64_t v_stride_n, const int64_t v_stride_h,
    const int64_t o_stride_b, const int64_t o_stride_n, const int64_t o_stride_h,
    const float sm_scale_log2e, const int tensor_layout) {
    direct_attn_kernel_impl_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, __half, __half, __half>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h,
        sm_scale_log2e, tensor_layout);
}

// bf16 direct 转置 kernel 的 launch wrapper (wpe2)
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void bf16_attn_kernel_wpe2_t(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v, __hip_bfloat16* __restrict__ output,
    const int64_t batch_size, const int64_t qo_len, const int64_t kv_len,
    const int64_t num_qo_heads, const int64_t num_kv_heads,
    const int64_t q_stride_b, const int64_t q_stride_n, const int64_t q_stride_h,
    const int64_t k_stride_b, const int64_t k_stride_n, const int64_t k_stride_h,
    const int64_t v_stride_b, const int64_t v_stride_n, const int64_t v_stride_h,
    const int64_t o_stride_b, const int64_t o_stride_n, const int64_t o_stride_h,
    const float sm_scale_log2e, const int tensor_layout) {
    direct_attn_kernel_impl_t<HeadDim, IsCausal, BLOCK_M, BLOCK_N, __hip_bfloat16, __hip_bfloat16, __hip_bfloat16>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h,
        sm_scale_log2e, tensor_layout);
}

}  // namespace

// V [B,N,H,D] -> V_T [B,H,D,N] (contiguous)
Tensor v_transpose_gfx11(Tensor value, Tensor value_t, int64_t tensor_layout) {
    const int64_t batch = value.size(0);
    const int64_t heads = (tensor_layout == kHND) ? value.size(1) : value.size(2);
    const int64_t seq_len = (tensor_layout == kHND) ? value.size(2) : value.size(1);
    const int64_t head_dim = value.size(3);
    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = (tensor_layout == kHND) ? value.stride(2) : value.stride(1);
    const int64_t v_stride_h = (tensor_layout == kHND) ? value.stride(1) : value.stride(2);
    const hipStream_t stream = current_hip_stream(value);
    const int64_t total8 = batch * heads * seq_len * (head_dim / 8);
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((total8 + block.x - 1) / block.x));
    v_transpose_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(value.data_ptr()),
        reinterpret_cast<__half*>(value_t.data_ptr()),
        batch, seq_len, heads, head_dim,
        v_stride_b, v_stride_n, v_stride_h,
        static_cast<int>(tensor_layout));
    return value_t;
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
    } else if (input.scalar_type() == ScalarType::BFloat16) {
        mean_hnd_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
            seq_len, heads, head_dim,
            in_stride_b, in_stride_n, in_stride_h);
    } else {
        mean_hnd_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(output.data_ptr()),
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

Tensor qk_int8_sv_bf16_attn_gfx11_t(
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

    // 实验: SAGEATTN_INT8_WPE 选择 launch wrapper (1=wpe1 默认, 2=wpe2, 4=wpe4)
    const int wpe_sel = getenv("SAGEATTN_INT8_WPE") ? atoi(getenv("SAGEATTN_INT8_WPE")) : 1;
    // 实验: SAGEATTN_INT8_32 用每 warp 32 行 kernel (BM 恒 128, 4 warps)
    // 每 warp 32 行 kernel (BM 128, 4 warps, 2 子块共享 k_frag): D=64 self 默认启用 (实测快 7-10%)
    // SAGEATTN_INT8_32=0 可关闭
    const bool use_32w = getenv("SAGEATTN_INT8_32") ? atoi(getenv("SAGEATTN_INT8_32")) != 0 : true;
    constexpr int BLOCK_M_32 = 128;

    #define LAUNCH_ATTN_T(HD, CAUSAL, BM, BN, VTYPE, OTYPE, WPE) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            if (WPE == 2) { \
                attn_kernel_wpe2_t<HD, CAUSAL, BM, BN, VTYPE, OTYPE><<<grid, block, 0, stream>>>( \
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
            } else if (WPE == 4) { \
                attn_kernel_wpe4_t<HD, CAUSAL, BM, BN, VTYPE, OTYPE><<<grid, block, 0, stream>>>( \
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
            } else { \
                attn_kernel_wpe1_t<HD, CAUSAL, BM, BN, VTYPE, OTYPE><<<grid, block, 0, stream>>>( \
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
            } \
        } while(0)

    #define LAUNCH_ATTN_WPE2_T(HD, CAUSAL, BM, BN, VTYPE, OTYPE) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            attn_kernel_wpe2_t<HD, CAUSAL, BM, BN, VTYPE, OTYPE><<<grid, block, 0, stream>>>( \
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

    // V 输入与输出 dtype 分离:
    //   fp16 输入: V=__half (uint4 拷贝), OUT=__half (现状)
    //   bf16 输入(方案B): V=__half (core 预转 fp16, 带宽受限高效), OUT=__hip_bfloat16 (kernel 直接写 bf16)
    //   bf16 直通(备选): V=__hip_bfloat16 (kernel 内转换, 实测 1.6-5.5% 开销)
    const bool v_is_bf16 = (value.scalar_type() == ScalarType::BFloat16);
    const bool out_is_bf16 = (output.scalar_type() == ScalarType::BFloat16);

    #define LAUNCH_ATTN_ALL(VT, OT) \
        do { \
            if (head_dim == 64) { \
                const bool is_self_attn = (qo_len == kv_len); \
                if (is_self_attn) { \
                    if (use_32w) { \
                        dim3 block_32(BLOCK_M_32); \
                        dim3 grid_32((qo_len + 128 - 1) / 128, q_heads, batch); \
                        if (is_causal) { \
                            attn_kernel_wpe1_32_t<64, true, 128, 32, VT, OT><<<grid_32, block_32, 0, stream>>>( \
                                reinterpret_cast<const int8_t*>(query.data_ptr()), \
                                reinterpret_cast<const int8_t*>(key.data_ptr()), \
                                reinterpret_cast<const VT*>(value.data_ptr()), \
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
                        } else { \
                            attn_kernel_wpe1_32_t<64, false, 128, 32, VT, OT><<<grid_32, block_32, 0, stream>>>( \
                                reinterpret_cast<const int8_t*>(query.data_ptr()), \
                                reinterpret_cast<const int8_t*>(key.data_ptr()), \
                                reinterpret_cast<const VT*>(value.data_ptr()), \
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
                        } \
                    } else if (is_causal) { LAUNCH_ATTN_T(64, true, 128, 32, VT, OT, wpe_sel); } \
                    else { LAUNCH_ATTN_T(64, false, 128, 32, VT, OT, wpe_sel); } \
                } else if (kv_len <= 77) { \
                    if (is_causal) { LAUNCH_ATTN_T(64, true, 64, 16, VT, OT, wpe_sel); } \
                    else { LAUNCH_ATTN_T(64, false, 64, 16, VT, OT, wpe_sel); } \
                } else { \
                    if (is_causal) { LAUNCH_ATTN_T(64, true, 64, 32, VT, OT, wpe_sel); } \
                    else { LAUNCH_ATTN_T(64, false, 64, 32, VT, OT, wpe_sel); } \
                } \
            } else { \
                const int wpe128 = (wpe_sel == 1) ? 2 : wpe_sel; \
                if (is_causal) { LAUNCH_ATTN_T(128, true, 64, 32, VT, OT, wpe128); } \
                else { LAUNCH_ATTN_T(128, false, 64, 32, VT, OT, wpe128); } \
            } \
        } while(0)

    if (v_is_bf16) { LAUNCH_ATTN_ALL(__hip_bfloat16, __hip_bfloat16); }
    else if (out_is_bf16) { LAUNCH_ATTN_ALL(__half, __hip_bfloat16); }
    else { LAUNCH_ATTN_ALL(__half, __half); }
    #undef LAUNCH_ATTN_ALL
    #undef LAUNCH_ATTN_T
    #undef LAUNCH_ATTN_WPE2_T

    return output;
}

Tensor fp16_attn_gfx11_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, int64_t bm_sel) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t qo_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    const hipStream_t stream = current_hip_stream(query);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;

    // 实验: bm_sel 运行时选择 BM (0=默认, 1=32, 2=128, 3=强制64); env SAGEATTN_FP16_BM 可覆盖
    int bm_ov = (bm_sel == 1) ? 32 : (bm_sel == 2) ? 128 : (bm_sel == 3) ? 64 : 0;
    if (getenv("SAGEATTN_FP16_BM")) bm_ov = atoi(getenv("SAGEATTN_FP16_BM"));
    // 实验: SAGEATTN_FP16_BN 覆盖 D=64 direct 路径的 BN (0=默认, 16/32/64/128)
    const int bn_ov = getenv("SAGEATTN_FP16_BN") ? atoi(getenv("SAGEATTN_FP16_BN")) : 0;

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

    #define LAUNCH_FP16_T(HD, CAUSAL, BM, BN) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            fp16_attn_kernel_wpe2_t<HD, CAUSAL, BM, BN><<<grid, block, 0, stream>>>( \
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

    // is_causal 是运行时参数, 必须用 if/else 生成编译期模板常量
    #define LAUNCH_FP16_BN(HD, BM, BN) \
        do { \
            if (is_causal) { LAUNCH_FP16_T(HD, true, BM, BN); } \
            else { LAUNCH_FP16_T(HD, false, BM, BN); } \
        } while(0)

    // 实验 BM 覆盖: SAGEATTN_FP16_BM
    #define LAUNCH_FP16_BM(HD, BM, BN) \
        do { \
            if (bm_ov == 32) { LAUNCH_FP16_BN(HD, 32, BN); } \
            else if (bm_ov == 128) { LAUNCH_FP16_BN(HD, 128, BN); } \
            else { LAUNCH_FP16_BN(HD, BM, BN); } \
        } while(0)

    if (head_dim == 64) {
        if (kv_len <= 128) {
            // 默认 BN=16; 实验可用 SAGEATTN_FP16_BN 覆盖
            if (bn_ov == 32) { LAUNCH_FP16_BM(64, 64, 32); }
            else if (bn_ov == 64) { LAUNCH_FP16_BM(64, 64, 64); }
            else if (bn_ov == 128) { LAUNCH_FP16_BM(64, 64, 128); }
            else { LAUNCH_FP16_BM(64, 64, 16); }
        } else {
            const bool is_self_attn = (qo_len == kv_len);
            if (is_self_attn) {
                if (bn_ov == 32) { LAUNCH_FP16_BM(64, 64, 32); }
                else if (bn_ov == 128) { LAUNCH_FP16_BM(64, 64, 128); }
                else { LAUNCH_FP16_BM(64, 64, 64); }
            } else {
                // cross-attn (kv > 128 且 <=1024, fp16 direct):
                //   BM=128 实测一致快 4-14% (SDXL03/06/09/12/15/18) —— 减少 K/V 冗余读取
                //   kv<=128 保持 BM=64 (SDXL11/17 用 BM128 反而慢 6-17%)
                // 实验 env (bn_ov/bm_ov) 优先于默认规则
                if (bn_ov == 128) { LAUNCH_FP16_BN(64, 64, 128); }
                else if (bn_ov == 64) { LAUNCH_FP16_BN(64, 64, 64); }
                else if (bn_ov == 32) { LAUNCH_FP16_BN(64, 64, 32); }
                else if (bm_ov != 0) { LAUNCH_FP16_BM(64, 64, 32); }
                else if (kv_len > 128) { LAUNCH_FP16_BN(64, 128, 32); }
                else { LAUNCH_FP16_BN(64, 64, 32); }
            }
        }
    } else {
        LAUNCH_FP16_BN(128, 64, 16);
    }
    #undef LAUNCH_FP16_T
    #undef LAUNCH_FP16_BN
    #undef LAUNCH_FP16_BM

    return output;
}

Tensor bf16_attn_gfx11_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, int64_t bm_sel) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t qo_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    const hipStream_t stream = current_hip_stream(query);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;

    // 实验: bm_sel 运行时选择 BM (0=默认, 1=32, 2=128, 3=强制64); env SAGEATTN_BF16_BM 可覆盖
    int bm_ov = (bm_sel == 1) ? 32 : (bm_sel == 2) ? 128 : (bm_sel == 3) ? 64 : 0;
    if (getenv("SAGEATTN_BF16_BM")) bm_ov = atoi(getenv("SAGEATTN_BF16_BM"));
    // 实验: SAGEATTN_BF16_BN 覆盖 D=64 direct 路径的 BN (0=默认, 16/32/64/128)
    const int bn_ov = getenv("SAGEATTN_BF16_BN") ? atoi(getenv("SAGEATTN_BF16_BN")) : 0;

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

    #define LAUNCH_BF16_T(HD, CAUSAL, BM, BN) \
        do { \
            dim3 block(BM / 16 * 32); \
            dim3 grid((qo_len + BM - 1) / BM, q_heads, batch); \
            bf16_attn_kernel_wpe2_t<HD, CAUSAL, BM, BN><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()), \
                reinterpret_cast<const __hip_bfloat16*>(key.data_ptr()), \
                reinterpret_cast<const __hip_bfloat16*>(value.data_ptr()), \
                reinterpret_cast<__hip_bfloat16*>(output.data_ptr()), \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                sm_scale_log2e, static_cast<int>(tensor_layout)); \
        } while(0)

    // is_causal 是运行时参数, 必须用 if/else 生成编译期模板常量
    #define LAUNCH_BF16_BN(HD, BM, BN) \
        do { \
            if (is_causal) { LAUNCH_BF16_T(HD, true, BM, BN); } \
            else { LAUNCH_BF16_T(HD, false, BM, BN); } \
        } while(0)

    // 实验 BM 覆盖: SAGEATTN_BF16_BM
    #define LAUNCH_BF16_BM(HD, BM, BN) \
        do { \
            if (bm_ov == 32) { LAUNCH_BF16_BN(HD, 32, BN); } \
            else if (bm_ov == 128) { LAUNCH_BF16_BN(HD, 128, BN); } \
            else { LAUNCH_BF16_BN(HD, BM, BN); } \
        } while(0)

    if (head_dim == 64) {
        if (kv_len <= 128) {
            // 默认 BN=16; 实验可用 SAGEATTN_BF16_BN 覆盖
            if (bn_ov == 32) { LAUNCH_BF16_BM(64, 64, 32); }
            else if (bn_ov == 64) { LAUNCH_BF16_BM(64, 64, 64); }
            else if (bn_ov == 128) { LAUNCH_BF16_BM(64, 64, 128); }
            else { LAUNCH_BF16_BM(64, 64, 16); }
        } else {
            const bool is_self_attn_bf = (qo_len == kv_len);
            if (is_self_attn_bf) {
                if (bn_ov == 32) { LAUNCH_BF16_BM(64, 64, 32); }
                else if (bn_ov == 128) { LAUNCH_BF16_BM(64, 64, 128); }
                else { LAUNCH_BF16_BM(64, 64, 64); }
            } else {
                // cross-attn: 与 fp16 相同的规则, kv>128 用 BM=128
                if (bn_ov == 128) { LAUNCH_BF16_BN(64, 64, 128); }
                else if (bn_ov == 64) { LAUNCH_BF16_BN(64, 64, 64); }
                else if (bn_ov == 32) { LAUNCH_BF16_BN(64, 64, 32); }
                else if (bm_ov != 0) { LAUNCH_BF16_BM(64, 64, 32); }
                else if (kv_len > 128) { LAUNCH_BF16_BN(64, 128, 32); }
                else { LAUNCH_BF16_BN(64, 64, 32); }
            }
        }
    } else {
        LAUNCH_BF16_BN(128, 64, 16);
    }
    #undef LAUNCH_BF16_T
    #undef LAUNCH_BF16_BN
    #undef LAUNCH_BF16_BM

    return output;
}
