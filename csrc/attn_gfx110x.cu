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
#error "attn_gfx110x.cu is only intended for ROCm/HIP."
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

// int8 V_T (量化转置) -> fp16 片段: 16 个 int8 (16B, 单次 dwordx4 向量读) -> 16 halfs
__device__ __forceinline__ sageattn_gfx11::v16h convert_i8x32_f16(const int8_t* p) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const signed char* c = reinterpret_cast<const signed char*>(&u);
    sageattn_gfx11::v16h out;
    __half* o = reinterpret_cast<__half*>(&out);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        o[j] = __float2half_rn(static_cast<float>(c[j]));
    }
    return out;
}

// 变体: 转换时直接乘 per-tile 反量化 scale (省一次 16-half 乘 pass)
__device__ __forceinline__ sageattn_gfx11::v16h convert_i8x32_f16_scaled(const int8_t* p, float s) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const signed char* c = reinterpret_cast<const signed char*>(&u);
    sageattn_gfx11::v16h out;
    __half* o = reinterpret_cast<__half*>(&out);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        o[j] = __float2half_rn(static_cast<float>(c[j]) * s);
    }
    return out;
}

// v16h 逐元素乘常数 (半精度) — int8 V 反量化 (per 16-col 块)
__device__ __forceinline__ sageattn_gfx11::v16h mul_v16h_scalar(sageattn_gfx11::v16h a, __half s) {
    __half* pa = reinterpret_cast<__half*>(&a);
#pragma unroll
    for (int i = 0; i < 16; ++i) pa[i] = __hmul(pa[i], s);
    return a;
}

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
    // aoti_torch_get_current_stream returns an opaque StreamHandle, not a
    // raw HIP stream; the CUDA variant returns the actual cudaStream_t.
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
    // v2: 32B 向量读 (16 half 连续 d 列), 消除原 2B 标量读 (实测原版仅 ~38GB/s,
    // torch mean 40-71GB/s; 对齐 quant 的 32B 读粒度经验)
    constexpr int TileD = 16;
    constexpr int Threads = 256;
    __shared__ float partial_sum[16][256];

    const int tid = threadIdx.x;
    const int64_t d_base = static_cast<int64_t>(blockIdx.x) * TileD;
    const int64_t h = blockIdx.y;
    const int64_t b = blockIdx.z;
    if (d_base + TileD > head_dim) return;

    // 每 thread 每迭代读行 s 的 16 个连续 half (32B, 对齐: d_base 为 16 倍数), 
    // 累加进 16 个列累加器; s 按 256 threads 步进
    float acc[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = 0.0f;
    for (int64_t s = tid; s < seq_len; s += Threads) {
        const int64_t offset = b * in_stride_b + s * in_stride_n + h * in_stride_h + d_base;
        const uint4* p4 = reinterpret_cast<const uint4*>(input + offset);
        const T* v = reinterpret_cast<const T*>(p4);
#pragma unroll
        for (int c = 0; c < 16; ++c) acc[c] += to_float(v[c]);
    }
#pragma unroll
    for (int c = 0; c < 16; ++c) partial_sum[c][tid] = acc[c];
    __syncthreads();
    if (tid < TileD) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < Threads; ++i) sum += partial_sum[tid][i];
        const int64_t mean_d = d_base + tid;
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

// 多值 blockReduceMax: 一次屏障对 RATIO 个标量并行归约 (原方案 RATIO 次串行
// blockReduceMax 产生 2*RATIO 次 barrier; D=64 BLK=128 RATIO=4 时每块 8+2 次 barrier
// 缩为一次向量归约的 2 次 barrier + 1 次 amax 守卫 barrier)
template <typename T, int N>
__device__ __forceinline__ void blockReduceMaxVec(T* val) {
    static __shared__ T shared[32][N];
    const int lane = threadIdx.x & 0x1f;
    const int wid  = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
#pragma unroll
        for (int i = 0; i < N; ++i) val[i] = fmaxf(val[i], __shfl_xor_sync(FINAL_MASK, val[i], m, 32));
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) shared[wid][i] = val[i];
    }
    __syncthreads();
    if (threadIdx.x < (blockDim.x / 32)) {
#pragma unroll
        for (int i = 0; i < N; ++i) val[i] = fmaxf(val[i], shared[lane][i]);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
#pragma unroll
        for (int i = 0; i < N; ++i) val[i] = fmaxf(val[i], __shfl_xor_sync(FINAL_MASK, val[i], m, 32));
    }
    __syncthreads();
}

template <typename T, int HeadDim, int BLK, int MIN_BLK>
__global__ void quant_qk_int8_hnd_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    const T* __restrict__ key_mean,
    float* __restrict__ scale_out,
    const int64_t batch,
    const int64_t heads,
    const int64_t seq_len,
    const int scale_groups,
    const int is_q,
    const float sm_scale_log2e,
    const int64_t in_stride_b,
    const int64_t in_stride_n,
    const int64_t in_stride_h,
    const int groups_per_block) {
    constexpr int Threads = 256;
    // 大 block (BLK 行) + MIN_BLK 粒度 scale: block 处理 BLK 行, 每 MIN_BLK 行一组
    // 独立 amax (RATIO 组), 用 RATIO 个 fmax 累加器 ILP —— 避免大 block 的串行 fmax 链
    constexpr int RATIO = BLK / MIN_BLK;
    constexpr int PackElems = 8;
    constexpr int Packs = (BLK * HeadDim) / 8;
    constexpr int PackPairs = Packs / 2;
    constexpr int PPT = (PackPairs + Threads - 1) / Threads;  // 每 thread 的 pack-pair 数
    // pass1 原始数据缓存于寄存器 (免 shared_data LDS 往返 + 尾部 barrier), 仅剩 shared_amax
    __shared__ float shared_amax[RATIO];

    const int head = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch || head >= heads) return;
    const int total_blocks = static_cast<int>((seq_len + BLK - 1) / BLK);
    // 每 thread 一次读 32B (2 个连续 pack): 读粒度 16B->32B, 与带宽上限对齐
    const float pass1_scale = is_q ? sm_scale_log2e : 1.0f;
    const float extra_scale = is_q ? sm_scale_log2e : 1.0f;

    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int blk = blockIdx.x * groups_per_block + gi;
        if (blk >= total_blocks) break;
        const int64_t base_row = static_cast<int64_t>(blk) * BLK;
        const int tid = threadIdx.x;
        constexpr int Packs = (BLK * HeadDim) / 8;

        // ---- pass1: 读全局 (32B/thread) -> 寄存器缓存 + 分组 amax (RATIO 累加器 ILP) ----
        float local_amax[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) local_amax[r] = 1e-7f;
        uint4 reg[PPT][2];
#pragma unroll
        for (int it = 0; it < PPT; ++it) {
            const int p = tid + it * Threads;
            uint4 raw0 = make_uint4(0, 0, 0, 0);
            uint4 raw1 = make_uint4(0, 0, 0, 0);
            if (p < PackPairs) {
                const int pack = p * 2;
                const int elem_base = pack * PackElems;
                const int row = elem_base / HeadDim;
                const int d = elem_base - row * HeadDim;
                const int64_t seq = base_row + row;
                if (seq < seq_len) {
                    const int64_t in_off = static_cast<int64_t>(b) * in_stride_b + seq * in_stride_n + head * in_stride_h + d;
                    raw0 = *reinterpret_cast<const uint4*>(input + in_off);
                    raw1 = *reinterpret_cast<const uint4*>(input + in_off + 8);
                    const int r = row / MIN_BLK;
                    const T* v0 = reinterpret_cast<const T*>(&raw0);
                    const T* v1 = reinterpret_cast<const T*>(&raw1);
                    float am = local_amax[r];
                    if (!is_q && key_mean != nullptr) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            float v = to_float(v0[i]) - to_float(key_mean[(b * heads + head) * HeadDim + d + i]);
                            am = fmaxf(am, fabsf(v * pass1_scale));
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            float v = to_float(v1[i]) - to_float(key_mean[(b * heads + head) * HeadDim + d + 8 + i]);
                            am = fmaxf(am, fabsf(v * pass1_scale));
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            float v = to_float(v0[i]);
                            am = fmaxf(am, fabsf(v * pass1_scale));
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            float v = to_float(v1[i]);
                            am = fmaxf(am, fabsf(v * pass1_scale));
                        }
                    }
                    local_amax[r] = am;
                }
            }
            reg[it][0] = raw0;
            reg[it][1] = raw1;
        }
        // ---- 归约: RATIO 个 amax 一次向量 blockReduce (2 barriers 只付一次) ----
        float red[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) red[r] = local_amax[r];
        blockReduceMaxVec<float, RATIO>(red);
        if (tid == 0) {
#pragma unroll
            for (int r = 0; r < RATIO; ++r) {
                shared_amax[r] = red[r];
                // 尾部子块写 scale 需守卫防越界
                if (base_row + static_cast<int64_t>(r) * MIN_BLK < seq_len) {
                    scale_out[(static_cast<int64_t>(b) * heads + head) * scale_groups + blk * RATIO + r] =
                        red[r] / 127.0f;
                }
            }
        }
        __syncthreads();
        float inv_scale[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) inv_scale[r] = 127.0f / shared_amax[r];

        // ---- pass2: 从寄存器读 -> 量化 (按子块 r 选 scale) -> 写回 ----
#pragma unroll
        for (int it = 0; it < PPT; ++it) {
            const int p = tid + it * Threads;
            if (p < PackPairs) {
                const int pack = p * 2;
                const int elem_base = pack * PackElems;
                const int row = elem_base / HeadDim;
                const int d = elem_base - row * HeadDim;
                const int64_t seq = base_row + row;
                if (seq < seq_len) {
                    const int r = row / MIN_BLK;
                    const int64_t out_off = (static_cast<int64_t>(b) * heads + head) * seq_len * HeadDim + seq * HeadDim + d;
                    const uint4 raw0 = reg[it][0];      // pass1 缓存在寄存器, 省 LDS 往返
                    const uint4 raw1 = reg[it][1];
                    const T* values = reinterpret_cast<const T*>(&raw0);
                    const T* values1 = reinterpret_cast<const T*>(&raw1);
                    char4 out0, out1, out2, out3;
                    float v0 = to_float(values[0]), v1 = to_float(values[1]);
                    float v2 = to_float(values[2]), v3 = to_float(values[3]);
                    float v4 = to_float(values[4]), v5 = to_float(values[5]);
                    float v6 = to_float(values[6]), v7 = to_float(values[7]);
                    float w0 = to_float(values1[0]), w1 = to_float(values1[1]);
                    float w2 = to_float(values1[2]), w3 = to_float(values1[3]);
                    float w4 = to_float(values1[4]), w5 = to_float(values1[5]);
                    float w6 = to_float(values1[6]), w7 = to_float(values1[7]);
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
                        w0 -= to_float(key_mean[mean_base + 8]);
                        w1 -= to_float(key_mean[mean_base + 9]);
                        w2 -= to_float(key_mean[mean_base + 10]);
                        w3 -= to_float(key_mean[mean_base + 11]);
                        w4 -= to_float(key_mean[mean_base + 12]);
                        w5 -= to_float(key_mean[mean_base + 13]);
                        w6 -= to_float(key_mean[mean_base + 14]);
                        w7 -= to_float(key_mean[mean_base + 15]);
                    }
                    const float iscale = inv_scale[r] * extra_scale;
                    out0.x = float_to_int8(v0 * iscale);
                    out0.y = float_to_int8(v1 * iscale);
                    out0.z = float_to_int8(v2 * iscale);
                    out0.w = float_to_int8(v3 * iscale);
                    out1.x = float_to_int8(v4 * iscale);
                    out1.y = float_to_int8(v5 * iscale);
                    out1.z = float_to_int8(v6 * iscale);
                    out1.w = float_to_int8(v7 * iscale);
                    out2.x = float_to_int8(w0 * iscale);
                    out2.y = float_to_int8(w1 * iscale);
                    out2.z = float_to_int8(w2 * iscale);
                    out2.w = float_to_int8(w3 * iscale);
                    out3.x = float_to_int8(w4 * iscale);
                    out3.y = float_to_int8(w5 * iscale);
                    out3.z = float_to_int8(w6 * iscale);
                    out3.w = float_to_int8(w7 * iscale);
                    *reinterpret_cast<char4*>(output + out_off) = out0;
                    *reinterpret_cast<char4*>(output + out_off + 4) = out1;
                    *reinterpret_cast<char4*>(output + out_off + 8) = out2;
                    *reinterpret_cast<char4*>(output + out_off + 12) = out3;
                }
            }
        }
    }
}

// V [B, N, H, D] (NHD) / [B, H, N, D] (HND) -> V_T [B, H, D, N] (contiguous, n 连续)
//   - 加载/写回 4 half: d 步长 4 恰好覆盖 32 LDS banks 无冲突 (实测最优: v5 8half/v7 大 tile 均更慢)
//   - 16-bit 标量 LDS 访问 (行宽 33 奇数, 无对齐问题)
template <typename V_IN>
__global__ void v_transpose_kernel(
    const V_IN* __restrict__ v,
    __half* __restrict__ v_t,
    const int64_t batch_size,
    const int64_t seq_len,
    const int64_t num_heads,
    const int64_t head_dim,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int tensor_layout) {
    constexpr int NT = 32;   // n-tile
    constexpr int DT = 32;   // d-tile
    constexpr int THREADS = 256;
    __shared__ __half tile[NT * (DT + 1)];  // pad 1

    // V_T 输出 n 维: padding 到 64 倍数 (与 core.py / attn kernel 的 v_t_n 规则一致),
    // padding 区填 0, 防止 attn kernel 的 v_frag_t 32B 直读越界 (kv_len 非 64 倍数时)
    const int64_t v_t_n = ((seq_len + 63) / 64) * 64;
    // 注意: ntiles 必须按 v_t_n 计算 (而非 seq_len), 使写回覆盖整个 padded 区
    // [ceil(seq/32)*32, ceil(seq/64)*64) 也要写 0, 否则 kv mod 64 ∈ [1,32] 时
    // 最后 kv-tile (BN=64) 的 v_frag_t 读未初始化 V_T -> NaN
    const int64_t ntiles = v_t_n / NT;
    const int64_t dtiles = (head_dim + DT - 1) / DT;
    const int64_t total = batch_size * num_heads * ntiles * dtiles;
    for (int64_t i = blockIdx.x; i < total; i += gridDim.x) {
        // 索引顺序: dt 变化最快 (实测: nt 变化最快反而慢 80%, 块调度顺序与写合并假设不符)
        const int64_t dt = i % dtiles;
        const int64_t nt = (i / dtiles) % ntiles;
        const int64_t h = (i / (dtiles * ntiles)) % num_heads;
        const int64_t b = i / (dtiles * ntiles * num_heads);
        const int64_t n0 = nt * NT;
        const int64_t d0 = dt * DT;

        const int tid = threadIdx.x;
        // ---- 加载: thread t -> n_local = t>>3 (0..31), d_local = (t&7)*4 (0..28) ----
        // 4 half (8B) 连续读 -> LDS 连续写; d 步长 4 覆盖 32 banks 无冲突
        const int n_l = tid >> 3;
        const int d_l = (tid & 7) * 4;
        const int64_t n_abs = n0 + n_l;
        const bool d_in = (d0 + d_l + 4 <= head_dim);
        if (n_abs < seq_len && d_in) {
            const int64_t v_off = (tensor_layout == kHND) ?
                (b * v_stride_b + h * v_stride_h + n_abs * v_stride_n + d0 + d_l) :
                (b * v_stride_b + n_abs * v_stride_n + h * v_stride_h + d0 + d_l);
            __half* dst = &tile[n_l * (DT + 1) + d_l];
            if constexpr (std::is_same<V_IN, __half>::value) {
                const __half* src = v + v_off;
#pragma unroll
                for (int j = 0; j < 4; ++j) dst[j] = src[j];
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dst[j] = __float2half_rn(__bfloat162float(v[v_off + j]));
                }
            }
        } else {
            __half* dst = &tile[n_l * (DT + 1) + d_l];
#pragma unroll
            for (int j = 0; j < 4; ++j) dst[j] = __half{0};
        }
        __syncthreads();

        // ---- 写回: thread t -> d_local = t>>3 (0..31), n_local = (t&7)*4 (0..28) ----
        const int d_w = tid >> 3;
        const int n_w = (tid & 7) * 4;
        const int64_t d_abs = d0 + d_w;
        if (d_abs < head_dim) {
            __half hvals[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) hvals[j] = tile[(n_w + j) * (DT + 1) + d_w];
            const int64_t vt_base = ((b * num_heads + h) * head_dim + d_abs) * v_t_n + n0 + n_w;
            // 4 个 2B 写 (n 连续, 合并为 8B); padding 区 (n 越界) 填 0
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_t[vt_base + j] = (n0 + n_w + j < seq_len) ? hvals[j] : __half{0};
            }
        }
        __syncthreads();  // 下一个 tile 复用 LDS
    }
}

// V 量化转置 (gfx110x int8 path): V [B,H,N,D] bf16/fp16 -> V_T_i8 [B,H,D,N_pad64] int8
// + 每 32 n-tile 的 fp32 scale [B,H,ceil(seq/32)] (scale = maxabs/127, 无效区 scale 1.0)。
// block 覆盖 (b, h, nt); head_dim <= 128; 单个 block 先载入整 tile (含 max),
// 再按 [D][N] 写回 int8, 避免二次全局读 V。
template <typename V_IN>
__global__ void v_quant_transpose_kernel(
    const V_IN* __restrict__ v,
    int8_t* __restrict__ v_t_i8,
    float* __restrict__ v_scale,
    const int64_t batch_size,
    const int64_t seq_len,
    const int64_t num_heads,
    const int64_t head_dim,
    const int64_t v_stride_b,
    const int64_t v_stride_n,
    const int64_t v_stride_h,
    const int tensor_layout) {
    constexpr int NT = 32;      // n-tile
    constexpr int THREADS = 256;
    __shared__ __half tile[NT][(128 / 16) * 16 + 1];
    __shared__ float red[THREADS];

    const int64_t v_t_n = ((seq_len + 63) / 64) * 64;
    const int64_t ntiles = v_t_n / NT;
    const int64_t total = batch_size * num_heads * ntiles;
    const int tid = threadIdx.x;
    const int n_l = tid >> 3;         // 0..31 (载入: 行)
    const int d_l = (tid & 7) * 16;   // 0..112 (载入: 16 halfs / 线程, 8 线程/行)
    const int d_w = tid >> 3;         // 0..31  (写回: d 覆盖)
    const int n_w = (tid & 7) * 4;    // 0..28  (写回: n 4 个连续)

    for (int64_t i = blockIdx.x; i < total; i += gridDim.x) {
        const int64_t nt = i % ntiles;
        const int64_t h = (i / ntiles) % num_heads;
        const int64_t b = i / (ntiles * num_heads);
        const int64_t n0 = nt * NT;

        // pass1: 载入 LDS (fp16) + 局部 max
        float lmax = 0.0f;
        {
            const int64_t d_abs = d_l;
            const int64_t n_abs = n0 + n_l;
            __half* dst = &tile[n_l][d_l];
            if (d_abs + 16 <= head_dim) {
                if (n_abs < seq_len) {
                    const int64_t v_off = (tensor_layout == kHND) ?
                        (b * v_stride_b + h * v_stride_h + n_abs * v_stride_n + d_abs) :
                        (b * v_stride_b + n_abs * v_stride_n + h * v_stride_h + d_abs);
                    if constexpr (std::is_same<V_IN, __half>::value) {
                        const __half* src = v + v_off;
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const float f = __half2float(src[j]);
                            dst[j] = src[j];
                            lmax = fmaxf(lmax, fabsf(f));
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const float f = __bfloat162float(v[v_off + j]);
                            dst[j] = __float2half_rn(f);
                            lmax = fmaxf(lmax, fabsf(f));
                        }
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 16; ++j) dst[j] = __half{0};
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) dst[j] = __half{0};
            }
        }
        red[tid] = lmax;
        __syncthreads();
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        const float scale = (red[0] > 1e-6f) ? (red[0] / 127.0f) : 1.0f;
        if (tid == 0) v_scale[(b * num_heads + h) * ntiles + nt] = scale;
        __syncthreads();

        // pass2: quantize + 转置写回 [D][N] (n 连续, 4 x int8)
        for (int64_t d_off = 0; d_off < head_dim; d_off += 32) {
            const int64_t d_abs = d_off + d_w;
            if (d_abs < head_dim) {
                __half hvals[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) hvals[j] = tile[n_w + j][d_abs];
                const int64_t vt_base = ((b * num_heads + h) * head_dim + d_abs) * v_t_n + n0 + n_w;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int64_t n_abs = n0 + n_w + j;
                    int8_t vq = 0;
                    if (n_abs < seq_len) {
                        const float q = roundf(__half2float(hvals[j]) / scale);
                        const int ic = (q > 127.0f) ? 127 : ((q < -127.0f) ? -127 : static_cast<int>(q));
                        vq = static_cast<int8_t>(ic);
                    }
                    v_t_i8[vt_base + j] = vq;
                }
            }
        }
        __syncthreads();
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
    // V_T tile 拷贝到 LDS (布局 [D][N], 行 = D 维, n 连续 16 half 行读):
    // PV 的 v_frag 从 LDS 读而非全局 (消除每 lane 32 次冗余全局行读, 与 triton LDS 缓存对齐)
    // VT_GLOBAL: 需 HeadDim 行 x VTileStride(80) 列; 非 VT_GLOBAL: 原 [BLOCK_N][VStride] 列读布局
    constexpr int VTileStride = 80;  // 行间 8 bank 偏移 (同 VStride 冲突模式), 20KB/128 行
    __shared__ __align__(32) __half v_tile[
        (SAGEATTN_VT_GLOBAL) ? (HeadDim * VTileStride) : (BLOCK_N * VStride)];

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
    const int64_t q_idx_q = q_start + q_row;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        if (q_idx_q < qo_len) {
            const int d_base = dt * BK;
            const int64_t q_off = (tensor_layout == kHND) ?
                (b * q_stride_b + hq * q_stride_h + q_idx_q * q_stride_n + d_base) :
                (b * q_stride_b + q_idx_q * q_stride_n + hq * q_stride_h + d_base);
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
    // V_T n 维 (core.py 分配时 padding 到 64 倍数): vt_off 行 stride 用 v_t_n 而非 kv_len,
    // 否则 kv_len 非 64 倍数时最后 kv-tile 的 v_frag_t 32B 直读越界 (UB/NaN)
    const int64_t v_t_n = ((kv_len + 63) / 64) * 64;
    constexpr int KVecsPerRow = HeadDim / 16;
    constexpr int VVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int VVecsTotal = BLOCK_N * VVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;
    constexpr int VPrefetchPerThread = (VVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    // VT_GLOBAL=1: V lives in the V_T tile (LDS), v_prefetch unused
    // VT_GLOBAL=1: V lives in the V_T tile (LDS), v_prefetch unused
    uint4 v_prefetch[(SAGEATTN_VT_GLOBAL) ? 1 : VPrefetchPerThread];
    constexpr int VVecsPerTileVT = (HeadDim * BLOCK_N) / 16;
    // int8 V 专用 prefetch 缓冲区; fp16/bf16 V 由 tile-start 直读, 不需要 (省 32 VGPR)
    constexpr int VPrefetchVT = (SAGEATTN_VT_GLOBAL && std::is_same<V_DTYPE, int8_t>::value)
        ? (VVecsPerTileVT / THREADS) : 0;
    v16h v_prefetch_vt[VPrefetchVT > 0 ? VPrefetchVT : 1];
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
    if (SAGEATTN_VT_GLOBAL) {
    if (std::is_same<V_DTYPE, int8_t>::value) {
    // int8 V: first V_T tile loads directly into LDS (later tiles: v_prefetch_vt -> LDS in write phase)
    // V_T_i8: int8 数据, 32B (32 int8) -> 16 halfs 转 fp16 后入 LDS (LDS/frag 仍为 fp16)
    const int8_t* vt8 = reinterpret_cast<const int8_t*>(v);
#pragma unroll
    for (int i = tid; i < VVecsPerTileVT; i += THREADS) {
        const int d = i / (BLOCK_N / BK);
        const int c16 = i % (BLOCK_N / BK);
        const int64_t v_off =
            ((b * num_kv_heads + hkv) * HeadDim + d) * v_t_n + (c16 * BK);
        *reinterpret_cast<v16h*>(
            reinterpret_cast<char*>(v_tile) + (d * VTileStride + c16 * BK) * 2) =
            convert_i8x32_f16(vt8 + v_off);
    }
    }
    // fp16/bf16 V_T: PV 直接从全局 V_T 行读 (vt_off 32B), 无需 LDS 暂存
    } else {
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
    if (SAGEATTN_VT_GLOBAL) {
    if (std::is_same<V_DTYPE, int8_t>::value) {
    const int8_t* vt8 = reinterpret_cast<const int8_t*>(v);
#pragma unroll
        for (int i = tid; i < VVecsPerTileVT; i += THREADS) {
            const int d = i / (BLOCK_N / BK);
            const int c16 = i % (BLOCK_N / BK);
            const int64_t v_off =
                ((b * num_kv_heads + hkv) * HeadDim + d) * v_t_n + (next_base + c16 * BK);
            v_prefetch_vt[i / THREADS] = convert_i8x32_f16(vt8 + v_off);
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

        // ---- V_T 当前 tile (kb_base) 拷入 LDS (仅 VT_GLOBAL) ----
        // v_tile 布局 [D][N]: 行 = D 维 (HeadDim), 列 = n (BLOCK_N), 行 stride VStride half。
        // PV 的 v_frag 从 LDS 行读 (b128), 消除"每 lane 每 dt 一次 32B 全局行读"的
        // L1/L2 带宽冗余 (128 lanes x 32 次/迭代 -> 128KB, 实际 tile 仅 16KB)。
        // 512 个 v16h (128 D 行 x 4 个 16 列组), 128 threads -> 4/thread, 全局行读 32B 对齐。
        if (SAGEATTN_VT_GLOBAL) {
        if (!std::is_same<V_DTYPE, int8_t>::value && BLOCK_M >= 128) {
            // fp16/bf16 V_T (BM128): 每 tile 行读入 LDS (与历史直读结构一致), barrier 后 PV 读 LDS
            // BM64: 不走 LDS, PV 直接全局行读 (实测 BM64 下直读更快, 免 barrier/20KB tile)
#pragma unroll
            for (int i = tid; i < VVecsPerTileVT; i += THREADS) {
                const int d = i / (BLOCK_N / BK);
                const int c16 = i % (BLOCK_N / BK);
                const int64_t v_off =
                    ((b * num_kv_heads + hkv) * HeadDim + d) * v_t_n + (kb_base + c16 * BK);
                const v16h src = *reinterpret_cast<const v16h*>(v + v_off);
                *reinterpret_cast<v16h*>(
                    reinterpret_cast<char*>(v_tile) + (d * VTileStride + c16 * BK) * 2) = src;
            }
            __syncthreads();  // v_tile 写完后 PV 才可读
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
        // out^T = V^T @ P^T: A = V^T (v_tile 行 (L&15) 的 16 连续 n = LDS 行读 b128),
        // B = P^T (p_frag). V_T tile 已由主循环拷入 v_tile [D][N] (见主循环 V 拷贝段)
        // C = out^T: lane L 持 out^T[2e+hw][L&15] = out[L&15][2e+hw] (转置解释, 匹配写回)
        // fp16/bf16 V_T: BM128 走 v_tile LDS 行读 (与历史直读 PV 一致); BM64 直接全局 V_T 行读
        if (!std::is_same<V_DTYPE, int8_t>::value && BLOCK_M >= 128) {
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_row = dt * BK + m_row;  // v_tile 的 D 维行 (与 V_T 同)
                const v16h* vp = reinterpret_cast<const v16h*>(
                    reinterpret_cast<const char*>(v_tile) + (d_row * VTileStride + ct * BK) * 2);
                const v16h v_frag_t = *vp;
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(v_frag_t, p_frag, out_acc[dt]);
            }
        }
        } else if (!std::is_same<V_DTYPE, int8_t>::value) {
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int64_t vt_off =
                    ((b * num_kv_heads + hkv) * HeadDim + (dt * BK + m_row)) * v_t_n + (kb_base + ct * BK);
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
            // int8 V 反量化: 每 16-col 块 (V 量化粒度 32, 两 16-col 半共用同 scale)
            const float rvs = v_scale[b * vs_stride_b + hkv * vs_stride_h +
                                      static_cast<int>((kb_base + ct * BK) / 32)];
            const __half rvs_h = __float2half_rn(rvs);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int d_row = dt * BK + m_row;  // v_tile 的 D 维行 (与 V_T 同)
                const v16h* vp = reinterpret_cast<const v16h*>(
                    reinterpret_cast<const char*>(v_tile) + (d_row * VTileStride + ct * BK) * 2);
                const v16h v_frag_t = mul_v16h_scalar(*vp, rvs_h);
                out_acc[dt] = sageattn_gfx11::wmma_f32_f16(v_frag_t, p_frag, out_acc[dt]);
            }
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
    if (SAGEATTN_VT_GLOBAL) {
    if (std::is_same<V_DTYPE, int8_t>::value) {
#pragma unroll
            for (int i = tid; i < VVecsPerTileVT; i += THREADS) {
                const int d = i / (BLOCK_N / BK);
                const int c16 = i % (BLOCK_N / BK);
                *reinterpret_cast<v16h*>(
                    reinterpret_cast<char*>(v_tile) + (d * VTileStride + c16 * BK) * 2) = v_prefetch_vt[i / THREADS];
            }
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

    // ---- 写回 (向量化 v2): permlanex16 交换得到完整行 16 列, 分工 16B 连续写 ----
    // 每 dt: 8 half/bf16 打包 4 u32 -> 4 次 permlanex16 -> 组装 16 列
    // hw=0 lane 写列 {0..7} (16B), hw=1 lane 写列 {8..15} (16B), 消除 2B 散布写
    const int64_t q_idx = q_start + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            unsigned pack[4], cross[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    __half lo = __float2half_rn(out_acc[dt][2 * k] * inv_l);
                    __half hi = __float2half_rn(out_acc[dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__half_as_ushort(lo));
                } else {
                    __hip_bfloat16 lo = from_float_bf16(out_acc[dt][2 * k] * inv_l);
                    __hip_bfloat16 hi = from_float_bf16(out_acc[dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__bfloat16_as_ushort(lo));
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);

            const int d = dt * BK + hw * 8;
            const int64_t o_off = (tensor_layout == kHND) ?
                (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
            if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                // 组装 8 个 u32 (16 half): u32[2k]={列4k,4k+1}, u32[2k+1]={列4k+2,4k+3}
                alignas(16) unsigned b16[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b16[2 * k] = (even & 0xFFFF) | ((odd & 0xFFFF) << 16);
                    b16[2 * k + 1] = (even >> 16) | (odd & 0xFFFF0000);
                }
                // hw=0 写列 0..7 (b16[0..3]), hw=1 写列 8..15 (b16[4..7])
                *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b16 + hw * 4);
            } else {
                alignas(16) __hip_bfloat16 b[16];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b[4 * k + 0] = __ushort_as_bfloat16(static_cast<unsigned short>(even & 0xFFFF));
                    b[4 * k + 1] = __ushort_as_bfloat16(static_cast<unsigned short>(odd & 0xFFFF));
                    b[4 * k + 2] = __ushort_as_bfloat16(static_cast<unsigned short>(even >> 16));
                    b[4 * k + 3] = __ushort_as_bfloat16(static_cast<unsigned short>(odd >> 16));
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__hip_bfloat16*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b + hw * 8);
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
    // VT_GLOBAL=1 时 V 直接从全局 V_T 行读, LDS v_tile 完全不用 (见 impl_t 注释)
    __shared__ __half v_tile[(SAGEATTN_VT_GLOBAL) ? 1 : (BLOCK_N * VStride)];

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
    // V_T n 维 (core.py padding 到 64 倍数), 见 attn_kernel_impl_t 说明
    const int64_t v_t_n = ((kv_len + 63) / 64) * 64;
    constexpr int KVecsPerRow = HeadDim / 16;
    constexpr int VVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int VVecsTotal = BLOCK_N * VVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;
    constexpr int VPrefetchPerThread = (VVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    // VT_GLOBAL=1 时 V 从全局 V_T 行读 (无 LDS), v_prefetch 完全不用:
    // 缩为 1 元素省 (VPrefetchPerThread-1)*4 个 VGPR (编译期常量三元)
    uint4 v_prefetch[(SAGEATTN_VT_GLOBAL) ? 1 : VPrefetchPerThread];

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
        if (blockIdx.x == 31 && tid == 0 && (row_l[0] != row_l[0] || row_l[1] != row_l[1])) {
            printf("DBG rowl NaN kb=%d rl=%f/%f rm=%f/%f oa0=%f oa1=%f oa2=%f oa3=%f\n",
                   (int)kb_base, row_l[0], row_l[1], row_m[0], row_m[1],
                   out_acc[0][0][0], out_acc[0][0][1], out_acc[0][0][2], out_acc[0][0][3]);
        }
        // ---- PV x2 子块 (2 个独立 WMMA, ILP 提升) ----
        if (SAGEATTN_VT_GLOBAL) {
        // out^T = V^T @ P^T: A = V^T (V_T 行读), B = P^T (p_frag)
        if (std::is_same<V_DTYPE, int8_t>::value) {
        const int8_t* vt8 = reinterpret_cast<const int8_t*>(v);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
                float p_vals[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[s][ct][e];
                const v16h p_frag = assemble_p_frag(p_vals, hw);
                // int8 V 反量化: 每 16-col 块 (V 量化粒度 32, 两 16-col 半共用同 scale)
                const float rvs = v_scale[b * vs_stride_b + hkv * vs_stride_h +
                                         static_cast<int>((kb_base + ct * BK) / 32)];
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                    const int64_t vt_off =
                        ((b * num_kv_heads + hkv) * HeadDim + (dt * BK + m_row)) * v_t_n + (kb_base + ct * BK);
                    const v16h v_frag_t = convert_i8x32_f16_scaled(vt8 + vt_off, rvs);
                    out_acc[s][dt] = sageattn_gfx11::wmma_f32_f16(v_frag_t, p_frag, out_acc[s][dt]);
                }
            }
        }
        } else {
        // fp16/bf16 V_T: 直读全局 V_T 行 (32B), 与历史 D128 直读 PV 一致
        // 优化: V_T 读按 (dt,ct) 只加载一次 (s=0/1 共用同一 fragment), 提前于
        // assemble 发射以隐藏全局延迟
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            const int64_t vt_base =
                (b * num_kv_heads + hkv) * HeadDim * v_t_n + (kb_base + ct * BK);
            v16h v_frag[DTiles];
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int64_t vt_off = vt_base + (dt * BK + m_row) * v_t_n;
                                v_frag[dt] = *reinterpret_cast<const v16h*>(v + vt_off);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                float p_vals[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[s][ct][e];
                const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                    out_acc[s][dt] = sageattn_gfx11::wmma_f32_f16(v_frag[dt], p_frag, out_acc[s][dt]);
                }
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
        }    }

    // ---- 写回 x2 子块 (向量化 v2): permlanex16 交换后 16B 连续写 ----
#pragma unroll
    for (int s = 0; s < 2; ++s) {
    const int64_t q_idx = q_start + s * RM + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l[s] > 0.0f) ? (1.0f / row_l[s]) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            unsigned pack[4], cross[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    __half lo = __float2half_rn(out_acc[s][dt][2 * k] * inv_l);
                    __half hi = __float2half_rn(out_acc[s][dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__half_as_ushort(lo));
                } else {
                    __hip_bfloat16 lo = from_float_bf16(out_acc[s][dt][2 * k] * inv_l);
                    __hip_bfloat16 hi = from_float_bf16(out_acc[s][dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__bfloat16_as_ushort(lo));
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);

            const int d = dt * BK + hw * 8;
            const int64_t o_off = (tensor_layout == kHND) ?
                (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
            if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                // 组装 8 个 u32 (16 half): u32[2k]={列4k,4k+1}, u32[2k+1]={列4k+2,4k+3}
                alignas(16) unsigned b16[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b16[2 * k] = (even & 0xFFFF) | ((odd & 0xFFFF) << 16);
                    b16[2 * k + 1] = (even >> 16) | (odd & 0xFFFF0000);
                }
                // hw=0 写列 0..7 (b16[0..3]), hw=1 写列 8..15 (b16[4..7])
                *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b16 + hw * 4);
            } else {
                alignas(16) __hip_bfloat16 b[16];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b[4 * k + 0] = __ushort_as_bfloat16(static_cast<unsigned short>(even & 0xFFFF));
                    b[4 * k + 1] = __ushort_as_bfloat16(static_cast<unsigned short>(odd & 0xFFFF));
                    b[4 * k + 2] = __ushort_as_bfloat16(static_cast<unsigned short>(even >> 16));
                    b[4 * k + 3] = __ushort_as_bfloat16(static_cast<unsigned short>(odd >> 16));
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__hip_bfloat16*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b + hw * 8);
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
        q, k, v, output, q_scale, k_scale, v_scale, vs_stride_b, vs_stride_h,
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
        q, k, v, output, q_scale, k_scale, v_scale, vs_stride_b, vs_stride_h,
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
        q, k, v, output, q_scale, k_scale, v_scale, vs_stride_b, vs_stride_h,
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
        q, k, v, output, q_scale, k_scale, v_scale, vs_stride_b, vs_stride_h,
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
    const float* __restrict__ v_scale,
    const int64_t vs_stride_b,
    const int64_t vs_stride_h,
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
        q, k, v, output, q_scale, k_scale, v_scale, vs_stride_b, vs_stride_h,
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
    // VT_GLOBAL=1 时 V 直接从全局 V_T 行读 (LDS v_tile 缓存经实测为负优化:
    // D=64 短序列 kv-tile 少, 原 PV 冗余读多命中 L2, V 拷贝+barrier 是纯开销,
    // SDXL10 +24% 慢; 与 VAE (D=128 超长, L2 带宽饱和) 相反)
    __shared__ __half v_tile[(SAGEATTN_VT_GLOBAL) ? 1 : (BLOCK_N * VStride)];

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
    // V_T n 维 (core.py padding 到 64 倍数), 见 attn_kernel_impl_t 说明
    const int64_t v_t_n = ((kv_len + 63) / 64) * 64;
    constexpr int KVecsPerRow = HeadDim / 8;
    constexpr int VVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int VVecsTotal = BLOCK_N * VVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;
    constexpr int VPrefetchPerThread = (VVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    // VT_GLOBAL=1 时 V 从全局 V_T 行读 (无 LDS), v_prefetch 完全不用:
    // 缩为 1 元素省 (VPrefetchPerThread-1)*4 个 VGPR (编译期常量三元)
    uint4 v_prefetch[(SAGEATTN_VT_GLOBAL) ? 1 : VPrefetchPerThread];

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
                    ((b * num_kv_heads + hkv) * HeadDim + (dt * BK + m_row)) * v_t_n + (kb_base + ct * BK);
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

    // ---- 写回 (向量化 v2, 与 int8 kernel 一致): permlanex16 交换得到完整行 16 列, 分工 16B 连续写 ----
    const int64_t q_idx = q_start + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            unsigned pack[4], cross[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    __half lo = __float2half_rn(out_acc[dt][2 * k] * inv_l);
                    __half hi = __float2half_rn(out_acc[dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__half_as_ushort(lo));
                } else {
                    __hip_bfloat16 lo = from_float_bf16(out_acc[dt][2 * k] * inv_l);
                    __hip_bfloat16 hi = from_float_bf16(out_acc[dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__bfloat16_as_ushort(lo));
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);

            const int d = dt * BK + hw * 8;
            const int64_t o_off = (tensor_layout == kHND) ?
                (b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d) :
                (b * o_stride_b + q_idx * o_stride_n + hq * o_stride_h + d);
            if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                alignas(16) unsigned b16[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b16[2 * k] = (even & 0xFFFF) | ((odd & 0xFFFF) << 16);
                    b16[2 * k + 1] = (even >> 16) | (odd & 0xFFFF0000);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b16 + hw * 4);
            } else {
                alignas(16) __hip_bfloat16 b[16];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b[4 * k + 0] = __ushort_as_bfloat16(static_cast<unsigned short>(even & 0xFFFF));
                    b[4 * k + 1] = __ushort_as_bfloat16(static_cast<unsigned short>(odd & 0xFFFF));
                    b[4 * k + 2] = __ushort_as_bfloat16(static_cast<unsigned short>(even >> 16));
                    b[4 * k + 3] = __ushort_as_bfloat16(static_cast<unsigned short>(odd >> 16));
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__hip_bfloat16*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b + hw * 8);
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

// V [B,N,H,D] -> V_T [B,H,D,N] (contiguous); 输入支持 fp16/bf16 (bf16 直接转 fp16)
Tensor v_transpose_gfx110x(Tensor value, Tensor value_t, int64_t tensor_layout) {
    const int64_t batch = value.size(0);
    const int64_t heads = (tensor_layout == kHND) ? value.size(1) : value.size(2);
    const int64_t seq_len = (tensor_layout == kHND) ? value.size(2) : value.size(1);
    const int64_t head_dim = value.size(3);
    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = (tensor_layout == kHND) ? value.stride(2) : value.stride(1);
    const int64_t v_stride_h = (tensor_layout == kHND) ? value.stride(1) : value.stride(2);
    const hipStream_t stream = current_hip_stream(value);
    // ntiles 按 padded v_t_n (64 倍数) 计算, 与 kernel 一致 (覆盖 padding 区写 0)
    const int64_t v_t_n = ((seq_len + 63) / 64) * 64;
    const int64_t ntiles = v_t_n / 32;
    const int64_t dtiles = (head_dim + 31) / 32;
    const int64_t total_tiles = batch * heads * ntiles * dtiles;
    dim3 block(256);
    // grid 分派: 小数据量 (total_tiles <= 4096) 用 grid=192 (每 block 多 tile),
    // 实测 SDXL10 (1920 tiles) v_transpose 0.237→0.147ms (-38%, 固定开销占比大);
    // 大数据量曾用 grid=total (每 block 1 tile)。2026-09 复核: D128 int8 self
    // (8k-18k tiles) 下 grid=total 在部分 GPU 状态下调度/同步开销主导 (实测慢 40-60%),
    // 折中 cap ~total/10 (上限 1536), 稳定态不同态均不劣化 (交错 A/B 全程 ±5% 内,
    // 设计 C 再优 1-7%)。SAGEATTN_VT_GRID: 0=auto, N=强制固定 grid
    int64_t grid_cap;
    const char* vt_grid_env = getenv("SAGEATTN_VT_GRID");
    if (vt_grid_env && atoi(vt_grid_env) > 0) {
        grid_cap = atoi(vt_grid_env);
    } else if (total_tiles <= 4096) {
        grid_cap = 192;
    } else {
        grid_cap = std::min<int64_t>(std::max<int64_t>(total_tiles / 10, 128), 1536);
    }
    dim3 grid(static_cast<unsigned>(std::min(total_tiles, grid_cap)));
    if (value.scalar_type() == ScalarType::BFloat16) {
        v_transpose_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(value.data_ptr()),
            reinterpret_cast<__half*>(value_t.data_ptr()),
            batch, seq_len, heads, head_dim,
            v_stride_b, v_stride_n, v_stride_h,
            static_cast<int>(tensor_layout));
    } else {
        v_transpose_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(value.data_ptr()),
            reinterpret_cast<__half*>(value_t.data_ptr()),
            batch, seq_len, heads, head_dim,
            v_stride_b, v_stride_n, v_stride_h,
            static_cast<int>(tensor_layout));
    }
    return value_t;
}

// V [B,N,H,D] -> V_T_i8 [B,H,D,N] int8 + fp32 tile scales [B,H,ceil(seq/32)]
// (gfx110x int8 path 单次的量化转置 prepass; 输入支持 fp16/bf16, 输出 int8 V_T)
Tensor v_quant_transpose_gfx110x(Tensor value, Tensor value_t_i8, Tensor v_scale_t,
                                 int64_t tensor_layout) {
    const int64_t batch = value.size(0);
    const int64_t heads = (tensor_layout == kHND) ? value.size(1) : value.size(2);
    const int64_t seq_len = (tensor_layout == kHND) ? value.size(2) : value.size(1);
    const int64_t head_dim = value.size(3);
    const int64_t v_stride_b = value.stride(0);
    const int64_t v_stride_n = (tensor_layout == kHND) ? value.stride(2) : value.stride(1);
    const int64_t v_stride_h = (tensor_layout == kHND) ? value.stride(1) : value.stride(2);
    const hipStream_t stream = current_hip_stream(value);
    const int64_t v_t_n = ((seq_len + 63) / 64) * 64;
    const int64_t ntiles = v_t_n / 32;
    const int64_t total_tiles = batch * heads * ntiles;
    dim3 block(256);
    int64_t grid_cap;
    const char* vt_grid_env = getenv("SAGEATTN_VT_GRID");
    if (vt_grid_env && atoi(vt_grid_env) > 0) {
        grid_cap = atoi(vt_grid_env);
    } else if (total_tiles <= 4096) {
        grid_cap = 192;
    } else {
        grid_cap = std::min<int64_t>(std::max<int64_t>(total_tiles / 10, 128), 1536);
    }
    dim3 grid(static_cast<unsigned>(std::min(total_tiles, grid_cap)));
    if (value.scalar_type() == ScalarType::BFloat16) {
        v_quant_transpose_kernel<__hip_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(value.data_ptr()),
            reinterpret_cast<int8_t*>(value_t_i8.data_ptr()),
            reinterpret_cast<float*>(v_scale_t.data_ptr()),
            batch, seq_len, heads, head_dim,
            v_stride_b, v_stride_n, v_stride_h,
            static_cast<int>(tensor_layout));
    } else {
        v_quant_transpose_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(value.data_ptr()),
            reinterpret_cast<int8_t*>(value_t_i8.data_ptr()),
            reinterpret_cast<float*>(v_scale_t.data_ptr()),
            batch, seq_len, heads, head_dim,
            v_stride_b, v_stride_n, v_stride_h,
            static_cast<int>(tensor_layout));
    }
    return value_t_i8;
}

Tensor mean_seq_gfx110x(Tensor input, int64_t tensor_layout) {
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

std::vector<Tensor> quant_qk_int8_gfx110x(
    Tensor query, Tensor key, Tensor key_mean,
    int64_t tensor_layout, double sm_scale, int64_t skip_q) {

    const int64_t batch = query.size(0);
    const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
    const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
    const int64_t q_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
    const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
    const int64_t head_dim = query.size(3);

    (void)skip_q;

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
    // 大 block quant (对齐 triton BLK): 每 block 处理 BLK 行, 每 MIN_BLK 行一组
    // 独立 scale (RATIO 组) + 多累加器 ILP —— MIN_BLK 粒度 (scale 语义) 不变
    // 实测 (同进程 A/B): D=64 用 BLK_Q=128/BLK_K=64 最优 (SDXL07 -19%, SDXL13 -18%);
    // D=128 旧逻辑 (LDS 往返 + 标量逐 RATIO 归约) 大 block 128/64 或 64/32 均 +1~3%,
    // 故曾保持 MIN_BLK; 改寄存器缓存 + 向量归约 (blockReduceMaxVec) 后, D=128
    // BLK 128/64 实测 -6~9% (block 数减少、barrier 不随 RATIO 增加), 故 auto 下 D=128 也用 128/64
    // SAGEATTN_QUANT_BLK: 1=auto (按 head_dim), 128=强制 128/64, 64=强制 64/32, 0=旧逻辑
    const int blk_sel = getenv("SAGEATTN_QUANT_BLK") ? atoi(getenv("SAGEATTN_QUANT_BLK")) : 1;
    constexpr int BLK_Q64 = 128;
    constexpr int BLK_K64 = 64;
    constexpr int BLK_Q128 = 64;
    constexpr int BLK_K128 = 32;
    int blk_q, blk_k;
    if (blk_sel == 128) { blk_q = BLK_Q64; blk_k = BLK_K64; }
    else if (blk_sel == 64) { blk_q = BLK_Q128; blk_k = BLK_K128; }
    else if (blk_sel == 0) { blk_q = MIN_BLK_Q; blk_k = MIN_BLK_K; }
    else { blk_q = BLK_Q64; blk_k = BLK_K64; }
    const int q_blocks = (q_len + blk_q - 1) / blk_q;
    const int k_blocks = (kv_len + blk_k - 1) / blk_k;
    // 多 group 合并: 每 block 顺序处理 groups_per_block 个连续大 block (GPB=8 时为 D128 增益 1.1-1.2ms)
    // auto 按 kv_len 扫描选择 (2026-09):
    //   D128: kv<6144: GPB=32 (A01 14.54->14.43); 6144..9216: GPB=16 (A03 32.79->32.66); >=9216: GPB=64 (A05 75.68->75.00)
    //   D64:  kv<6144: GPB=4 (S01 4.62->4.60); 6144..9216: GPB=8 (S07 9.86->9.85); >=9216: GPB=16 (S13 21.65->21.57, 10.08TF)
    // (quant 的 GPB 最优只与 quant 形状相关, 与 attn kernel 无关)
    // env SAGEATTN_QUANT_GPB 可覆盖 (0/缺省=auto)
    int q_gpb = getenv("SAGEATTN_QUANT_GPB") ? atoi(getenv("SAGEATTN_QUANT_GPB")) : 0;
    if (q_gpb < 1) {
        if (head_dim == 64) q_gpb = (kv_len >= 9216) ? 16 : ((kv_len >= 6144) ? 8 : 4);
        else q_gpb = (kv_len >= 9216) ? 64 : ((kv_len >= 6144) ? 16 : 32);
    }
    dim3 grid_q((q_blocks + q_gpb - 1) / q_gpb, q_heads, batch);
    dim3 grid_k((k_blocks + q_gpb - 1) / q_gpb, kv_heads, batch);

    // 模板参数须为编译期常量: 按 blk_q/blk_k 分派
    #define LAUNCH_QUANT(HD, T, BQ, BK) \
        do { \
            quant_qk_int8_hnd_kernel<T, HD, BQ, MIN_BLK_Q><<<grid_q, block, 0, stream>>>( \
                reinterpret_cast<const T*>(query.data_ptr()), \
                reinterpret_cast<int8_t*>(q_int8.data_ptr()), \
                nullptr, \
                reinterpret_cast<float*>(q_scale.data_ptr()), \
                batch, q_heads, q_len, q_groups, 1, sm_scale_log2e, \
                q_sb, q_sn, q_sh, q_gpb); \
            quant_qk_int8_hnd_kernel<T, HD, BK, MIN_BLK_K><<<grid_k, block, 0, stream>>>( \
                reinterpret_cast<const T*>(key.data_ptr()), \
                reinterpret_cast<int8_t*>(k_int8.data_ptr()), \
                has_mean ? reinterpret_cast<const T*>(key_mean.data_ptr()) : nullptr, \
                reinterpret_cast<float*>(k_scale.data_ptr()), \
                batch, kv_heads, kv_len, k_groups, 0, 1.0f, \
                k_sb, k_sn, k_sh, q_gpb); \
        } while(0)

    #define LAUNCH_QUANT_DISPATCH(HD, T) \
        do { \
            if (blk_q == BLK_Q64) { LAUNCH_QUANT(HD, T, 128, 64); } \
            else if (blk_q == BLK_Q128) { LAUNCH_QUANT(HD, T, 64, 32); } \
            else { LAUNCH_QUANT(HD, T, MIN_BLK_Q, MIN_BLK_K); } \
        } while(0)

    if (query.scalar_type() == ScalarType::Half) {
        if (head_dim == 64) { LAUNCH_QUANT_DISPATCH(64, __half); }
        else { LAUNCH_QUANT_DISPATCH(128, __half); }
    } else {
        if (head_dim == 64) { LAUNCH_QUANT_DISPATCH(64, __hip_bfloat16); }
        else { LAUNCH_QUANT_DISPATCH(128, __hip_bfloat16); }
    }
    #undef LAUNCH_QUANT_DISPATCH
    #undef LAUNCH_QUANT
    return {q_int8, q_scale, k_int8, k_scale};
}

Tensor qk_int8_sv_bf16_attn_gfx110x_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    Tensor q_scale, Tensor k_scale, Tensor v_scale,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, Tensor q_fp) {

    (void)q_fp;

    const int64_t batch = query.size(0);
    const int64_t q_heads = query.size(1);
    const int64_t qo_len = query.size(2);
    const int64_t head_dim = query.size(3);
    const int64_t kv_heads = key.size(1);
    const int64_t kv_len = key.size(2);

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
    const int64_t vs_stride_b = v_scale.stride(0);
    const int64_t vs_stride_h = v_scale.stride(1);

    // 实验: SAGEATTN_INT8_WPE 选择 launch wrapper (1=wpe1 默认, 2=wpe2, 4=wpe4)
    const int wpe_sel = getenv("SAGEATTN_INT8_WPE") ? atoi(getenv("SAGEATTN_INT8_WPE")) : 1;
    // 实验: SAGEATTN_INT8_32 用每 warp 32 行 kernel (BM 恒 128, 4 warps)
    // 每 warp 32 行 kernel (BM 128, 4 warps, 2 子块共享 k_frag): D=64 self 默认启用 (实测快 7-10%)
    // SAGEATTN_INT8_32=0 可关闭
    const bool use_32w = getenv("SAGEATTN_INT8_32") ? atoi(getenv("SAGEATTN_INT8_32")) != 0 : true;
    // 实验: D=64 self 32w kernel 的 launch wrapper (SAGEATTN_D64_32W_WPE 1=wpe1 默认, 4=wpe4 高 occupancy)
    const int d64_wpe = getenv("SAGEATTN_D64_32W_WPE") ? atoi(getenv("SAGEATTN_D64_32W_WPE")) : 1;
// 实验: SAGEATTN_INT8_BN128 覆盖 D=128 BM=64 路径的 BN (0=auto, 16/32/64/128)
    // BN auto (2026-09 扫描 25+ 点): kv%512==0 且 kv>=5120 -> BN=64, 其余 BN=32.
    // kv%512==0 时大 tile 配置占优 (少迭代/少 barrier), 非 512 倍数的 KV 无此收益.
    const int bn128_ov = getenv("SAGEATTN_INT8_BN128") ? atoi(getenv("SAGEATTN_INT8_BN128")) : 0;
    const bool kv512 = (kv_len % 512) == 0;
    const int bn_auto = (kv512 && kv_len >= 5120) ? 64 : 32;
    const int bn128 = (bn128_ov != 0) ? bn128_ov : bn_auto;
    // D=128 int8 self-attn 的 BM 选择: kv%512==0 且 kv>=8192 -> BM=128 (实测 8192/8704/9216/
    // 9728/10240/16384 全胜, +1-4ms); 其余 BM=64. 旧 kv 阈值 9216 无法覆盖 9344-9600 等
    // 非 512 倍数段 (彼处 BM64 反胜 1.3-2.5ms), 故以对齐规则替代单调阈值.
    const int bm128_sel = getenv("SAGEATTN_INT8_BM128") ? atoi(getenv("SAGEATTN_INT8_BM128")) : -1;
    const int bm128_thr = getenv("SAGEATTN_INT8_BM128_THR") ? atoi(getenv("SAGEATTN_INT8_BM128_THR")) : 8192;
    bool use_bm128_d128;
    if (bm128_sel == 1) use_bm128_d128 = true;
    else if (bm128_sel == 0) use_bm128_d128 = false;
    else use_bm128_d128 = (kv512 && kv_len >= bm128_thr);
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
                reinterpret_cast<const float*>(v_scale.data_ptr()), vs_stride_b, vs_stride_h, \
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
                reinterpret_cast<const float*>(v_scale.data_ptr()), vs_stride_b, vs_stride_h, \
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
                reinterpret_cast<const float*>(v_scale.data_ptr()), vs_stride_b, vs_stride_h, \
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
                reinterpret_cast<const float*>(v_scale.data_ptr()), vs_stride_b, vs_stride_h, \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                static_cast<int>(tensor_layout)); \
        } while(0)

    #define LAUNCH_ATTN_T32_WPE1(HD, CAUSAL, BM, BN, VTYPE, OTYPE) \
        do { \
            attn_kernel_wpe1_32_t<HD, CAUSAL, BM, BN, VTYPE, OTYPE><<<grid_32, block_32, 0, stream>>>( \
                reinterpret_cast<const int8_t*>(query.data_ptr()), \
                reinterpret_cast<const int8_t*>(key.data_ptr()), \
                reinterpret_cast<const VTYPE*>(value.data_ptr()), \
                output.data_ptr(), \
                reinterpret_cast<const float*>(q_scale.data_ptr()), \
                reinterpret_cast<const float*>(k_scale.data_ptr()), \
                reinterpret_cast<const float*>(v_scale.data_ptr()), vs_stride_b, vs_stride_h, \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                static_cast<int>(tensor_layout)); \
        } while(0)

    // wpe4 (高 occupancy) 变体的通用 launch: BM 编译期常量, grid_32/block_32 由外层按 BM 计算
    #define LAUNCH_ATTN_T32_WPE4(HD, CAUSAL, BM, BN, VTYPE, OTYPE) \
        do { \
            attn_kernel_wpe4_32_t<HD, CAUSAL, BM, BN, VTYPE, OTYPE><<<grid_32, block_32, 0, stream>>>( \
                reinterpret_cast<const int8_t*>(query.data_ptr()), \
                reinterpret_cast<const int8_t*>(key.data_ptr()), \
                reinterpret_cast<const VTYPE*>(value.data_ptr()), \
                output.data_ptr(), \
                reinterpret_cast<const float*>(q_scale.data_ptr()), \
                reinterpret_cast<const float*>(k_scale.data_ptr()), \
                reinterpret_cast<const float*>(v_scale.data_ptr()), vs_stride_b, vs_stride_h, \
                batch, qo_len, kv_len, q_heads, kv_heads, \
                q_stride_b, q_stride_n, q_stride_h, \
                k_stride_b, k_stride_n, k_stride_h, \
                v_stride_b, v_stride_n, v_stride_h, \
                o_stride_b, o_stride_n, o_stride_h, \
                qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                static_cast<int>(tensor_layout)); \
        } while(0)

    // 实验: D=64 self 32w kernel 的 BN 选择 (SAGEATTN_D64_BN 16/32/64, 默认 32)
    const int d64_bn = getenv("SAGEATTN_D64_BN") ? atoi(getenv("SAGEATTN_D64_BN")) : 32;
    // 实验: D=64 self 32w kernel 的 BM (SAGEATTN_D64_32W_BM 64/128, 默认 128)
    const int d64_bm = getenv("SAGEATTN_D64_32W_BM") ? atoi(getenv("SAGEATTN_D64_32W_BM")) : 128;

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
                        const int bx32 = (d64_bm == 64) ? 64 : 128; \
                        dim3 block_32(bx32); \
                        dim3 grid_32((qo_len + d64_bm - 1) / d64_bm, q_heads, batch); \
                        if (d64_wpe == 4) { \
                            if (is_causal) { \
                                if (d64_bm == 64) { LAUNCH_ATTN_T32_WPE4(64, true, 64, 32, VT, OT); } \
                                else { LAUNCH_ATTN_T32_WPE4(64, true, 128, 32, VT, OT); } \
                            } else { \
                                if (d64_bm == 64) { LAUNCH_ATTN_T32_WPE4(64, false, 64, 32, VT, OT); } \
                                else { LAUNCH_ATTN_T32_WPE4(64, false, 128, 32, VT, OT); } \
                            } \
                        } else if (d64_bn == 16) { \
                            if (is_causal) { LAUNCH_ATTN_T32_WPE1(64, true, 128, 16, VT, OT); } \
                            else { LAUNCH_ATTN_T32_WPE1(64, false, 128, 16, VT, OT); } \
                        } else if (d64_bn == 64) { \
                            if (is_causal) { LAUNCH_ATTN_T32_WPE1(64, true, 128, 64, VT, OT); } \
                            else { LAUNCH_ATTN_T32_WPE1(64, false, 128, 64, VT, OT); } \
                        } else if (is_causal) { LAUNCH_ATTN_T32_WPE1(64, true, 128, 32, VT, OT); } \
                        else { LAUNCH_ATTN_T32_WPE1(64, false, 128, 32, VT, OT); } \
                    } else if (is_causal) { LAUNCH_ATTN_T(64, true, 128, 32, VT, OT, wpe_sel); } \
                    else { LAUNCH_ATTN_T(64, false, 128, 32, VT, OT, wpe_sel); } \
                } else if (kv_len <= 77) { \
                    /* 实验: SAGEATTN_D64_X_BN 覆盖 D64 cross 短 kv 的 BN (0=default 16) */ \
                    const int d64x_bn = getenv("SAGEATTN_D64_X_BN") ? atoi(getenv("SAGEATTN_D64_X_BN")) : 16; \
                    if (d64x_bn == 32) { \
                        if (is_causal) { LAUNCH_ATTN_T(64, true, 64, 32, VT, OT, wpe_sel); } \
                        else { LAUNCH_ATTN_T(64, false, 64, 32, VT, OT, wpe_sel); } \
                    } else if (d64x_bn == 64) { \
                        if (is_causal) { LAUNCH_ATTN_T(64, true, 64, 64, VT, OT, wpe_sel); } \
                        else { LAUNCH_ATTN_T(64, false, 64, 64, VT, OT, wpe_sel); } \
                    } else { \
                        if (is_causal) { LAUNCH_ATTN_T(64, true, 64, 16, VT, OT, wpe_sel); } \
                        else { LAUNCH_ATTN_T(64, false, 64, 16, VT, OT, wpe_sel); } \
                    } \
                } else { \
                    if (is_causal) { LAUNCH_ATTN_T(64, true, 64, 32, VT, OT, wpe_sel); } \
                    else { LAUNCH_ATTN_T(64, false, 64, 32, VT, OT, wpe_sel); } \
                } \
            } else { \
                const int wpe128 = (wpe_sel == 1) ? 2 : wpe_sel; \
                if (use_bm128_d128) { \
                    /* VAE 超长 self-attn: BM=128/8 warps, K/V 重读减半 (dispatch, 旧用例不受影响) */ \
                    if (is_causal) { LAUNCH_ATTN_T(128, true, 128, 64, VT, OT, wpe128); } \
                    else { LAUNCH_ATTN_T(128, false, 128, 64, VT, OT, wpe128); } \
                } else if (bn128 == 128) { \
                    /* 实验: BN=128 (迭代/softmax 次数减半, LDS k_tile 18KB) */ \
                    if (is_causal) { LAUNCH_ATTN_T(128, true, 64, 128, VT, OT, wpe128); } \
                    else { LAUNCH_ATTN_T(128, false, 64, 128, VT, OT, wpe128); } \
                } else if (bn128 == 16) { \
                    if (is_causal) { LAUNCH_ATTN_T(128, true, 64, 16, VT, OT, wpe128); } \
                    else { LAUNCH_ATTN_T(128, false, 64, 16, VT, OT, wpe128); } \
                } else if (bn128 == 32) { \
                    if (is_causal) { LAUNCH_ATTN_T(128, true, 64, 32, VT, OT, wpe128); } \
                    else { LAUNCH_ATTN_T(128, false, 64, 32, VT, OT, wpe128); } \
                } else { \
                    /* BN=64 (auto: kv>=6144) */ \
                    if (is_causal) { LAUNCH_ATTN_T(128, true, 64, 64, VT, OT, wpe128); } \
                    else { LAUNCH_ATTN_T(128, false, 64, 64, VT, OT, wpe128); } \
                } \
            } \
        } while(0)

    const bool v_is_i8 = (value.scalar_type() == ScalarType::Char);

    if (v_is_i8) {
        if (out_is_bf16) { LAUNCH_ATTN_ALL(int8_t, __hip_bfloat16); }
        else { LAUNCH_ATTN_ALL(int8_t, __half); }
    } else if (v_is_bf16) { LAUNCH_ATTN_ALL(__hip_bfloat16, __hip_bfloat16); }
    else if (out_is_bf16) { LAUNCH_ATTN_ALL(__half, __hip_bfloat16); }
    else { LAUNCH_ATTN_ALL(__half, __half); }
    #undef LAUNCH_ATTN_ALL
    #undef LAUNCH_ATTN_T
    #undef LAUNCH_ATTN_WPE2_T

    return output;
}

Tensor fp16_attn_gfx110x_t(
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
        // LAUNCH_FP16_BN(128, 64, 16);
        // big2cater's fix:
        // D=128 direct path. Default (BM=128, BN=16) was tuned on 780M;
        // env SAGEATTN_FP16_BM / SAGEATTN_FP16_BN select alternatives at
        // runtime for per-GPU (gfx1100) sweep experiments.
        if (bn_ov == 32) { LAUNCH_FP16_BN(128, 64, 32); }
        else if (bn_ov == 64) { LAUNCH_FP16_BN(128, 64, 64); }
        else if (bn_ov == 128) { LAUNCH_FP16_BN(128, 64, 128); }
        else if (bm_ov == 32) { LAUNCH_FP16_BN(128, 32, 16); }
        else if (bm_ov == 128) { LAUNCH_FP16_BN(128, 128, 16); }
        else if (bm_ov == 64) { LAUNCH_FP16_BN(128, 64, 16); }
        else { LAUNCH_FP16_BN(128, 64, 16); }
    }
    #undef LAUNCH_FP16_T
    #undef LAUNCH_FP16_BN
    #undef LAUNCH_FP16_BM

    return output;
}

Tensor bf16_attn_gfx110x_t(
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
        // LAUNCH_BF16_BN(128, 64, 16);
        // big2cater's fix:
        // D=128 bf16 direct path; runtime BM/BN override for gfx1100 sweep.
        if (bn_ov == 32) { LAUNCH_BF16_BN(128, 64, 32); }
        else if (bn_ov == 64) { LAUNCH_BF16_BN(128, 64, 64); }
        else if (bn_ov == 128) { LAUNCH_BF16_BN(128, 64, 128); }
        else if (bm_ov == 32) { LAUNCH_BF16_BN(128, 32, 16); }
        else if (bm_ov == 128) { LAUNCH_BF16_BN(128, 128, 16); }
        else if (bm_ov == 64) { LAUNCH_BF16_BN(128, 64, 16); }
        else { LAUNCH_BF16_BN(128, 64, 16); }
    }
    #undef LAUNCH_BF16_T
    #undef LAUNCH_BF16_BN
    #undef LAUNCH_BF16_BM

    return output;
}
