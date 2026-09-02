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
#error "attn_gfx103x.cu is only intended for ROCm/HIP."
#endif

#include "reduction_utils.cuh"

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

#include "mma_gfx10.h"
#include "attn_gfx10_new.h"
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
    // (报告 §八: MIN_BLK 调大触发 fmax 串行依赖链 +37-45%; 此处 MIN_BLK 不变, 只放大
    // 每 block 的吞吐/减少 block 调度开销, 对齐 triton 的 BLK=128 大 block quant)
    constexpr int RATIO = BLK / MIN_BLK;
    constexpr int PackElems = 8;
    // pass1 读入的原始数据缓存在 LDS (Q 128 行 D=128 = 32KB, K 64 行 = 16KB), pass2 从 LDS 读
    __shared__ float shared_amax[RATIO];
    __shared__ uint4 shared_data[(BLK * HeadDim) / 8];

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

        // ---- pass1: 读全局 (32B/thread) -> shared_data + 分组 amax (RATIO 累加器 ILP) ----
        float local_amax[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) local_amax[r] = 1e-7f;
        for (int p = tid; p < Packs / 2; p += Threads) {
            const int pack = p * 2;
            const int elem_base = pack * PackElems;
            const int row = elem_base / HeadDim;
            const int d = elem_base - row * HeadDim;
            const int64_t seq = base_row + row;
            if (seq < seq_len) {
                const int64_t in_off = static_cast<int64_t>(b) * in_stride_b + seq * in_stride_n + head * in_stride_h + d;
                const uint4 raw0 = *reinterpret_cast<const uint4*>(input + in_off);
                const uint4 raw1 = *reinterpret_cast<const uint4*>(input + in_off + 8);
                shared_data[pack] = raw0;
                shared_data[pack + 1] = raw1;
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
            } else {
                shared_data[pack] = make_uint4(0, 0, 0, 0);
                shared_data[pack + 1] = make_uint4(0, 0, 0, 0);
            }
        }
        // 归约 RATIO 个 amax (blockReduceMax 内部有 barrier, 尾部子块写 scale 需守卫防越界)
        for (int r = 0; r < RATIO; ++r) {
            const float block_amax = vllm::blockReduceMax(local_amax[r]);
            if (tid == 0) {
                shared_amax[r] = block_amax;
                if (base_row + static_cast<int64_t>(r) * MIN_BLK < seq_len) {
                    scale_out[(static_cast<int64_t>(b) * heads + head) * scale_groups + blk * RATIO + r] =
                        block_amax / 127.0f;
                }
            }
        }
        __syncthreads();
        float inv_scale[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) inv_scale[r] = 127.0f / shared_amax[r];

        // ---- pass2: 从 LDS 读 -> 量化 (按子块 r 选 scale) -> 写回 ----
        for (int p = tid; p < Packs / 2; p += Threads) {
            const int pack = p * 2;
            const int elem_base = pack * PackElems;
            const int row = elem_base / HeadDim;
            const int d = elem_base - row * HeadDim;
            const int64_t seq = base_row + row;
            if (seq < seq_len) {
                const int r = row / MIN_BLK;
                const int64_t out_off = (static_cast<int64_t>(b) * heads + head) * seq_len * HeadDim + seq * HeadDim + d;
                const uint4 raw0 = shared_data[pack];      // pass1 缓存在 LDS, 省全局重读
                const uint4 raw1 = shared_data[pack + 1];
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
        __syncthreads();  // 防下轮 blk 的 pass1 覆盖 shared_data (上轮 pass2 未读完)
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
}  // namespace


Tensor v_transpose_gfx103x(Tensor value, Tensor value_t, int64_t tensor_layout) {
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
    // 大数据量保持每 block 1 tile (Anima01 8192 tiles grid 减小反而 +5-10%, barrier
    // 串行 + L2 局部性损失)。SAGEATTN_VT_GRID: 0=auto, N=强制固定 grid
    int64_t grid_cap;
    const char* vt_grid_env = getenv("SAGEATTN_VT_GRID");
    if (vt_grid_env && atoi(vt_grid_env) > 0) {
        grid_cap = atoi(vt_grid_env);
    } else if (total_tiles <= 4096) {
        grid_cap = 192;
    } else {
        grid_cap = total_tiles;
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

Tensor mean_seq_gfx103x(Tensor input, int64_t tensor_layout) {
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

std::vector<Tensor> quant_qk_int8_gfx103x(
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
    // 大 block quant (对齐 triton BLK): 每 block 处理 BLK 行, 每 MIN_BLK 行一组
    // 独立 scale (RATIO 组) + 多累加器 ILP —— MIN_BLK 粒度 (scale 语义) 不变
    // 实测 (同进程 A/B): D=64 用 BLK_Q=128/BLK_K=64 最优 (SDXL07 -19%, SDXL13 -18%);
    // D=128 旧逻辑已达 ~100GB/s (接近带宽上限), 大 block (128/64 或 64/32) 均 +1~3%
    // (32KB LDS / 额外 blockReduce barrier), 故 D=128 保持旧逻辑 (BLK=MIN_BLK)
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
    else if (head_dim == 64) { blk_q = BLK_Q64; blk_k = BLK_K64; }
    else { blk_q = MIN_BLK_Q; blk_k = MIN_BLK_K; }
    const int q_blocks = (q_len + blk_q - 1) / blk_q;
    const int k_blocks = (kv_len + blk_k - 1) / blk_k;
    // 多 group 合并: 每 block 顺序处理 groups_per_block 个连续大 block (实验选项, 默认 1)
    int q_gpb = getenv("SAGEATTN_QUANT_GPB") ? atoi(getenv("SAGEATTN_QUANT_GPB")) : 1;
    if (q_gpb < 1) q_gpb = 1;
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

Tensor qk_int8_sv_bf16_attn_gfx103x_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    Tensor q_scale, Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, double sm_scale) {
            // q_int8/k_int8 are ALWAYS laid out as [B, H, S, D] (quant writes HND-style
            // regardless of the input tensor_layout), so the int8 tensors always use
            // HND-style sizes/strides. V_T is [B,H,D,N] always. Output follows input layout.
            const int64_t batch = query.size(0);
            const int64_t q_heads = query.size(1);
            const int64_t kv_heads = key.size(1);
            const int64_t qo_len = query.size(2);
            const int64_t kv_len = key.size(2);
            const int64_t head_dim = query.size(3);
            const int64_t q_stride_b = query.stride(0);
            const int64_t q_stride_n = query.stride(2);   // seq stride (HND int8 tensor)
            const int64_t q_stride_h = query.stride(1);   // head stride
            const int64_t k_stride_b = key.stride(0);
            const int64_t k_stride_n = key.stride(2);
            const int64_t k_stride_h = key.stride(1);
            // value is V_T [B,H,D,N] (always, regardless of input layout) -- n-dim contiguous.
            const int64_t v_stride_b = value.stride(0);
            const int64_t v_stride_n = value.stride(1);   // H stride
            const int64_t v_stride_h = value.stride(2);   // D stride
            const int64_t o_stride_b = output.stride(0);
            const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
            const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);
            const int64_t qs_stride_b = q_scale.stride(0), qs_stride_h = q_scale.stride(1);
            const int64_t ks_stride_b = k_scale.stride(0), ks_stride_h = k_scale.stride(1);
            const hipStream_t stream = current_hip_stream(query);

            // V_T 恒 fp16 (v_transpose 已转), VDT 恒 __half。ODT 由 output dtype 决定。
            const bool out_bf = (output.scalar_type() == ScalarType::BFloat16);
            // SAGEATTN_GFX10_V2: 0=v1, 1=v2 (默认), 2=auto (self 长序列用 v2, 短/交叉用 v1)
            const int v2_mode = getenv("SAGEATTN_GFX10_V2") ? atoi(getenv("SAGEATTN_GFX10_V2")) : 2;
            // SAGEATTN_GFX10_VT_GLOBAL: 0=off, 1=force v2.2 (PV from global V_T), 2=auto (self 长序列试用)
            const int vt_mode = getenv("SAGEATTN_GFX10_VT_GLOBAL") ? atoi(getenv("SAGEATTN_GFX10_VT_GLOBAL")) : 0;
            // auto 模式: BM=64 v2 在 self 长序列 (qo_len==kv_len>=512) 优 1.5-2x，
            // 但短序列/cross-attn (block 数 < 6*8=48) 用 v2 浪费 lanes，应回退 v1 (BM=32)
            const bool use_v2 = (v2_mode == 1) || (v2_mode == 2 && qo_len == kv_len && qo_len >= 512);
            // v2.2 only when v2 already on (PV-from-global needs 2-row layout)
            const bool use_v22 = use_v2 && ((vt_mode == 1) || (vt_mode == 2 && qo_len == kv_len && qo_len >= 1536));
            // SAGEATTN_GFX10_V3: 0=off, 1=force v3 (true tile layout), 2=auto (self 长序列试用)
            // v3 与 v2/v1 互斥：v3 单独走，BM=64 (4 warps × 16 rows)
            const int v3_mode = getenv("SAGEATTN_GFX10_V3") ? atoi(getenv("SAGEATTN_GFX10_V3")) : 2;
            const bool use_v3 = (v3_mode == 1) || (v3_mode == 2 && qo_len == kv_len && qo_len >= 512);
            #define L10(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 31) / 32, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_t<HD, C, BN, ODT><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout)); \
                } while (0)
            #define L10_V2(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v2_t<HD, C, BN, ODT><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout)); \
                } while (0)
            #define L10_V22(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v2_2_t<HD, C, BN, ODT><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout)); \
                } while (0)
            #define L10_V3(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v3_t<HD, C, BN, ODT><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout)); \
                } while (0)
            if (head_dim == 64) {
                // v3.4: BN=16 matches triton config; v3 wins on self long AND short cross
                const bool use_v3_long = (qo_len == kv_len && qo_len >= 512);
                const bool use_v3_short_cross = (qo_len != kv_len && qo_len >= 256);
                const int bn = (qo_len == kv_len) ? ((kv_len <= 77) ? 16 : 32) : 32;
                if (use_v3 && (use_v3_long || use_v3_short_cross)) {
                    // v3 with BN=16 (long self-attn, D=64)
                    if (is_causal) { if (out_bf) L10_V3(64, true, 16, __hip_bfloat16); else L10_V3(64, true, 16, __half); }
                    else { if (out_bf) L10_V3(64, false, 16, __hip_bfloat16); else L10_V3(64, false, 16, __half); }
                } else if (use_v22) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V22(64, true, 16, __hip_bfloat16); else L10_V22(64, true, 16, __half); }
                        else { if (out_bf) L10_V22(64, false, 16, __hip_bfloat16); else L10_V22(64, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V22(64, true, 32, __hip_bfloat16); else L10_V22(64, true, 32, __half); }
                        else { if (out_bf) L10_V22(64, false, 32, __hip_bfloat16); else L10_V22(64, false, 32, __half); }
                    }
                } else if (use_v2) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V2(64, true, 16, __hip_bfloat16); else L10_V2(64, true, 16, __half); }
                        else { if (out_bf) L10_V2(64, false, 16, __hip_bfloat16); else L10_V2(64, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V2(64, true, 32, __hip_bfloat16); else L10_V2(64, true, 32, __half); }
                        else { if (out_bf) L10_V2(64, false, 32, __hip_bfloat16); else L10_V2(64, false, 32, __half); }
                    }
                } else {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10(64, true, 16, __hip_bfloat16); else L10(64, true, 16, __half); }
                        else { if (out_bf) L10(64, false, 16, __hip_bfloat16); else L10(64, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10(64, true, 32, __hip_bfloat16); else L10(64, true, 32, __half); }
                        else { if (out_bf) L10(64, false, 32, __hip_bfloat16); else L10(64, false, 32, __half); }
                    }
                }
            } else {
                // D=128 int8 精度修复 (gfx1035):
                //   V2 (BN=32) 在 kv_len>2048 时存在 online-softmax 精度损失 (Anima01/03/05,
                //   AnimaVAE01 等 BF16 D128 int8 长序列 MaxErr 0.2+, cos<0.98)。
                //   V3 (BN=16) 在 kv_len>=2304 时精确 (mae~0.003, 与 triton 同精度)。
                //   因此 kv_len>2048 默认走 V3 (BN=16), 否则走 V2 (BN=32)。
                //   SAGEATTN_GFX10_V3 可覆盖: 1=强制 V3, 0=强制关闭 V3 (回到旧 V2 路径)。
                const bool use_v3_d128 = (v3_mode == 1) || (v3_mode != 0 && kv_len > 2048);
                const int bn = (kv_len <= 77) ? 16 : 32;
                if (use_v3_d128) {
                    if (is_causal) { if (out_bf) L10_V3(128, true, 16, __hip_bfloat16); else L10_V3(128, true, 16, __half); }
                    else { if (out_bf) L10_V3(128, false, 16, __hip_bfloat16); else L10_V3(128, false, 16, __half); }
                } else if (use_v22) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V22(128, true, 16, __hip_bfloat16); else L10_V22(128, true, 16, __half); }
                        else { if (out_bf) L10_V22(128, false, 16, __hip_bfloat16); else L10_V22(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V22(128, true, 32, __hip_bfloat16); else L10_V22(128, true, 32, __half); }
                        else { if (out_bf) L10_V22(128, false, 32, __hip_bfloat16); else L10_V22(128, false, 32, __half); }
                    }
                } else if (use_v2) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V2(128, true, 16, __hip_bfloat16); else L10_V2(128, true, 16, __half); }
                        else { if (out_bf) L10_V2(128, false, 16, __hip_bfloat16); else L10_V2(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V2(128, true, 32, __hip_bfloat16); else L10_V2(128, true, 32, __half); }
                        else { if (out_bf) L10_V2(128, false, 32, __hip_bfloat16); else L10_V2(128, false, 32, __half); }
                    }
                } else {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10(128, true, 16, __hip_bfloat16); else L10(128, true, 16, __half); }
                        else { if (out_bf) L10(128, false, 16, __hip_bfloat16); else L10(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10(128, true, 32, __hip_bfloat16); else L10(128, true, 32, __half); }
                        else { if (out_bf) L10(128, false, 32, __hip_bfloat16); else L10(128, false, 32, __half); }
                    }
                }
            }
            #undef L10
            #undef L10_V2
            #undef L10_V22
            #undef L10_V3
            return output;
}
Tensor fp16_attn_gfx103x_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, int64_t bm_sel) {
            const int64_t batch = query.size(0);
            const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
            const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
            const int64_t qo_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
            const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
            const int64_t head_dim = query.size(3);
            const int64_t q_stride_b = query.stride(0);
            const int64_t q_stride_n = (tensor_layout == kHND) ? query.stride(2) : query.stride(1);
            const int64_t q_stride_h = (tensor_layout == kHND) ? query.stride(1) : query.stride(2);
            const int64_t k_stride_b = key.stride(0);
            const int64_t k_stride_n = (tensor_layout == kHND) ? key.stride(2) : key.stride(1);
            const int64_t k_stride_h = (tensor_layout == kHND) ? key.stride(1) : key.stride(2);
            const int64_t v_stride_b = value.stride(0);
            const int64_t v_stride_n = value.stride(1);
            const int64_t v_stride_h = value.stride(2);
            const int64_t o_stride_b = output.stride(0);
            const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
            const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);
            const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;
            const hipStream_t stream = current_hip_stream(query);
            // SAGEATTN_GFX10_V2: 0=v1, 1=v2, 2=auto (self 长序列用 v2, 其他用 v1)
            const int v2_mode = getenv("SAGEATTN_GFX10_V2") ? atoi(getenv("SAGEATTN_GFX10_V2")) : 2;
            // SAGEATTN_GFX10_VT_GLOBAL: 0=off, 1=force v2.2 (PV from global V_T), 2=auto
            const int vt_mode = getenv("SAGEATTN_GFX10_VT_GLOBAL") ? atoi(getenv("SAGEATTN_GFX10_VT_GLOBAL")) : 0;
            // auto 模式: direct 路径 v2 (BM=64) 仅在 self 长序列 (qo_len==kv_len>=1024) 有正收益
            // 短 cross (kv<512) v2 反而慢 2-3x
            const bool use_v2 = (v2_mode == 1) || (v2_mode == 2 && qo_len == kv_len && qo_len >= 1024);
            const bool use_v22 = use_v2 && ((vt_mode == 1) || (vt_mode == 2 && qo_len == kv_len && qo_len >= 1536));
            // SAGEATTN_GFX10_V3: 0=off, 1=force v3, 2=auto
            // v3.4 status: BROKEN for fp16/bf16 (16 lanes × 2 halfs = D/2, only covers half row)
            //                  BROKEN for int8 PV output (each lane writes 4 cols, 60 cols uninit)
            //                  Works for int8 QK (lane reduce correct) but PV output is incomplete
            // Disable by default; use SAGEATTN_GFX10_V3=1 to opt-in if you know what you're doing
            const int v3_mode = getenv("SAGEATTN_GFX10_V3") ? atoi(getenv("SAGEATTN_GFX10_V3")) : 0;
            const bool use_v3 = (v3_mode == 1) || (v3_mode == 2 && qo_len == kv_len && qo_len >= 512);
            #define LD(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 31) / 32, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            #define LD_V2(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_v2_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            #define LD_V22(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_v2_2_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            #define LD_V3(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 127) / 128, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_v3_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            if (use_v22) {
                if (head_dim == 64) {
                    if (is_causal) LD_V22(64, true, 32, __half, __half);
                    else LD_V22(64, false, 32, __half, __half);
                } else {
                    if (is_causal) LD_V22(128, true, 32, __half, __half);
                    else LD_V22(128, false, 32, __half, __half);
                }
            } else if (use_v3) {
                if (is_causal) LD_V3(64, true, 16, __half, __half);
                else LD_V3(64, false, 16, __half, __half);
            } else if (use_v2) {
                if (head_dim == 64) {
                    if (is_causal) LD_V2(64, true, 32, __half, __half);
                    else LD_V2(64, false, 32, __half, __half);
                } else {
                    if (is_causal) LD_V2(128, true, 32, __half, __half);
                    else LD_V2(128, false, 32, __half, __half);
                }
            } else {
                if (head_dim == 64) {
                    if (is_causal) LD(64, true, 32, __half, __half);
                    else LD(64, false, 32, __half, __half);
                } else {
                    if (is_causal) LD(128, true, 32, __half, __half);
                    else LD(128, false, 32, __half, __half);
                }
            }
            #undef LD
            #undef LD_V2
            #undef LD_V22
            #undef LD_V3
            return output;
}
Tensor bf16_attn_gfx103x_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, int64_t bm_sel) {
            const int64_t batch = query.size(0);
            const int64_t q_heads = (tensor_layout == kHND) ? query.size(1) : query.size(2);
            const int64_t kv_heads = (tensor_layout == kHND) ? key.size(1) : key.size(2);
            const int64_t qo_len = (tensor_layout == kHND) ? query.size(2) : query.size(1);
            const int64_t kv_len = (tensor_layout == kHND) ? key.size(2) : key.size(1);
            const int64_t head_dim = query.size(3);
            const int64_t q_stride_b = query.stride(0);
            const int64_t q_stride_n = (tensor_layout == kHND) ? query.stride(2) : query.stride(1);
            const int64_t q_stride_h = (tensor_layout == kHND) ? query.stride(1) : query.stride(2);
            const int64_t k_stride_b = key.stride(0);
            const int64_t k_stride_n = (tensor_layout == kHND) ? key.stride(2) : key.stride(1);
            const int64_t k_stride_h = (tensor_layout == kHND) ? key.stride(1) : key.stride(2);
            const int64_t v_stride_b = value.stride(0);
            const int64_t v_stride_n = value.stride(1);
            const int64_t v_stride_h = value.stride(2);
            const int64_t o_stride_b = output.stride(0);
            const int64_t o_stride_n = (tensor_layout == kHND) ? output.stride(2) : output.stride(1);
            const int64_t o_stride_h = (tensor_layout == kHND) ? output.stride(1) : output.stride(2);
            const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2e;
            const hipStream_t stream = current_hip_stream(query);
            // SAGEATTN_GFX10_V2: 0=v1, 1=v2, 2=auto (self 长序列用 v2, 其他用 v1)
            const int v2_mode = getenv("SAGEATTN_GFX10_V2") ? atoi(getenv("SAGEATTN_GFX10_V2")) : 2;
            // SAGEATTN_GFX10_VT_GLOBAL: 0=off, 1=force v2.2 (PV from global V_T), 2=auto
            const int vt_mode = getenv("SAGEATTN_GFX10_VT_GLOBAL") ? atoi(getenv("SAGEATTN_GFX10_VT_GLOBAL")) : 0;
            // auto 模式: direct 路径 v2 (BM=64) 仅在 self 长序列 (qo_len==kv_len>=1024) 有正收益
            const bool use_v2 = (v2_mode == 1) || (v2_mode == 2 && qo_len == kv_len && qo_len >= 1024);
            const bool use_v22 = use_v2 && ((vt_mode == 1) || (vt_mode == 2 && qo_len == kv_len && qo_len >= 1536));
            // SAGEATTN_GFX10_V3: 0=off, 1=force v3, 2=auto
            // v3.4 status: BROKEN for fp16/bf16 (16 lanes × 2 halfs = D/2, only covers half row)
            //                  BROKEN for int8 PV output (each lane writes 4 cols, 60 cols uninit)
            //                  Works for int8 QK (lane reduce correct) but PV output is incomplete
            // Disable by default; use SAGEATTN_GFX10_V3=1 to opt-in if you know what you're doing
            const int v3_mode = getenv("SAGEATTN_GFX10_V3") ? atoi(getenv("SAGEATTN_GFX10_V3")) : 0;
            const bool use_v3 = (v3_mode == 1) || (v3_mode == 2 && qo_len == kv_len && qo_len >= 512);
            #define LDB(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 31) / 32, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            #define LDB_V2(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_v2_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            #define LDB_V22(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_v2_2_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            #define LDB_V3(HD, C, BN, QDT, ODT) \
                do { \
                    dim3 bd(128); dim3 gd((qo_len + 127) / 128, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_direct_v3_t<HD, C, BN, QDT, ODT><<<gd, bd, 0, stream>>>( \
                        reinterpret_cast<const QDT*>(query.data_ptr()), \
                        reinterpret_cast<const QDT*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        sm_scale_log2e, static_cast<int>(tensor_layout)); \
                } while (0)
            if (use_v22) {
                if (head_dim == 64) {
                    if (is_causal) LDB_V22(64, true, 32, __hip_bfloat16, __hip_bfloat16);
                    else LDB_V22(64, false, 32, __hip_bfloat16, __hip_bfloat16);
                } else {
                    if (is_causal) LDB_V22(128, true, 32, __hip_bfloat16, __hip_bfloat16);
                    else LDB_V22(128, false, 32, __hip_bfloat16, __hip_bfloat16);
                }
            } else if (use_v3) {
                if (is_causal) LDB_V3(64, true, 16, __hip_bfloat16, __hip_bfloat16);
                else LDB_V3(64, false, 16, __hip_bfloat16, __hip_bfloat16);
            } else if (use_v2) {
                if (head_dim == 64) {
                    if (is_causal) LDB_V2(64, true, 32, __hip_bfloat16, __hip_bfloat16);
                    else LDB_V2(64, false, 32, __hip_bfloat16, __hip_bfloat16);
                } else {
                    if (is_causal) LDB_V2(128, true, 32, __hip_bfloat16, __hip_bfloat16);
                    else LDB_V2(128, false, 32, __hip_bfloat16, __hip_bfloat16);
                }
            } else {
                if (head_dim == 64) {
                    if (is_causal) LDB(64, true, 32, __hip_bfloat16, __hip_bfloat16);
                    else LDB(64, false, 32, __hip_bfloat16, __hip_bfloat16);
                } else {
                    if (is_causal) LDB(128, true, 32, __hip_bfloat16, __hip_bfloat16);
                    else LDB(128, false, 32, __hip_bfloat16, __hip_bfloat16);
                }
            }
            #undef LDB
            #undef LDB_V2
            #undef LDB_V22
            #undef LDB_V3
            return output;
}