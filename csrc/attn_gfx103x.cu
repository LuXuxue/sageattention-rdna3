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
// TEMPORARY: fetch the int8-PV debug dump (diag bit 5, env SAGEATTN_GFX10_IPV_DBG)
// (definition placed after ipv_dbg_fetch_kernel; see further below)
}// namespace


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
    int64_t tensor_layout, double sm_scale, int64_t skip_q) {

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
    if (skip_q) {
        // in-kernel Q quant (v10 INQ): drop the q int8/scale tensors entirely;
        // K still goes through the normal prepass below.
        q_int8 = new_empty_like(query, {0}, ScalarType::Char);
        q_scale = new_empty_like(query, {0}, ScalarType::Float);
    }

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
            if (!skip_q) { \
                quant_qk_int8_hnd_kernel<T, HD, BQ, MIN_BLK_Q><<<grid_q, block, 0, stream>>>( \
                    reinterpret_cast<const T*>(query.data_ptr()), \
                    reinterpret_cast<int8_t*>(q_int8.data_ptr()), \
                    nullptr, \
                    reinterpret_cast<float*>(q_scale.data_ptr()), \
                    batch, q_heads, q_len, q_groups, 1, sm_scale_log2e, \
                    q_sb, q_sn, q_sh, q_gpb); \
            } \
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

// ---------------------------------------------------------------------------
// int8-PV: V quantized to int8 per 32-key tile (per (batch, kv_head, tile)) so
// each [D][BN] V tile gets one fp32 scale. Two passes: tile max, then quantize.
// Operates on the natural [B,H,N,D] fp16 V (layout passed in v_stride_b/n/h).
// ---------------------------------------------------------------------------
// TEMPORARY IPV debug: device buffers dumped via ipv_dbg_fetch (env
// SAGEATTN_GFX10_IPV_DBG sets diag bit 5 in the host dispatch).
__device__ unsigned g_ipv_dbg[4096];
__device__ unsigned g_ipv_once;

__global__ void ipv_dbg_zero_kernel() {
    for (int i = threadIdx.x; i < 4096; i += 256) g_ipv_dbg[i] = 0u;
    if (threadIdx.x == 0) g_ipv_once = 777u;
}
__global__ void ipv_dbg_fetch_kernel(int* out) {
    for (int i = threadIdx.x; i < 4096; i += 256) out[i] = static_cast<int>(g_ipv_dbg[i]);
}

// TEMPORARY: fetch the int8-PV debug dump (diag bit 5, env SAGEATTN_GFX10_IPV_DBG)
Tensor ipv_dbg_fetch_gfx103x(Tensor like) {
    Tensor t = new_empty_like(like, {4096}, ScalarType::Int);
    ipv_dbg_fetch_kernel<<<8, 256, 0, current_hip_stream(like)>>>(reinterpret_cast<int*>(t.data_ptr()));
    return t;
}
template <int HD, int BN, int NT = 256>
__global__ void v_tile_max_kernel(
    const __half* __restrict__ v, float* __restrict__ v_scale,
    int64_t batch, int64_t kv_heads, int64_t kv_len,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h) {
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * BN;
    const int64_t h = blockIdx.y;
    const int64_t b = blockIdx.z;
    int ni = n0 < kv_len ? static_cast<int>((kv_len - n0 < BN) ? (kv_len - n0) : BN) : 0;
    float m = 0.0f;
    // stride semantics match the v8 native V decode (proven fp16 path):
    // head index -> v_stride_n, sequence index -> v_stride_h.
    const __half* row = v + b * v_stride_b + h * v_stride_n;
    for (int i = threadIdx.x; i < HD * ni; i += NT) {
        const int n = i / HD;
        m = fmaxf(m, fabsf(__half2float(row[(n0 + n) * v_stride_h + (i % HD)])));
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && g_ipv_once == 777u) {
        g_ipv_dbg[3000] = *reinterpret_cast<const unsigned*>(&m);
        g_ipv_dbg[3001] = *reinterpret_cast<const unsigned*>(&v_stride_n);
        g_ipv_dbg[3002] = *reinterpret_cast<const unsigned*>(&v_stride_h);
    }
    // full block tree-reduce of per-thread partials (lane-0-only reduce is WRONG)
    __shared__ float shm[NT];
    shm[threadIdx.x] = m;
    __syncthreads();
    #pragma unroll
    for (int s = NT / 2; s >= 1; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] = fmaxf(shm[threadIdx.x], shm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int tiles = (kv_len + BN - 1) / BN;
        v_scale[(b * kv_heads + h) * tiles + blockIdx.x] = shm[0];
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && g_ipv_once == 777u) {
            m = shm[0];
            g_ipv_dbg[3016] = *reinterpret_cast<const unsigned*>(&m);
            for (int c = 0; c < 32; ++c) {
                const float fv = __half2float(row[c]);
                g_ipv_dbg[3020 + c] = *reinterpret_cast<const unsigned*>(&fv);
            }
        }
    }
}

template <int HD, int BN, int NT = 256>
__global__ void v_tile_quant_kernel(
    const __half* __restrict__ v, const float* __restrict__ v_scale,
    int8_t* __restrict__ v_i8,
    int64_t batch, int64_t kv_heads, int64_t kv_len,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h) {
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * BN;
    const int64_t h = blockIdx.y;
    const int64_t b = blockIdx.z;
    const int tiles = (kv_len + BN - 1) / BN;
    const float sc = v_scale[(b * kv_heads + h) * tiles + blockIdx.x];
    const float inv = (sc > 0.0f) ? (127.0f / sc) : 0.0f;
    int ni = n0 < kv_len ? static_cast<int>((kv_len - n0 < BN) ? (kv_len - n0) : BN) : 0;
    // stride semantics match the v8 native V decode (proven fp16 path):
    // head index -> v_stride_n, sequence index -> v_stride_h.
    const __half* row = v + b * v_stride_b + h * v_stride_n;
    int8_t* i8row = v_i8 + b * kv_heads * (kv_len * HD) + h * (kv_len * HD);
    for (int i = threadIdx.x; i < HD * ni; i += NT) {
        const int n = i / HD;
        const int d = i % HD;
        float f = __half2float(row[(n0 + n) * v_stride_h + d]) * inv;
        int x = static_cast<int>(fabsf(f) + 0.5f);
        if (x > 127) x = 127;
        i8row[(n0 + n) * HD + d] = (f < 0.0f) ? static_cast<int8_t>(-x) : static_cast<int8_t>(x);
    }
}

Tensor qk_int8_sv_bf16_attn_gfx103x_t(
    Tensor query, Tensor key, Tensor value, Tensor output,
    Tensor q_scale, Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, double sm_scale, Tensor q_fp) {
            // q_int8/k_int8 are ALWAYS laid out as [B, H, S, D] (quant writes HND-style
            // regardless of the input tensor_layout), so the int8 tensors always use
            // HND-style sizes/strides. V_T is [B,H,D,N] always. Output follows input layout.
            // When INQ skips the q8 prepass, query/q_scale are {0}-shaped: derive
            // the geometry/strides from the always-present fp16/bf16 q_fp view.
            const Tensor& qgeo = (query.dim() < 4) ? q_fp : query;
            const int64_t batch = qgeo.size(0);
            const int64_t q_heads = qgeo.size(1);
            const int64_t kv_heads = key.size(1);
            const int64_t qo_len = qgeo.size(2);
            const int64_t kv_len = key.size(2);
            const int64_t head_dim = qgeo.size(3);
            const int64_t q_stride_b = (query.dim() >= 4) ? query.stride(0) : 0;
            const int64_t q_stride_n = (query.dim() >= 4) ? query.stride(2) : 0;   // seq stride (HND int8 tensor)
            const int64_t q_stride_h = (query.dim() >= 4) ? query.stride(1) : 0;   // head stride
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
            const int64_t qs_stride_b = (q_scale.numel() > 0) ? q_scale.stride(0) : 0,
              qs_stride_h = (q_scale.numel() > 0) ? q_scale.stride(1) : 0;
            const int64_t ks_stride_b = k_scale.stride(0), ks_stride_h = k_scale.stride(1);
            const hipStream_t stream = current_hip_stream(query);
            // v10 INQ (in-kernel Q int8 quant): the main kernel reads the original
            // fp16/bf16 Q directly (q_fp, HND-views of the [B,H,S,D] layout) and
            // skips the q8 prepass global round-trip. Strides are the fp16/bf16
            // tensor's, not the int8 one's. Env: 0=off, 1=force, 2=auto (kv<=1024).
            const int v10_inq_mode = getenv("SAGEATTN_V10_INQ") ? atoi(getenv("SAGEATTN_V10_INQ")) : 2;
            const bool v10_inq_wanted = (v10_inq_mode == 1) || (v10_inq_mode == 2 && kv_len <= 1024);
            int64_t qi_stride_b = q_stride_b, qi_stride_n = q_stride_n, qi_stride_h = q_stride_h;
            const int q_fp_bf16 = (q_fp.scalar_type() == ScalarType::BFloat16) ? 1 : 0;
            const float v10_sms = static_cast<float>(sm_scale) * kLog2e;
            if (v10_inq_wanted && q_fp.numel() > 0) {
                qi_stride_b = q_fp.stride(0);
                qi_stride_n = q_fp.stride(2);
                qi_stride_h = q_fp.stride(1);
            }

            // V_T 恒 fp16 (v_transpose 已转), VDT 恒 __half。ODT 由 output dtype 决定。
            const bool out_bf = (output.scalar_type() == ScalarType::BFloat16);
            // SAGEATTN_GFX10_V2: 0=v1, 1=v2 (默认), 2=auto (self 长序列用 v2, 短/交叉用 v1)
            const int v2_mode = getenv("SAGEATTN_GFX10_V2") ? atoi(getenv("SAGEATTN_GFX10_V2")) : 2;
            // SAGEATTN_GFX10_VT_GLOBAL: 0=off, 1=force v2.2 (PV from global V_T), 2=auto (self 长序列试用)
            const int vt_mode = getenv("SAGEATTN_GFX10_VT_GLOBAL") ? atoi(getenv("SAGEATTN_GFX10_VT_GLOBAL")) : 0;
            // auto 模式: BM=64 v2 在 self 长序列 (qo_len==kv_len>=512) 优 1.5-2x。
            // D=64 cross-attn 的 int8 v2 (BM=64) 实测比 v1 (BM=32) 快 7-9x (见 try.md);
            // 故 v2 也用于 D=64 cross (qo_len>=256 保证 blocks 足够)。
            const bool use_v2 = (v2_mode == 1) || (v2_mode == 2 && qo_len == kv_len && qo_len >= 512)
                || (v2_mode == 2 && qo_len != kv_len && qo_len >= 256);
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
            #define L10_V4(HD, C, BN, ODT) \
                do { \
                    dim3 b10(256); \
                    dim3 g10((qo_len + 127) / 128, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v4_t<HD, C, BN, 128, ODT><<<g10, b10, 0, stream>>>( \
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
            #define L10_V4B64(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v4_t<HD, C, BN, 64, ODT><<<g10, b10, 0, stream>>>( \
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
            #define L10_V7P(HD, C, ODT) \
                do { \
                    dim3 b10(HD == 64 ? 128 : 64); \
                    dim3 g10((qo_len + (HD == 64 ? 127 : 63)) / (HD == 64 ? 128 : 64), q_heads, batch); \
                    const int v7p_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v7p_t<HD, C, 16, (HD == 64 ? 128 : 64), ODT><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v7p_diag); \
                } while (0)
            #define L10_V8P(HD, C, BNV, ODT, HG, VN) \
                do { \
                    dim3 b10((HD == 64 ? 128 : 64)); \
                    dim3 g10((qo_len + (HD == 64 ? 127 : 63)) / (HD == 64 ? 128 : 64), \
                             (q_heads + HG - 1) / HG, batch); \
                    const int v8_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v8_t<HD, C, BNV, (HD == 64 ? 128 : 64), HG, ODT><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v8_diag, (VN), nullptr, 0, 0); \
                } while (0)
            #define L10_V8PB128SB(HD, C, BNV, ODT, HG, VN) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 127) / 128, (q_heads + HG - 1) / HG, batch); \
                    const int v8_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v8_t<HD, C, BNV, 128, HG, ODT, false><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v8_diag, (VN), nullptr, 0, 0); \
                } while (0)
            #define L10_V8PB128(HD, C, BNV, ODT, HG, VN) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 127) / 128, (q_heads + HG - 1) / HG, batch); \
                    const int v8_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v8_t<HD, C, BNV, 128, HG, ODT><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v8_diag, (VN), nullptr, 0, 0); \
                } while (0)
            #define L10_V8PBM(HD, C, BNV, ODT, HG, VN, BMM) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), (q_heads + HG - 1) / HG, batch); \
                    const int v8_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v8_t<HD, C, BNV, (BMM), HG, ODT><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v8_diag, (VN), nullptr, 0, 0); \
                } while (0)
            // int8-PV variant: V is a pre-quantized int8 [B,H,N,D] tensor (v8_vint8),
            // staged to LDS as int8; P quantized to [0,127]; v_scale[] per-tile fp32.
            #define L10_V8PBM_IPV(HD, C, BNV, ODT, HG, VN, BMM) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), (q_heads + HG - 1) / HG, batch); \
                    const int64_t vi_stride_b = v_int8.stride(0); \
                    const int64_t vi_stride_n = v_int8.stride(1); \
                    const int64_t vi_stride_h = v_int8.stride(2); \
                    const int v8_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        int d = e ? atoi(e) : 0; \
                        if (getenv("SAGEATTN_GFX10_IPV_DBG")) d |= 32; \
                        return d; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v8_t<HD, C, BNV, (BMM), HG, ODT, true><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(v_int8.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        vi_stride_b, vi_stride_n, vi_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout), v8_diag, (VN), \
                        reinterpret_cast<const float*>(v_scale.data_ptr()), \
                        v_scale.stride(0), v_scale.stride(1)); \
                } while (0)
            // v9: 1-row-per-lane int8-QK + (optional) int8-PV. Env SAGEATTN_GFX10_V9=1.
            // SAGEATTN_V9_IPV: 0=off (fp16 PV), 1=on (int8 PV), 2=auto (self long on)
            #define L10_V9F(HD, C, BNV, BMM, ODT) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), q_heads, batch); \
                    const int v9_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i9_t<HD, C, BNV, (BMM), ODT, false><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v9_diag, 1, \
                        nullptr, 0, 0); \
                } while (0)
            #define L10_V9I(HD, C, BNV, BMM, ODT) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), q_heads, batch); \
                    Tensor v_i8 = new_empty_like(value, {batch, kv_heads, kv_len, head_dim}, ScalarType::Char); \
                    const int v_tiles = static_cast<int>((kv_len + 31) / 32); \
                    Tensor v_sc = new_empty_like(value, {batch, kv_heads, v_tiles}, ScalarType::Float); \
                    dim3 bv(256); \
                    dim3 gv(v_tiles, kv_heads, batch); \
                    v_tile_max_kernel<HD, 32><<<gv, bv, 0, stream>>>( \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<float*>(v_sc.data_ptr()), \
                        batch, kv_heads, kv_len, v_stride_b, v_stride_n, v_stride_h); \
                    v_tile_quant_kernel<HD, 32><<<gv, bv, 0, stream>>>( \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<float*>(v_sc.data_ptr()), \
                        reinterpret_cast<int8_t*>(v_i8.data_ptr()), \
                        batch, kv_heads, kv_len, v_stride_b, v_stride_n, v_stride_h); \
                    const int64_t vi_stride_b = v_i8.stride(0); \
                    const int64_t vi_stride_n = v_i8.stride(1); \
                    const int64_t vi_stride_h = v_i8.stride(2); \
                    const int v9_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i9_t<HD, C, BNV, (BMM), ODT, true><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(v_i8.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        vi_stride_b, vi_stride_n, vi_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout), v9_diag, 1, \
                        reinterpret_cast<const float*>(v_sc.data_ptr()), \
                        v_sc.stride(0), v_sc.stride(1)); \
                } while (0)
            // v10: register-tiled PV (Triton-style). Env SAGEATTN_GFX10_V10=1.
            // SAGEATTN_V10_BN: 16 or 32 (default 16).
            // SAGEATTN_V10_TM: rows/lane for PV tiling (2/4/8, default 2).
            #define L10_V10F(HD, C, BNV, BMM, ODT, TMV) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), q_heads, batch); \
                    const int v10_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i10_t<HD, C, BNV, (BMM), ODT, TMV><<<g10, b10, 0, stream>>>( \
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
                        static_cast<int>(tensor_layout), v10_diag, 1, \
                        nullptr, 0, 0, \
                        nullptr, 0, 0.0f); \
                } while (0)
            // v10 int8-PV (spdot4): V quantized per 32-key tile (v_i8/v_sc prepass),
            // same register-tiled PV as L10_V10F but P (int8) + V (int8) LDS reads
            // halve the V traffic. Env SAGEATTN_V10_IPV=1 (2=auto self-long, default 0).
            #define L10_V10IPVF(HD, C, BNV, BMM, ODT, TMV) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), q_heads, batch); \
                    Tensor v_i8 = new_empty_like(value, {batch, kv_heads, kv_len, head_dim}, ScalarType::Char); \
                    const int v_tiles = static_cast<int>((kv_len + 31) / 32); \
                    Tensor v_sc = new_empty_like(value, {batch, kv_heads, v_tiles}, ScalarType::Float); \
                    dim3 bv(256); \
                    dim3 gv(v_tiles, kv_heads, batch); \
                    v_tile_max_kernel<HD, 32><<<gv, bv, 0, stream>>>( \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<float*>(v_sc.data_ptr()), \
                        batch, kv_heads, kv_len, v_stride_b, v_stride_n, v_stride_h); \
                    v_tile_quant_kernel<HD, 32><<<gv, bv, 0, stream>>>( \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<float*>(v_sc.data_ptr()), \
                        reinterpret_cast<int8_t*>(v_i8.data_ptr()), \
                        batch, kv_heads, kv_len, v_stride_b, v_stride_n, v_stride_h); \
                    const int64_t vi_stride_b = v_i8.stride(0); \
                    const int64_t vi_stride_n = v_i8.stride(1); \
                    const int64_t vi_stride_h = v_i8.stride(2); \
                    const int v10_ipv_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i10_t<HD, C, BNV, (BMM), ODT, TMV, true><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(v_i8.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        q_stride_b, q_stride_n, q_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        vi_stride_b, vi_stride_n, vi_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout), v10_ipv_diag, 1, \
                        reinterpret_cast<const float*>(v_sc.data_ptr()), \
                        v_sc.stride(0), v_sc.stride(1), \
                        nullptr, 0, 0.0f); \
                } while (0)
            // v10 INQ: same register-tiled kernel but Q is quantized in-kernel from
            // the fp16/bf16 q_fp view (skips the q8 prepass round-trip) - wins when
            // kv is small (prepass dominates end-to-end). Env SAGEATTN_V10_INQ.
            #define L10_V10INQF(HD, C, BNV, BMM, ODT, TMV) \
                do { \
                    dim3 b10((BMM)); \
                    dim3 g10((qo_len + (BMM) - 1) / (BMM), q_heads, batch); \
                    const int v10_diag = []() { \
                        const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                        return e ? atoi(e) : 0; \
                    }(); \
                    sageattn_gfx10::attn_kernel_gfx10_i10_t<HD, C, BNV, (BMM), ODT, TMV, false, true><<<g10, b10, 0, stream>>>( \
                        reinterpret_cast<const int8_t*>(query.data_ptr()), \
                        reinterpret_cast<const int8_t*>(key.data_ptr()), \
                        reinterpret_cast<const __half*>(value.data_ptr()), \
                        reinterpret_cast<ODT*>(output.data_ptr()), \
                        reinterpret_cast<const float*>(q_scale.data_ptr()), \
                        reinterpret_cast<const float*>(k_scale.data_ptr()), \
                        batch, qo_len, kv_len, q_heads, kv_heads, \
                        qi_stride_b, qi_stride_n, qi_stride_h, q_stride_n, \
                        k_stride_b, k_stride_n, k_stride_h, \
                        v_stride_b, v_stride_n, v_stride_h, \
                        o_stride_b, o_stride_n, o_stride_h, \
                        qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h, \
                        static_cast<int>(tensor_layout), v10_diag, 1, \
                        nullptr, 0, 0, \
                        reinterpret_cast<const void*>(q_fp.data_ptr()), q_fp_bf16, v10_sms); \
                } while (0)
            #define L10_V2F(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v2f_t<HD, C, BN, ODT><<<g10, b10, 0, stream>>>( \
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
            #define L10_V2X(HD, C, BN, ODT) \
                do { \
                    dim3 b10(128); \
                    dim3 g10((qo_len + 63) / 64, q_heads, batch); \
                    sageattn_gfx10::attn_kernel_gfx10_i8_v2x_t<HD, C, BN, ODT><<<g10, b10, 0, stream>>>( \
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
                // v9: 1-row-per-lane int8-QK + optional int8-PV. Env SAGEATTN_GFX10_V9=1.
                // SAGEATTN_V9_BN: 16 or 32 (default 32). SAGEATTN_V9_IPV: 0=fp16 PV, 1=int8, 2=auto.
                #define V9DISPATCH(HD_, BNV, BMM, IPV) \
                    if (is_causal) { \
                        if (out_bf) { if (IPV) L10_V9I(HD_, true, BNV, BMM, __hip_bfloat16); else L10_V9F(HD_, true, BNV, BMM, __hip_bfloat16); } \
                        else { if (IPV) L10_V9I(HD_, true, BNV, BMM, __half); else L10_V9F(HD_, true, BNV, BMM, __half); } \
                    } else { \
                        if (out_bf) { if (IPV) L10_V9I(HD_, false, BNV, BMM, __hip_bfloat16); else L10_V9F(HD_, false, BNV, BMM, __hip_bfloat16); } \
                        else { if (IPV) L10_V9I(HD_, false, BNV, BMM, __half); else L10_V9F(HD_, false, BNV, BMM, __half); } \
                    }
                const bool v9_mode = getenv("SAGEATTN_GFX10_V9") ? (atoi(getenv("SAGEATTN_GFX10_V9")) == 1) : false;
                if (v9_mode) {
                    const int v9_ipv = getenv("SAGEATTN_V9_IPV") ? atoi(getenv("SAGEATTN_V9_IPV")) : 2;
                    const bool ipv_on = (v9_ipv == 1) || (v9_ipv == 2 && qo_len == kv_len && qo_len >= 512);
                    const int v9_bn = getenv("SAGEATTN_V9_BN") ? atoi(getenv("SAGEATTN_V9_BN")) : 32;
                    if (v9_bn == 16) { V9DISPATCH(64, 16, 128, ipv_on); } else { V9DISPATCH(64, 32, 128, ipv_on); }
                } else {
                // Mode chain for D=64 (legacy + v10):
                // v2/v3/v4/v7p/v8/v8r are the historical kernels; v10 adds a
                // register-tiled PV (Triton-style). All env-gated for A/B.
                const bool force_v3 = getenv("SAGEATTN_GFX10_V4") ? (atoi(getenv("SAGEATTN_GFX10_V4")) == 0) : false;
                const int v4_mode = getenv("SAGEATTN_GFX10_V4") ? atoi(getenv("SAGEATTN_GFX10_V4")) : 0;
                const bool use_v4_long = (qo_len == kv_len && qo_len >= 512);
                const bool use_v4_short_cross = (qo_len != kv_len && qo_len >= 256);
                const int bn = (qo_len == kv_len) ? ((kv_len <= 77) ? 16 : 32) : 32;
                const bool v7p_mode = getenv("SAGEATTN_GFX10_V7P") ? (atoi(getenv("SAGEATTN_GFX10_V7P")) == 1) : false;
                const bool v8_mode = getenv("SAGEATTN_GFX10_V8") ? (atoi(getenv("SAGEATTN_GFX10_V8")) == 1) : false;
                const bool v8r_mode = getenv("SAGEATTN_GFX10_V8R") ? (atoi(getenv("SAGEATTN_GFX10_V8R")) == 1) : false;
                const bool v10_mode = getenv("SAGEATTN_GFX10_V10") ? (atoi(getenv("SAGEATTN_GFX10_V10")) == 1) : true;
                const int v10_ipv = getenv("SAGEATTN_V10_IPV") ? atoi(getenv("SAGEATTN_V10_IPV")) : 0;
                const bool v10_ipv_on = (v10_ipv == 1) || (v10_ipv == 2 && qo_len == kv_len && qo_len >= 512);
                if (v10_mode) {
                    // register-tiled PV (Triton-style). SAGEATTN_V10_BN: 16 or 32.
                    const int v10_bn = getenv("SAGEATTN_V10_BN") ? atoi(getenv("SAGEATTN_V10_BN")) : 32;
                    const int v10_tm = getenv("SAGEATTN_V10_TM") ? atoi(getenv("SAGEATTN_V10_TM")) : 8;
                    if (is_causal) {
                        if (v10_bn == 32) {
                            if (out_bf) {
                                if (v10_tm == 4) L10_V10F(64, true, 32, 128, __hip_bfloat16, 4);
                                else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(64, true, 32, 128, __hip_bfloat16, 8); else L10_V10F(64, true, 32, 128, __hip_bfloat16, 8); }
                                else L10_V10F(64, true, 32, 128, __hip_bfloat16, 2);
                            } else {
                                if (v10_tm == 4) L10_V10F(64, true, 32, 128, __half, 4);
                                else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(64, true, 32, 128, __half, 8); else L10_V10F(64, true, 32, 128, __half, 8); }
                                else L10_V10F(64, true, 32, 128, __half, 2);
                            }
                        } else {
                            if (out_bf) {
                                if (v10_tm == 4) L10_V10F(64, true, 16, 128, __hip_bfloat16, 4);
                                else if (v10_tm == 8) L10_V10F(64, true, 16, 128, __hip_bfloat16, 8);
                                else L10_V10F(64, true, 16, 128, __hip_bfloat16, 2);
                            } else {
                                if (v10_tm == 4) L10_V10F(64, true, 16, 128, __half, 4);
                                else if (v10_tm == 8) L10_V10F(64, true, 16, 128, __half, 8);
                                else L10_V10F(64, true, 16, 128, __half, 2);
                            }
                        }
                    } else {
                        if (v10_inq_wanted && !v10_ipv_on) {
                            if (out_bf) {
                                if (v10_tm == 4) L10_V10INQF(64, false, 32, 128, __hip_bfloat16, 4);
                                else if (v10_tm == 8) L10_V10INQF(64, false, 32, 128, __hip_bfloat16, 8);
                                else L10_V10INQF(64, false, 32, 128, __hip_bfloat16, 2);
                            } else {
                                if (v10_tm == 4) L10_V10INQF(64, false, 32, 128, __half, 4);
                                else if (v10_tm == 8) L10_V10INQF(64, false, 32, 128, __half, 8);
                                else L10_V10INQF(64, false, 32, 128, __half, 2);
                            }
                        } else
                        if (v10_bn == 32) {
                            if (out_bf) {
                                if (v10_tm == 4) L10_V10F(64, false, 32, 128, __hip_bfloat16, 4);
                                else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(64, false, 32, 128, __hip_bfloat16, 8); else L10_V10F(64, false, 32, 128, __hip_bfloat16, 8); }
                                else L10_V10F(64, false, 32, 128, __hip_bfloat16, 2);
                            } else {
                                if (v10_tm == 4) L10_V10F(64, false, 32, 128, __half, 4);
                                else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(64, false, 32, 128, __half, 8); else L10_V10F(64, false, 32, 128, __half, 8); }
                                else L10_V10F(64, false, 32, 128, __half, 2);
                            }
                        } else {
                            if (out_bf) {
                                if (v10_tm == 4) L10_V10F(64, false, 16, 128, __hip_bfloat16, 4);
                                else if (v10_tm == 8) L10_V10F(64, false, 16, 128, __hip_bfloat16, 8);
                                else L10_V10F(64, false, 16, 128, __hip_bfloat16, 2);
                            } else {
                                if (v10_tm == 4) L10_V10F(64, false, 16, 128, __half, 4);
                                else if (v10_tm == 8) L10_V10F(64, false, 16, 128, __half, 8);
                                else L10_V10F(64, false, 16, 128, __half, 2);
                            }
                        }
                    }
                } else if (v8_mode) {
                    // v8: int4-vectorized staging. v_native=1 (default): natural [B,H,N,D]
                    // V (d-contiguous full-sector reads, 18ms @ H=8); V_T n-contiguous is
                    // 2.4x slower (31.7ms), so V_T is only kept for V7P/V2 paths.
                    const int v8_vt = getenv("SAGEATTN_GFX10_V8_VT") ? (atoi(getenv("SAGEATTN_GFX10_V8_VT")) == 1) : 0;
                    const int v8_bn = getenv("SAGEATTN_V8_BN") ? atoi(getenv("SAGEATTN_V8_BN")) : 16;
                    #define L10_V8PB(BNV) \
                        if (is_causal) { if (out_bf) L10_V8P(64, true, BNV, __hip_bfloat16, 1, v8_vt ? 0 : 1); \
                                      else L10_V8P(64, true, BNV, __half, 1, v8_vt ? 0 : 1); } \
                        else { if (out_bf) L10_V8P(64, false, BNV, __hip_bfloat16, 1, v8_vt ? 0 : 1); \
                             else L10_V8P(64, false, BNV, __half, 1, v8_vt ? 0 : 1); }
                    if (v8_bn == 32) { L10_V8PB(32); } else { L10_V8PB(16); }
                    #undef L10_V8PB
                } else if (v8r_mode) {
                    // v8r-v3: row-per-thread PV (16 keys/lane, BM=128, NTHREAD=128).
                    // HD=64, BN=16 only.
                    #define L10_V8R(ISCV, ODT) \
                        do { \
                            dim3 b10(128); \
                            dim3 g10((qo_len + 127) / 128, q_heads, batch); \
                            const int v8r_diag = []() { \
                                const char* e = getenv("SAGEATTN_V7P_DIAG"); \
                                return e ? atoi(e) : 0; \
                            }(); \
                            sageattn_gfx10::attn_kernel_gfx10_i8_v8r_t<64, ISCV, 16, 128, ODT><<<g10, b10, 0, stream>>>( \
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
                                static_cast<int>(tensor_layout), v8r_diag, 1); \
                        } while (0)
                    if (is_causal) { if (out_bf) { L10_V8R(true, __hip_bfloat16); } else { L10_V8R(true, __half); } } \
                    else { if (out_bf) { L10_V8R(false, __hip_bfloat16); } else { L10_V8R(false, __half); } }
                    #undef L10_V8R
                } else if (v7p_mode) {
                    if (is_causal) { if (out_bf) L10_V7P(64, true, __hip_bfloat16); else L10_V7P(64, true, __half); }
                    else { if (out_bf) L10_V7P(64, false, __hip_bfloat16); else L10_V7P(64, false, __half); }
                } else if (force_v3) {
                    // v3 fallback for A/B benchmarking
                    if (is_causal) { if (out_bf) L10_V3(64, true, 16, __hip_bfloat16); else L10_V3(64, true, 16, __half); }
                    else { if (out_bf) L10_V3(64, false, 16, __hip_bfloat16); else L10_V3(64, false, 16, __half); }
                } else if (v4_mode == 1) {
                    // v4 forced
                    if (bn == 32) {
                        if (is_causal) { if (out_bf) L10_V4(64, true, 32, __hip_bfloat16); else L10_V4(64, true, 32, __half); }
                        else { if (out_bf) L10_V4(64, false, 32, __hip_bfloat16); else L10_V4(64, false, 32, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V4(64, true, 16, __hip_bfloat16); else L10_V4(64, true, 16, __half); }
                        else { if (out_bf) L10_V4(64, false, 16, __hip_bfloat16); else L10_V4(64, false, 16, __half); }
                    }
                } else if (use_v22) {
                    // v2.2 from global V_T (auto when qo_len==kv_len>=1536)
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V22(64, true, 16, __hip_bfloat16); else L10_V22(64, true, 16, __half); }
                        else { if (out_bf) L10_V22(64, false, 16, __hip_bfloat16); else L10_V22(64, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V22(64, true, 32, __hip_bfloat16); else L10_V22(64, true, 32, __half); }
                        else { if (out_bf) L10_V22(64, false, 32, __hip_bfloat16); else L10_V22(64, false, 32, __half); }
                    }
                } else {
// gfx1035 D=64 default: v8 BM=128 BN=32 native V (2.3-6x faster than v2/v8-BN16).
                    // Override with SAGEATTN_GFX10_V8_D64=0 to restore the old v2 default.
                    const bool v8d64_default = getenv("SAGEATTN_GFX10_V8_D64") ? (atoi(getenv("SAGEATTN_GFX10_V8_D64")) != 0) : true;
                    if (v8d64_default) {
                        if (is_causal) { if (out_bf) L10_V8P(64, true, 32, __hip_bfloat16, 1, 1); else L10_V8P(64, true, 32, __half, 1, 1); }
                        else { if (out_bf) L10_V8P(64, false, 32, __hip_bfloat16, 1, 1); else L10_V8P(64, false, 32, __half, 1, 1); }
                    } else {
                    // v2 default (highest occupancy on gfx1035)
                    if (bn == 32) {
                        if (is_causal) { if (out_bf) L10_V2(64, true, 32, __hip_bfloat16); else L10_V2(64, true, 32, __half); }
                        else { if (out_bf) L10_V2(64, false, 32, __hip_bfloat16); else L10_V2(64, false, 32, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V2(64, true, 16, __hip_bfloat16); else L10_V2(64, true, 16, __half); }
                        else { if (out_bf) L10_V2(64, false, 16, __hip_bfloat16); else L10_V2(64, false, 16, __half); }
                    }
                }
            }}} else {
                // D=128 int8 (gfx1035):

                // v10 (SAGEATTN_GFX10_V10=1): register-tiled PV, same as D64.
                // SAGEATTN_V10_BN (16/32), SAGEATTN_V10_BM_T (64/128), SAGEATTN_V10_TM (2/4/8).
                const bool v10_mode_d128 = getenv("SAGEATTN_GFX10_V10") ? (atoi(getenv("SAGEATTN_GFX10_V10")) == 1) : true;
                const int v10_ipv = getenv("SAGEATTN_V10_IPV") ? atoi(getenv("SAGEATTN_V10_IPV")) : 0;
                const bool v10_ipv_on = (v10_ipv == 1) || (v10_ipv == 2 && qo_len == kv_len && qo_len >= 512);
                if (v10_mode_d128) {
                    // SAGEATTN_V10_D128_* override the shared SAGEATTN_V10_* when set,
                    // so D64 (BN32/TM8) and D128 (BN16/BM128/TM8) can differ.
                    const int v10_bn = getenv("SAGEATTN_V10_D128_BN") ? atoi(getenv("SAGEATTN_V10_D128_BN")) : (getenv("SAGEATTN_V10_BN") ? atoi(getenv("SAGEATTN_V10_BN")) : 16);
                    const int v10_bm = getenv("SAGEATTN_V10_D128_BM") ? atoi(getenv("SAGEATTN_V10_D128_BM")) : (getenv("SAGEATTN_V10_BM") ? atoi(getenv("SAGEATTN_V10_BM")) : 128);
                    const int v10_tm = getenv("SAGEATTN_V10_D128_TM") ? atoi(getenv("SAGEATTN_V10_D128_TM")) : (getenv("SAGEATTN_V10_TM") ? atoi(getenv("SAGEATTN_V10_TM")) : 8);
                    if (v10_bm == 128) {
                        if (v10_bn == 32) {
                            if (is_causal) {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, true, 32, 128, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) L10_V10F(128, true, 32, 128, __hip_bfloat16, 8);
                                    else L10_V10F(128, true, 32, 128, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, true, 32, 128, __half, 4);
                                    else if (v10_tm == 8) L10_V10F(128, true, 32, 128, __half, 8);
                                    else L10_V10F(128, true, 32, 128, __half, 2);
                                }
                            } else {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, false, 32, 128, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) L10_V10F(128, false, 32, 128, __hip_bfloat16, 8);
                                    else L10_V10F(128, false, 32, 128, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, false, 32, 128, __half, 4);
                                    else if (v10_tm == 8) L10_V10F(128, false, 32, 128, __half, 8);
                                    else L10_V10F(128, false, 32, 128, __half, 2);
                                }
                            }
                        } else {
                            if (is_causal) {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, true, 16, 128, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(128, true, 16, 128, __hip_bfloat16, 8); else L10_V10F(128, true, 16, 128, __hip_bfloat16, 8); }
                                    else L10_V10F(128, true, 16, 128, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, true, 16, 128, __half, 4);
                                    else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(128, true, 16, 128, __half, 8); else L10_V10F(128, true, 16, 128, __half, 8); }
                                    else L10_V10F(128, true, 16, 128, __half, 2);
                                }
                            } else {
                                if (v10_inq_wanted && !v10_ipv_on) {
                                    if (out_bf) {
                                        if (v10_tm == 4) L10_V10INQF(128, false, 16, 128, __hip_bfloat16, 4);
                                        else if (v10_tm == 8) L10_V10INQF(128, false, 16, 128, __hip_bfloat16, 8);
                                        else L10_V10INQF(128, false, 16, 128, __hip_bfloat16, 2);
                                    } else {
                                        if (v10_tm == 4) L10_V10INQF(128, false, 16, 128, __half, 4);
                                        else if (v10_tm == 8) L10_V10INQF(128, false, 16, 128, __half, 8);
                                        else L10_V10INQF(128, false, 16, 128, __half, 2);
                                    }
                                } else
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, false, 16, 128, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(128, false, 16, 128, __hip_bfloat16, 8); else L10_V10F(128, false, 16, 128, __hip_bfloat16, 8); }
                                    else L10_V10F(128, false, 16, 128, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, false, 16, 128, __half, 4);
                                    else if (v10_tm == 8) { if (v10_ipv_on) L10_V10IPVF(128, false, 16, 128, __half, 8); else L10_V10F(128, false, 16, 128, __half, 8); }
                                    else L10_V10F(128, false, 16, 128, __half, 2);
                                }
                            }
                        }
                    } else {
                        if (v10_bn == 32) {
                            if (is_causal) {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, true, 32, 64, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) L10_V10F(128, true, 32, 64, __hip_bfloat16, 8);
                                    else L10_V10F(128, true, 32, 64, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, true, 32, 64, __half, 4);
                                    else if (v10_tm == 8) L10_V10F(128, true, 32, 64, __half, 8);
                                    else L10_V10F(128, true, 32, 64, __half, 2);
                                }
                            } else {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, false, 32, 64, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) L10_V10F(128, false, 32, 64, __hip_bfloat16, 8);
                                    else L10_V10F(128, false, 32, 64, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, false, 32, 64, __half, 4);
                                    else if (v10_tm == 8) L10_V10F(128, false, 32, 64, __half, 8);
                                    else L10_V10F(128, false, 32, 64, __half, 2);
                                }
                            }
                        } else {
                            if (is_causal) {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, true, 16, 64, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) L10_V10F(128, true, 16, 64, __hip_bfloat16, 8);
                                    else L10_V10F(128, true, 16, 64, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, true, 16, 64, __half, 4);
                                    else if (v10_tm == 8) L10_V10F(128, true, 16, 64, __half, 8);
                                    else L10_V10F(128, true, 16, 64, __half, 2);
                                }
                            } else {
                                if (out_bf) {
                                    if (v10_tm == 4) L10_V10F(128, false, 16, 64, __hip_bfloat16, 4);
                                    else if (v10_tm == 8) L10_V10F(128, false, 16, 64, __hip_bfloat16, 8);
                                    else L10_V10F(128, false, 16, 64, __hip_bfloat16, 2);
                                } else {
                                    if (v10_tm == 4) L10_V10F(128, false, 16, 64, __half, 4);
                                    else if (v10_tm == 8) L10_V10F(128, false, 16, 64, __half, 8);
                                    else L10_V10F(128, false, 16, 64, __half, 2);
                                }
                            }
                        }
                    }
                } else {
                // v9 (SAGEATTN_GFX10_V9=1): 1-row-per-lane int8-QK + int8-PV.
                const bool v9_mode_d128 = getenv("SAGEATTN_GFX10_V9") ? (atoi(getenv("SAGEATTN_GFX10_V9")) == 1) : false;
                if (v9_mode_d128) {
                    const int v9_ipv = getenv("SAGEATTN_V9_IPV") ? atoi(getenv("SAGEATTN_V9_IPV")) : 2;
                    const bool ipv_on = (v9_ipv == 1) || (v9_ipv == 2 && qo_len == kv_len && qo_len >= 512);
                    const int v9_bn = getenv("SAGEATTN_V9_BN") ? atoi(getenv("SAGEATTN_V9_BN")) : 32;
                    const int v9_bm = getenv("SAGEATTN_V9_BM") ? atoi(getenv("SAGEATTN_V9_BM")) : 64;
                    if (v9_bn == 16) { if (v9_bm == 64) { V9DISPATCH(128, 16, 64, ipv_on); } else { V9DISPATCH(128, 16, 128, ipv_on); } } else { if (v9_bm == 64) { V9DISPATCH(128, 32, 64, ipv_on); } else { V9DISPATCH(128, 32, 128, ipv_on); } }
                } else {                
                //   v8 (BM=64/BN=32, env SAGEATTN_GFX10_V8_D128) is now the default:
                //   ~3.7x faster than direct, ~6.7x faster than v4 (104 vs 386/696 ms @4096 self).
                //   Legacy env-gated paths (exp_v4, exp_v2x, force_v22, force_v2) kept for
                //   benchmarking. If env flags set, they take priority.
                const bool force_v2   = (v2_mode == 1);
                const bool force_v22  = (vt_mode == 1);
                const bool exp_v2x = getenv("SAGEATTN_EXP_PVFP32") ? atoi(getenv("SAGEATTN_EXP_PVFP32")) == 1 : false;
                const bool exp_v4 = getenv("SAGEATTN_EXP_V4_D128") ? atoi(getenv("SAGEATTN_EXP_V4_D128")) == 1 : false;
                const int bn = (kv_len <= 77) ? 16 : 32;
                // SAGEATTN_GFX10_V7P: Triton-style 2-stage SW pipeline (BM=64/BN=16 for D=128).
                const bool v7p_mode = getenv("SAGEATTN_GFX10_V7P") ? (atoi(getenv("SAGEATTN_GFX10_V7P")) == 1) : false;
                // SAGEATTN_GFX10_V8_D128: v8 row-per-thread int8-QK kernel at HD=128 (default ON
                // for gfx103x; set to 0 to fall back to v4). Int8 QK (4 MAC/inst) targets
                // the fp16 D=128 self gap (direct path is fdot2-throughput-bound).
                const bool v8d128_mode = getenv("SAGEATTN_GFX10_V8_D128") ? (atoi(getenv("SAGEATTN_GFX10_V8_D128")) == 1) : true;
                // SAGEATTN_GFX10_V8_D128_BM: block rows (64/128/256, default 64).
                // SAGEATTN_GFX10_V8_D128_BN: keys per tile (16/32, default 32).
                // SAGEATTN_GFX10_V8_D128_PV8: int8-PV (ON=1, default 0). Forces BN=32.
                int v8d128_bm = 64, v8d128_bnv = 32;
                if (getenv("SAGEATTN_GFX10_V8_D128_BM")) { int e = atoi(getenv("SAGEATTN_GFX10_V8_D128_BM")); if (e == 64 || e == 128 || e == 256) v8d128_bm = e; }
                if (getenv("SAGEATTN_GFX10_V8_D128_BN")) { int e = atoi(getenv("SAGEATTN_GFX10_V8_D128_BN")); if (e == 16 || e == 32) v8d128_bnv = e; }
                const bool pv8_mode = getenv("SAGEATTN_GFX10_V8_D128_PV8") ? (atoi(getenv("SAGEATTN_GFX10_V8_D128_PV8")) == 1) : false;
                if (v8d128_mode) {
                    if (pv8_mode) {
                        // V int8 quant per 32-key tile, then the int8-PV kernel (BN=32).
                        if (getenv("SAGEATTN_GFX10_IPV_DBG")) {
                            ipv_dbg_zero_kernel<<<1, 256, 0, stream>>>();
                            fprintf(stderr, "[qpv] sb=%lld sn=%lld sh=%lld kv=%lld h=%lld\n",
                                    (long long)v_stride_b, (long long)v_stride_n, (long long)v_stride_h,
                                    (long long)kv_len, (long long)q_heads);
                        }
                        Tensor v_int8 = new_empty_like(value, {batch, kv_heads, kv_len, head_dim}, ScalarType::Char);
                        const int v_tiles = static_cast<int>((kv_len + 31) / 32);
                        Tensor v_scale = new_empty_like(value, {batch, kv_heads, v_tiles}, ScalarType::Float);
                        dim3 bv(256);
                        dim3 gv(v_tiles, kv_heads, batch);
                        v_tile_max_kernel<128, 32><<<gv, bv, 0, stream>>>(
                            reinterpret_cast<const __half*>(value.data_ptr()),
                            reinterpret_cast<float*>(v_scale.data_ptr()),
                            batch, kv_heads, kv_len, v_stride_b, v_stride_n, v_stride_h);
                        v_tile_quant_kernel<128, 32><<<gv, bv, 0, stream>>>(
                            reinterpret_cast<const __half*>(value.data_ptr()),
                            reinterpret_cast<float*>(v_scale.data_ptr()),
                            reinterpret_cast<int8_t*>(v_int8.data_ptr()),
                            batch, kv_heads, kv_len, v_stride_b, v_stride_n, v_stride_h);
                        // int8 tensor is contiguous [B,H,N,D]: real byte strides set inside macro.
                        if (v8d128_bm == 128) {
                            if (is_causal) { if (out_bf) L10_V8PBM_IPV(128, true, 32, __hip_bfloat16, 1, 1, 128); else L10_V8PBM_IPV(128, true, 32, __half, 1, 1, 128); }
                            else { if (out_bf) L10_V8PBM_IPV(128, false, 32, __hip_bfloat16, 1, 1, 128); else L10_V8PBM_IPV(128, false, 32, __half, 1, 1, 128); }
                        } else if (v8d128_bm == 256) {
                            if (is_causal) { if (out_bf) L10_V8PBM_IPV(128, true, 32, __hip_bfloat16, 1, 1, 256); else L10_V8PBM_IPV(128, true, 32, __half, 1, 1, 256); }
                            else { if (out_bf) L10_V8PBM_IPV(128, false, 32, __hip_bfloat16, 1, 1, 256); else L10_V8PBM_IPV(128, false, 32, __half, 1, 1, 256); }
                        } else {
                            if (is_causal) { if (out_bf) L10_V8PBM_IPV(128, true, 32, __hip_bfloat16, 1, 1, 64); else L10_V8PBM_IPV(128, true, 32, __half, 1, 1, 64); }
                            else { if (out_bf) L10_V8PBM_IPV(128, false, 32, __hip_bfloat16, 1, 1, 64); else L10_V8PBM_IPV(128, false, 32, __half, 1, 1, 64); }
                        }
                    } else {
                    if (v8d128_bnv == 32) {
                        if (v8d128_bm == 64) {
                            if (is_causal) { if (out_bf) L10_V8PBM(128, true, 32, __hip_bfloat16, 1, 1, 64); else L10_V8PBM(128, true, 32, __half, 1, 1, 64); }
                            else { if (out_bf) L10_V8PBM(128, false, 32, __hip_bfloat16, 1, 1, 64); else L10_V8PBM(128, false, 32, __half, 1, 1, 64); }
                        } else if (v8d128_bm == 256) {
                            if (is_causal) { if (out_bf) L10_V8PBM(128, true, 32, __hip_bfloat16, 1, 1, 256); else L10_V8PBM(128, true, 32, __half, 1, 1, 256); }
                            else { if (out_bf) L10_V8PBM(128, false, 32, __hip_bfloat16, 1, 1, 256); else L10_V8PBM(128, false, 32, __half, 1, 1, 256); }
                        } else {
                            if (is_causal) { if (out_bf) L10_V8PBM(128, true, 32, __hip_bfloat16, 1, 1, 128); else L10_V8PBM(128, true, 32, __half, 1, 1, 128); }
                            else { if (out_bf) L10_V8PBM(128, false, 32, __hip_bfloat16, 1, 1, 128); else L10_V8PBM(128, false, 32, __half, 1, 1, 128); }
                        }
                    } else {
                        if (v8d128_bm == 64) {
                            if (is_causal) { if (out_bf) L10_V8PBM(128, true, 16, __hip_bfloat16, 1, 1, 64); else L10_V8PBM(128, true, 16, __half, 1, 1, 64); }
                            else { if (out_bf) L10_V8PBM(128, false, 16, __hip_bfloat16, 1, 1, 64); else L10_V8PBM(128, false, 16, __half, 1, 1, 64); }
                        } else if (v8d128_bm == 256) {
                            if (is_causal) { if (out_bf) L10_V8PBM(128, true, 16, __hip_bfloat16, 1, 1, 256); else L10_V8PBM(128, true, 16, __half, 1, 1, 256); }
                            else { if (out_bf) L10_V8PBM(128, false, 16, __hip_bfloat16, 1, 1, 256); else L10_V8PBM(128, false, 16, __half, 1, 1, 256); }
                        } else {
                            if (is_causal) { if (out_bf) L10_V8PBM(128, true, 16, __hip_bfloat16, 1, 1, 128); else L10_V8PBM(128, true, 16, __half, 1, 1, 128); }
                            else { if (out_bf) L10_V8PBM(128, false, 16, __hip_bfloat16, 1, 1, 128); else L10_V8PBM(128, false, 16, __half, 1, 1, 128); }
                        }
                    }
                    }
                } else if (v7p_mode) {
                    if (is_causal) { if (out_bf) L10_V7P(128, true, __hip_bfloat16); else L10_V7P(128, true, __half); }
                    else { if (out_bf) L10_V7P(128, false, __hip_bfloat16); else L10_V7P(128, false, __half); }
                } else if (exp_v4) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V4B64(128, true, 16, __hip_bfloat16); else L10_V4B64(128, true, 16, __half); }
                        else { if (out_bf) L10_V4B64(128, false, 16, __hip_bfloat16); else L10_V4B64(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V4B64(128, true, 32, __hip_bfloat16); else L10_V4B64(128, true, 32, __half); }
                        else { if (out_bf) L10_V4B64(128, false, 32, __hip_bfloat16); else L10_V4B64(128, false, 32, __half); }
                    }
                } else if (exp_v2x) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V2X(128, true, 16, __hip_bfloat16); else L10_V2X(128, true, 16, __half); }
                        else { if (out_bf) L10_V2X(128, false, 16, __hip_bfloat16); else L10_V2X(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V2X(128, true, 32, __hip_bfloat16); else L10_V2X(128, true, 32, __half); }
                        else { if (out_bf) L10_V2X(128, false, 32, __hip_bfloat16); else L10_V2X(128, false, 32, __half); }
                    }
                } else if (force_v22) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V22(128, true, 16, __hip_bfloat16); else L10_V22(128, true, 16, __half); }
                        else { if (out_bf) L10_V22(128, false, 16, __hip_bfloat16); else L10_V22(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V22(128, true, 32, __hip_bfloat16); else L10_V22(128, true, 32, __half); }
                        else { if (out_bf) L10_V22(128, false, 32, __hip_bfloat16); else L10_V22(128, false, 32, __half); }
                    }
                } else if (force_v2) {
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V2(128, true, 16, __hip_bfloat16); else L10_V2(128, true, 16, __half); }
                        else { if (out_bf) L10_V2(128, false, 16, __hip_bfloat16); else L10_V2(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V2(128, true, 32, __hip_bfloat16); else L10_V2(128, true, 32, __half); }
                        else { if (out_bf) L10_V2(128, false, 32, __hip_bfloat16); else L10_V2(128, false, 32, __half); }
                    }
                } else {
                    // default: v4 BM=64 (per-column k_scale fix, ~15% faster than v3 BN=16)
                    if (bn == 16) {
                        if (is_causal) { if (out_bf) L10_V4B64(128, true, 16, __hip_bfloat16); else L10_V4B64(128, true, 16, __half); }
                        else { if (out_bf) L10_V4B64(128, false, 16, __hip_bfloat16); else L10_V4B64(128, false, 16, __half); }
                    } else {
                        if (is_causal) { if (out_bf) L10_V4B64(128, true, 32, __hip_bfloat16); else L10_V4B64(128, true, 32, __half); }
                        else { if (out_bf) L10_V4B64(128, false, 32, __hip_bfloat16); else L10_V4B64(128, false, 32, __half); }
                    }
                }
                }
            }
            #undef L10
            #undef L10_V2
            #undef L10_V22
            #undef L10_V3
            #undef L10_V4
            #undef L10_V4B64
            #undef L10_V2F
            #undef L10_V2X
            #undef L10_V7P
            #undef L10_V10F
            #undef L10_V10IPVF
            #undef L10_V10INQF
        }
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
            const bool use_v2 = (v2_mode == 1) || (v2_mode == 2 && qo_len == kv_len && qo_len >= 1024)
                || (v2_mode == 2 && head_dim == 128 && qo_len != kv_len);  // D=128 cross: BM64 >> BM32
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
                    int d128_bn = getenv("SAGEATTN_D128_BN") ? atoi(getenv("SAGEATTN_D128_BN")) : 32;
                    if (d128_bn == 16) {
                        if (is_causal) LD_V2(128, true, 16, __half, __half);
                        else LD_V2(128, false, 16, __half, __half);
                    } else {
                        if (is_causal) LD_V2(128, true, 32, __half, __half);
                        else LD_V2(128, false, 32, __half, __half);
                    }
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
            const bool use_v2 = (v2_mode == 1) || (v2_mode == 2 && qo_len == kv_len && qo_len >= 1024)
                || (v2_mode == 2 && head_dim == 128 && qo_len != kv_len);  // D=128 cross: BM64 >> BM32
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