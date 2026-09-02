#pragma once
#if defined(__HIP_PLATFORM_AMD__)
#define FINAL_MASK 0xffffffffffffffffull
#else
#define FINAL_MASK 0xffffffff
#endif

namespace vllm {

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;
    int                 wid  = threadIdx.x >> 5;
    val = warpReduceMax(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);
    // 尾随 barrier: 防止本调用 warp0 对 shared[lane] 的读阶段
    // 与下一次调用 (RATIO 循环) 其它 warp lane0 对 shared[wid] 的写阶段重叠
    // (data race -> D=64 int8 quant 的 q_scale/q_int8 非确定, 见 try.md §4)
    __syncthreads();
    return val;
}

}
