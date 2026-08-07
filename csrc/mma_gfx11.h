#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#error "mma_gfx11.h is only intended for ROCm/HIP."
#endif

#include <cstdint>

namespace sageattn_gfx11 {

typedef int int32_t_v4 __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

__device__ __forceinline__ int acc_row(int lane, int e) {
    return 2 * e + (lane >> 4);
}

__device__ __forceinline__ int acc_col(int lane) {
    return lane & 15;
}

__device__ __forceinline__ float fast_exp2(float x) {
    return exp2f(x);
}

__device__ __forceinline__ int32_t_v4 load_i8_frag(
    const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    int32_t_v4 r;
    const int* q = reinterpret_cast<const int*>(p);
    r[0] = q[0];
    r[1] = q[1];
    r[2] = q[2];
    r[3] = q[3];
    return r;
}

__device__ __forceinline__ v16h load_fp16_col_frag(
    const void* lds, int col, int stride) {
    const char* base = static_cast<const char*>(lds) + col * 2;
    v16h r;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        r[k] = *reinterpret_cast<const _Float16*>(base + k * stride);
    }
    return r;
}

__device__ __forceinline__ v8i wmma_i32_iu8(int32_t_v4 a, int32_t_v4 b, v8i c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

__device__ __forceinline__ v8f wmma_f32_bf16(v16bf a, v16bf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

__device__ __forceinline__ v8f wmma_f32_f16(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

}
