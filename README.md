# SageAttention RDNA3

SageAttention 的 HIP 原生实现，面向 AMD RDNA3 (gfx11xx) GPU。

## 概述

本项目提供针对 RDNA3 GPU 优化的高性能量化 attention kernel。
开发平台为 **Radeon 780M (gfx1103)**——一块 iGPU（集成显卡），共享系统内存，
无 Infinity Cache，内存延迟远高于独立显卡。

## 环境要求

- ROCm 6.0+（开发环境：ROCm 7.14）
- PyTorch 2.4+（ROCm 版本）
- RDNA3 GPU (gfx11xx)
- Python 3.9+

## 安装

```bash
# 设置目标架构（默认 gfx1103）
set GPU_ARCHS=gfx1103        # Windows
export GPU_ARCHS=gfx1103     # Linux

# 安装
pip install . --no-build-isolation

# 开发模式（editable）
pip install -e . --no-build-isolation
```

> **注意**：`--no-build-isolation` 是必须的，否则构建系统无法找到已安装的 PyTorch HIP 头文件。

## 使用方法

```python
import torch
from sageattention import sageattn

# NHD 布局: [batch, seq_len, heads, head_dim]
q = torch.randn(1, 4096, 10, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 4096, 10, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 4096, 10, 64, dtype=torch.float16, device="cuda")

out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

# HND 布局: [batch, heads, seq_len, head_dim]
q = torch.randn(1, 10, 4096, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 10, 4096, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 10, 4096, 64, dtype=torch.float16, device="cuda")

out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
```

## API

```python
def sageattn(
    q: torch.Tensor,             # Query [B,S,H,D] (NHD) 或 [B,H,S,D] (HND)
    k: torch.Tensor,             # Key（同 q 布局）
    v: torch.Tensor,             # Value（同 q 布局）
    tensor_layout: str = "HND",  # "HND" 或 "NHD"
    is_causal: bool = False,     # 是否使用 causal mask
    sm_scale: float = None,      # Softmax scale（默认 head_dim^-0.5）
    smooth_k: bool = True,       # 减去 K 均值以提升量化精度
) -> torch.Tensor
```

## 支持的配置

| 项目 | 支持范围 |
|------|----------|
| Head 维度 | 64, 128 |
| 数据类型 | fp16, bf16, fp32（自动转 fp16） |
| 张量布局 | HND, NHD |
| Causal mask | 支持 |
| GQA | 支持（h_q != h_kv） |
| Smooth K | 支持（减去 K 均值提升量化精度） |

## 测试

```bash
# 正确性测试（36 个用例, 默认 native 后端; 设 SAGEATTN_BACKEND=triton 可测 triton）
python test_sageattn_rdna3.py
```

## 架构细节

### 开发平台硬件参数

**Radeon 780M (gfx1103)** — RDNA3 iGPU：

| 参数 | 值 |
|------|-----|
| Compute Units | 12 |
| LDS | 128 KiB / CU |
| VGPR | 512 KiB / CU（4 SIMD x 32768 VGPRs） |
| Infinity Cache | 无 |
| L2 Cache | 2 MiB |
| 显存 | 共享系统内存（DDR5/LPDDR5） |
| 内存带宽 | ~76.8 GB/s (DDR5-4800) |
| 内存延迟 | ~200-400ns（远高于 dGPU） |

### 后端切换

通过环境变量 `SAGEATTN_BACKEND` 切换计算后端：

```bash
set SAGEATTN_BACKEND=triton   # Triton autotune kernel（默认）
set SAGEATTN_BACKEND=native   # HIP native WMMA kernel
```

>**说明：**
>Native backend 已重构为转置布局 kernel（permlanex16 归约 + 寄存器 P fragment），
>在全部 24 个 benchmark 用例上与 Triton 匹配（同进程交错轮转测量 ≤1.15x）。
>默认仍采用 triton 以获得与 FlashAttn 生态的最佳兼容性；
>native 可用于无 Triton 依赖的独立部署，性能与 triton 相当。

### Kernel 模板参数

Attention kernel 使用 C++ 模板参数化，编译时生成多个实例：

```cpp
template <int HeadDim, bool IsCausal, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void attn_kernel_impl(...);

// 两个 global 包装器，不同 waves_per_eu 设置：
template <...> __global__ void attn_kernel_wpe1(...);  // waves_per_eu=1
template <...> __global__ void attn_kernel_wpe2(...);  // waves_per_eu=2
```

| 参数 | 含义 | 约束 |
|------|------|------|
| `HeadDim` | head 维度 | 64 或 128 |
| `IsCausal` | causal mask | true/false |
| `BLOCK_M` | 每个 workgroup 处理的 Q 行数 | 16 的倍数 |
| `BLOCK_N` | 每次 KV tile 的行数 | 16 的倍数 |

派生参数：`WARPS = BLOCK_M/16`，`THREADS = WARPS*32`。

### 分发逻辑

Python 层根据 `head_dim` 和 `kv_len` 自动选择计算路径：

```
if head_dim == 64:
    kv_len <= 1024 -> fp16/bf16 直接运算
    kv_len >  1024 -> int8 量化运算
if head_dim == 128:
    kv_len <= 512  -> fp16/bf16 直接运算
    kv_len >  512  -> int8 量化运算
```

C++ 层的模板配置分发（native 后端采用 Triton 风格**转置布局** kernel：
`qk^T = k @ q^T` + `v_permlanex16_b32` 归约 + 寄存器 P fragment，
消除了 `ds_bpermute` butterfly 归约和 P LDS 中转）：

```
fp16/bf16 直接路径 (转置 kernel, wpe2):
    D=64,  kv<=128           -> (BM=64, BN=16)
    D=64,  kv>128, self-attn -> (BM=64, BN=64)
    D=64,  kv>128, cross-attn-> (BM=64, BN=32)
    D=128                    -> (BM=64, BN=16)

int8 量化路径 (转置 kernel):
    D=64,  self-attn         -> (BM=128, BN=32)  [wpe1]
    D=64,  cross, kv<=77     -> (BM=64, BN=16)   [wpe1]
    D=64,  cross, kv>77      -> (BM=64, BN=32)   [wpe1]
    D=128                    -> (BM=64, BN=32)   [wpe2]
```

## 参考

- [SageAttention native gfx12 attention backend](https://github.com/jammm/SageAttention/tree/jam/gfx12-abi3)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — HIP backend 参考（gfx11/gfx12 WMMA）
- [ComfyUI-FeatherOps](https://github.com/woct0rdho/ComfyUI-FeatherOps) — VRAM 带宽优化参考

## 许可证

Apache License 2.0
