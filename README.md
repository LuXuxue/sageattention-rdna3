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
| 显存 | 共享系统内存（DDR5-5600, 64GB 双通道） |
| 内存带宽 | 理论 ~89.6 GB/s（128-bit @ 5600MT/s）；实测 copy 73 GB/s、读 79 GB/s |
| 内存延迟 | ~200-400ns（远高于 dGPU） |

### 后端切换

通过环境变量 `SAGEATTN_BACKEND` 切换计算后端：

```bash
set SAGEATTN_BACKEND=triton   # Triton autotune kernel（默认）
set SAGEATTN_BACKEND=native   # HIP native WMMA kernel
```

>默认仍采用 triton 以获得最佳兼容性；native 用于无 Triton 依赖的独立部署。

### 环境变量

| 变量 | 作用 | 默认 |
|------|------|------|
| `SAGEATTN_BACKEND` | 后端选择（native/triton） | triton |
| `SAGEATTN_INT8_32` | int8 self 用每 warp 32 行 kernel（0 关闭回退 8 warps） | 1（启用） |
| `SAGEATTN_INT8_WPE` | int8 实验：launch wrapper waves_per_eu（1/2/4） | 1 |
| `SAGEATTN_INT8_BN128` | D=128 int8 的 BN 覆盖（0 默认 64, 16/32） | 0（=64） |
| `SAGEATTN_FP16_BM` / `SAGEATTN_FP16_BN` | fp16 direct 实验配置覆盖 | 0 / 0 |
| `SAGEATTN_BF16_BM` / `SAGEATTN_BF16_BN` | bf16 direct 实验配置覆盖 | 0 / 0 |
| `SAGEATTN_BM_SEL` | direct kernel 的 BM 强制覆盖（0 默认, 1=32, 2=128, 3=64） | 0 |
| `SAGEATTN_DIRECT_THRESHOLD_D64` | D=64 self 的 direct/int8 阈值 | 3072 |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL` | D=64 causal 的阈值 | 6144 |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CROSS` | D=64 cross 的阈值 | 6144 |
| `SAGEATTN_DIRECT_THRESHOLD_D128` | D=128 self/causal 的阈值 | 2048 |
| `SAGEATTN_DIRECT_THRESHOLD_D128_CROSS` | D=128 cross 的阈值 | 4096 |

> `SAGEATTN_VT_GLOBAL` 已由 `setup.py` 编译宏固定为 1（V_T 方案默认启用），无需设置。

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

Python 层（`core.py`）根据 `head_dim`、`kv_len`、`q_len` 和 `is_causal` 自动选择路径。
**核心权衡**：direct（fp16 QK）省 quant+mean 辅助 kernel（固定 0.4-0.6ms），
int8 用 i8 WMMA（RDNA3 上为 fp16 的 2 倍吞吐，优势与计算量成正比）。
计算量小（causal / cross 短 q）→ direct；计算量大（self 长序列）→ int8。

```
D=64:
  self 非 causal: kv <= 3072 -> direct; kv > 3072 -> int8   (交叉点 3072-3456)
  causal:         kv <= 6144 -> direct; kv > 6144 -> int8   (4096/6144 direct 优 8%/2.7%, 8192 int8 优 5.5%)
  cross (q<kv):   kv <= 6144 -> direct                       (q=3072/kv=4096 仍 direct 优 3%)
D=128:
  self/causal:    kv <= 2048 -> direct; kv > 2048 -> int8    (交叉点 2048-2560)
  cross q*2<kv:   kv <= 4096 -> direct                       (q=512/1024 vs kv=4096 direct 优 8-25%; q=2048=kv/2 起 int8 优)
```

**int8 量化路径的适用场景**：不仅限 self-attn——
- D=64：self（kv>3072）、causal/cross（kv>6144）
- D=128：self/causal（kv>2048）、cross 且 q>=kv/2（kv>2048）
- 在 benchmark 的 24 用例中，int8 实际只服务 self-attn（SDXL01/07/13 和 Anima01/03/05），
  其余用例（cross/kv 短）均走 direct。

int8 路径的 kernel 选择（`qk_int8_sv_bf16_attn_gfx11_t`）：
```
D=64  self-attn -> 每 warp 32 行 kernel (BM=128, BN=32, 4 warps) [默认]
                  (SAGEATTN_INT8_32=0 回退 8 warps wpe1)
D=64  cross, kv<=77 -> (BM=64, BN=16) [wpe1]
D=64  cross, kv>77  -> (BM=64, BN=32) [wpe1]
D=128               -> (BM=64, BN=64) [wpe2]   (BN=64 实测优于 BN=32, kv-tile 减半)
```

fp16/bf16 direct 路径（转置 kernel, wpe2）：
```
D=64  self -> (BM=64, BN=64)
D=64  cross, kv<=128 -> (BM=64, BN=32)
D=64  cross, kv>128  -> (BM=128, BN=32)   [BM=128 快 9-13%]
D=128                -> (BM=64, BN=16)
```

## 参考

- [SageAttention native gfx12 attention backend](https://github.com/jammm/SageAttention/tree/jam/gfx12-abi3)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — HIP backend 参考（gfx11/gfx12 WMMA）
- [ComfyUI-FeatherOps](https://github.com/woct0rdho/ComfyUI-FeatherOps) — VRAM 带宽优化参考

## 许可证

Apache License 2.0
