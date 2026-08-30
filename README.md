# SageAttention RDNA2 / RDNA3

SageAttention 的 HIP 原生实现，支持 AMD **RDNA3 (gfx11xx, WMMA)** 与
**RDNA2 (gfx1035, V_DOT4 / V_DOT2 SIMD dot 指令)** 两种 GPU 架构。

> 性能优化细节、benchmark 数据、踩坑记录与剩余优化方向见
> [NativeBackendOptimizeReport-gfx1103.md](NativeBackendOptimizeReport-gfx1103.md)
> [NativeBackendOptimizeReport-gfx1035.md](NativeBackendOptimizeReport-gfx1035.md)

## 环境要求

- Python
- ROCm
- PyTorch
- RDNA3 GPU (gfx110x) 或 RDNA2 GPU (gfx103x)
- 构建需 MSVC（Windows）/ GCC（Linux）+ ROCm SDK

## 安装

```bash
# 设置目标架构（默认 gfx1103 + gfx1035 双架构）
set GPU_ARCHS=gfx1103;gfx1035       # Windows
export GPU_ARCHS=gfx1103:gfx1035    # Linux

# 安装（开发模式）
pip install -e . --no-build-isolation
```

> **注意**：`--no-build-isolation` 是必须的，否则构建系统无法找到已安装的 PyTorch HIP 头文件。
> Windows 下需要 MSVC + ROCm SDK 环境（setup.py 会自动定位 `rocm-sdk` 与 MSVC 工具链）。
> 运行pytest需要开发模式，正常使用可以用 `pip install . --no-build-isolation` 进行安装。

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
    smooth_k: bool = False,      # 是否减 K 均值（默认 False；K 有明显 DC 偏置时可设 True）
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
| Smooth K | 支持（可选，默认关闭；见 API `smooth_k`） |
| 目标 GPU | gfx1103 (RDNA3 WMMA), gfx1035 (RDNA2 V_DOT4/V_DOT2) |

> **注意**：native 后端要求 q 的 stride（除 last dim）为 8 的倍数（16B 向量写回
> 对齐），非 contiguous 输入会触发断言或未对齐写（UB）；请使用 contiguous 张量。

## 后端切换

`core.py` 默认使用 native 后端；设 `SAGEATTN_BACKEND=triton` 可使用 triton 后端
进行性能对比测试：

```bash
set SageATTN_BACKEND=triton   # Windows
export SAGEATTN_BACKEND=triton  # Linux
```

## 测试

```bash
# 正确性测试（36 个用例；含 NaN/Inf 检测）
# 默认 native 后端（test_sageattn_rdna3.py 头部 setdefault）。
python test_sageattn_rdna3.py

# 切到 triton 后端测试：SAGEATTN_BACKEND=triton python test_sageattn_rdna3.py
```

## 架构差异

| 项目 | gfx1103 (RDNA3) | gfx1035 (RDNA2) |
|------|-----------------|-----------------|
| 矩阵/点积指令 | WMMA 16×16×16 (i32_i8, f32_f16) | `v_dot4c_i32_i8`, `v_dot2c_f32_f16` SIMD 指令 |
| 缺矩阵核处理 | 支持张量核 (WMMA) | 无张量核，矩阵乘降级为 SIMD 串/并点积链 |
| bf16 dot2 | WMMA 路径原生 | 需 LDS 暂存时转为 fp16 运算 |
| int8 dot4 | WMMA 路径原生 | SIMD 点积风格，每 lane 独立累加 |

`setup.py` 通过 `--offload-arch` 同时编译双架构，单 wheel 同时包含 RDNA3 与 RDNA2
的 native kernel；gfx1103 WMMA 路径在 RDNA2 上运行时 hipcc 会因 ISA 不兼容自动报错。

## 参考

- [SageAttention native gfx12 attention backend](https://github.com/jammm/SageAttention/tree/jam/gfx12-abi3)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — HIP backend 参考（gfx11/gfx12 WMMA）
- [ComfyUI-FeatherOps](https://github.com/woct0rdho/ComfyUI-FeatherOps) — VRAM 带宽优化参考

## 许可证

Apache License 2.0