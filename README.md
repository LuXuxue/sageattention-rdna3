# SageAttention RDNA3

SageAttention 的 HIP 原生实现，面向 AMD RDNA3 (gfx11xx) GPU。

> 性能优化细节、benchmark 数据与剩余优化方向见
> **[NativeBackendOptimizeReport.md](NativeBackendOptimizeReport.md)**；
> 实验过程与 A/B 验证数据见 **try.md**。

## 环境要求

- ROCm 6.0+（开发环境：ROCm 7.14）
- PyTorch 2.4+（ROCm 版本）
- RDNA3 GPU (gfx11xx)
- Python 3.9+
- 构建需 MSVC（Windows）/ GCC（Linux）+ ROCm SDK

## 安装

```bash
# 设置目标架构（默认 gfx1103）
set GPU_ARCHS=gfx1103        # Windows
export GPU_ARCHS=gfx1103     # Linux

# 安装（开发模式）
pip install -e . --no-build-isolation
```

> **注意**：`--no-build-isolation` 是必须的，否则构建系统无法找到已安装的 PyTorch HIP 头文件。
> Windows 下需要 MSVC + ROCm SDK 环境（setup.py 会自动定位 `rocm-sdk` 与 MSVC 工具链）。

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

> **注意**：native 后端要求 q 的 stride（除 last dim）为 8 的倍数（16B 向量写回
> 对齐），非 contiguous 输入会触发断言或未对齐写（UB）；请使用 contiguous 张量。

## 后端切换

`core.py` 默认使用 triton 后端；设 `SAGEATTN_BACKEND=native` 使用本项目的
HIP native WMMA kernel：

```bash
set SAGEATTN_BACKEND=native   # Windows
export SAGEATTN_BACKEND=native  # Linux
```

## 测试

```bash
# 正确性测试（36 个用例；默认 native 后端，设 SAGEATTN_BACKEND=triton 可测 triton）
python test_sageattn_rdna3.py
```

> **注意**：native 正确性验证必须显式设 `SAGEATTN_BACKEND=native`
> （core.py 默认 `_BACKEND=triton`，不设 env 时测的是 triton 后端）。
> flash-attn 为 **NHD 布局** API，benchmark 脚本按此调用。

## 参考

- [SageAttention native gfx12 attention backend](https://github.com/jammm/SageAttention/tree/jam/gfx12-abi3)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — HIP backend 参考（gfx11/gfx12 WMMA）
- [ComfyUI-FeatherOps](https://github.com/woct0rdho/ComfyUI-FeatherOps) — VRAM 带宽优化参考

## 许可证

Apache License 2.0
