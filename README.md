# SageAttention RDNA3 / RDNA2

SageAttention 的 HIP 原生实现，支持 AMD **RDNA3 (gfx110x)** 与
**RDNA2 (gfx103x)** 两种 GPU 架构。

## 环境要求

- Python
- ROCm
- PyTorch
- RDNA3 GPU (gfx110x) 或 RDNA2 GPU (gfx103x)
- 构建需 MSVC（Windows）/ GCC（Linux）+ ROCm SDK

## 平台特性对照

| 项目 | gfx110x（RDNA3） | gfx103x（RDNA2） |
| --- | --- | --- |
| 矩阵/点积指令 | WMMA 16×16×16 (i32_i8, f32_f16) | `v_dot4c_i32_i8`, `v_dot2c_f32_f16` SIMD 指令 |
| int8 QK | WMMA i32（**fp16 的 2× 吞吐**） | `v_dot4c_i32_i8`（MAC 吞吐与 fdot2 持平） |
| fp16/bf16 PV | WMMA f32 | `v_dot2c_f32_f16` |
| bf16 | 原生支持 | **无硬件指令**，须 LDS 暂存时转 fp16 |
| LDS | 128 KiB/CU | 64 KiB/CU |
| CU 数（典型） | 12 CU（gfx1103） | 6 CU（gfx1035） |

两架构 .cu/.h 互不引用；公共辅助（mean / quant / v_transpose kernel + host 函数）各复制一份。`core.py` 按运行时设备 `gcnArchName` 懒加载对应 pyd 并注册 `torch.ops.sageattention`，同进程只加载一个 pyd。

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
# 正确性测试（默认使用 native 后端）。
python test_sageattn_rdna3.py

# 切到 triton 后端测试：SAGEATTN_BACKEND=triton python test_sageattn_rdna3.py
```

## 测量方法论

1. **同进程内交错轮转**：被测实现与参考实现在同一进程交替运行。
2. **充分预热**：≥20 次 warmup。
3. **取中位数**：≥50 次迭代取 median。
4. **iGPU 波动**：共享内存 iGPU 长时间运行因温度/时钟下降，小 case 跨 run 波动可达 ±25%；**跨会话绝对时间不可比**，只有同会话内比率口径成立。
5. **固定随机种子**：`torch.manual_seed(0)` 保证可复现。
6. **正确性基准**：参考实现必须 `transpose(1,2)` 到与 kernel 输入一致的 HND/NHD 布局；早期多次因参考布局用错而误判。

## 许可证

Apache License 2.0