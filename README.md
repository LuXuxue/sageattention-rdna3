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
set SAGEATTN_BACKEND=triton   # Windows
export SAGEATTN_BACKEND=triton  # Linux
```

## 环境变量调优参数

> 所有变量在每次调用时通过 `getenv` 读取（可用 `os.environ` 在进程内切换）。
> Windows 设置：`set NAME=value`；Linux：`export NAME=value`。
> 下表默认值为代码内默认；`gfx1103` 指 RDNA3（WMMA），`gfx1035` 指 RDNA2（v10/v8）。

### 后端与通用

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_BACKEND` | `native` | `native`, `triton` | 后端选择（triton 用于性能对比） |
| `SAGEATTN_DEFAULT_PV_ACCUM_DTYPE` | `fp32` | `fp32`, `fp16`, `fp16+fp32` | triton 后端的 PV 累加精度（非法值回落 fp32） |
| `SAGEATTN_BM_SEL` | `0` | `0`, `1`, `2` | direct kernel 的 BM 选择（0=默认, 1=32, 2=128）；API 参数 `bm_sel` 优先 |
| `SAGEATTN_INT8_V` | `0` | `0`, `1` | 0=fp16 V_T 快路径（默认）；1=int8 V 量化转置（实验，实测更慢） |
| `SAGEATTN_VT_OVERLAP` | `1` | `0`, `1` | int8-V 路径的 `v_quant_transpose` 是否用侧流重叠（1=重叠, 0=串行） |

### Direct / Int8 分派阈值（`core.py`，kv 小于等于阈值走 direct 路径）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SAGEATTN_DIRECT_THRESHOLD_D64` | 2048 (HND) / 3072 (NHD) | D64 self 的 direct 阈值 |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL` | 6144 | D64 causal 的 direct 阈值 |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CROSS` | 6144 | D64 cross 的 direct 阈值 |
| `SAGEATTN_DIRECT_THRESHOLD_D128` | 2048 | D128 self 的 direct 阈值 |
| `SAGEATTN_DIRECT_THRESHOLD_D128_CROSS` | 4096 | D128 cross（q_len×2 < kv_len）的 direct 阈值 |

> 注意：gfx1035 上 direct 路径被强制关闭（一律走 int8）。

### D128 int8 kernel（`attn_kernel_*_t` 系列，gfx1103）

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_INT8_WPE` | `1` | `1`, `2`, `4` | launch wrapper 的 waves-per-EU（occupancy） |
| `SAGEATTN_INT8_BN128` | `0` (auto) | `16`, `32`, `64`, `128` | BN 覆盖；auto = kv%512==0 且 kv≥5120 时 64，否则 32 |
| `SAGEATTN_INT8_BM128` | `-1` (auto) | `1`, `0` | BM 强制；auto = kv%512==0 且 kv≥`SAGEATTN_INT8_BM128_THR` 时 128，否则 64 |
| `SAGEATTN_INT8_BM128_THR` | `8192` | 正整数 | auto 模式启用 BM=128 的 kv 阈值 |
| `SAGEATTN_INT8_32` | `1` | `0`, `1` | D64 self 是否用 32w kernel（0 回退旧 `impl_t` kernel） |

### D64 32w kernel（`attn_kernel_impl_32_t`，gfx1103，D64 self）

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_D64_32W_WPE` | `1` | `1`, `4` | occupancy（4=高 occupancy，实测在多数形状为负收益或噪声） |
| `SAGEATTN_D64_32W_BM` | `128` | `64`, `128` | BM 覆盖；**仅配合 WPE=4 生效**（WPE=1+BM=64 是 grid/block 失配陷阱，会得到错误的快计时，勿用） |
| `SAGEATTN_D64_BN` | `32` | `16`, `32`, `64` | 32w WPE=1 路径的 BN 覆盖（实测 16/64 无益或更差） |

### Direct fp16/bf16 kernel（`fp16_attn_kernel_wpe2_t` / `bf16_attn_kernel_wpe2_t`，gfx1103）

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_FP16_BN` | `0` (auto) | `16`, `32`, `64`, `128` | fp16 direct 的 BN 覆盖（auto: D64 kv≤128→16, 其余 32/64） |
| `SAGEATTN_FP16_BM` | `0` (默认规则) | `32`, `64`, `128` | fp16 direct 的 BM 覆盖 |
| `SAGEATTN_BF16_BN` | `0` (auto) | `16`, `32`, `64`, `128` | bf16 direct 的 BN 覆盖（auto: D128 默认 16） |
| `SAGEATTN_BF16_BM` | `0` (默认规则) | `32`, `64`, `128` | bf16 direct 的 BM 覆盖 |
| `SAGEATTN_D64_X_BN` | `16` | `16`, `32`, `64` | D64 direct cross（kv≤77）的 BN 覆盖（实测 32/64 更差） |

### QK 量化预处理（`quant_qk_int8`，两架构通用）

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_QUANT_BLK` | `1` (auto) | `128`, `64`, `0` | 量化分块行数；auto = D64 用 Q128/K64、D128 用 Q32/K16；128=Q128/K64；64=Q64/K32；0=Q32/K16 |
| `SAGEATTN_QUANT_GPB` | `0` (auto) | 正整数 | 每 workgroup 顺序处理的块数（gfx1103 auto: D64 kv<6144→4、<9216→8、否则 16；D128→32/16/64；gfx1035 默认 1） |

### v_transpose（gfx1103）

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_VT_GRID` | `0` (auto) | 正整数 | v_transpose grid 上限；auto = total_tiles≤4096 时 192，否则 min(max(total/10,128),1536) |

### gfx1035（RDNA2 v10 kernel）

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `SAGEATTN_BN` | `16` | 正整数 | D128 v10 的 BN 默认 |
| `SAGEATTN_BN_D64` / `SAGEATTN_BN_D128` | `32` / `16` | 正整数 | 分维 BN 覆盖（优先于 `SAGEATTN_BN`） |
| `SAGEATTN_TM` | `8` | 正整数 | v10 的 TILE_M 默认 |
| `SAGEATTN_TM_D64` / `SAGEATTN_TM_D128` | `8` | 正整数 | 分维 TM 覆盖（优先于 `SAGEATTN_TM`） |
| `SAGEATTN_V10_INQ` | `2` (auto) | `0`, `1`, `2` | INQ 预扫描：0=关，1=强制，2=auto（kv≤1024 时启用） |

### 基准脚本（`benchmark_attn/benchmark_attn.py`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SAGEATTN_BENCH_ROUNDS` | `3` | 交错轮转轮数（抗热漂移） |
| `SAGEATTN_BENCH_ITERS` | `50` | 每轮迭代次数 |

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