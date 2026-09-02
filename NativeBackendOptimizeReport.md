# SageAttention Native Backend 优化报告（gfx110x / gfx103x）

本报告为项目 native 后端（HSA / HIP）的总优化记录：涵盖硬件基础、核心算法、实验过程与结论、分发逻辑、易踩坑点与剩余优化方向。所有性能结论遵循统一的测量方法论（§四）：**同进程内交错轮转 + 充分预热 + 取中位数**，跨进程/跨会话的绝对时间不可比。

> 覆盖架构：
> - **gfx110x（RDNA3）**：本机开发平台 gfx1103（Radeon 780M），WMMA 16×16×16 tensor core。
> - **gfx103x（RDNA2）**：gfx1035（Rembrandt），V_DOT4 / V_DOT2 SIMD 指令，无 WMMA / MFMA。

---

## 一、最终结论（当前最优状态）

### 1.1 核心优化链（按收益排序，gfx110x）

1. **V 全局转置 PV（`SAGEATTN_VT_GLOBAL=1`）**：V 一次性转置为 `V_T [B,H,D,N]`，PV 的 A=V^T 从 128 次 u16 LDS 列读变为 1 条行读 → direct 路径真实提升 17–36%。
2. **V_T tile LDS 缓存 PV（仅 D=128 超长 self-attn，kv≥16384）**：PV 的 v_frag 从 LDS 行读而非全局，消除 8 倍 L1/L2 冗余（128KB/迭代 → 16KB/迭代）→ VAE 用例 attn 提升 ~35%（SDXLVAE01 94.7→62ms），追平并反超 triton。**注意：该优化仅对 D=128 超长序列有效，D=64 全场景为负优化（见 §八 E1/C8）。**
3. **每 warp 32 行 QK kernel（仅 D=64 int8 self）**：BM=128/4 warps，2 子块共享 k_frag 提升 ILP → int8 self 从慢 7–13% 追平。
4. **bf16→fp16 转换融合进 v_transpose**：省掉 `v.to(fp16)` 独立 kernel（Anima01 约 5%）。
5. **quant 优化链**：pass1→LDS 缓存（省一半全局读），叠加大 block + 多累加器 ILP（D=64 用 BLK_Q=128/BLK_K=64）→ quant 单独再 -5~-18%（带宽 23.5→87–95GB/s）；D=128 保持旧逻辑（已 ~100GB/s 接近上限）。
6. **mean kernel 32B 向量读**：带宽 38→68–77GB/s，优于 torch mean 8–43%（仅在显式 `smooth_k=True` 时生效）。
7. **v_transpose grid 分派**：小数据量（tiles≤4096）用 grid=192（每 block 多 tile）→ SDXL10 v_transpose -11%。
8. **D=128 int8 BN=32→64**：kv-tile 数减半。
9. **差异化 direct/int8 分发**（按 kv_len/q_len/is_causal/tensor_layout 选择路径，见 §五）。
10. **写回向量化**（permlanex16 + 16B 连续写，全部 kernel 一致）。
11. **smooth_k=False（默认）**：跳过 mean kernel，端到端省 0.5–3.5%，SDXL01 借此反超 FlashAttn（见 §3.12）。
12. **D=128 int8 self BM 尺寸自适应**：当 kv≥6144 时动态切换 BM=128（默认 BM=64）。在 Anima05 提升 4.2%，SDXLVAE01 提升 5.8%，且保持 bit-identical 正确性。
13. **blockReduceMax 跨调用竞态修复**：修复 D=64 int8 Q-quant 路径因 shared memory 复用无尾随 barrier 导致的非确定性问题，保障输出严格一致且无性能回退。

### 1.2 当前 benchmark 状态

基线为最新配置（默认编译、无 env、`smooth_k=False`）。比率 = SageAttn/FlashAttn（或 native/triton），<1 为快。波动 ±5–7%（iGPU 热状态，见 §四）。

**gfx1103（RDNA3，本机）严格子进程 perf（拆分后最新）：**

| 用例 | native(ms) | triton(ms) | native/triton | 状态 |
| --- | --- | --- | --- | --- |
| D=64 self 4096 non-causal | 2.49 | 2.24 | 1.11 | 持平 |
| D=64 self 4096 causal | 1.88 | 2.44 | **0.77** | **反超** |
| D=128 self 4096 non-causal | 4.40 | 4.20 | 1.05 | 持平 |
| D=64 self 6144 non-causal | 4.33 | 4.26 | 1.02 | 持平 |
| D=128 self 6144 non-causal | 9.31 | 8.46 | 1.10 | 略慢 |
| D=64 cross q1024/kv2560 | 1.18 | 0.76 | 1.56 | 慢（cross 辅助占比高） |

整体与 triton 持平；causal D=64 self 反超 23%。

**gfx1103 历史 benchmark 总览（§一~九实验基线，报告基线）：**

| 类别 | native/FA | native/triton | 说明 |
| --- | --- | --- | --- |
| D=64 int8 self（SDXL01 4096） | 0.95–0.99 | ~1.01 | **反超 FA**（smooth_k=False 后） |
| D=64 int8 self（SDXL07/13 6144/9216） | 1.03–1.05 | ~1.02 | 慢 FA 3–5%：attn 主 kernel 已持平，差距 = vt+quant 辅助累计 |
| D=128 int8 self bf16（Anima01 4096） | 1.02–1.03 | ~1.05 | 慢 FA 2–3%：i8 2x 吞吐优势被 aux（vt 0.49+quant 0.68ms）吃掉 |
| D=128 int8 self bf16（Anima03/05 6144/9216） | 0.95–0.97 | — | 快 FA 3–5%（序列越长 i8 吞吐优势越显、aux 占比越低） |
| D=64 direct self（SDXL10/16） | 1.07–1.11 | 0.93–1.15 | v_transpose 固有成本 + 短序列固定开销 |
| D=64 direct cross 短 kv（SDXL02/08/14 等） | 0.82–0.90 | 0.70–0.92 | 快于 FA（BM=128 大 block 省 K/V 重读 + FA 短 kv 低效） |
| D=128 direct cross（Anima02/04/06） | 0.78–0.84 | 0.75–0.89 | 快于 FA（同左） |
| VAE（D=128 int8 self 16384/24576） | 0.97–0.99 | 0.97–1.03 | 持平/略快（LDS PV 优化后） |

> 注 1：native/triton 与 native/FA 来自不同的独立复测文件，FA 绝对时间因散热批次略有差异，比率口径各自文件内成立，不可跨文件直接相减。
> 注 2：短 cross（kv=77/154，SDXL05/06/11/12/17/18）native 与 triton 均慢 FA 15–35%（绝对差仅 0.03–0.12ms）——SageAttention 两后端共有的短序列固定开销，非 native 特有（见 §9.1）。
> 注 3：`core.py` 在 import 时读 `SAGEATTN_BACKEND` 一次；单跑 `benchmark_attn.py` 必须在进程启动前设 env，轮转脚本须用 `core._BACKEND` 模块变量切换。

**gfx1035（RDNA2，编译通过；性能基线来自移植期）**：

| 类别 | native/triton | 说明 |
| --- | --- | --- |
| D=128 int8 self bf16 | ~0.35（**快 2.85x**） | V2（BM=64, NR=2, NW=4, BN=16）显著优于 triton |
| D=64 int8 self bf16 | ~0.25（**慢 4x**） | 当前唯一性能瓶颈；D=64 计算密度不足 + LDS/occupancy 约束 |
| D=64/D=128 direct cross | 0.7–0.9 | 持平或略快（BM=64 V2 短 kv 友好） |

> gfx1035 主要瓶颈：D=64 长 self-attn 受 SIMD 点积链延迟、16-lane reduction 与 LDS 容量限制；D=128 长 self 由 V2 解决。V3（16-lane 协作 QK/PV）已尝试但 PV 输出不完整（见 §七 RDNA2 踩坑）。

### 1.3 关键结论

**gfx110x（RDNA3）：**
- i8 WMMA 在 RDNA3 为 fp16 的 2 倍吞吐（QK 占一半计算量）→ int8 在长 self-attn 占优；direct 省 quant 辅助（固定 ~0.4ms）→ 在计算量小（causal/cross 短 q）时占优。
- WMMA 硬件吞吐是 int8 的剩余上限（D=128 attn 达 ~10.9 TFLOP）。
- VAE 超长 self-attn 的瓶颈是 PV 的 WMMA + L1/L2 带宽冗余；**LDS 缓存 V_T tile 是唯一有效结构优化（-35%），且仅对 D=128 超长序列有效**。
- **默认编译配置（无 env）即为各用例最优配置**：全部 env 默认值与 `smooth_k=False` 均经同进程 A/B 验证。
- 相对 FA 的剩余差距：attn 主 kernel 已与 triton/FA 持平或反超；剩余差距 = ① v_transpose 固有成本（native 独有）② int8 路径的 quant 辅助累计 ③ 短序列固定开销（两后端共性）——详见 §九。
- 指令级微优化全部失败；有效的优化是**结构性**的（见 §八"规律"）。

**gfx103x（RDNA2）：**
- 消费级 RDNA2 缺失 MFMA / WMMA，矩阵乘降级为 SIMD 点积指令链（V_DOT4_I32_I8 / V_DOT2_F32_F16）；无 tensor core，吞吐上限远低于 RDNA3。
- D=128 长 self 由 V2 架构（BM=64, NR=2, BN=16）反超 triton ~2.85x；D=64 长 self 仍慢 triton ~4x，为当前唯一性能瓶颈。
- 16-lane 协作（V3/V3.4）方案在 PV 输出完整性上失败（见 §七）；后续 D=64 优化需新结构（如 LDS 排布重构、warp 间协作）。

---

## 二、硬件与指令基础

### 2.1 硬件规格（开发平台）

| 项目 | gfx1103 (RDNA3, 开发平台) | gfx1035 (RDNA2, 移植目标) |
| --- | --- | --- |
| GPU | AMD Radeon 780M (iGPU) | AMD Radeon Graphics (Rembrandt iGPU) |
| ROCm | 7.15 (rocm-sdk) | 同 |
| CU | 12 | 6 |
| LDS | 128 KiB / CU | 64 KiB / CU |
| VGPR | 512 KiB / CU | 同 |
| L2 | 2 MiB | 同 |
| 显存 | 共享系统内存 DDR5-5600 64GB 双通道 | 共享系统内存 ~14.4 GiB |
| 内存带宽 | 理论 ~89.6 GB/s；实测 copy 73 GB/s、读 79 GB/s | 2200 MHz 时钟 |
| 矩阵指令 | **WMMA 16×16×16**（无 MFMA） | **V_DOT4_I32_I8 / V_DOT2_F32_F16**（无 WMMA / MFMA） |

实测内存带宽：copy 73 GB/s、sum 79 GB/s、torch transpose 仅 16.5 GB/s。**转置类操作是本平台的带宽瓶颈**；native v_transpose 31–37GB/s 已优于 torch 2 倍。

### 2.2 架构差异与指令映射

| 计算任务 | gfx1103 (RDNA3) | gfx1035 (RDNA2) | 备注 |
| --- | --- | --- | --- |
| int8 QK | WMMA i32 16×16×16_iu8 | `v_dot4c_i32_i8` (`__builtin_amdgcn_sdot4`) | RDNA2 SIMD 点积风格，每 lane 独立累加 |
| fp16/bf16 PV | WMMA f32 16×16×16_f16 | `v_dot2c_f32_f16` (`__builtin_amdgcn_fdot2`) | RDNA2 无 tensor core，fp16 矩阵乘降级 |
| 矩阵核 | 支持 WMMA | 无（消费级 RDNA2 缺失） | 彻底放弃 MFMA/WMMA 方案 |

gfx1035 不支持 bf16 硬件指令，bf16 输入需在 LDS 暂存时转换为 fp16 处理。

### 2.3 WMMA 布局（probe 实验验证，gfx1103）

- **C 输出**：lane L 持有 `C[2e+(L>>4)][L&15]`，e=0..7（行 = 2e+hw，列 = L&15）。
- **A operand**：lane L 提供 A 行 `L&15` 的 16 列（`a[k] = A[L&15][k]`）。
- **B operand**：lane L 提供 B 列 `L&15` 的 16 行（`b[k] = B[k][L&15]`）。
- A 与 B 的 operand 布局不对称（A 按行、B 按列）——这是转置 PV 设计的基础。
- 每 lane 的 A/B operand 为 16 个元素（`v16h`），C 为 8 个元素（`v8f`）。
- 转置 QK 的用途：交换 operand（A=k, B=q^T）后，WMMA 输出为 `S^T = K @ Q^T`，`lane L 持有 S[L&15][2e+(L>>4)]` 恰好使 lane L 与 L^16 持有同一 softmax 行的偶/奇列，permlanex16（XOR-16 交换）一次即可合并整行归约。

### 2.4 v_permlanex16_b32 指令（gfx1103）

- 执行 XOR-16 跨半波交换（lane 0↔16, …, 15↔31），纯寄存器操作。
- 限制：只能 XOR-16，不能做半波内（XOR-8/4/2/1）通信。
- 踩坑：早期误判该指令在 gfx1103 不可用（driver bug），实际是 `__builtin_amdgcn_permlanex16` 参数格式错误；参考 Triton 生成的 ISA 编码即可。
- 性能：`__shfl_xor_sync` 替代反而慢 19%。

### 2.5 ds_bpermute_b32 与 __shfl_xor

`__shfl_xor` 编译为约 6 VALU + 1 ds_bpermute（含 exec mask 边界检查），延迟 20–30 cycles。内联汇编无法复制其 exec mask 管理逻辑。`__shfl_xor` 是唯一可靠方式，但本方案用 permlanex16 而非它。

### 2.6 MFMA

MFMA 在 RDNA3（gfx1103）不可用（编译错误/LLVM 崩溃），仅支持 WMMA 16×16×16。这是硬件限制，native 与 triton 相同。


---

## 三、核心算法与优化

### 3.1 转置布局算法

**核心设计**：将 QK 与 PV 都以"转置"形式输入 WMMA（qk^T = k @ q^T，out^T = V^T @ P^T）。优势：

- **QK**：lane L 与 L^16 持有同一 softmax 行的偶/奇列，permlanex16 一次合并整行归约。
- **PV**：A operand（V^T）按行提供，B operand（P^T）按列提供，恰好与 RDNA3 WMMA operand 布局天然匹配，无 shuffle。

### 3.2 量化路径（int8 QK + fp16 PV）

RDNA3 i8 WMMA 为 fp16 的 2 倍吞吐（QK 占一半计算量），int8 路径在长 self-attn 占优。量化 Q/K 到 int8（per-32-row / per-16-col group scale），PV 保持 fp16。输出 dtype 与输入一致（fp16 输入 → fp16 输出；bf16 输入 → bf16 输出，V/OUT dtype 分离方案 B）。

### 3.3 direct 路径（fp16/bf16 QK + fp16 PV）

计算量小（causal / cross 短 q）时跳过 quant 辅助（固定 ~0.4ms），省 quant+mean 调度开销。fp16 QK 用 WMMA 16×16x16_f16；bf16 用 wmma_f32_bf16。

### 3.4 LDS 缓存 V_T tile PV（仅 D=128 超长 self-attn）

D=128 超长 self-attn（kv≥16384）的瓶颈是 PV 的 WMMA + L1/L2 带宽冗余（每次迭代 128KB v_frag 全局读）。将 V_T tile 缓存在 LDS（16KB/迭代），消除 8 倍 L1/L2 冗余 → SDXLVAE01 94.7→62ms（-35%）。**仅对 D=128 超长序列有效，D=64 全场景为负优化**（D=64 v_frag 仅 8KB，LDS 缓存反而浪费 occupancy）。

### 3.5 V 全局转置 PV 方案（`SAGEATTN_VT_GLOBAL=1`，默认）

V 一次性转置为 `V_T [B,H,D,N]`（core.py 调用 `ops.v_transpose`），PV 的 A=V^T 从 128 次 u16 LDS 列读变为 1 条行读（b128）→ direct 路径真实提升 17–36%。配合 v_transpose 一次全局转置开销，短序列固定开销略有增加（v_transpose 约 0.15-0.5ms），但长序列 PV 收益远超。

注意：V_T 的 n 维 padding 到 64 的倍数（防 attn kernel 的 v_frag_t 32B 直读越界，kv_len 非 64 倍数时最后 kv-tile 越界读未初始化内存 → NaN；padding 区由 v_transpose 填 0）。

### 3.6 mean kernel 32B 向量读

mean kernel 用 32B 向量读（`uint4` 一次 8 个 fp16/bf16），带宽 38→68–77GB/s，优于 torch mean 8–43%。**仅在显式 `smooth_k=True` 时生效**（默认 False，见 §3.12）。

### 3.7 v_transpose grid 分派

小数据量（total_tiles ≤ 4096）用 grid=192（每 block 多 tile，barrier 复用），实测 SDXL10（1920 tiles）v_transpose 0.237→0.147ms（-38%，固定开销占比大）；大数据量保持每 block 1 tile（Anima01 8192 tiles grid 减小反而 +5-10%，barrier 串行 + L2 局部性损失）。`SAGEATTN_VT_GRID`：0=auto，N=强制固定 grid。

### 3.8 quant 优化链

- **pass1 → LDS 缓存**：第一轮 quant pass 读 K 全局，第二轮从 LDS 读，省一半 K 全局读。
- **大 block + 多累加器 ILP**：D=64 用 BLK_Q=128/BLK_K=64（每 block 顺序处理多 RATIO 组，独立 scale + 多累加器）→ quant 单独再 -5~-18%（带宽 23.5→87–95GB/s）。
- D=128 保持旧逻辑（BLK=MIN_BLK，已 ~100GB/s 接近上限，大 block 仅 +1~3%）。
- `SAGEATTN_QUANT_BLK`：1=auto（按 head_dim），128=强制 128/64，64=强制 64/32，0=旧逻辑。

### 3.9 D=128 int8 BN=32→64

D=128 int8 默认 BN=64（实测相对 triton 更优，Anima01 1.038→1.02, Anima03 1.051→0.99, Anima05 1.045→1.02）：kv-tile 数减半 → barrier / k_tile 填充减半。`SAGEATTN_INT8_BN128` 可覆盖：128/16/32。

### 3.10 每 warp 32 行 QK kernel（仅 D=64 int8 self）

BM=128/4 warps，每 warp 处理 2 子块（2×16=32 行），2 子块共享 k_frag 提升 ILP → int8 self D=64 从慢 7–13% 追平。`SAGEATTN_INT8_32=0` 可关闭。

### 3.11 写回向量化

attn 输出写回为 16B 向量写（`uint4` 一次 8 个 fp16/bf16）：out_off = b*stride_b + n*stride_n + h*stride_h + d，d 恒为 8 倍数。permlanex16（XOR-16 跨半波交换）让 lane L 与 L^16 拥有同行的偶/奇 D 列，一次合并写回。**要求 q strides 是 8 倍数**（core.py 断言；非 contiguous 的 q 会导致未对齐 16B 写 UB）。

### 3.12 smooth_k=False（默认）

实测 randn 下精度不降反升（减 mean 反而引入 mean 值自身的 fp16/bf16 舍入误差），端到端省 0.5-3.5%（见 §9.3）。**默认 smooth_k=False**（跳过 mean kernel）；仅在 K 有明显非零 DC 偏置时减 mean 才有价值，此时可显式传 `smooth_k=True`。

### 3.13 D=128 int8 self BM 尺寸自适应
针对 D=128 int8 self 路径，当 kv_len ≥ 6144 时，动态将 Block 行维度（BM）从 64 切换为 128。
- **收益**：Anima05 提升 4.2%，SDXLVAE01 提升 5.8%，D=64 路径无回退。
- **正确性**：BM=128 用例输出 bit-identical (maxdiff=0)。
- **机制**：默认阈值 6144，可通过环境变量微调。

### 3.14 blockReduceMax 跨调用竞态修复
- **现象**：D=64 int8 全路径重复调用时，输出存在非确定性误差（maxdiff≈1.2e-2）。
- **根因**：`reduction_utils.cuh` 中的 `blockReduceMax` 使用 `static __shared__`。D=64 的 BLK_Q=128 导致该函数在同一个 block 内被 RATIO=4 循环调用多次。第 r 次调用的读阶段与第 r+1 次调用的写阶段之间**缺乏 barrier**，导致 q_scale 偶发读取到下一组 r 的 amax。
- **修复**：在 `blockReduceMax` 末尾补充 `__syncthreads()`（尾随 barrier），使每次调用自同步。
- **验证**：修复后 D=64 quant 重复调用 maxdiff=0（逐位一致）；e2e 全路径性能落在 ±0.3% 内，无回退。

---

## 四、测量方法论

**统一标准**（所有 benchmark 结论必须遵循）：

1. **同进程内交错轮转**：native 与 triton 在同一 Python 进程内交替跑（避免跨进程初始化 / 散热 / 驱动状态差异），每轮 warmup + iters。
2. **充分预热**：≥20 次 warmup（前几轮受 kernel JIT / cache 冷启影响）。
3. **取中位数**：≥50 次迭代取 median（避免尾延迟 / GC 干扰）。
4. **iGPU 热状态波动 ±5–7%**：开发平台为共享内存 iGPU（Radeon 780M），长时间运行温度上升导致频率下降；跨会话绝对时间不可比，但同会话内比率口径成立。
5. **固定随机种子**（测试用）：`torch.manual_seed(0)` 保证可复现。

**踩坑**：早期用 `time.perf_counter()` 测 GPU kernel 不加 `torch.cuda.synchronize()` 会严重低估（GPU 异步）；必须在每次计时前后同步。

---

## 五、分发逻辑（实测阈值）

`core.py` 根据 headdim / kv_len / q_len / is_causal / tensor_layout 选择 direct 或 int8 路径。阈值扫描见 `bench_threshold*.py`。

### 5.1 direct vs int8 选择

| headdim | 路径 | 条件 |
| --- | --- | --- |
| D=64 HND self | int8 | kv_len > 2048 |
| D=64 NHD self | int8 | kv_len > 3072 |
| D=64 causal | direct | kv_len ≤ 6144（int8 在 6144 持平，8192 int8 优 5.5%） |
| D=64 cross (q<kv) | direct | kv_len ≤ 6144（q=3072/kv=4096 仍 direct 优 3%） |
| D=128 self/causal | direct | kv_len ≤ 2048（kv=2048 direct 优 5%，2560 int8 优 5%） |
| D=128 cross q<<kv | direct | q < kv/2 且 kv ≤ 4096（q=512/1024 vs kv=4096 direct 优 8-25%） |

**核心权衡**：int8 路径省 QK 2x 吞吐（WMMA i8 2x fp16），但有 quant + mean 辅助开销（固定 ~0.4ms）。计算量小（causal/cross 短 q）时 direct 占优；计算量大（self 长序列）时 int8 占优。

环境变量覆盖（不推荐，仅实验用）：
- `SAGEATTN_DIRECT_THRESHOLD_D64`（HND=2048, NHD=3072）
- `SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL`（6144）
- `SAGEATTN_DIRECT_THRESHOLD_D64_CROSS`（6144）
- `SAGEATTN_DIRECT_THRESHOLD_D128`（2048）
- `SAGEATTN_DIRECT_THRESHOLD_D128_CROSS`（4096）

### 5.2 int8 path：V/OUT dtype 分离（方案B）

- bf16 输入：V 转为 fp16（融合进 v_transpose，bf16→fp16 一次 kernel 完成，省掉独立 v.to(fp16)）；OUT 由 kernel 直接写 bf16（o 复用，省 o.to(bf16) 转换 kernel）。
- fp16 输入：V=fp16, OUT=fp16（现状）。
- `v_is_bf16` / `out_is_bf16` 由 output dtype 决定；attn kernel 模板参数 VTYPE / OTYPE 分派。

### 5.3 V 全局转置

所有路径（direct + int8）统一使用 `v_transpose` 预转置 V 为 `V_T [B,H,D,N]`（`SAGEATTN_VT_GLOBAL=1`，编译时常量 + core.py runtime 行为）。n 维 padding 到 64 倍数。

---

## 六、编译

### 6.1 拆分后的源文件结构

`attn_gfx11.cu` 已拆分为两个架构独立源 + 各自 pybind + 各自头文件：

```
csrc/
├── attn_gfx110x.cu / .h    # RDNA3 (gfx110x): WMMA kernel + gfx11 dispatch + 公共辅助
├── attn_gfx103x.cu / .h    # RDNA2 (gfx103x): V_DOT4/V_DOT2 kernel + gfx10 dispatch + 公共辅助副本
├── pybind_gfx110x.cpp       # _qattn_gfx110x pyd (RDNA3)
├── pybind_gfx103x.cpp       # _qattn_gfx103x pyd (RDNA2)
├── mma_gfx11.h              # RDNA3 WMMA primitives (permlanex16, wmma_*, p_frag)
├── mma_gfx10.h              # RDNA2 V_DOT4/V_DOT2 kernel (attn_kernel_gfx10_*)
├── attn_gfx10_new.h         # RDNA2 v3.5 实验 (16-lane 协作, 默认不启用)
└── reduction_utils.cuh      # 公共 blockReduceMax
```

两架构 .cu/.h 互不依赖：RDNA3 不 include mma_gfx10.h / attn_gfx10_new.h；RDNA2 不 include mma_gfx11.h。公共辅助（mean/quant/v_transpose kernel + host 函数）各复制一份。

### 6.2 setup.py 按 GPU_ARCHS 分发

```bash
# 仅编 RDNA3 (gfx110x)
GPU_ARCHS=gfx1103 pip install -e . --no-build-isolation

# 仅编 RDNA2 (gfx103x)
GPU_ARCHS=gfx1035 pip install -e . --no-build-isolation

# 同时编两个 (默认, 适用于 wheel 分发)
GPU_ARCHS=gfx1103,gfx1035 pip install -e . --no-build-isolation
# 或省略 GPU_ARCHS, 默认含 gfx1103 + gfx1035
```

生成对应 pyd：`sageattention/_qattn_gfx110x.pyd`（RDNA3）、`sageattention/_qattn_gfx103x.pyd`（RDNA2），或同时两个。

### 6.3 core.py 运行时架构检测

`core.py` 不在 import 时硬加载 pyd；首次调用 `sageattn` 时检测当前 HIP 设备的 `gcnArchName`（`gfx1103` → `gfx110x`；`gfx1035` → `gfx103x`），懒加载对应 pyd 并注册 `torch.ops.sageattention`。同进程只加载匹配设备的一个 pyd，op 注册不冲突。

`SAGEATTN_BACKEND` 环境变量仍由 import 时读取（`triton` / `native` 默认 native）；如需运行时切换后端，轮转脚本须用 `core._BACKEND` 模块变量。

### 6.4 关键编译选项（HIP）

```python
HIP_FLAGS = [
    "-O3", "-std=c++17", "-ffast-math", "-fgpu-flush-denormals-to-zero",
    "-fno-offload-uniform-block",
    "-D__HIP_PLATFORM_AMD__=1",
    "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__",
    "-mllvm", "--lsr-drop-solution=1",
    "-mllvm", "-enable-post-misched=1",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "-amdgpu-max-memory-clause=32",
    "-mllvm", "-amdgpu-vgpr-index-mode=1",
    "-DSAGEATTN_VT_GLOBAL=1",  # V 全局转置 PV 方案（默认）
]
```


---

## 七、易踩坑点（后续开发必读）

### 7.1 gfx110x（RDNA3）

1. **permlanex16 参数格式**：`v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]`，`%2` 为 s 字面常量（如 `0x76543210`），`%3` 为 n 字面常量（如 `0xfedcba98`）。早期误用动态 src desc 报 "unrecognized instruction"，参考 Triton 生成的 ISA 编码即可。

2. **`__shfl_xor_sync` 替代 permlanex16 反而慢 19%**：`__shfl_xor` 编译为约 6 VALU + 1 ds_bpermute（含 exec mask 边界检查），延迟 20–30 cycles。permlanex16 是纯寄存器 XOR-16 交换。

3. **MFMA 不可用**：RDNA3（gfx1103）无 MFMA（硬件不支持），`+mai-insts` 编译 flag 开启会导致 LLVM AMDGPU 后端 WMMA codegen 段错误。

4. **V_T n 维 padding 到 64 倍数**：`v_transpose` 必须将 n 维 padding 到 64 倍数（防 attn kernel 的 v_frag_t 32B 直读越界 → NaN）。padding 区由 v_transpose 填 0。

5. **写回 16B 对齐**：q strides 必须是 8 倍数（half 粒度），否则 16B 向量写 UB。`core.py` 断言保证 contiguous 输入满足。

6. **wmma_* builtin 仅在 `__GFX11__` 编译时可用**：`mma_gfx11.h` 中提供 `__GFX11__` 宏分支的 builtin（`__builtin_amdgcn_wmma_*`），其他架构返回零值（编译通过、运行失效）。gfx110x 编译必须传 `--offload-arch=gfx11xx`。

7. **stable torch library 注册**：`STABLE_TORCH_LIBRARY(sageattention, m)` 在同一进程只能注册一次（同名 schema 重复注册报错）。拆分后两个 pyd 都注册 `sageattention`，但 `core.py` 懒加载机制保证同进程只 import 一个 pyd，无冲突。

8. **dispatch 中 `is_causal` 是运行时参数**：必须用 `if/else` 在 host 端生成编译期模板常量（`LAUNCH_*_T(HD, true/false, ...)`），不能直接当模板参数传。

9. **namespace 边界**：原文件 `namespace {` 在第 30 行，约 1892 行闭合（`}  // namespace`）。**所有 host / dispatch 公共函数必须在 namespace 外（全局作用域）**—— 否则 pybind 链接时 LNK2001（匿名 namespace 内部链接 vs 外部声明的 mangle 名称不匹配）。拆分时务必保留 1892 行的 `}  // namespace`。

10. **拆分时的 `// ---- gfx10` 块必须整段删除**：原 gfx11.cu 的 `qk_int8_sv_bf16_attn_gfx11_t` / `fp16_attn_gfx11_t` / `bf16_attn_gfx11_t` 各自函数体开头有 `if (prop.major == 10) { ... }` RDNA2 分支（运行时设备检测 + gfx10 kernel 分发）。RDNA3 版必须整段删除（含外层 `{` 和 `if` 闭合 `}`），否则会因 `sageattn_gfx10::*` 符号找不到而编译失败。

11. **blockReduceMax 竞态陷阱**：若 `blockReduceMax` 在循环中被多次调用（RATIO > 1），**必须在函数末尾添加 `__syncthreads()`**。否则 shared memory 的跨调用复用会导致读写重叠，引发严重的非确定性精度问题。

### 7.2 gfx103x（RDNA2）

1. **消费级 RDNA2 缺失 MFMA / WMMA**：gfx1035 等消费级 RDNA2 不支持 MFMA / WMMA 张量核，`+mai-insts` 编译 flag 不可加（且无意义）。矩阵乘降级为 SIMD 点积指令（V_DOT4_I32_I8 / V_DOT2_F32_F16），每 lane 独立累加。

2. **bf16 无硬件指令**：gfx1035 不支持 bf16 V_DOT，必须在 LDS 暂存时转换为 fp16（增加 LDS 占用与转换开销）。这是 D=64 bf16 路径慢 triton 的部分原因。

3. **V3 / V3.4 16-lane 协作 PV 输出不完整**：V3 设计的 16-lane 协作 QK（warp_reduce_sdot4 在 16 lanes 内归约）正确，但 PV 段每 lane 只写 4 cols（D=64）或 8 cols（D=128），剩余 D/2 列未初始化（lane 16..31 冗余但不写）。`SAGEATTN_GFX10_V3=1` 可强制启用但 PV 输出不完整，**默认禁用**。

4. **V2 架构定型**：最终采用 V2（BM=64, NR=2, NW=4, BN=16）作为核心 kernel。Block 128 线程（4 warps × 32 lanes），每线程 2 行（NR=2）多行独立点积链提升 ILP 隐藏 sdot4 延迟；BN=16 显著降低单 tile LDS 占用（K tile ≈ 1KB, V tile ≈ 2KB），避免 occupancy 崩溃。

5. **D=64 长 self-attn 仍慢 triton ~4x**：V2 在 D=128 长 self 显著优于 triton（~2.85x），但 D=64 长 self 计算密度不足 + 16-lane reduction + LDS 容量限制，仍慢 triton。后续优化方向：LDS 排布重构、warp 间协作、专用 D=64 kernel。

6. **V3.5（attn_gfx10_new.h）默认不启用**：16-lane 协作设计，未被 dispatch 调用。`attn_gfx10_new.h` 仍 include 在 RDNA2 源中（占编译体积但不影响功能）；性能较差的实验分支不保留，**默认不调用**。

7. **gfx10 dispatch 函数体提取**：拆分时把原 `attn_gfx11.cu` 中 `if (prop.major == 10) { ... }` 内层块提取为独立的 gfx103x dispatch 函数，必须去除 `is_gfx10` 判断（整个 pyd 就是 gfx10）和外层 `if` 包裹。内层内容缩进多一层不影响编译但代码可读性略差。

8. **公共辅助副本**：mean_hnd_kernel / quant_qk_int8_hnd_kernel / v_transpose_kernel 在 gfx103x 源中各复制一份（与 gfx110x 完全相同的实现）。分开编译符号不冲突。

### 7.3 通用

1. **`__shfl_xor_sync` 编译延迟高**：约 6 VALU + 1 ds_bpermute，延迟 20–30 cycles。本方案用 permlanex16 / lane-reduce 替代。

2. **`exp2f` vs `expf`**：所有 softmax 用 `exp2f`（乘 log2e 预缩放），比 `expf` 快约 1 个 cycle。

3. **`getenv` 在 Windows 平台 deprecation 警告**（`_CRT_INSECURE_DEPRECATE`）：无害，hipcc 不报 error。`-D_CRT_SECURE_NO_WARNINGS` 可消除警告（未加，无影响编译）。

4. **`torch.cuda.get_device_properties().gcnArchName`**：AMD HIP 返回 `"gfx1103"` 等字符串；`major` 字段为 gfx major 整型（gfx1103 → 11，gfx1035 → 10）。core.py 优先用 gcnArchName 字符串前缀判断，回退到 major 整型。

5. **`importlib.reload(sageattention)` 切换 backend**：reload 重读 `SAGEATTN_BACKEND` 环境变量，`sageattn` 函数引用模块级 `_BACKEND` 会正确切换；但首次 reload 有 pyd 初始化开销。**严格 perf 必须用独立子进程**（每 backend 一个子进程），避免 reload 干扰。


---

## 八、失败的优化尝试（避免重复）

> 按"投入与收益"排序的失败实验记录。每条标注失败原因（避免后续重复投入）。

### 8.1 gfx110x（RDNA3）

**A. 指令级微优化**（全部失败或可忽略）

| 编号 | 尝试 | 失败原因 |
| --- | --- | --- |
| A1 | `__shfl_xor_sync` 替代 permlanex16 | 慢 19%（permlanex16 是纯寄存器 XOR-16 交换，__shfl_xor 走 ds_bpermute） |
| A2 | 内联汇编 hand-craft permlanex16 | 无法复制 exec mask 管理逻辑（边界 lane 处理），放弃 |
| A3 | `expf` 替代 `exp2f` + log2e 预缩放 | exp2f 已比 expf 快 1 cycle |
| A4 | hand-craft `__shfl_xor` PTX 替代 builtin | 同 A2，exec mask 不可控 |
| A5 | mfma intrinsic（CDNA 指令） | RDNA3 不支持，编译错误 / LLVM 崩溃 |
| A6 | 减少 `__syncthreads` 数量 | 已最小化（每个 kv-tile 1 次），无法再减 |
| A7 | K/V prefetch 改用 `__builtin_amdgcn_global_load_lds` | 与现有 LDS 缓存路径等价，无收益 |

**B. 量化 / 数据布局**（部分有效，部分失败）

| 编号 | 尝试 | 失败原因 |
| --- | --- | --- |
| B1 | fp8e5m2 存储 V（替代 fp16） | 精度损失大（V 动态范围大），cos<0.95；详见附录（已删除） |
| B2 | int4 量化 Q/K | WMMA 不支持 int4，需软件模拟，性能反降 3-5x |
| B3 | per-head 量化 scale（替代 per-32-row） | 精度无提升（随机数据），quant 开销增加 |
| B4 | smooth_k=True 默认 | randn 下精度不降反升（减 mean 引入舍入误差），端到端慢 0.5-3.5% |
| B5 | v_transpose 融合进 attn kernel（消除独立 v_transpose） | LDS 占用翻倍，occupancy 崩溃 |

**C. 结构优化**（有效记录于 §一，其余失败）

| 编号 | 尝试 | 状态 |
| --- | --- | --- |
| C1 | V 全局转置 PV（`SAGEATTN_VT_GLOBAL=1`） | **✓ 17-36% 提升**（§3.5） |
| C2 | V_T tile LDS 缓存 PV（D=128 超长） | **✓ 35% 提升**（§3.4） |
| C3 | 每 warp 32 行 QK kernel（D=64 int8 self） | **✓ 7-10% 提升**（§3.10） |
| C4 | bf16→fp16 融合 v_transpose | **✓ 5% 提升**（§3.5） |
| C5 | quant 大 block + 多累加器 ILP | **✓ 5-18% 提升**（§3.8） |
| C6 | mean kernel 32B 向量读 | **✓ 带宽 38→68-77 GB/s**（§3.6） |
| C7 | v_transpose grid 分派 | **✓ 11% 提升**（§3.7） |
| C8 | V_T tile LDS 缓存 PV（D=64） | ✗ D=64 v_frag 仅 8KB，LDS 缓存浪费 occupancy，全场景负优化 |
| C9 | 16B 写回（全部 kernel） | **✓**（§3.11） |
| C10 | wpe2/wpe4（每 lane 2/4 个 kv-tile 流水线） | 收益 < 1%，无意义 |
| C11 | 双重 quant（先 K mean 再 quant K） | mean 路径 overhead 抵消，放弃 |
| C12 | fp8 量化 Q/K（e4m3 / e5m2） | WMMA 不支持，需软件 unpack；与 int8 路径吞吐优势持平但 quant 更慢 |

**D. 分发 / 调度**

| 编号 | 尝试 | 失败原因 |
| --- | --- | --- |
| D1 | 所有路径统一 int8（禁用 direct） | causal / 短 cross 计算量小时，quant 辅助开销 > int8 吞吐收益 |
| D2 | 所有路径统一 direct（禁用 int8） | 长 self-attn i8 2x 吞吐优势消失，慢 30-50% |
| D3 | 运行时 auto-tune（每 shape 选最优配置） | 开销 > 收益；env 覆盖阈值足够 |
| D4 | 动态 batch merge | 与上游 SD pipeline 冲突，放弃 |

**E. 已删除 / 默认不启用的分支**

- fp8e5m2 存储 V（实验记录，详见历史 commit）：精度不达标，已删除。
- `SAGEATTN_VT_GLOBAL=0`（原转置 PV 路径）：已被 `=1` 全面超越，保留仅为兼容性（不再维护）。
- `SAGEATTN_INT8_BM128=1`（D=128 VAE 用例）：LDS PV 优化后已被 LDS PV 路径超越，保留仅为实验。
- `SAGEATTN_GFX10_V3=1`（RDNA2 V3 16-lane）：PV 输出不完整，默认禁用（见 §7.2）。
- `attn_gfx10_new.h`（RDNA2 V3.5）：未被 dispatch 调用，默认不启用（编译进 pyd 占体积）。

### 8.2 gfx103x（RDNA2）

| 编号 | 尝试 | 失败原因 |
| --- | --- | --- |
| R1 | MFMA 路径 | 消费级 RDNA2 缺失，编译错误 |
| R2 | V1（每 warp 1 行） | D=128 长 self 慢 triton ~10x |
| R3 | V3（16-lane 协作 QK/PV） | QK 正确但 PV 输出不完整（每 lane 仅写 D/16 cols） |
| R4 | V3.4（attn_gfx10_new.h，16-lane + tile 排布） | 同 R3，PV 输出仍不完整 |
| R5 | BM=128（V1/V2 加大 block） | LDS 占用翻倍，occupancy 崩溃；D=64 长 self 反而慢 |
| R6 | BN=32（D=64） | 边界处理复杂，无收益；D=64 V2 BN=16 已最优 |
| R7 | 全局 V_T + v_frag_t 32B 直读（V2.2） | 长序列有效但短序列负优化；默认 auto 模式（D=64 self 长 + 短 cross 用 V2.2） |
| R8 | 双 warp 协作 QK（NR=4） | LDS 占用翻倍，sdot4 延迟未被有效隐藏 |

### 8.3 规律

- **指令级微优化全部失败**（A1-A7）：RDNA3 编译器已对 WMMA 路径做充分调度。
- **结构优化是唯一有效路径**（C1-C9）：减少冗余读（LDS 缓存、V 转置）、提升 ILP（大 block、warp 内多子块）、融合冗余 kernel（v_transpose + bf16→fp16）。
- **失败的量化 / 精度方案**：fp8、int4、per-head scale 均因精度或硬件不支持而失败；int8 per-group scale + fp16 PV 是当前最优点。


---

## 九、性能差距分析与剩余可优化方向

### 9.1 native 与 FA / triton 的剩余差距来源

**相对 FA / triton 的剩余差距**（attn 主 kernel 已持平）：

1. **v_transpose 固有成本**（native 独有）：每次调用 ~0.15-0.5ms（随 seq_len 增长）。triton / FA 无此开销（直接在 attn kernel 内转置或避免转置）。短序列下此开销占比大（SDXL10 v_transpose 0.147ms / 总 0.6ms = 25%）。
2. **int8 路径的 quant 辅助累计**：quant + v_transpose + mean（即使 smooth_k=False 也仅省 mean）共 ~0.4-1.1ms。triton / FA 跳过 quant（直接 fp16/bf16 算）或 quant 开销更低。
3. **短序列固定开销**（两后端共性）：kernel launch + grid setup + 同步开销，绝对差仅 0.03-0.12ms，但占比大。SageAttention 两后端均存在，非 native 特有。

### 9.2 剩余可优化方向（理论收益 + 风险）

| 方向 | 理论收益 | 风险 | 优先级 |
| --- | --- | --- | --- |
| **消除 v_transpose**（attn kernel 内原地转置 V） | 短序列 -15-25% | LDS 占用翻倍，occupancy 崩溃（B5 失败记录） | 中 |
| **int8 路径减 quant 开销**（在线 quant / K 预 quant） | 长 self -3-5% | 在线 quant 与 attn kernel 同步复杂 | 低 |
| **fp8 V 存储**（替换 fp16 V_T） | 带宽 -50%（V 占用减半） | 精度不达标（B1 失败）；需 RDNA3 fp8 硬件支持（gfx1103 不支持） | 不推荐 |
| **per-stage 流水线**（KV 重叠计算与访存） | 长 self -2-5% | 当前 LDS 已较紧，加 stage 可能 occupancy 下降 | 低 |
| **更激进的 int8 WPE**（每 lane 8 个 kv-tile） | 长 self -1-3% | C10 失败记录，收益 < 1% | 不推荐 |
| **RDNA2 D=64 专用 kernel** | D=64 长 self 大幅提升 | 需新设计（LDS 排布、warp 协作），工程量大 | 高（gfx103x 端） |
| **RDNA2 bf16 native 路径**（避免 fp16 转换） | D=64 bf16 -3-8% | 需 bf16 V_DOT2 支持，gfx1035 硬件不支持 | 不可行 |

### 9.3 smooth_k=True / False 决策

**实验**：随机数据下，`smooth_k=True`（减 K mean）端到端慢 0.5-3.5%。

- **原因**：K mean 值本身需 fp16/bf16 舍入存储，减 mean 引入额外舍入误差；randn 下 K 期望为 0（mean ≈ 0），减 mean 收益微乎其微，反被舍入误差拖累。
- **何时 smooth_k=True 有价值**：K 有明显非零 DC 偏置时（如某些 quantized K 来自非零中心分布）。
- **默认 `smooth_k=False`**：跳过 mean kernel 节省调度开销；如需启用，显式传 `smooth_k=True`。

### 9.4 RDNA2（gfx103x）后续优化方向

1. **D=64 长 self-attn**：当前 V2 仍慢 triton ~4x，需新结构。候选：
   - LDS 排布重构（D=64 v_frag 仅 4KB，可放更大 BN 或更多 K prefetch）。
   - warp 间协作（4 warps 共享 Q row，1 warp 算 QK，3 warps 算 PV）。
   - 专用 D=64 kernel（独立于 D=128 模板）。
2. **D=64 bf16 路径**：当前需 LDS 暂存转 fp16，损失吞吐。可考虑：
   - fp16 V 路径（用户接受 fp16 输出）。
   - 接受当前性能，文档说明 bf16 D=64 gfx1035 慢 triton。
3. **消除 v_transpose 固有成本**：与 gfx1103 同样问题，但 D=64 计算密度更低，v_transpose 占比可能更大。

---

## 十、文件结构与构建速查

```
sageattention-rdna3/
├── csrc/
│   ├── attn_gfx110x.cu / .h     # RDNA3 源 + 头
│   ├── attn_gfx103x.cu / .h     # RDNA2 源 + 头
│   ├── pybind_gfx110x.cpp        # RDNA3 pyd 入口
│   ├── pybind_gfx103x.cpp        # RDNA2 pyd 入口
│   ├── mma_gfx11.h               # RDNA3 WMMA primitives
│   ├── mma_gfx10.h               # RDNA2 V_DOT kernel
│   ├── attn_gfx10_new.h          # RDNA2 V3.5 实验（默认不启用）
│   └── reduction_utils.cuh       # blockReduceMax
├── sageattention/
│   ├── __init__.py
│   ├── core.py                   # 架构检测懒加载 + 公共 sageattn 入口
│   ├── triton_backend.py         # triton 后端
│   ├── _qattn_gfx110x.pyd        # RDNA3 native pyd (编译产物)
│   └── _qattn_gfx103x.pyd        # RDNA2 native pyd (编译产物)
├── setup.py                      # 按 GPU_ARCHS 分组编译
├── benchmark_attn/               # benchmark 脚本与历史结果
├── test_sageattn_rdna3.py        # 正确性 + 性能测试
└── NativeBackendOptimizeReport.md # 本报告
```

**构建命令速查**：
```bash
# 仅 RDNA3 (gfx1103 本机开发)
GPU_ARCHS=gfx1103 pip install -e . --no-build-isolation

# 仅 RDNA2 (gfx1035)
GPU_ARCHS=gfx1035 pip install -e . --no-build-isolation

# 同时编两个 (默认)
pip install -e . --no-build-isolation

# 跳过编译（仅安装 Python 部分）
SAGEATTN_SKIP_BUILD=1 pip install -e . --no-build-isolation
```

**运行时切换后端**：
```bash
SAGEATTN_BACKEND=triton python ...  # triton 后端
SAGEATTN_BACKEND=native python ...  # native 后端（默认）
```

