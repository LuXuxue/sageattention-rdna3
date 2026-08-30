# SageAttention Native Backend 优化报告（gfx1103）

本报告为项目优化的总记录：涵盖硬件基础、核心算法、实验过程与结论、分发逻辑、易踩坑点与剩余优化方向。所有性能结论遵循统一的测量方法论（§四）：**同进程内交错轮转 + 充分预热 + 取中位数**，跨进程/跨会话的绝对时间不可比。

---

## 一、最终结论（当前最优状态）

### 1.1 核心优化链（按收益排序）

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

### 1.2 当前 benchmark 状态

基线为最新配置（默认编译、无 env、`smooth_k=False`）。比率 = SageAttn/FlashAttn（或 native/triton），<1 为快。波动 ±5–7%（iGPU 热状态，见 §四）。

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
> 注 3：`core.py` 在 import 时读 `SAGEATTN_BACKEND` 一次；单跑 `benchmark_attn.py` 必须在进程启动前设 env，轮转脚本须用 `core._BACKEND` 模块变量切换（见 §4.3）。
> 注 4：VAE 用例 03/04（36864/55296）因 iGPU 长时间运行死机已从 `benchmark_attn.py` 注释，结论由 01/02（16384/24576）外推（见 §9.2）。

### 1.3 关键结论

- i8 WMMA 在 RDNA3 为 fp16 的 2 倍吞吐（QK 占一半计算量）→ int8 在长 self-attn 占优；direct 省 quant 辅助（固定 ~0.4ms）→ 在计算量小（causal/cross 短 q）时占优。
- WMMA 硬件吞吐是 int8 的剩余上限（D=128 attn 达 ~10.9 TFLOP）。
- VAE 超长 self-attn 的瓶颈是 PV 的 WMMA + L1/L2 带宽冗余；**LDS 缓存 V_T tile 是唯一有效结构优化（-35%），且仅对 D=128 超长序列有效**。
- **默认编译配置（无 env）即为各用例最优配置**：全部 env 默认值与 `smooth_k=False` 均经同进程 A/B 验证。
- 相对 FA 的剩余差距：attn 主 kernel 已与 triton/FA 持平或反超；剩余差距 = ① v_transpose 固有成本（native 独有）② int8 路径的 quant 辅助累计 ③ 短序列固定开销（两后端共性）——详见 §九。
- 指令级微优化全部失败；有效的优化是**结构性**的（见 §八"规律"）。

---

## 二、硬件与指令基础

### 2.1 硬件规格（开发平台）

| 项目 | 规格 |
| --- | --- |
| GPU | AMD Radeon 780M（gfx1103，RDNA3 iGPU） |
| ROCm | 7.14（`rocm-sdk path --root` 定位工具链） |
| CU | 12 |
| LDS | 128 KiB / CU |
| VGPR | 512 KiB / CU（每 SIMD 256 VGPRs/lane 上限，warp 级 1024 槽） |
| L2 | 2 MiB（无 Infinity Cache） |
| 显存 | 共享系统内存（DDR5-5600，64GB 双通道） |
| 内存带宽 | 理论 ~89.6 GB/s（128-bit @ 5600MT/s）；实测 copy 73 GB/s、读 79 GB/s |
| 矩阵指令 | WMMA 16×16×16（**无 MFMA**） |

实测内存带宽（`benchmark_attn/bench_bandwidth.py`）：copy 73 GB/s、sum 79 GB/s、torch transpose 仅 16.5 GB/s。**转置类操作是本平台的带宽瓶颈**；native v_transpose 31–37GB/s 已优于 torch 2 倍。

### 2.2 WMMA 布局（probe 实验验证）

- **C 输出**：lane L 持有 `C[2e+(L>>4)][L&15]`，e=0..7（行 = 2e+hw，列 = L&15）。
- **A operand**：lane L 提供 A 行 `L&15` 的 16 列（`a[k] = A[L&15][k]`）。
- **B operand**：lane L 提供 B 列 `L&15` 的 16 行（`b[k] = B[k][L&15]`）。
- A 与 B 的 operand 布局不对称（A 按行、B 按列）——这是转置 PV 设计的基础。
- 每 lane 的 A/B operand 为 16 个元素（`v16h`），C 为 8 个元素（`v8f`）。
- 转置 QK 的用途：交换 operand（A=k, B=q^T）后，WMMA 输出为 `S^T = K @ Q^T`，`lane L 持有 S[L&15][2e+(L>>4)]` 恰好使 lane L 与 L^16 持有同一 softmax 行的偶/奇列，permlanex16（XOR-16 交换）一次即可合并整行归约。

### 2.3 v_permlanex16_b32 指令

- 执行 XOR-16 跨半波交换（lane 0↔16, …, 15↔31），纯寄存器操作。
- 限制：只能 XOR-16，不能做半波内（XOR-8/4/2/1）通信。
- 踩坑：早期误判该指令在 gfx1103 不可用（driver bug），实际是 `__builtin_amdgcn_permlanex16` 参数格式错误；参考 Triton 生成的 ISA 编码即可。
- 性能：`__shfl_xor_sync` 替代反而慢 19%。

### 2.4 ds_bpermute_b32 与 __shfl_xor

`__shfl_xor` 编译为约 6 VALU + 1 ds_bpermute（含 exec mask 边界检查），延迟 20–30 cycles。内联汇编无法复制其 exec mask 管理逻辑。`__shfl_xor` 是唯一可靠方式，但本方案用 permlanex16 而非它。

### 2.5 MFMA

MFMA 在 RDNA3（gfx1103）不可用（编译错误/LLVM 崩溃），仅支持 WMMA 16×16×16。这是硬件限制，native 与 triton 相同。

---

## 三、核心算法与优化

### 3.1 转置布局算法

标准 flash-attention 的 QK WMMA（A=q, B=k）输出布局使归约需跨 16 lane 的 butterfly 通信。交换 operand（A=k, B=q）后输出为转置矩阵 `S^T = K @ Q^T`，归约只需 1 次 permlanex16：

- max：`gm = max(row_m, local_mx)` 后 `gm = max(gm, permlanex16(gm))`
- sum：`row_l += partial_sm + permlanex16(partial_sm)`

P fragment 寄存器内组装（`assemble_p_frag`）：lane L 的 8 个 exp 值对应 P 行 `(L&15)` 的偶/奇列，与 lane^16 交换后获得完整 16 列，打包成 fp16 对按 WMMA B-operand 布局交错。

转置 PV：`out^T = V^T @ P^T`（A=V^T, B=P^T），输出布局自动回到"out 行 `(L&15)` 的偶/奇 D 列"，写回按 `d = dt*16 + 2e + hw` 排列。

### 3.2 V 全局转置 PV（核心优化之一）

**动机**：原转置 PV 中 A = V^T 需要 V 的列读（跨行），经 LDS 中转后每迭代 128 次 u16 标量 LDS 读（低效且 bank 冲突）。

**解法**：V 一次性转置为 `V_T [B,H,D,N]`（n 连续），PV 的 A=V^T 变为行读（1 条 v16h）。

要点：
- wmma 参数顺序必须是 `wmma(v_frag_t, p_frag, out_acc)`（A=V^T, B=P^T, C=out^T）；若写成 `wmma(p_frag, v_frag_t)`（C=out）则 C 布局与写回转置，输出乱序（踩坑点 5）。
- `vt_off` 必须含 `kb_base`：`((b*H+hkv)*D + dt*16+m_row)*N + (kb_base + ct*16)`，漏掉 kb_base 会只处理前 32 个 key（踩坑点 6）。
- V_T 的 n 维必须 padding 到 64 倍数（见 §3.6）。
- 实现：`v_transpose` op（NHD/HND 均支持，LDS 分块转置，tile 32n×32d，pad 33 消 bank 冲突；vt 单独耗时 0.49ms@Anima01（D=128）≈ 34GB/s、D=64 约 31–37GB/s，接近转置类带宽上限）。
- `core.py` 在 int8 与 direct 路径均无条件转置 V；`setup.py` 默认编译 `-DSAGEATTN_VT_GLOBAL=1`。

> **V_T 行读是 direct 路径的刚需**：曾尝试在 direct 路径消除 v_transpose（改用 LDS 列读原始 V），结果全用例大幅负优化（self +27~33%、D=128 cross +31~41%），印证 v_transpose 是一次性支付的"划算"成本，不能省（见 §八 E4）。

### 3.3 每 warp 32 行 QK kernel（核心优化之二，仅 D=64 int8 self）

设计（`attn_kernel_impl_32_t`，BM=128，4 warps）：
- 每 warp 处理 32 行（2 个 16 行子块 sub0/sub1）。
- 2 子块共享 k_frag 加载 → 每迭代产生 2 条独立 WMMA 依赖链。
- softmax/PV/写回均 2 子块化（permlanex16 归约不变）。
- **VGPR 压力**：2 组 per-q 状态，D=64 约 160–190 VGPR 可编译；D=128 不可行（q_frag[2][8]+out_acc[2][8]+score_cache ≈ 256+ VGPR → 溢出，实测慢 10–24x）。
- D=64 self-attn 默认启用（`SAGEATTN_INT8_32=0` 回退 8 warps）。

### 3.4 V/OUT dtype 分离（方案 B）

- V 转 fp16 并入 v_transpose（读 bf16 直接转 fp16 写 V_T，省 `v.to(fp16)` 独立 kernel——实测后者 0.87ms@Anima01，即 5%）。
- 输出由 kernel 直接写 bf16（`OUT_DTYPE` 模板参数，o 复用，单次舍入精度更好）。
- fp32 输入仍走 core 的 `v.to(fp16)`（v_transpose 不支持 fp32，场景少）。

### 3.5 写回向量化

原写回为 2B 散布写（每 dt 8 次 × DTiles）。改为：permlanex16 交换（4 次/dt）获得完整行 16 列 → hw=0/1 lane 分工各写 16B 连续。fp16 路径组装 half 位模式；bf16 路径从 float 单次舍入转 bf16 再组装。应用到全部 3 个 attn kernel（int8 impl_t / int8 32行 / direct）。

实测无显著性能变化（写回被内存带宽/流水线掩盖），但消除 2B 散布写、统一代码。

### 3.6 V_T padding（non_aligned 修复）

**问题**：kv_len 非 64 倍数时（如 1057），attn kernel 的 `v_frag_t` 32B 直读越界——最后 kv-tile 读超出 V_T 的 n 维 → 未初始化内存 → NaN。k_tile 填充有掩码，v_frag_t 直读无掩码。

**修复**：V_T 的 n 维 padding 到 64 倍数（`padded_n = ceil(kv/64)*64`），三处规则一致：
1. `core.py`：`v_t = torch.empty(B, H, D, padded_n)`
2. `v_transpose_kernel`：写回用 `v_t_n = ceil(seq/64)*64` 寻址，padding 区写 0
3. 3 个 attn kernel：`vt_off` 行 stride 用 `v_t_n`（kernel 内 `((kv_len+63)/64)*64` 计算）

**衍生 bug**（review 发现）：v_transpose 的 ntiles 若按 `ceil(seq/32)` 计算，只覆盖 32 对齐区，padding 区 `[ceil32, ceil64)` 未写 0（kv mod 64 ∈ [1,32] 时如 2049/2113 仍 NaN）。必须按 `v_t_n/32` 计算 ntiles（kernel 与 host 同步）。

### 3.7 quant 优化链（LDS 缓存 + 大 block + 多累加器 ILP）

**瓶颈定位**：pass1 读 Q+K 仅 20GB/s（torch sum 同数据量 75GB/s）。逐个排除：非读粒度（16B→32B 已做）、非 blockReduce、非计算、非 block 调度/DRAM 行切换（GPB=2/4/8 合并无改善）。根因 = block 粒度小（32/16 行）+ 每 thread 迭代数少 + 固定开销占比高。

**优化 1（LDS 缓存）**：pass1 读入数据缓存到 LDS（`shared_data` uint4），pass2 从 LDS 读 → quant 2.28→2.15ms。同时 pass1 读粒度 16B→32B（2 连续 pack 合并）。

**优化 2（大 block + 多累加器 ILP）**：对齐 triton 的 BLK=128 大 block，但保持 MIN_BLK 粒度（scale 语义不变）避免 fmax 串行链：
- kernel 模板 `<T, HeadDim, BLK, MIN_BLK>`：每 block 处理 BLK 行，每 MIN_BLK 行一组独立 amax（RATIO 组），RATIO 个 fmax 累加器并行（fmax 依赖链长度不变 = 16 元素/组）。
- host 端 Q/K 拆分为两次 launch；`SAGEATTN_QUANT_BLK` 可选 1=auto（默认）、128/64/0（旧逻辑）。
- scale 写回加越界守卫（`base_row + r*MIN_BLK < seq_len`）。

**结果**（同进程 A/B）：D=64 用 BLK_Q=128/BLK_K=64 时 quant 单独 -5~-18%（带宽 23.5→87–95GB/s，SDXL07/13 最显著）；**D=128 保持旧逻辑**——旧逻辑已 ~100GB/s（超 torch sum 上限），大块（128/64 或 64/32）反而 +1~3%（32KB LDS 或额外 blockReduce barrier）。

### 3.8 V_T tile LDS 缓存 PV（仅 VAE 超长 self-attn）

**背景**：VAE 用例（D=128 int8 self，kv=16384~55296）。native 在 16384 用例比 triton 慢 52–72%（SDXLVAE01 94.7 vs 62ms）。分阶段 profiling 确认 attn 主 kernel 占 96–98%，辅助仅 2–5%。

**瓶颈定位（逐个排除）**：BM=128/8 warps、BN=16/32/128、wpe=1/2/4、occupancy 全部无效 → 非 DRAM 带宽/occupancy 受限，WMMA 计算本身接近吞吐上限。真正的结构性差距：PV 的 v_frag 每 lane 每 dt 从全局 V_T 行读 32B，每迭代 128 lanes × 32 次 × 32B = 128KB 全局读，而实际 tile 数据仅 16KB —— **8 倍冗余**。

**解法**：V_T 当前 tile（64n×128D）每迭代一次拷入 LDS（布局 `[D][N]`，行 = D 维、n 连续 16 half 行读），PV 的 v_frag 改从 LDS 行读：
- 全局读从 128KB/迭代（冗余）降到 16KB/迭代。
- `VTileStride=80`（行间 8 bank 偏移；20KB/128 行）。
- `__align__(32)` 保证 v16h 32B 对齐。
- 每迭代 +1 次 `__syncthreads()`（V 拷贝后）。

**结果**（同进程轮转）：SDXLVAE01 attn 94.7→62ms（-35%），SDXLVAE02 反超 triton 4%，AnimaVAE01 -36%（69.6→44.8ms）。LDS PV 优化后 BM=64 全面优于 BM=128（-6%）。

**踩坑**：初版 v_tile 大小沿用 `BLOCK_N*VStride`（64×144），而 `[D][N]` 布局需要 `HeadDim` 行 → 越界写 2 倍污染 LDS，4096+ 序列输出错误。必须 `(SAGEATTN_VT_GLOBAL) ? (HeadDim*VTileStride) : (BLOCK_N*VStride)` 三元区分布局。

> **适用范围**（重要）：本优化**仅对 D=128 超长序列（kv≥16384）有效**。D=64 direct 短序列（C8）与 D=64 int8 self（E1）均为负优化——D=64 的 V_T 数据量小（冗余读命中 L2）、PV 瓶颈是 WMMA 计算吞吐而非 L2 带宽。

### 3.9 v_tile / v_prefetch 条件声明（省 LDS/VGPR）

`VT_GLOBAL=1` 时 LDS `v_tile` 与寄存器 `v_prefetch` 完全未使用，但无条件声明占用 18KB LDS + 8×uint4 VGPR。改为编译期常量三元（`? 1 : N`，死分支被 DCE），三个 kernel 一致：
- D=128（impl_t）：无性能变化（occupancy 非瓶颈）。
- D=64 的 32 行 kernel（SDXL01/07/13）：实测反超 triton 8–9%（省 LDS/VGPR 的调度收益）。

### 3.10 mean kernel 32B 向量读

> 注：本优化仅在显式 `smooth_k=True` 时生效（默认 `smooth_k=False` 已跳过 mean kernel，见 §3.12）。

**动机**：triton 用 torch `k.mean()`（40–71GB/s），native mean 仅 ~38GB/s——原实现每 thread 每迭代只读 2B（标量）。

**实现**：每 thread 每迭代读行 s 的 16 个连续 half（32B，uint4 对齐），16 个列累加器（寄存器），256 threads 步进 s；LDS 16KB 中转做列归约。

**结果**（正确性 err=0）：带宽 38→68–77GB/s（接近读上限 79GB/s），全用例优于 torch mean 8–43%。

**mean 融合进 quant 不可行**：quant K pass1 需 mean 完成才能减 mean（跨 block 全局归约，K 数据 > L2 2MB 无法单 kernel 缓存），拆两 kernel 后 K 读仍 2 次，收益仅省 1 个 launch（~0.02ms）→ 分别优化更优。

### 3.11 v_transpose grid 分派

**动机**：SDXL10（1536/20h）的 v_transpose 仅 3.9MB @ 17GB/s——小数据量（1920 tiles）固定开销（block 调度/launch 尾部）占比大；SDXL10 attn kernel 本身已快于 triton，v_transpose 0.233ms 是主要差距来源。

**实现**：grid 分派——tiles≤4096 用 grid=192（每 block 多 tile，stride 分配）；大数据量保持每 block 1 tile（Anima01 8192 tiles grid 减小反而 +5–10%：barrier 串行 + L2 局部性损失）。`SAGEATTN_VT_GRID` env 可覆盖。

**结果**（同进程 A/B）：SDXL10 v_transpose -11%（0.142→0.126ms）；其余用例持平或略好。grid 进一步微调（96/48）端到端差异在 ±7% 噪音内，无一致规律 → 保持 auto=192。

### 3.12 smooth_k=False（跳过 mean kernel，默认）

**动机**：int8 量化路径原在 quant 前减 K 均值（`smooth_k`，对齐上游语义），需多跑一个 `mean_seq` kernel（0.077–0.22ms）。

**实测结论**（同进程 A/B + 精度验证）：
- **精度**：randn（零均值）下 `smooth_k=False` 不降反升（减 mean 反而引入 mean 值自身的 fp16/bf16 舍入误差）；仅当 K 有明显非零 DC 偏置时减 mean 才有价值（K=randn+10 时 cos 0.99993→0.9995，仍远超 0.99/0.05 阈值）。真实模型 K 经 LayerNorm 后均值≈0，仅 RoPE 后可能偏移。
- **性能**：端到端 -0.5~-3.5%（SDXL01 -3.5%、SDXL07/13 -0.5~-1.3%、Anima01 -1.3%）。

**实现**：`core.py` 默认 `smooth_k = kwargs.get("smooth_k", False)`；显式传 `smooth_k=True` 仍可覆盖（供 K 有明显 DC 偏置的场景）。triton 路径不受影响。

**效果**：SDXL01（D=64 int8 self 4096）借此反超 FA；多个用例相对 FA 的差距收窄。

---

## 四、测量方法论

### 4.1 热漂移与频率抖动

- iGPU 冷启动后频率剧烈波动（同一 kernel 单次时间可跳变 20 倍）。
- 需 ~600 次 kernel 调用预热才收敛；跨会话/跨进程绝对时间不可比。
- 固定顺序交错轮转有轮内顺序效应（先测的后端更冷更慢）。

### 4.2 修正后的测量方法

1. 每用例充分预热：各后端交替跑 ≥600 次调用。
2. 测量轮：5–6 轮 × 每轮 20–30 iters。
3. 轮间后端顺序旋转：第 r 轮从 r 号后端开始（对称采样）。
4. 各后端取中位数。

**唯一可信对比 = 同进程内交错轮转的倍率。**

### 4.3 layout_code 陷阱

- `tensor_layout="NHD"` 必须传 layout_code=0（kNHD）；传 1（kHND）会让 kernel 按 HND stride 解释 NHD 数据 → 输出错误（err≈3.0）但执行更快（假象）。
- 任何"手动拆分 ops 调用"的验证必须与 `core.py` 逐参数一致（含 layout_code）。
- native 正确性验证必须显式设 `SAGEATTN_BACKEND=native`：`core.py` 默认 `_BACKEND=triton`，不设 env 时 check 脚本测的是 triton 后端。

---

## 五、分发逻辑（实测阈值）

### 5.1 direct/int8 选择（core.py）

核心权衡：direct（fp16 QK）省 quant+mean 辅助 kernel（固定 0.4–0.6ms）；int8 用 i8 WMMA（RDNA3 上为 fp16 的 2 倍吞吐，优势与计算量成正比）。计算量小（causal/cross 短 q）→ direct；计算量大（self 长序列）→ int8。

阈值实测扫描（NHD 布局为 benchmark/实际应用布局，以下数据均按 NHD 重扫确认）：

| D | 场景 | 阈值（kv_len） | 关键数据 |
| --- | --- | --- | --- |
| 64 | self 非 causal | 3072 | 2560 direct 优 0.4%、3072 int8 优 3.8%、3456 int8 优 4.7% |
| 64 | causal | 6144 | 3072/4096 direct 优 5%/2%、6144 持平、8192 int8 优 5.5% |
| 64 | cross（q<kv） | 6144 | q=3072/kv=4096 仍 direct 优 3% |
| 128 | self/causal | 2048 | 2048 direct 优 5%、2560 int8 优 5%、4096 int8 优 16% |
| 128 | cross q*2<kv | 4096 | q=512/1024 vs kv=4096 direct 优 25%/8%、q=2048 int8 优 5% |

**注意：阈值受 tensor_layout 影响**——HND 布局下 D=64 self 平衡点降至 ~2048（2048 direct 优 3.7%、2304 int8 优 0.4%），因 stride/缓存访问模式不同。

> **[✓ 已实施] HND 布局阈值适配**：`core.py` 对 D=64 self 按布局区分默认阈值（`HND=2048` / `NHD=3072`），env `SAGEATTN_DIRECT_THRESHOLD_D64` 仍可覆盖；验证：HND 2304/2560 走 int8、NHD 保持 direct，正确性 OK。

int8 量化路径的适用场景（不仅限 self-attn）：
- D=64：self（kv>3072）、causal/cross（kv>6144）
- D=128：self/causal（kv>2048）、cross 且 q>=kv/2（kv>2048）

**causal D=64 边界补测**：6144 处 int8 优 1.4%（噪音内）、6656+ int8 优 3–9%，6144 以下 direct 优 → 阈值 6144 合理。

**GQA 阈值标定**：GQA 长序列 n/t≈1.0–1.03（与 MHA 同），当前阈值对 GQA 适用，无需调整。GQA 实测（Krea 48/12、GQA-D64 32/8 等）长序列与 triton 持平、短序列快 36%；kv_heads 共享加载收益小（kv 数据小，L2 命中掩盖重复读）→ 不实施。

### 5.2 int8 路径 kernel 选择

```
D=64  self-attn            -> 每 warp 32 行 kernel (BM=128, BN=32, 4 warps) [默认]
                              (SAGEATTN_INT8_32=0 回退 8 warps wpe1)
D=64  cross, kv<=77        -> (BM=64, BN=16) [wpe1]
D=64  cross, kv>77         -> (BM=64, BN=32) [wpe1]
D=128                      -> (BM=64, BN=64) [wpe2]  (BN=64 实测优于 BN=32; BN=16 全面变差)
VAE 超长 self (kv≥16384)    -> (BM=64, BN=64) [wpe2]，但 PV 走 §3.8 的 LDS 缓存路径
```

LDS PV 优化后 BM=64 全面优于 BM=128（-6%），BM=128/BN=128 仅保留 env 实验开关（默认禁用）。

### 5.3 fp16/bf16 direct 路径 kernel 选择

```
D=64  self                -> (BM=64, BN=64)
D=64  cross, kv<=128      -> (BM=64, BN=32)
D=64  cross, kv>128       -> (BM=128, BN=32)   [BM=128 快 9-13%]
D=128                     -> (BM=64, BN=16)    (BN=16 实测最优; BN=32/64 均更差)
```

**direct self 短序列 BM/BN 扫描确认默认已最优**：SDXL10/16/04 首轮扫描的个别 -7.5% 经严格复测证实为热漂移噪音，BM=32/BN=32 实际慢 1–2%。默认 BM=64/BN=64 即 direct self 最优；direct self 慢 FA 7–11% 是 v_transpose 固定成本（attn 主 kernel 已持平/快 FA），无法通过 BM/BN 调节消除。

### 5.4 环境变量汇总

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| SAGEATTN_BACKEND | 后端选择（native/triton） | triton |
| SAGEATTN_INT8_32 | int8 self 用每 warp 32 行 kernel | 1（启用） |
| SAGEATTN_INT8_WPE | int8 实验：waves_per_eu（1/2/4） | 1 |
| SAGEATTN_INT8_BN128 | D=128 int8 的 BN 覆盖（0 默认 64） | 0（=64） |
| SAGEATTN_INT8_BM128 | D=128 int8 强制 BM=128（实测 BM=64 更优） | -1（禁用，=64） |
| SAGEATTN_QUANT_GPB | quant 多 group 合并实验（实测无改善） | 1 |
| SAGEATTN_QUANT_BLK | quant block 粒度（1=auto：D64 用 128/64，D128 旧逻辑） | 1 |
| SAGEATTN_VT_GRID | v_transpose grid 覆盖（0=auto：tiles≤4096 用 192） | 0 |
| SAGEATTN_FP16_BM / _BN | fp16 direct 实验配置覆盖 | 0 / 0 |
| SAGEATTN_BF16_BM / _BN | bf16 direct 实验配置覆盖 | 0 / 0 |
| SAGEATTN_DIRECT_THRESHOLD_D64 | D=64 self 的 direct/int8 阈值 | 3072 |
| SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL | D=64 causal 的阈值 | 6144 |
| SAGEATTN_DIRECT_THRESHOLD_D64_CROSS | D=64 cross 的阈值 | 6144 |
| SAGEATTN_DIRECT_THRESHOLD_D128 | D=128 self/causal 的阈值 | 2048 |
| SAGEATTN_DIRECT_THRESHOLD_D128_CROSS | D=128 cross 的阈值 | 4096 |

`SAGEATTN_VT_GLOBAL` 已由 `setup.py` 编译宏固定为 1（V_T 方案默认启用），无需设置。**默认编译配置（无 env）即为各用例最优配置。**

---

## 六、编译

- `setup.py` 默认 `HIP_FLAGS` 含 `-DSAGEATTN_VT_GLOBAL=1`（V_T 方案）。
- 构建：`pip install -e . --no-build-isolation`（Windows 需 MSVC + ROCm SDK 环境）。
- 若需回退旧路径（无 V_T），改 setup.py 宏为 0 并关闭 core.py 转置。
- **hipcc target 必须显式指定**：`--offload-arch=gfx1103`，否则 "device kernel image is invalid"（所有 kernel 静默失败输出 0）——独立 HIP 测试时尤其注意。

---

## 七、易踩坑点（后续开发必读）

1. **测量必须同进程旋转轮转 + ≥600 次预热**；跨进程绝对时间不可比（见 §四）。顺序测量的差异（如曾出现的 0.99ms 假差异）多为热漂移，轮转后方可证实。
2. **layout_code=0 是 NHD**；手动复刻 ops 调用传错 layout 会产生"错误但快"的假象。
3. **native 正确性验证必须设 `SAGEATTN_BACKEND=native`**（默认是 triton）。
4. **hipcc 对模板 `__device__` 函数体内的 `#if`/`if constexpr` 有解析 bug**：即使逻辑为空也会报虚假的 `undeclared`。解法：用运行时 `if (编译宏)`（宏展开为常量，死分支被 DCE）；`#if` 只用于 host 函数与 namespace 顶层。
5. **WMMA operand 顺序决定 C 布局**：`wmma(A=V^T, B=P^T)` 才得到 out^T（匹配写回）；交换成 `out = P @ V` 会转置输出。定位方法：corr(o,ref)≈0.005 但 corr(sorted)≈0.9997。
6. **vt_off 必须含 kb_base**：漏掉会每迭代都读 V_T 前 32 个 n。定位方法：V[n][d]=n 时 o≈均匀和，V[n][d]=(n≥512) 时 o=0。
7. **iGPU 频率抖动 vs 顺序效应**：A/B 对比必须轮转，否则结论可被污染。
8. **V_T 的 n 维必须 padding 到 64 倍数**（三处规则一致）；v_transpose 的 ntiles 必须按 `v_t_n/32` 计算，否则 padding 区未写 0，kv mod 64 ∈ [1,32] 时仍 NaN。
9. **LDS 用奇数 pad 消 bank 冲突时，行内偏移奇偶交替，8B/4B 向量访问必然未对齐**：要么 pad 取 4 的倍数（重做 bank 分析），要么全部用 16-bit 标量访问。
10. **uint4/uint2 的字节数（16B/8B）与 half 数量（8/4）易混淆**：v_transpose 初版用 uint4 处理 4 half → 越界读+覆盖邻数据（cos 骤降至 0.92）。
11. **16B 向量写回要求 o_off 8-half 对齐**：q 的 stride_b/n/h 均需为 8 倍数（core.py 有断言）；非 contiguous 输入会触发未对齐写（UB）。
12. **"性能提升"必须先验证正确性再谈性能**：早期部分"性能提升"来自错误输出（layout 错或只处理前 32 key）。
13. **LDS 数组大小必须匹配实际布局**：V_T tile LDS 缓存（§3.8）初版沿用 `BLOCK_N*VStride` 而 `[D][N]` 布局需 `HeadDim` 行 → 越界写污染相邻 LDS，短序列恰好无感、4096+ 输出错误。定位方法：先小序列验证再逐步放大；LDS 尺寸用三元表达式按布局区分。
14. **VT_GLOBAL 相关的死代码必须有编译期守卫**：direct kernel 的 `v_prefetch`（C8 已修复，两处缺 `if (!SAGEATTN_VT_GLOBAL)` 守卫，`VT_GLOBAL=1` 时数组缩为 1 元素但循环仍越界写）与 **V 初始加载循环**（E4 排查中发现同类越界，该守卫随实验回退，**尚未单独落地，属已知遗留问题**）。原代码"碰巧没崩"（LDS 布局恰好容忍），改动 v_tile 后暴露。后续若改动 direct kernel 的 LDS 声明，务必补齐守卫。
15. **flash_attn 布局**：`flash_attn_func` 官方 API 为 NHD 布局 `[B,S,H,D]`（benchmark_attn.py 传原始 q/k/v 即正确）；传 HND 会输出错误（cos≈0.005，执行快但结果错误——"错误但快"陷阱）。

---

## 八、失败的优化尝试（避免重复）

**规律：指令级微优化全部失败；有效的优化是结构性的。**（V_T 消除 LDS 列读、32 行共享 k_frag 提升 ILP、LDS 缓存省全局重读、bf16 融合省 kernel、差异化分发省辅助、V_T tile LDS 缓存消除 PV 冗余全局读、quant 大 block 减 block 数、mean/vt 读粒度提升与固定开销摊薄、smooth_k=False 省 mean kernel。）

| 优化 | 结果 |
| --- | --- |
| D=128 每 warp 32 行 kernel | VGPR 溢出（256 上限），慢 10–24x |
| MIN_BLK 调大（64/128，对齐 triton 粒度） | quant +37–45%（fmax 串行链）；attn 退化 4% |
| BN 扫描（D=64 int8 16/32/64/128） | 大 BN 有害；BN=16/32 最优（D=128 则 BN=64 最优） |
| wpe1/2/4（occupancy） | 全部无改善或更差——occupancy 非瓶颈 |
| v_transpose 大 tile（64n×64d/128n×32d） | 均更慢（LDS bank 冲突或调度） |
| v_transpose uint2 显式向量化 | 更慢（编译器已自动优化） |
| v_transpose 连续 chunk 分配 | 变差 -5~-40%（负载不均/动态边界），回退保持 stride 分配 |
| quant 双累加器 / 跳过 blockReduce | 无帮助（瓶颈非依赖链/归约） |
| D=128 quant 大 block（128/64 或 64/32） | +1~3%（32KB LDS 或额外 barrier）→ D=128 保持旧逻辑（已 100GB/s） |
| quant pass1/pass2 软件流水（预读） | 大 block 后每 thread 仅 2 次迭代且带宽已 87–95GB/s（近上限），无意义 |
| quant 多 group 合并（GPB=2/4/8） | 无改善 → quant 瓶颈非 block 调度/DRAM 行切换 |
| mean_seq 融合进 quant（跨 block 归约） | 无法省 K 读（K > L2 2MB），分别优化更优 |
| QK WMMA 依赖链拆分（2 累加器） | +3% |
| k_frag 提前加载到寄存器 | 0%（LDS 读被 WMMA 掩盖） |
| KStride+20（消 LDS bank 冲突） | +3% |
| softmax max/sum 树形归约 | +3% |
| K 直读 global（消 LDS+barrier） | +8%（gather 的 L2 利用率低） |
| QK 用 fp16 WMMA（i8 转 fp16） | +33%（转换开销超 WMMA 提速） |
| fast_exp2 位近似 / v_max3 / permlanex16→shfl / 双缓冲 | 全部失败 |
| BM=128/8 warps（VAE 超长，K/V 重读减半） | ±1% 无效 → 非 DRAM 带宽/重读受限 |
| BN=16/32/128（D=128 int8，VAE） | BN=16/32 慢 4–14%；BN=128 持平 → 保持 BN=64 |
| v_tile 缩为 1 元素（提升 occupancy） | 无性能变化 → occupancy 非瓶颈（但省 LDS 仍有益，§3.9） |
| triton PV_ACCUM fp16 / BLOCK_M=128（VAE） | 无收益 / autotune 未选中 |
| fp8e5m2 V_T 存储 + fp16 WMMA（FeatherOps 思路，见附录） | 整体无提升（13 用例仅 1 个快 7%，多数慢 0–24%） |
| bf16 V_T + bf16 WMMA（AnimaVAE 免转换） | 收益仅 ~0.03–0.05ms，低于测量噪音 |
| bf16 V_T 直通（存 bf16 免转换，attn 仍 fp16） | 同进程 A/B 仅 0.007–0.010ms（顺序测量 0.99ms 为热漂移假象） |
| v_transpose grid 微调（96/48 深度扫描） | 端到端差异在 ±7% 噪音内，无一致规律 → 保持 auto=192 |
| A1. v_transpose 与 attn 融合 | 不可行：设计1（每 q-block 转置全量 kv）转置重复 q_blocks 倍（SDXL10=24×，~3ms >> vt+attn 1.5ms）；设计2（split-KV）需二次 softmax 归约 + q 重复读，大工程收益未验证 |
| B4. CUDA Graph 多 kernel launch 优化 | 负收益：Anima01 replay 17.944 vs direct 17.588（+2%）、SDXL10 +12% → launch 间隙非瓶颈；"全流程 vs 分段差 2.4ms"实为测量方法差异（分段计时更热） |
| C5. causal 三角跳过 | 已达理论 87%：causal/noncausal 0.567（理论 0.5），block 级 kv_limit 已获主要收益；剩余 13% 需 warp 级差异化（受 barrier 同步限制），无 causal 用例可验证 |
| C6. GQA kv_heads 共享加载 | 无差距：GQA 长序列 n/t=1.0–1.03、短序列快 36%，kv 数据小 L2 命中掩盖重复读 |
| C8. direct kernel V_T tile LDS 缓存 | 负优化回退：D=64 direct（SDXL10/16/04）+24~30% 慢（短序列 kv-tile 少、冗余读多命中 L2，V 拷贝+barrier 纯开销）；排查中**修复 direct kernel v_prefetch 越界 UB** |
| E1. D=64 int8 self（32 行 kernel）PV LDS 缓存 | 负优化 +3~+14%（SDXL01 4.71→5.38ms）——D=64 冗余读命中 L2、4 warp 每迭代 1 次 barrier 开销大；与 C8 共同确认 LDS PV 缓存仅对 D=128 超长 VAE 有效 |
| E2. mean∥v_transpose multi-stream 并行 | 负优化 +2~+8%（12 CU iGPU 并发争抢 CU/带宽，串行反而各打满带宽） |
| E3. vt 带宽再提升 / V_T tile 双缓冲 / QK k_frag 双缓冲 / 多累加器 ILP | 分析不实施：vt 已 34.6GB/s 近转置上限、VAE 瓶颈为 WMMA 吞吐（非 barrier）、双缓冲历史记录失败 → 预期收益 <1% |
| E4. direct 路径消除 v_transpose（LDS 列读原始 V） | 负优化回退：direct self +27~+33%（SDXL10 1.64→2.16ms）、D=128 cross +31~+41%，仅 kv=77 短 cross 快 3–6%（噪音边界）——LDS 列读 PV 的 128 次 u16 标量读+bank 冲突+V 拷贝 barrier 远超省下的 vt 成本；印证 §3.2「V_T 行读是 direct 刚需」 |
| direct self 短序列 BM/BN 扫描（BM=128/32 等） | 首轮个别 -7.5% 为热漂移噪音，严格复测证实默认 BM=64/BN=64 最优 |

**前期错误结论的更正**（作为踩坑保留）：
- 曾认为 `v_permlanex16_b32` 在 gfx1103 不可用（driver bug），实为参数格式错误。
- 曾认为 report 中"全流程 vs 分段计时差 2.4ms"是 launch 间隙开销，实为测量方法差异（分段计时更热）。
- 早期部分"性能提升"结论来自错误输出（见踩坑点 12）。

---

## 九、性能差距分析与剩余可优化方向

### 9.1 与 triton / flash-attn 的三方对比

**flash_attn 说明**：按 NHD 布局调用后 FA 在本环境可用（布局陷阱见踩坑点 15）。

同进程三方轮转（native / triton / flash-attn，warmup 150/rounds 5/iters 15）：

| case | native | triton | FA | n/t | n/fa | t/fa |
| --- | --- | --- | --- | --- | --- | --- |
| SDXL01（D64 int8 self 4096） | 4.713 | 4.707 | 4.931 | 1.001 | 0.956 | 0.955 |
| SDXL07（D64 int8 self 6144） | 9.974 | 9.792 | 9.906 | 1.019 | 1.007 | 0.989 |
| SDXL13（D64 int8 self 9216） | 22.543 | 21.984 | 22.484 | 1.025 | 1.003 | 0.978 |
| SDXL10（D64 direct self 1536） | 1.622 | 1.732 | 2.339 | 0.936 | 0.693 | 0.740 |
| SDXL16（D64 direct self 2304） | 3.266 | 3.317 | 3.409 | 0.985 | 0.958 | 0.973 |
| Anima01（D128 int8 self 4096） | 17.731 | 16.818 | 17.741 | 1.054 | 0.999 | 0.948 |
| Anima05（D128 int8 self 9216） | 79.974 | 80.247 | 83.758 | 0.997 | 0.955 | 0.958 |
| SDXLVAE01（D128 int8 16384） | 60.887 | 62.241 | 61.816 | 0.978 | 0.985 | 1.007 |
| AnimaVAE01（D128 int8 bf16 16384） | 44.474 | 44.358 | 45.431 | 1.003 | 0.979 | 0.976 |
| Anima02（D128 direct cross 512） | 2.640 | 3.056 | 3.543 | 0.864 | 0.745 | 0.863 |
| SDXL02（D64 direct cross 77） | 0.225 | 0.293 | 0.513 | 0.767 | 0.438 | 0.571 |

**结论**：
- **native vs FA**：全面持平或快（n/fa 0.44–1.0）；短 kv 场景优势最大（SDXL02 快 56%、SDXL10 快 31%、Anima02 快 25%）——FA 短 kv kernel 低效。
- **native vs triton**：D64 int8 持平，Anima01 慢 5.4%，cross 快 15–25%。
- **FA vs triton**：FA 通常慢 2–5%，短 kv 慢 26–43%。

### 9.2 全流程分解与差距归因

全流程：native = v_transpose + quant + attn kernel + 写回（smooth_k=False 后无 mean）；triton = torch mean + quant + attn kernel，无 v_transpose；FA = 仅 attn kernel，无任何辅助。

**① attn 主 kernel（计算层面）——已持平/反超**：i8 WMMA = fp16 2x 吞吐；D=128 int8 峰值 ~10.9 TFLOP（WMMA 硬件上限）；SDXL10 attn 1.39ms 比 triton 总时长还快。native 与 triton 的剩余差异已不在计算层面；瓶颈是 WMMA 吞吐 + L2 带宽（VAE 场景 LDS PV 已解决）。

**② v_transpose（native 独有，结构性成本）**：triton/FA 直接读 V 原布局，无转置；native 需一次性转置（成本 0.1–0.5ms，转置类读写混合 ~60GB/s 已接近带宽上限，为 torch transpose 的 2–3 倍）。影响最大的 case：SDXL10/16（attn 已持平但 vt 拖累）、所有 bf16 输入（Anima01 vt 0.49ms）。

**③ 辅助 kernel 累计**：smooth_k=False 后，native 2 个 kernel（vt/quant）vs triton 2 个（torch mean/quant）vs FA 0 个。关键差异：native 的 vt 比 triton 的 mean 更贵，故相对 triton 仍吃亏；相对 FA 多 vt+quant 两个 kernel（Anima01 ~1.17ms）。quant 大 block 后带宽已不落后，差距是 kernel 数量与 vt 固有成本。kernel launch 间隙经 CUDA Graph 探针证实非瓶颈。

**④ 写回与分发**：写回 16B 向量化与 triton 相当；direct/int8 阈值按 NHD 标定，causal/cross 短 q 走 direct 是正确选择。

### 9.3 剩余可优化方向

历史方向（A~E 与方向 1~3）已全部实验处理，结论归并至对应章节：[✓ 已实施]（HND 阈值 → §5.1；smooth_k=False → §3.12）；[✗ 负优化/不实施]（见 §八）。

**当前差距画像**（smooth_k=False 后）：
- direct self（SDXL10/16）：慢 FA 7–11%——vt 固有成本，已证伪消除手段（E4）。
- D=64 int8 self（SDXL07/13）：慢 FA 3–5%，主因 vt+quant。
- D=128 int8 self（Anima01）：慢 FA 2–3%，主因 vt+quant。
- 其余（cross / VAE / 长 int8 self）：已持平或反超。

**方向 1：int8 路径的 vt+quant 固有成本（已近极限）**：int8 的 i8 WMMA 2x 吞吐优势（长 self-attn 主导收益）必须付出 quant + V_T 的代价，二者不可兼得；该差距达本硬件（无 MFMA、WMMA 吞吐上限）的极限，非可继续压榨的方向。

**方向 2：短 cross（kv=77/154）固定开销（低优先级）**：两后端均慢 FA 15–35%（绝对差仅 0.03–0.12ms），为两后端共有的短序列固定开销，非 native 特有。可评估 direct 路径多 kernel launch 合并、或对 kv≤154 的特化 kernel；优先级低（绝对差小）。

**方向 3：VAE 03/04（36864/55296）实测验证**：iGPU 长时间运行死机已从 `benchmark_attn.py` 注释，当前结论由 01/02 外推；LDS PV 收益在更大序列应更显著，若有散热条件可实测确认外推。

---

## 附录：fp8e5m2 存储方案实验总结（ComfyUI-FeatherOps 思路移植）

**动机**：`ComfyUI-FeatherOps` 用 fp8e5m2 存 matmul 的 B 矩阵，加载时 2 条 V_PERM_B32 快速 upcast 到 fp16（两者指数 bias 相同=15，只需尾数零扩展），省加载带宽/LDS 占用。在 Strix Halo 上 fp16@fp8e5m2 matmul 达 43 vs 36 TFLOPS（+20%）。RDNA3 无原生 fp8 WMMA，故 fp8 只做存储格式，计算仍用 fp16 WMMA。

**精度模拟结论**（决策依据）：
- QK 用 fp8e5m2：max_err 为 int8 的 4–5 倍，❌ 不可行（softmax 输入，指数放大；印证 SageAttention 用 int8 而非 fp8 做 QK 的原因）。
- PV 的 V 用 fp8e5m2（per-token scale）：cos≈0.9987，✅ 唯一可行方案。

**移植方案**：V_T 以 fp8e5m2 存（v_transpose 融合量化，per-token scale），PV 的 v_frag 读 16B + perm upcast，P 列乘 scale（数学等价 `out=(P·scale)@V_fp8`）。direct 与 int8 路径共用。

**关键技术点**：
- `perm(0u, p, 0x010c000c)` → fp16 对零扩展（与 FeatherOps 语义一致，独立 HIP 测试验证）。
- fp16→fp8e5m2 截断法：`fp8 bits = fp16 bits >> 8`；需 clamp：fp16 值 ≥57344（E=31）会进入 fp8 的 inf/NaN 编码区，否则 `inf×0=P 列` 产生 NaN。
- 运行时分支 `if (v_scale != nullptr)` 双路径，避免新增模板参数触发 hipcc 模板 `#if` bug。

**正确性**：fp8 路径 cos≈0.998（pytest 28/36 过，8 个失败为 mae 0.36–0.51 超 0.35 阈值，2bit 尾数固有误差）；fp16 路径零破坏（36/36 过）。

**性能**（同进程轮转，13 用例）：仅 SDXL10（direct self 1536）快 6.7%；VAE 慢 24%、Anima02 慢 32%、Anima01 慢 13%、短 kv cross 慢 8–19%。**整体无提升。**

**失败根因（与 FeatherOps matmul 的本质差异）**：

| 维度 | FeatherOps matmul | attention QK/PV |
| --- | --- | --- |
| K 长度 | 4096（大矩阵） | 64/128（head_dim） |
| 主导 | 加载带宽（K-loop） | WMMA 计算（短 K） |
| fp8 upcast 开销占比 | 小（相对加载） | 大（相对 WMMA） |
| 结论 | +20% | -5% ~ -24%（多数用例） |

FeatherOps 收益前提是长 K-loop，加载指令占比大；attention 的 QK/PV K 维只有 head_dim，WMMA 计算主导，v_frag 加载（32B→16B）占比小。fp8 引入的 perm upcast（8 条/v16h）+ P×scale（8 float mul）+ 额外 v_scale kernel 无法被省下的加载带宽覆盖。LDS 中转路径（int8 D=128/VAE）更差：V_T tile 每迭代从 L2 读（2MB L2 覆盖，带宽本不稀缺），perm 是纯开销。唯一正收益用例的 L2 稀缺带宽场景收益不稳定（同形状 2304 反而慢 4.6%）。

**状态**：代码已回退（性能无改善，仅保留文档记录）。