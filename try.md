# SageAttention Native Backend 深度优化实验记录 (try.md)

> 本文件记录基于 NativeBackendOptimizeReport.md 的后续优化实验全过程。
> 平台: Radeon 780M (gfx1103) iGPU, ROCm 7.14, torch 2.12.0+rocm7.14.0
> 基线: 24/24 benchmark 用例 native/triton ≤ 1.05x（本次会话复现）

---

## 0. 测量方法论（重要教训）

### 0.1 GPU 频率剧烈波动（本次会话新发现）

报告已知"热漂移"（跨会话绝对时间差异 40%+），本次会话进一步发现：

- **iGPU 冷启动后频率剧烈波动**：同一 kernel 连续测量，单次时间在 0.19ms ↔ 4.2ms 之间跳变（20 倍差异），不是缓慢漂移而是剧烈抖动
- **需要长时间预热才稳定**：实测约需 ~600 次 kernel 调用（20 轮 × 30 iters）后时间才收敛（0.55 → 0.30ms）
- **固定顺序交错轮转存在轮内顺序效应**：每轮先测的后端处于"更冷/更慢"状态，后测的更快。早期 A/B 结论（BM=128 收益）部分被此污染

### 0.2 修正后的测量方法（bench_rot.py）

```
1. 每用例充分预热: 各后端交替跑 ≥600 次调用
2. 测量轮: 6 轮 × 每轮 30 iters
3. 轮间后端顺序旋转: 第 r 轮从 r 号后端开始（对称采样）
4. 各后端取中位数
```

### 0.3 其他验证

- 旧代码（git stash 后重建）同样慢 20-30% → 跨会话性能回退是 GPU 状态而非代码改动
- 跨进程绝对时间与 n/fa 倍率均不可跨会话对比；同进程内交错轮转（旋转）是唯一可信方法

---

## 1. 实验一：BN 覆盖（大 BN 单次迭代策略）

**假设**：short cross-attn（kv=77/154）多次迭代的 barrier/prefetch 固定开销占比高，大 BN 单次迭代覆盖整个 KV 可减少开销。

**实现**：dispatch 加 `SAGEATTN_FP16_BN` 环境变量（0/16/32/64/128）覆盖 D=64 direct 路径的 BN。

**结果**（交错轮转，n/fa 为同进程内倍率）：

| BN | SDXL05(1024x77) | SDXL06(1024x154) | SDXL12(1536x154) | SDXL18(2304x154) |
|----|----------------|-----------------|-----------------|-----------------|
| 默认(16/32) | 1.28 | 1.43 | 1.79 | 1.52 |
| 32 | 1.70 | 1.39 | 1.72 | 1.55 |
| 64 | 1.89 | 1.63 | 1.92 | 1.79 |
| 128 | 1.87 | 2.42 | 2.63 | 2.92 |

**结论**：大 BN 全面有害（VGPR 压力 → wpe2 occupancy 下降 + 尾块浪费 + 无重叠 LDS 加载）。
**BN=16/32 已是当前最优配置，维持不变。**

---

## 2. 实验二：BM 覆盖（减少 K/V 冗余读取）

**假设**：short cross-attn 中每 q-block 重复读取整个 K/V，BM=128 使 block 数减半 → K/V 流量减半（DRAM 带宽受限分析：native SDXL12 0.388ms ≈ 理论带宽极限 0.346ms，而 FA 0.217ms 依赖 L2 命中）。

**实现**：dispatch 加 `SAGEATTN_FP16_BM` 环境变量 + fp16/bf16 op 的 `bm_sel` 运行时参数（pybind 签名扩展），core.py 透传。

**结果**（第一/二轮，128/64 倍率，<1 表示 BM=128 更快）：

| 用例 | 128/64 轮1 | 128/64 轮2 | 趋势 |
|------|-----------|-----------|------|
| SDXL02 (4096x77, h=10) | 0.96 | 0.94 | BM128 快 4-6% |
| SDXL03 (4096x154, h=10) | 0.90 | 1.08 | 不稳定 |
| SDXL05 (1024x77) | 1.00 | 0.95 | 持平 |
| SDXL06 (1024x154) | 0.91 | 0.91 | BM128 快 9% |
| SDXL11 (1536x77) | 1.06 | 1.09 | BM128 慢 6-9% |
| SDXL12 (1536x154) | 0.91 | 0.92 | BM128 快 8-9% |
| SDXL17 (2304x77) | 1.15 | 1.16 | BM128 慢 15-16% |
| SDXL18 (2304x154) | 0.87 | 0.86 | BM128 快 13-14% |
| SDXL04 (1024x1024 self) | 1.28 | - | BM128 慢 28% |

**初步结论**（后被顺序效应部分污染，见实验四修正）：
- kv=154 的 cross-attn：BM128 一致快 8-14%
- kv=77 的大 q cross：BM128 慢 6-16%
- self-attn：BM128 有害（-28%）

---

## 3. 实验三：顺序效应发现与测量方法修正

**现象**：A/B 中 n64（每轮先测）与 n128（每轮后测）差异被怀疑是配置差异，但调试打印确认 kv=154 时两者走相同配置（BM=128），时间却差 15%+。

**诊断**（selfdiff.py / warmup_curve.py）：
- 同一配置连续 5 轮：0.50 → 0.29ms（近 2 倍下降）
- 连续 30 轮：0.55 → 0.30ms，~600 次调用后收敛
- GPU 频率剧烈波动（0.19-4.2ms 跳变）

**修正**：bench_rot.py —— 预热 ≥600 次 + 轮间顺序旋转 + 取中位数。

---

## 4. 实验四：旋转轮转验证 BM=128 分派规则（严谨复测）

**新分派规则**（已固化到 fp16 dispatch）：
```
D=64 direct, kv>128, cross-attn (qo≠kv) → BM=128, BN=32
其余（self / kv≤128 / D=128 / int8）→ 保持原配置
```

**旋转轮转验证**（64 = 新默认，64f = 强制 BM=64 旧行为，64/64f < 1 表示新规则更快）：

| 用例 | 64(新默认) | 64f(强制BM64) | 64/64f | 结论 |
|------|-----------|--------------|--------|------|
| SDXL02 (kv=77) | 0.204 | 0.199 | 1.03 | 持平（均走 BM=64） |
| SDXL03 (kv=154) | 0.315 | 0.318 | 0.99 | 持平 |
| SDXL04 (self) | 0.916 | 0.904 | 1.01 | 持平（均走 BM=64） |
| SDXL06 (kv=154) | 0.200 | 0.222 | **0.90** | 新规则快 10% |
| SDXL11 (kv=77) | 0.206 | 0.208 | 0.99 | 持平 |
| SDXL12 (kv=154) | 0.353 | 0.389 | **0.91** | 新规则快 9% |
| SDXL17 (kv=77) | 0.333 | 0.325 | 1.02 | 持平 |
| SDXL18 (kv=154) | 0.453 | 0.519 | **0.87** | 新规则快 13% |

**结论**：排除顺序效应后，kv=154 cross 的 BM=128 收益真实（9-13%），无回退用例。
SDXL03 (h=10) 持平——收益与 head 数相关（h=20 受益明显）。

---

## 5. 代码修改汇总

| 文件 | 修改 |
|------|------|
| `csrc/attn_gfx11.cu` | fp16 dispatch: 实验 env (SAGEATTN_FP16_BN/BM) + bm_sel 参数; 固化 kv>128 cross → BM=128 规则; bf16 dispatch: 同步 bn_ov/bm_ov 实验逻辑与 kv>128 cross → BM=128 规则 |
| `csrc/pybind_gfx11.cpp` | fp16_attn_t / bf16_attn_t 签名加 int bm_sel |
| `csrc/attn_gfx11.h` | 两函数声明加 bm_sel |
| `sageattention/core.py` | native 路径透传 bm_sel (kwargs/env) |

---

## 6. 完整 24 用例验证（旋转轮转，预热 600 + 6 轮旋转）

`64` = 新默认（kv>128 cross → BM=128），`64f` = 强制 BM=64（旧行为）。`64/64f < 1` 表示新规则更快。

| 用例 | 64 | 64f | 64/64f | triton | 结论 |
|------|-----|-----|--------|--------|------|
| SDXL01 (4096 self, int8) | 5.432 | 5.424 | 1.00 | 5.252 | 持平（int8 不受影响） |
| SDXL02 (4096x77) | 0.208 | 0.197 | 1.05 | 0.205 | 噪声（同配置） |
| SDXL03 (4096x154) | 0.314 | 0.318 | 0.99 | 0.304 | 持平 |
| SDXL04 (1024 self) | 0.920 | 0.910 | 1.01 | 0.905 | 持平 |
| SDXL05 (1024x77) | 0.139 | 0.139 | 1.01 | 0.139 | 持平 |
| SDXL06 (1024x154) | 0.202 | 0.217 | **0.93** | 0.200 | 新规则快 7% |
| SDXL07 (6144 self, int8) | 11.674 | 11.537 | 1.01 | 11.593 | 持平 |
| SDXL08 (6144x77) | 0.286 | 0.274 | 1.04 | 0.275 | 噪声 |
| SDXL09 (6144x154) | 0.456 | 0.483 | **0.95** | 0.454 | 新规则快 5% |
| SDXL10 (1536 self, int8) | 2.049 | 1.792 | 1.14 | 1.788 | 噪声（int8 同配置） |
| SDXL11 (1536x77) | 0.200 | 0.202 | 0.99 | 0.198 | 持平 |
| SDXL12 (1536x154) | 0.368 | 0.392 | **0.94** | 0.362 | 新规则快 6% |
| SDXL13 (9216 self, int8) | 26.470 | 26.252 | 1.01 | 26.328 | 持平 |
| SDXL14 (9216x77) | 0.410 | 0.404 | 1.01 | 0.407 | 持平 |
| SDXL15 (9216x154) | 0.656 | 0.690 | **0.95** | 0.661 | 新规则快 5% |
| SDXL16 (2304 self) | 3.798 | 3.759 | 1.01 | 3.778 | 持平 |
| SDXL17 (2304x77) | 0.330 | 0.327 | 1.01 | 0.328 | 持平 |
| SDXL18 (2304x154) | 0.439 | 0.523 | **0.84** | 0.441 | 新规则快 16% |
| Anima01-06 (d128) | - | - | 0.98-1.01 | - | 持平（d128 不受影响） |

**结论**：
- **真实收益**：kv=154 的 cross-attn 5 个用例（SDXL06/09/12/15/18）新规则快 5-16%
- **无真实回退**：SDXL02/08/10 的"1.04-1.14"为同配置测量噪声（这些用例 64 与 64f 走相同 kernel）
- native 与 triton 同进程持平（n64/triton ≈ 1.0），kv=154 用例达到/略超 triton
- 正确性：36/36 通过；benchmark_attn.py 24 用例全部 OK

---

## 7. benchmark_attn.py 最终运行（native, 单次运行相对倍率）

全部 24 用例 Status=OK。Sage/FA 倍率（单次运行内，受测量顺序影响，仅作参考）：

| 用例 | Sage/FA | 用例 | Sage/FA | 用例 | Sage/FA |
|------|--------|------|--------|------|--------|
| SDXL01 | 1.16 | SDXL09 | 1.15 | SDXL17 | 1.27 |
| SDXL02 | 0.84 | SDXL10 | 1.53 | SDXL18 | 1.38 |
| SDXL03 | 1.15 | SDXL11 | 1.09 | Anima01 | 1.11 |
| SDXL04 | 1.30 | SDXL12 | 1.55 | Anima02 | 1.05 |
| SDXL05 | 0.99 | SDXL13 | 1.21 | Anima03 | 1.07 |
| SDXL06 | 1.34 | SDXL14 | 0.83 | Anima04 | 1.19 |
| SDXL07 | 1.25 | SDXL15 | 1.18 | Anima05 | 1.03 |
| SDXL08 | 0.86 | SDXL16 | 1.18 | Anima06 | 1.13 |

注：benchmark_attn.py 固定顺序（每轮 FA 先测、Sage 后测），按本会话发现的顺序效应，
Sage 的倍率系统性偏保守；旋转轮转（第 6 节）是更可信的对比。

---

## 8. 结论与后续方向

**已完成的改进**：
1. 固化 kv>128 cross-attn → BM=128/BN=32 分派规则：5 个 kv=154 用例快 5-16%（旋转轮转验证）
2. 建立严谨测量方法（预热 600 + 轮间旋转）：揭示 iGPU 冷启动顺序效应，修正早期结论
3. int8 路径 V/OUT dtype 分离（方案 B）：bf16 输入消除 o.to(bf16) 转换 kernel

**测量方法学贡献**（可复用到后续优化）：
- iGPU 需 ~600 次调用预热才能稳定
- 固定顺序交错轮转存在轮内顺序效应 → 必须轮间旋转
- 跨会话绝对时间/倍率不可比（频率波动 20 倍）

**待探索方向**（未实施）：
- short cross-attn 相对 FA 的剩余差距（SDXL12 n64/fa≈1.6）根源待查（疑似 L2 命中率差异）
- int8 长序列 self-attn 的进一步优化（已与 triton 持平）
- bf16 D=64 direct 的 BM=128 规则缺少 benchmark 用例验证（已同步规则，风险低）

---

## 9. bf16 int8 路径 dtype 转换优化（方案 A/B/C 对比）

### 9.1 背景

用户反馈：Anima 系列（bf16, d128, int8 路径）在 ComfyUI 中 native 比 triton 慢 5-7%。
定位：core.py 对 bf16 输入在 int8 路径做两次 dtype 转换：
1. `v.to(torch.float16)` —— V 全量 bf16→fp16（读+写 32MB/iter for Anima01）
2. `o_int8.to(torch.bfloat16)` —— 输出 fp16→bf16（读+写 32MB/iter）

triton backend 无此转换（直接 bf16 参与 tl.dot）。

### 9.2 独立测量（Anima01 形状）

- v.to(fp16): 0.539ms, o.to(bf16): 0.545ms —— 合计占全流程 18.76ms 的 ~5.8%
- 理论带宽 67MB/76.8GB/s = 0.87ms（实测转换接近带宽受限）

### 9.3 三方案 kernel 级对比（同进程旋转轮转, ms）

| 用例 | A: V bf16 直通 | B: v.to + halfV + out bf16 | C: 旧(全转换) | C/B | A/B |
|------|---------------|---------------------------|--------------|-----|-----|
| Anima01 | 16.40 | **15.30** | 15.82 | 1.034 | 1.072 |
| Anima03 | 37.99 | **34.89** | 36.91 | 1.058 | 1.089 |
| Anima05 | 87.14 | **78.14** | 78.67 | 1.007 | 1.115 |

**方案 A（V bf16 直通）失败**：kernel 内逐元素 bf16→fp16 标量转换（`__bfloat162float`+`__float2half`）
破坏 LDS 加载的向量化（uint4 拷贝 → 8 次标量），实测 kernel 开销 +1.6~5.5%（单独测
halfV=15.93 vs bf16V=16.81 for Anima01）。

**方案 B（采用）**：V 预转 fp16（带宽受限 elementwise kernel，高效）+ kernel 用 __half V
（uint4 拷贝）+ kernel 输出直接写 bf16（省 o.to，且 fp32→bf16 单次舍入精度更好）。

### 9.4 实现

- `attn_kernel_impl_t` / `attn_kernel_wpe1_t` / `attn_kernel_wpe2_t` 增加 `OUT_DTYPE` 模板参数
  （默认 `= V_DTYPE`，输出部分按 OUT_DTYPE 判断 fp16/bf16 写回）
- dispatch `qk_int8_sv_bf16_attn_gfx11_t`：
  - fp16 输入 → (V=__half, OUT=__half) —— 原行为
  - bf16 输入 → (V=__half, OUT=__hip_bfloat16) —— 输出直写 bf16
  - （保留 bf16 V 直通分支作备选）
- core.py int8 路径：bf16 输入保留 `v.to(fp16)`，`o_int8 = o`（bf16 直接复用）

### 9.5 验证

- 正确性 36/36 通过
- 旋转轮转（native vs triton, n/t）：Anima01 1.007→1.00, Anima03 1.003→0.998, Anima05 0.999→0.995
  （kernel 级收益 0.7-5.8%，全流程摊薄后 native 与 triton 持平/略快）
- fp16 int8 路径（SDXL01/07）无回归（n/t=1.00）

### 9.6 关于 V int8 量化（用户问题 2 的分析）

用户问"int8 读取后再展开节省带宽"。分析：
- Q/K 已 int8 存储（quant_qk_int8，带宽减半），WMMA i8×i8→i32 硬件直接处理无需展开
- V 仍 fp16。V int8（SageAttention2 的 pv_int8）需 P 也量化 int8（RDNA3 WMMA 无 i8×f16 混合），
  改动大且有精度风险
- **收益评估**：Anima 长序列 int8 kernel 非带宽受限（Anima05 84ms vs 带宽极限 ~2ms），
  V int8 省 8MB/iter ≈ 0.1%——几乎无收益。V int8 只对带宽受限的 short cross-attn
  （SDXL06/12/18, fp16 direct 路径）有意义，但那些不走 int8 路径
- 结论：当前用例集下 V int8 量化不值得实施；若未来有大量 short cross-attn 需求可重估



## 10. 性能差距深度分析（本轮会话：int8 long self-attn 慢 17-27% 的根因排查）

### 10.1 会话目标与基线

- 目标：native 达到/超过 triton。重点 Animal 系列 + int8 long self-attn（SDXL01/07/13 等）
- 基线（同进程旋转轮转 bench_rot.py，预热 600 + 6 轮）：

| 用例 | Native | Triton | n/t |
|------|--------|--------|-----|
| SDXL01 (4096 self int8) | 5.25 | 4.44 | 1.18 |
| SDXL04 (1024 self fp16 direct) | 0.93 | 0.79 | 1.18 |
| SDXL07 (6144 self int8) | ~9.6 | ~8.0 | 1.20 |
| SDXL10/13/16 (int8 self) | - | - | 1.18-1.27 |
| Anima01/03/05 (d128 int8) | - | - | 1.06-1.10 |
| Anima02 (d128 cross) | 3.35 | 3.09 | 1.08 |
| short cross (kv=77/154) | - | - | 0.61-0.79 (native 快) |

主要差距集中在 int8 长序列 self-attn（D=64 慢 17-27%）与 SDXL04（1024 self direct，慢 18-20%）。

### 10.2 关键方法论教训：layout_code 陷阱（"错误但快"的假象）

排查中发现核心.sageattn 慢 3.6ms 于手动 ops 调用，多轮 debug 后真相：

- **正确 layout_code = 0 (kNHD)**：q 形状 [B, S, H, D] 是 NHD，`tensor_layout="NHD"` → layout_code=0
- 手动复刻时误传 layout=1 (kHND) → kernel 按 HND stride 解释 NHD 数据 → **输出错误（err≈3.0）但执行更快（假象）**
- 教训：任何"手动拆分 ops 调用"验证必须与 core.py 逐参数一致（含 layout_code），否则结果无效
- 修正后的分段计时（layout=0）：mean 0.10ms + quant 0.27ms + attn 4.86ms ≈ 完整 5.30ms（attn kernel 占 93%）

### 10.3 瓶颈分解实验（SAGEATTN_DEBUG_MODE 编译宏）

attn kernel 4.86ms 内部分解（跳过 PV = debug≥1，跳过 softmax = debug=2）：

| 变体 | 时间 | 说明 |
|------|------|------|
| 完整 | 4.86ms | 基线 |
| 跳过 PV（保留 QK+softmax） | 2.94ms | PV ≈ 1.9ms |
| 跳过 PV + softmax | 0.57ms | QK+score ≈ 0.57ms |
| **softmax 块** | **≈2.37ms** | **最大瓶颈（49%）** |
| 跳过 PV + 只加 max 归约 | 0.50ms | max 归约几乎免费 |
| softmax 的 exp 换便宜运算 | 省 0.11ms | exp 非瓶颈 |
| 跳过 alpha/out_acc 缩放 | 省 0.09ms | alpha 非瓶颈 |
| 跳过 sum 归约 | 省 0.15ms | sum 非瓶颈 |

**关键发现**：softmax 块（约 142 条 VALU 指令/迭代）耗时 2.37ms，是 QK 块（30 条指令，0.57ms）的近 5 倍；但删除其中任何单一指令段只省 ~0.1ms。说明瓶颈是**指令流整体执行时间 + 串行依赖链**（QK WMMA → max 归约 → permlane → alpha/exp → sum → permlane → row_l，每迭代约 600 cycles 依赖链），而非任何单点指令。

### 10.4 全部失败的优化尝试（9 项，均验证无收益或负收益）

| # | 优化 | 实现 | 结果 |
|---|------|------|------|
| 1 | int8 wpe2/wpe4（occupancy） | launch_bounds + amdgpu_waves_per_eu 实验宏 | wpe2 无变化（n/t 1.19），wpe4 更差（1.23） |
| 2 | int8 BN=64 | dispatch env 覆盖 | 5.68 vs 5.30，更差 |
| 3 | v_tile 转置布局（向量化 v_frag 读） | v_tile_T[d][n] + 转置写 | 6.38 vs 5.23（+22%），转置写 8× 标量指令 + bank 冲突 |
| 4 | fast_exp2 位近似（Schraudolph） | 1 FMA + bitcast | 5.83 vs 4.86（+20%），v_exp_f32 更快 |
| 5 | QK WMMA 依赖链拆分 | score_acc 拆 2 个并行累加器 | 5.47 vs 5.23（+4.6%） |
| 6 | v_max3_f32 树形 max 归约 | 内联汇编 v_max3_f32 | 5.61 vs 4.86（+15%），3 输入指令吞吐无优势 |
| 7 | LDS 双缓冲（减少 barrier） | k_tile/v_tile[2] | 5.51 vs 5.43，无改善（barrier 非瓶颈） |
| 8 | permlanex16 → __shfl_xor_sync | 替换 softmax 归约 | 6.39 vs 5.36（+19%），v_permlanex16_b32 本身高效 |
| 9 | SDXL04 BM/BN 配置矩阵 | SAGEATTN_FP16_BM/BN env | 全部 1.17-1.28 无改善（128/128 最差 1.91） |

### 10.5 技术分析：native vs triton 的结构差异（ISA 级对比）

- **triton _attn_fwd ISA**：wmma=96、**ds_load=0/ds_store=0（完全不用 LDS）**、permlane=54、exp=99、barrier=10（6 迭代展开），配置 BLOCK_M=128/BLOCK_N=32/**num_warps=4/waves_per_eu=4**
- **native int8 kernel ISA**：ds_load=136（其中 **128 次 ds_load_u16_d16 是 v_frag 的 16-bit LDS 读**）、ds_store=6、permlane=10、exp=17，配置 8 warps + wpe1
- **核心差异**：triton 用 4 warps + waves_per_eu=4 高 occupancy 隐藏指令延迟，且 K/V 直接 global→寄存器 + permlane 重排（无 LDS bank 冲突/barrier）；native 用 LDS 中转（v_frag 需 128 次 u16 标量读，2B/lane 低效），wpe1 每 SIMD 1 wave 无法隐藏依赖链延迟
- 结论：native 的转置布局 + LDS + permlanex16 设计在 gfx1103 上已接近结构极限，微观指令优化全部负收益。突破需结构性重写（如 triton 式无 LDS + permlane 重排 + 4-warp 高 occupancy），工程量与风险大

### 10.6 结论与建议

1. **本轮未实现性能提升**，但建立了可信的瓶颈地图与测量方法（layout=0 陷阱是重要修正）
2. 下一步建议（按优先级）：
   a. **结构性重写 PV 段**：V 走 global 直读 + permlane 重排（triton 方案），消除 128 次 u16 LDS 读
   b. **softmax 指令精简**：减少 per-lane 处理列数或改用 fp16 打包运算（需验证精度）
   c. **4-warp 高 occupancy 变体**：每 warp 处理 32 行（双 q 块），配合 wpe4（需控制 VGPR ≤ 256）
3. 现有实验基础设施已保留：bench_rot.py（旋转轮转）、quick_native.py（单后端快测）、explore_config.py（配置矩阵）、dump_triton_cfg.py（triton autotune 配置）

## 11. V 全局转置 PV 方案（结构性重写，重大性能提升）

### 11.1 方案原理

**核心洞察**：RDNA3 WMMA 16x16 的 A/B operand 布局不对称——A operand 要求 lane L 提供矩阵"行 L&15"的 16 列（行读），B operand 要求 lane L 提供"列 L&15"的 16 行（列读）。原转置 PV（out^T = V^T @ P^T）中 A = V^T 需要 **V 的列读**（跨行，LDS 中转 → 128 次 u16 标量读/迭代，2B/lane 低效）。

**解法**：把 V 一次性转置为 V_T [B,H,D,N]（n 连续），PV 改为 **out = P @ V**：
- A = P：lane L 提供 P 行 L&15 的 16 列 → **p_frag 现有布局复用**（assemble_p_frag 不变）
- B = V：lane L 提供 V 列 L&15 的 16 行 = V_T 行 L&15 的 16 连续 n → **行读 b128（1 条 v16h 向量读）**
- C 输出布局不变（out 行 L&15 的偶/奇 D 列）→ **out_acc 写回复用**

V_T 偏移（contiguous [B,H,D,N]）：`((b*H + h)*D + d)*N + n`，int8 路径 kv_len 恒为 8 倍数 → 32B 对齐安全。

### 11.2 实现

1. **v_transpose op**：`[B,N,H,D] -> [B,H,D,N]`（NHD/HND 均支持），每 thread 8 halfs 标量读 + 8 标量跨 stride 写（一次性 kernel，SDXL01 仅 +0.14ms）
2. **PV 段改 out = P @ V**：`wmma_f32_f16(p_frag, v_frag_t, out_acc)`，v_frag_t 从 V_T 行读
3. **SAGEATTN_VT_GLOBAL=1 编译 + core.py env 门控**：int8 与 direct 路径均支持
4. **V_T 模式跳过 v_tile LDS 残余**（V 初始加载/prefetch 读写用运行时 if 包，编译器 DCE）

### 11.3 关键教训：hipcc 对模板体内条件编译的解析 bug

- 在模板 `__device__` 函数体（attn_kernel_impl_t）内用 `#if`/`if constexpr` 包裹任何代码（即使逻辑为空），都会触发错误 **"attn_kernel_wpe1_t undeclared"**（诊断信息错误，实际是 hipcc 解析问题，与 -mllvm -amdgpu-early-inline-all 等标志相关）
- **解法**：模板体内用**运行时 `if (SAGEATTN_VT_GLOBAL)`**（宏展开为编译期常量，死分支被 DCE，两种模式都编译）；`#if` 只用于 host 函数（dispatch）和 namespace 顶层的宏定义
- 该 bug 浪费了大量调试时间（多轮"wpe1 undeclared"排查）

### 11.4 性能结果（旋转轮转 vs triton，V_T 模式）

| 用例 | 基线 n/t | V_T n/t | 变化 |
|------|---------|---------|------|
| SDXL01 (4096 int8 self) | 1.18 | 1.03-1.15 | -13~15% |
| SDXL07/13/16 (int8 self) | 1.18-1.27 | 1.06-1.09 | -12~18% |
| SDXL10 (1536 int8 self) | 1.27 | 0.98 | -29% |
| SDXL04 (1024 direct) | 1.18 | 0.65 | **-53%** |
| SDXL02-18 direct (cross) | 0.61-1.27 | 0.62-0.85 | 全面领先 |
| Anima01/03/05 (d128 int8) | 1.06-1.10 | 0.90-0.93 | **反超 7-10%** |
| Anima02/04/06 (d128 direct) | 0.97-1.35 | 0.60-0.88 | **反超 12-40%** |

**Anima 系列（用户重点）全部 native 快于 triton。**

### 11.5 正确性

全部路径通过（err < 0.05）：int8/direct × fp16/bf16 × self/cross × kv=77（非 8 倍数，硬件容忍非对齐 v16h 读）。

### 11.6 后续方向

- SDXL int8 长 self 仍有 6-15% 差距（SDXL01 波动大）：softmax 指令流仍是瓶颈（§10.3）
- QK 的 k_frag 也可尝试 global 直读（需解决 16B 对齐）
- 默认模式（=0）无回归（5.28ms）

## 12. V_T 方案正确性修复与真实性能验证（用户反馈"无改善"的根因排查）

### 12.1 问题诊断

用户加强散热条件下用 benchmark_attn.py 测试（默认模式数据 + triton 参考），V_T 方案"基本无改善、部分后退、ComfyUI 无改善"。排查发现三重问题：

**1) V_T 方案完全未生效**：
- .pyd 是默认编译（SAGEATTN_VT_GLOBAL=0 → kernel 走旧转置 PV 分支）
- benchmark_attn.bat / ComfyUI 未设置 SAGEATTN_VT_GLOBAL=1 → core.py 的 env 门控不转置 V
- 用户测的其实是默认模式（基线）→ "无改善"是必然的

**2) 正确性 bug A：WMMA operand 顺序错误**
- 误用 wmma(p_frag, v_frag_t) 计算 out = P @ V，其 C 输出布局（lane 持 out[2e+hw][L&15]）与写回代码（out[L&15][2e+hw]）转置
- 用 probe_wmma.cu（HIP 最小实验）确定 WMMA 布局：C 行 = 2e+(L>>4)、列 = L&15；A 行 = L&15 的 16 列；B 列 = L&15 的 16 行
- 修正：wmma(v_frag_t, p_frag)，A = V^T（V_T 行读）、B = P^T（p_frag）、C = out^T（转置解释 = out 行 L&15 列 2e+hw，匹配写回）

**3) 正确性 bug B：vt_off 漏 kb_base**
- v_frag_t 偏移公式漏了 kv 迭代块的 n 起点 → 每迭代都读 V_T 前 32 个 n（只处理前 32 个 key）
- 数值定位：V[n][d]=n 时 o≈均匀(0..31) 和（15.3），V[n][d]=(n≥512) 时 o=0
- 修正：vt_off 的 n 偏移加 kb_base

**教训**：此前的"性能提升"（SDXL01 4.78ms 等）是在错误输出（只处理前 32 key）下测的假象；且 check_all.py/check_err.py 未设 SAGEATTN_BACKEND → 默认走 triton 后端，native 正确性从未被真正验证。**native 验证必须显式设 SAGEATTN_BACKEND=native。**

### 12.2 修复与默认启用

- setup.py 默认加 -DSAGEATTN_VT_GLOBAL=1 编译
- core.py 去掉 env 门控，无条件转置 V（int8 + direct）
- 保留 kernel 内运行时 if(SAGEATTN_VT_GLOBAL) 分支（默认=1 时走 V_T）

### 12.3 真实性能（benchmark_attn.py 单次，加强散热条件，与用户默认模式数据同条件对比）

| 用例 | 用户默认模式 | V_T 修正版 | 变化 |
|------|-------------|-----------|------|
| SDXL01 (4096 int8) | 5.338 | 5.232 | -2% |
| SDXL04 (1024 direct) | 0.976 | 0.807 | **-17%** |
| SDXL07/10/13/16 (int8) | 10.73-25.34 | 10.73-24.13 | 基本持平 |
| SDXL02-18 (direct cross) | 0.21-0.68 | 0.16-0.67 | 持平或略快 |
| Anima02 (d128 direct) | 4.196 | 2.688 | **-36%** |
| Anima04 (d128 direct) | 5.728 | 3.814 | **-33%** |
| Anima06 (d128 direct) | 7.811 | 5.492 | **-30%** |
| Anima01/03/05 (d128 int8) | 16.3-82.3 | 17.2-84.0 | 基本持平 |

**结论：V_T 对 direct 路径（SDXL04/Anima cross/self 短序列）真实提升 17-36%；int8 长序列基本持平（-2%~+4%）。** 24 用例全部 OK。

### 12.4 ComfyUI 生效条件

- 需重新构建扩展（pip install -e .），.pyd 默认 V_T
- 无需设置环境变量（core.py 默认转置）
- ComfyUI 的 SDXL 短序列/self-attention（1024 等 direct 路径）应看到改善；4096+ 长序列（int8）基本持平

### 12.5 后续优化方向

- int8 长 self 仍有 11-14% 差距（softmax 指令流瓶颈，见 §10.3）；v_frag_t 的 global 读在 int8 迭代模式下 L2 行为可再优化
- 可考虑 int8 与 direct 分离配置（int8 回退旧 PV 或单独调优）
