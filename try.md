# try.md — RDNA3 (gfx1103) 全路径再提速实验记录

> 本文件记录 2026-09 对 `NativeBackendOptimizeReport.md` 中 gfx1103 全路径（fp16 direct / bf16 direct / int8 self / VAE）的再提速实验。
> **方法学**（遵循报告 §四）：唯一可信对比 = 同进程内交错轮转 + 充分预热 + 取中位数；绝对时间跨会话不可比。
> **防死机/中断**：每次改 kernel 后先跑单用例冒烟（正确性 + 单次计时），确认无 UB/死机，再跑全量；每步记录到本文件，避免重复实验。

## 0. 基线（commit 7149a4a，拆分后，实测于 2026-09）

在 gfx1103 上，用同进程交错轮转把「拆分前(80993de) vs 拆分后(HEAD)」逐路径实测，结论：**所有路径 ratio≈1.00（±2%），拆分无回归**。

int8 自注意路径（量化 + v_transpose 辅助占比是主要差距来源，报告 §9.1）：
- SDXL01/07/13, Anima01/03/05 大序列 self-attn
- VAE（SDXLVAE/AnimaVAE）D=128 超长

## 1. 实验方法（可复用 harness）

- `build_one.py <name>`：把 `csrc/attn_gfx110x.cu` 复制到 temp，按参数化模块名（`_qattn_<name>` / `sageattn_<name>`）编译成独立扩展，避免 torch.ops 命名冲突与 stale 缓存。
- `cmp.py <name> [all|direct|int8]`：同进程内交错轮转，`<name>` vs 基线模块 `_qattn_new2`，输出 ratio(base/cur)。**ratio<0.97=改进, >1.03=回退, 否则持平**。
- 冒烟：每次改 kernel 后先 build + 跑 cmp 单用例，确认无死机/错误。

## 1. 实验目录（本文件持续追加）

| # | 实验 | 状态 | 结果 |
|---|------|------|------|
| E0 | int8 路径各阶段 breakdown | done | 见 §2 |
| E1 | int8 D=64 self：32w kernel BN=32→64 | done | **回退 ~18-20%** (SDXL01 0.823, 07 0.829, 13 0.803)。BN=32 最优，已还原 |
| E2 | int8 D=128 self：BM 尺寸自适应 (kv≥6144 用 128) | done | **改进**：Anima05 -4.2%, VAE -5.8%, Anima01/D64 无回退。默认阈值 6144 (env 可调)。正确性：BM=128 用例 bit-identical (maxdiff=0)，vs SDPA 误差 3.4e-3 正常。**附带发现：D=64 int8 路径同模块重复调用 maxdiff≈1.2e-2 非确定（非 E2 引入，预存问题，已查明并修复，见 §4）** |
| E3 | D=64 direct 大 self：BM=64→128 | done | 持平 (±1% 噪声)：SDXL10 1.641→1.624, SDXL16 3.281→3.357。BM=64 维持 |
| E4 | D=64 int8 Q-quant 非确定性排查 | done/fixed | **定位：竞态在 blockReduceMax 的 RATIO 循环（跨调用 shared 复用无尾随 barrier），D=128 RATIO=1 不触发**。见 §4。**已修复（2026-09）**：reduction_utils.cuh 补尾随 barrier，D=64 determinism 验证通过，e2e 性能无回退（±0.3% 内） |
| E5 | D=64 self int8：wpe1_32 occupancy 1→4 波/EU | done | **持平/噪声** (SDXL01 +1.5%, SDXL07 -2.1%, SDXL13 0)。计算密集 kernel 提升 occupancy 无收益。已还原 |
| E6 | quant: LDS→寄存器缓存 + blockReduceMaxVec 向量归约 | done/kept | D64 quant -4%; D128 持平; 但使 D128 BLK 128/64 从 +1~3% 反转为 -6~9%。见 §6 |
| E7 | D128 int8 attn BN 默认 64→32 | done/kept | self 4096 -4.2%, cross D128 -4.7~4.9%, 长序列中性。env SAGEATTN_INT8_BN128=64 恢复旧值。见 §6 |
| E8 | E6+E7+D128 quant BLK 默认 128/64 组合定案 | done/merged | Anima01 -5%, cross D128 -6~7%, 其余路径无回退; 全 42 pytest 通过。见 §6 |

## 2. 阶段开销 breakdown（baseline，用于定位瓶颈）

同进程分阶段计时（vt=v_transpose, quant=quant_qk_int8, attn=attn kernel, full=三者串行）：

| 用例 | v_transpose | quant | attn kernel | 合计(full) |
|------|------------|-------|------------|-----------|
| SDXL01i8 (D=64 4096) | 219us | 220us | 4.192ms | 4.580ms |
| SDXL07i8 (D=64 6144) | 238us | 330us | 9.256ms | 9.873ms |
| Anima01i8 (D=128 4096) | 1300us | 2339us | 14.564ms | 18.083ms |
| SDXLVAE01 (D=128 16384) | 1167us | 1958us | 63.092ms | 66.350ms |

**结论**：attn 主 kernel 占绝对大头（计算密集 WMMA）；aux(vt+quant) 在 D=64 约 9%、D=128 约 18-20%。quant 在 D=128 是第二大成本（2-2.3ms）。

## 3. 本轮实验汇总（2026-09）

**本轮净改进（均已合并保留在 `attn_gfx110x.cu` 源码，已随 `pip install -e .` 安装生效）**：
- E2（D=128 int8 self 的 BM 尺寸自适应）— Anima05 -4.2%、SDXLVAE01 -5.8%
- E6（quant 寄存器缓存 + 向量归约）— D64 quant 单独 -4%；顺带使 D128 quant BLK 128/64 由 +1~3% 反转为 -6~9%
- E7（D128 attn BN 默认 64→32）— self 4096 -4.2%，cross D128 -4.7~4.9%
- E8（E6+E7+D128 quant BLK 默认 128/64 组合定案）— Anima01 -5%，cross D128 -6~7%，长序列/D64/direct 无回退

E1（32w BN=32→64）、E3（D=64 direct BM=128）、E5（occupancy 1→4）均无收益或回退，已全部还原。

**核心事实**：attn 主 kernel 计算密集（WMMA），配置已接近最优；合并后 int8 D128 cross 的 quant（~1.1-1.9ms）仍是单点最大辅助成本，D128 VAE/direct 路径收益空间已不大。剩余可挖：D128 quant 再提速（threads 256→512/1024 等）或跨 kernel 重叠。

## 4. D=64 int8 Q-quant 非确定性（已修复，2026-09）

**现象**：同输入重复调用 D=64 int8 全路径，输出 maxdiff≈1.2e-2（fp16 较大，非舍入）。分阶段定位：
- v_transpose：确定性 (maxdiff=0)
- **quant 的 Q 路径 (D=64, BLK=128/MIN_BLK_Q=32)：非确定性** — q 量化值 maxdiff=1.4e1, qscale 5.4e-4
- quant 的 K 路径 / D=128 全路径：确定性 (maxdiff=0)
- attn kernel：确定性 (给定相同 q_int8/k_int8/v_t/scale，输出一致)

**根因（数据竞态）**：`csrc/reduction_utils.cuh` 的 `blockReduceMax` 用 `static __shared__ shared[32]`。D=64 的 BLK_Q=128 ⇒ RATIO=4，该函数在同一个 block 内被 RATIO 循环调用多次；第 r 次调用的**读阶段**（warp0 读 `shared[lane]`）与第 r+1 次调用的**写阶段**（各 warp lane0 写 `shared[wid]`）之间**没有 barrier**，读/写可重叠 ⇒ q_scale 偶发取到下一组 r 的 amax。D=128 全路径 RATIO=1（单次调用）无跨调用复用，与实测确定性一致。K 路径（BLK_K=64/MIN_BLK_K=16）同样 RATIO=4，竞态结构相同，只是当时未在重复调用中观测到（窗口/时序偶发）。

**修复**：`blockReduceMax` 末尾补 `__syncthreads()`（尾随 barrier，使每次调用自同步）。该函数仅被两个 quant kernel（attn_gfx110x.cu / attn_gfx103x.cu）的 RATIO 循环调用，一处改动同时修复 gfx110x/gfx103x；不碰 attn 主 kernel。

**验证**：
- 确定性：修复前 D=64 quant 128 次重复调用 q_scale 差 1.9e-3、q_int8 差 32；修复后 maxdiff=0（逐位一致）
- 正确性：pytest 42/42 通过
- 性能（同进程交错轮转 A/B，修复前后扩展对比，e2e=int8 全路径=vt+quant+attn）：

| 用例 | quant fix/bsl | e2e fix/bsl |
|------|------|------|
| D64 s4096 | 0.950 | 1.003 |
| D64 s6144 | 0.960 | 1.001 |
| D128 s4096 | 1.000 | 0.999 |
| D128 s6144 | 1.004 | 1.003 |
| D128 s16384 | 0.990 | 1.001 |

quant 单独计时因核短、带宽受限，噪声大（±5-10%）；e2e 全路径（attn 主核占大头）均落在 ±0.3% 内，**无性能回退**。代价：每 blockReduceMax 多 1 次 barrier（D=64 每 block-iteration 多 4 次），对带宽为主的 quant 可忽略。
## 6. 本轮新实验 (2026-09, E6-E8, 均已合并保留)
### E6: quant 内核重写 (寄存器缓存 + 向量归约)
改动 (csrc/attn_gfx110x.cu quant_qk_int8_hnd_kernel):
- pass1 数据缓存 LDS (shared_data 往返) → 寄存器 uint4 reg[PPT][2]，去掉尾部 data barrier
- RATIO 串行 blockReduceMax → 多值向量归约 blockReduceMaxVec<T,N>(shared_amax[N])，barrier 2R→2
- 隔离量化 A/B (g1 实验扩展, 同进程交错, 中位数):
  | 用例 (quant 单独) | g1/bsl |
  |---|---|
  | D64 SDXL07 | 0.958 (359->344us) |
  | D128 Anima02x | 0.985 (持平) |
  | D128 Anima01 | 0.996 (持平) |
  | D128 AnimaCross03 | 1.005 (持平) |
结论: 单独看仅 D64 quant -4%, D128 无收益 (RATIO=1 本就走通, 瓶颈是带宽/block 开销非 barrier)。
**补充扫描: D128 quant 在新内核下 BLK 128/64 (RATIO=4) 从旧逻辑的 +1~3% 反转为 -6~9%**
  (旧 LDS+标量归约时 32KB LDS + 每 RATIO 一次 barrier 是负担; 寄存器+向量归约后 block 数减 4x 净赚)
### E7: D128 int8 attn BN 默认 64→32 (BM=64 路径, causal+非causal)
env SAGEATTN_INT8_BN128=64 可恢复旧值。
BN=32 相对 BN=64 (bench.py e2e all, g3 实验扩展): Anima01 0.9585, Anima02 0.9514,
Anima04 0.9532, Anima06 0.9593, SDXLCross01 0.9644, AnimaCross03 0.9915; 长序列 6144/9216/VAE16384 中性;
D64 路径未改 1.00。正确性: 全用例 allclose vs bsl (atol/rtol 0.05)。
解释: BN=64 少一半 kv-tile→更少 barrier, 但 score_cache 寄存器 (4x8 float/线程) 拖累占用;
BN=32 占用更优, cross/中短 self 净赢, 长序列两相抵。
### E8: 组合定案 = E6 + E7 + D128 quant BLK 默认 128/64 (auto 分支 head_dim==128 原 MIM_BLK → 128/64)
bench.py all 全矩阵 (g4 扩展, ratio g4/bsl):
  | 路径 | 用例 | ratio |
  |---|---|---|
  | int8 D128 self | Anima01 4096 | 0.951 (18.18->17.28ms) |
  | int8 D128 self | Anima03 6144 | 0.997 (中性) |
  | int8 D128 self | Anima05 9216 | 1.000 (中性) |
  | int8 D128 VAE | 16384 | 0.996 (中性) |
  | int8 D128 cross | Anima02 4096x512 | 0.931 |
  | int8 D128 cross | Anima04 | 0.923 |
  | int8 D128 cross | Anima06 | 0.940 |
  | int8 D128 cross | AnimaCross03 | 0.931 |
  | int8 D64 | SDXL01/07/13 | ~1.00 (attn 主导, quant 改善被吸收; SDXL07 0.995) |
  | direct D64 small | SDXL04/02/08/Cross01/05 | 0.91-0.96 但为噪声 (direct 代码未动, 已用 verify_direct AABB 证实 只有 SDXL08 可复现, 其余 flat; 小用例逐次噪声 ±10%+) |
端到端 (core.sageattn 已安装扩展 vs bsl 复刻): 5 用例全部 allclose=True (含 causal D128 4096)。
正确性: 安装后 pytest 42/42 通过。
D128 quant BLK 128/64 单独扫描 (g1 扩展, AABB): Anima02x 0.909, Anima01 0.937, D64 0.83 (D64 auto 本来就 128/64)。

### E9: quant Threads 256->512 (探针, 已拒绝)
改动: quant_qk_int8_hnd_kernel Threads 256->512 (D128 BLK=128/64 时 PPT 4->2, 寄存器占用减半)。
计时 (bench.py e2e all, g5 vs bsl): Anima01 0.890, Anima03 0.958, cross D128 0.73-0.79,
D64 SDXL01 0.939 —— 量化隔离 g5/g4 Anima02x 0.578 (747->432us), 收益远超 E6/E8。
**正确性否决**: pytest 42->40 (D128 int8 maxerr[0]/[1], NHD seq=2304 用例 cos~0.70, 实为严重错误);
复测中 g5 在小型 D128 上出现运行间 maxdiff 数百的非确定性 + iGPU 硬崩溃。
量化为�*行本身数值有限且与 bsl 仅差 ±1 LSB (BLK 模板编译期常量折叠, 0.011% 边界元素),
但 e2e 精度门失败 —— T=512 几何在 iGPU 上不可靠。已回退 (git 恢复 T=256, pytest 42/42 复绿)。
结论: **quant 保持 Threads=256**。如未来再试需先在多进程/长序列下复验 GPT 稳定性。
