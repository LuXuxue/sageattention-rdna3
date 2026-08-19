# try.md — 剩余优化方向 1~4 实验记录

> 本文件记录针对 `NativeBackendOptimizeReport.md` §九「剩余可优化方向」1~4 的
> 实验过程、代码修改、性能数据与技术分析。所有性能结论遵循报告 §四的测量方法论
> （同进程交错轮转 + 充分预热 + 取中位数），绝对时间跨会话不可比，**唯一可信对比
> = 同进程内交错轮转的倍率**。

## 0. 环境与基线

- 平台: AMD Radeon 780M (gfx1103), ROCm 7.14, torch 2.12.0+rocm7.14.0
- 基线 commit: `21d98b2` (fp8e5m2 存储优化实验记录, 即报告所述"已回退"状态)
- 构建: `pip install -e . --no-build-isolation`

### 基线数据（修改前, commit 21d98b2, 2026-08 重测）

正确性: `test_sageattn_rdna3.py` 36/36 通过 (native backend)。

聚焦基准 (benchmark_attn/bench_focus.py, 同进程轮转, warmup 300/rounds 6/iters 30;
VAE 用例 warmup 200/rounds 5/iters 20; ratio = native/triton):

| 用例 | native ms | triton ms | ratio | 报告对照 |
|------|-----------|-----------|-------|---------|
| SDXL01 (D64 int8 self 4096) | 4.857 | 4.781 | 1.016 | 持平 |
| SDXL07 (D64 int8 self 6144) | 10.491 | 9.750 | 1.076 | 慢 3-6% (此处 7.6%) |
| SDXL13 (D64 int8 self 9216) | 23.126 | 21.880 | 1.057 | 慢 3-6% |
| SDXL10 (D64 direct self 1536) | 1.617 | 1.489 | 1.086 | 慢 7-13% |
| SDXL16 (D64 direct self 2304) | 3.587 | 3.007 | 1.193 | 慢 7-13% (此处 19%) |
| Anima01 (D128 int8 self 4096 bf16) | 17.470 | 16.639 | 1.050 | 持平 |
| AnimaVAE01 (D128 int8 self 16384 bf16) | 45.158 | 44.324 | 1.019 | 慢 2-3% |
| AnimaVAE02 (D128 int8 self 24576 bf16) | 101.647 | 98.972 | 1.027 | 慢 2-3% |
| SDXLVAE01 (D128 int8 self 16384 fp16) | 62.308 | 62.802 | 0.992 | 持平 |

> 注: bench_focus.py 为本次实验新增的聚焦轮转脚本 (后端切换通过运行时改
> `core._BACKEND` 模块变量实现, env 仅在 import 时读一次)。

---

## 1. 方向 3：quant 大 block + 多累加器 ILP

**动机**：quant 读 Q+K 仅 20-23.5GB/s（torch sum 75GB/s）。报告 §八 试过 MIN_BLK
调大（fmax 串行链 +37-45%）、GPB 合并（无改善）。方向 1 建议"大 block quant +
多累加器 ILP"（对齐 triton BLK=128）。

**实现**（csrc/attn_gfx11.cu）：
- `quant_qk_int8_hnd_kernel` 重写为模板 `<T, HeadDim, BLK, MIN_BLK>`：每 block 处理
  BLK 行，每 MIN_BLK 行一组独立 amax（RATIO 组），**RATIO 个 fmax 累加器并行**
  （fmax 依赖链长度不变 = 16 元素/组）。scale 语义不变（per-32/16 行）。
- host 端 Q/K 拆分为两次 launch（原为混合 grid）。BLK 选择：
  - `SAGEATTN_QUANT_BLK=1`（默认 auto）：D=64 用 BLK_Q=128/BLK_K=64；D=128 用旧逻辑
    （BLK=MIN_BLK）
  - `=128` 强制 128/64、`=64` 强制 64/32、`=0` 旧逻辑（用于 A/B）
- scale 写回加越界守卫（`base_row + r*MIN_BLK < seq_len`，防最后一个 block 尾部
  子块写越界）。

**同进程 A/B 结果**（bench_quant_ab.py, 轮转, quant 单独）：

| 用例 | 大 block ms | 旧逻辑 ms | 变化 |
|------|------------|----------|------|
| SDXL01 (D64) | 0.240 | 0.258 | **-7.1%** |
| SDXL07 (D64) | 0.350 | 0.370 | **-5.5%** |
| SDXL13 (D64) | 0.496 | 0.605 | **-18.0%** |
| Anima01 (D128) | 0.676 | 0.669 | +1.1% (64/32) |
| AnimaVAE01 (D128) | 0.531 | 0.525 | +1.1% (64/32) |

- D=64 大 block 显著改善（-5~-18%），带宽达 87-95 GB/s（旧 23.5）
- D=128 旧逻辑已达 ~100GB/s（超 torch sum 75GB/s 上限），大 block（128/64 或 64/32）
  反而 +1~3%（32KB LDS 或额外 blockReduce barrier）→ **auto 分派 D=128 保持旧逻辑**
- 端到端收益被 attn kernel 体量掩盖（SDXL01/07/13 -1~-3%，噪音 ±2%）

**端到端同进程 A/B**（bench_e2e_ab.py）：SDXL01 -0.5%、SDXL07 +2%、SDXL13 -0.04%、
Anima01 +0.15%、AnimaVAE01 -0.2%（噪音内）。

---

## 2. 方向 2：mean_seq 向量化（32B 读）+ quant 联动

**动机**：SDXL01/07/13 慢 3-6% = 辅助 kernel（quant 0.30 + v_transpose 0.23 +
mean 0.09）。triton 用 torch `k.mean()`（40-71GB/s），native mean 仅 ~38GB/s
（2B 标量读）。

**实现**：mean_hnd_kernel v2 —— 每 thread 每迭代读行 s 的 16 个连续 half（32B，
uint4 对齐），16 个列累加器（寄存器），256 threads 步进 s；LDS 16KB 中转做列归约。

**结果**（单 kernel 计时, 与 torch mean 对比, 正确性 err=0）：

| 用例 | native v2 | torch mean | 变化 |
|------|----------|-----------|------|
| SDXL01 | 0.077ms (68GB/s) | 0.136ms | **快 43%** |
| SDXL07 | 0.116ms (68GB/s) | 0.141ms | **快 18%** |
| Anima01 | 0.220ms (76GB/s) | 0.249ms | **快 12%** |
| AnimaVAE01 | 0.164ms (77GB/s) | 0.178ms | **快 8%** |

mean 从 ~38GB/s → 68-77GB/s（接近读带宽上限 79GB/s），全用例优于 torch mean。

**方向 2 的 mean 融合（跨 block 归约）分析**：quant 的 K pass2 需要 mean 值，而
mean 需全局归约（K 数据 > L2 2MB 无法单 kernel 缓存），原子/二次 kernel 方案均无法
省 K 读 → 放弃融合，改为分别优化（mean v2 + quant 大 block）。

---

## 3. 方向 4：AnimaVAE bf16（探索结论）

**分段数据**（AnimaVAE01, native）：attn 41.6ms（96%）、quant 0.60、v_transpose
0.49、mean 0.17。v_transpose 总流量 25MB @ 52GB/s 已接近读写混合带宽上限；
bf16→fp16 转换仅 ~0.03ms（AnimaVAE01 0.676 vs SDXLVAE01 0.642）。**bf16 V_T 方案
收益 ~0.03-0.05ms，与 1-2% 的测量噪音相当，暂不实施**（保留为低优先级）。

---

## 4. 方向 1：SDXL10/16 direct self（v_transpose grid 分派优化）

**分段数据**（SDXL10: 1536×20 h）：

| 组件 | 时间 | 说明 |
|------|------|------|
| direct_attn (native) | 1.378ms | **比 triton 总时长还快 6%** |
| v_transpose | 0.233ms | 3.9MB @ 17GB/s, 小数据量效率低 |
| native 合计 | ~1.61ms | vs triton 1.473ms |
| SDXL16 attn | 3.030ms | 比 triton 慢 2% |

**关键发现**：SDXL10 的 attn kernel 本身快于 triton，**v_transpose 0.233ms 是
主要差距来源**（小数据量 1920 tiles 带宽仅 17GB/s，固定开销占比大）。

**实现**：v_transpose grid 分派 —— 小数据量（total_tiles ≤ 4096）用 grid=192
（每 block 多 tile，减少 block 调度/launch 尾部开销）；大数据量保持每 block 1 tile
（Anima01 8192 tiles grid 减小反而 +5-10%：barrier 串行 + L2 局部性损失）。
`SAGEATTN_VT_GRID` env 可强制覆盖（0=auto）。tile 内 stride 分配保持不变。

**同进程 A/B**（bench_vt_ab.py, 轮转）：SDXL10 **-11.4%**（0.142→0.126ms）；
SDXL16/SDXL01/Anima01/AnimaVAE01/SDXLVAE01 均持平或略好（0.98-1.01）。

**附加探索**：
- int8 路径测试（HND 布局）：SDXL10 int8 慢 12%、SDXL16 int8 快 6.5% —— 但 NHD
  布局下反转（SDXL16 int8 慢 3.6%）。**布局影响 int8 平衡点**：NHD 重扫 D=64 self
  平衡点仍在 3072（2560 direct 优 0.4%、3072 int8 优 3.8%），与报告一致 → 阈值
  保持 3072；HND 布局平衡点降至 ~2048（记录备查，NHD 为实际 benchmark/应用布局）
- BN=128 对 SDXL16 attn 略优 0.7%（噪音内，不采纳）
- v_transpose 连续 chunk 分配反而变差（-5-40%），已回退（保持 stride 分配）

**端到端**（bench_focus, NHD）：SDXL10 native 1.617→1.550（-4%）、SDXL16
3.587→3.290（-8%，绝对时间）；ratio 受 triton ±7% 热波动影响（1.04-1.15）。

---

## 5. 最终汇总（vs 基线 commit 21d98b2）

| 优化 | 目标用例 | 收益 (同进程 A/B) | 端到端 |
|------|---------|------------------|--------|
| quant 大 block + 多累加器 ILP (BLK_Q=128/BLK_K=64) | SDXL01/07/13 (D64 int8) | quant -5~-18% | native -1~-3% |
| mean kernel 32B 向量读 | 全部 int8 用例 | mean -8~-43% (68-77GB/s) | native -0~-2% |
| v_transpose grid 分派 | SDXL10/16 (短 kv direct) | vt -11% (SDXL10) | SDXL10 -4%, SDXL16 -8% |
| D=128 quant 保持旧逻辑 (auto) | Anima01/VAE | 0% (已 100GB/s) | 0% |
| D=64 self 阈值重扫 | 分发 | NHD 平衡点 3072 不变 | 0% (确认无误) |

**最终基准状态**（多轮测量, triton 热波动 ±7%）：

| 用例 | 基线 native | 最终 native | 基线 ratio | 最终 ratio (波动区间) |
|------|------------|------------|-----------|---------------------|
| SDXL01 | 4.857 | 4.67-4.72 | 1.016 | 1.05-1.06 |
| SDXL07 | 10.491 | 10.20-10.32 | 1.076 | 1.05-1.06 |
| SDXL13 | 23.126 | 22.81-22.94 | 1.057 | 1.04-1.05 |
| SDXL10 | 1.617 | 1.55-1.67 | 1.086 | 1.04-1.12 |
| SDXL16 | 3.587 | 3.27-3.50 | 1.193 | 1.10-1.15 |
| Anima01 | 17.470 | 17.7±0.05 | 1.050 | 1.06 |
| AnimaVAE01 | 45.158 | 44.2 | 1.019 | 1.01 |
| SDXLVAE01 | 62.308 | 60.7 | 0.992 | 0.98 |

> 注：native 绝对时间波动 ±3-5%（iGPU 热状态），triton 波动 ±7%（跨会话）。
> 同进程 A/B 的组件级收益（quant/mean/vt）是可靠结论；端到端 ratio 为区间估计。

**未实施项（记录原因）**：
- 方向 4 bf16 V_T + bf16 WMMA：v_transpose 转换仅 ~0.03ms，收益 < 噪音
- mean 融合进 quant：跨 block 归约无法省 K 读（K > L2），两个 kernel 分别优化更优
- D=128 quant 大 block：旧逻辑已 100GB/s（超带宽上限），无提升空间
- quant 软件流水（pass1 预读）：大 block 后每 thread 迭代数仍少（D=64 2 次），
  带宽已达 87-95GB/s（接近上限），流水无意义

---

## 6. 默认配置核查与"倒退"case 验证 (用户反馈)

**结论：不设置任何 env 变量的默认编译+运行即为各用例最优配置**。全部 env 默认值
均经同进程 A/B 验证：

| env | 默认 | 验证 |
|-----|------|------|
| SAGEATTN_QUANT_BLK | 1 (auto) | D=64 大 block -5~-18%, D=128 旧逻辑 (100GB/s 上限) |
| SAGEATTN_VT_GRID | auto | tiles≤4096→192; 端到端与其他 grid 值差异在 ±7% 噪音内 |
| SAGEATTN_QUANT_GPB | 1 | GPB>1 无改善 |
| SAGEATTN_INT8_32 / WPE / BN128 / BM128 | 1/1/0/-1 | 均为已扫最优 |
| 阈值 (D64/D64C/D64X/D128/D128X) | 3072/6144/6144/2048/4096 | NHD 重扫确认 |
| 编译宏 SAGEATTN_VT_GLOBAL | 1 (setup.py) | V_T 方案核心 |

**"加强散热下个别 case 倒退"验证**（同进程轮转 native vs triton）：
- SDXL04/09/11/12/14/17/18 等"倒退"case：同进程轮转全部**快于 triton 20-27%**
  （SDXL04 0.922, SDXL09 0.797, SDXL11 0.750, SDXL12 0.743, SDXL14 0.735,
  SDXL17 0.797, SDXL18 0.782）——旧 benchmark 文件为历史快照（无 VAE 用例），
  非代码回归
- Anima01 (n/t=1.066) 慢 6.6%：为长期状态（8888cee 前 21d98b2 代码实测同样
  17.47ms/ratio 1.05），非本次改动引入；旧文件 16.016ms 为特殊历史快照。
  8888cee 对 Anima01 仅改动 mean v2（更快方向），quant(D=128)/vt(tiles>4096)
  无变化 → 无回归
- vt grid 微调 (192 vs 96 vs 48) 端到端差异在 ±7% 噪音内（SDXL04 0.762/0.761/
  0.765, SDXL01 4.819/5.011/4.725 无一致规律）→ 保持 auto=192

**flash_attn 可用性（注意）**：flash_attn_func 官方 API 为 **NHD 布局** [B,S,H,D]
（benchmark_attn.py 传原始 q/k/v 即正确）；传 HND 会输出错误（cos=0.005, mae=3.07，
"错误但快"陷阱）。修正布局后 FA 可用，三方同进程对比见 try.md §7。

---

## 7. 三方同进程对比 (native / triton / flash-attn, bench_3way.py)

flash_attn 按 NHD 布局调用 (正确)。同进程轮转 warmup 150/rounds 5/iters 15:

| case | native | triton | FA | n/t | n/fa | t/fa |
|------|--------|--------|-----|-----|------|------|
| SDXL01 (D64 int8 self 4096) | 4.713 | 4.707 | 4.931 | 1.001 | **0.956** | 0.955 |
| SDXL07 (D64 int8 self 6144) | 9.974 | 9.792 | 9.906 | 1.019 | 1.007 | 0.989 |
| SDXL13 (D64 int8 self 9216) | 22.543 | 21.984 | 22.484 | 1.025 | 1.003 | 0.978 |
| SDXL10 (D64 direct self 1536) | 1.622 | 1.732 | 2.339 | 0.936 | **0.693** | 0.740 |
| SDXL16 (D64 direct self 2304) | 3.266 | 3.317 | 3.409 | 0.985 | 0.958 | 0.973 |
| Anima01 (D128 int8 self 4096) | 17.731 | 16.818 | 17.741 | **1.054** | 0.999 | 0.948 |
| Anima05 (D128 int8 self 9216) | 79.974 | 80.247 | 83.758 | 0.997 | 0.955 | 0.958 |
| SDXLVAE01 (D128 int8 16384) | 60.887 | 62.241 | 61.816 | 0.978 | 0.985 | 1.007 |
| AnimaVAE01 (D128 int8 bf16 16384) | 44.474 | 44.358 | 45.431 | 1.003 | 0.979 | 0.976 |
| Anima02 (D128 direct cross 512) | 2.640 | 3.056 | 3.543 | 0.864 | **0.745** | 0.863 |
| SDXL02 (D64 direct cross 77) | 0.225 | 0.293 | 0.513 | 0.767 | **0.438** | 0.571 |

**结论**：
- native vs FA: 全面持平或快 (n/fa 0.44-1.0); 短 kv 场景优势最大 (SDXL02 快 56%,
  SDXL10 快 31%, Anima02 快 25%) —— FA 短 kv kernel 低效
- native vs triton: 与之前一致 (D64 int8 持平, Anima01 慢 5.4%, cross 快 15-25%)
- FA vs triton: FA 通常慢 2-5%, 短 kv 慢 26-43%

---

## 8. 剩余可优化方向 A~D 实验记录

按报告 §9.2 A~D 逐项处理（同进程轮转验证, 结论与数据如下）:

### A1. v_transpose 与 attn 融合 — 判定不可行
- 设计1 (每 q-block 转置全量 kv): 转置重复 q_blocks 倍 (SDXL10=24x), 列读带宽
  ~30GB/s (v_transpose 实测) vs 行读近峰值 → 重复成本 ~3ms >> 当前 vt 0.126+attn 1.39
- 设计2 (split-KV): 转置不重复但需 flash-decoding 式二次 softmax 归约 + q 重复读,
  大工程且收益未验证 → 不实施

### A2. bf16 V_T 直通 — 收益 0.01ms, 不实施
- 同进程 A/B (轮转): bf16(转换) vs fp16(直拷) v_transpose 差仅 0.007-0.010ms
  (AnimaVAE-shape 0.421/0.414, Anima01-shape 1.305/1.295)
- 注意: 顺序测量曾出现 0.99ms 假差异 (热漂移), 轮转后证实转换非瓶颈

### B3. mean+quant 融合 — 收益小, 不实施
- quant K pass1 需 mean 完成才能减 mean (跨 block 全局归约), 拆两 kernel 后 K 读
  仍 2 次, 收益仅省 1 个 launch (~0.02ms) → 低于噪音

### B4. CUDA Graph launch 优化 — 负收益, 不实施
- 探针 (bench_graph_probe.py): Anima01 replay_only 17.944 vs direct 17.588 (+2%),
  SDXL10 replay_only 1.733 vs 1.545 (+12%) — launch 间隙非瓶颈, graph 重放反而慢
- 报告"全流程 vs 分段差 2.4ms"实为测量方法差异 (分段计时更热)

### C5. causal 三角跳过 — 已达理论 87%, 不实施
- causal/noncausal 时间比 0.567 (理论最优 0.5): block 级 kv_limit 已获主要收益
- 剩余 13% 需 warp 级差异化 kv 范围, 受 barrier 同步限制 (最慢 warp 决定时间),
  benchmark 无 causal 用例无法验证收益

### C6. GQA kv_heads 共享加载 — 无差距, 不实施
- GQA 实测 (bench: Krea 48/12 hq/hkv 4213 bf16 n/t=1.008; Krea-short 117 n/t=0.638;
  GQA-D64 32/8 4096 n/t=1.027; cos 全 OK): 长序列与 triton 持平, 短序列快 36%
- kv_heads 共享加载收益小 (kv 数据小, L2 命中掩盖重复读)

### C7. D=128 int8 BN 在 VAE 场景重测 — 默认 BN=64 确认最优
- bench_bn_scan.py: SDXLVAE01/02, AnimaVAE01/02 全用例 BN=16/32/128 分别 +9~+27%/+21~+72%
  vs BN=64 — LDS PV 后 BN=64 依然最优, 无代码改动

### C8. direct kernel V_T tile LDS 缓存 (新探索) — 负优化回退 + UB 修复
- 将 VAE 的 LDS PV 优化 (3.8) 应用到 D=64 direct kernel (SDXL10/16 的 PV 也有 8 倍
  全局行读冗余): 实测 SDXL10 1.549→1.933ms (+24%), SDXL16 +27%, SDXL04 +30% —
  D=64 短序列 kv-tile 少 (24 迭代), 原冗余读多命中 L2, V 拷贝+barrier 是纯开销,
  与 VAE (D=128 超长, L2 带宽饱和) 相反 → 回退
- **附带修复**: 排查崩溃中发现 direct kernel 两处 v_prefetch 使用缺少
  if (!SAGEATTN_VT_GLOBAL) 守卫 (v_prefetch 数组 VT_GLOBAL=1 时缩为 1 元素但循环
  VPrefetchPerThread 次 → 越界 UB; has_next 块还错误写 v_tile [BLOCK_N][VStride]
  索引) — 原代码"碰巧没崩" (LDS 布局恰好容忍), v_tile 修改后暴露。已加守卫
  (与 impl_t/impl_32_t 一致), 正确性 36/36, 性能无变化 (死代码真正 DCE)

### D8. HND 布局阈值适配 — 已实施
- core.py: D=64 self 阈值按布局区分 (HND=2048, NHD=3072), env 可覆盖
- 验证: HND 2304/2560 走 int8, NHD 保持 direct, 正确性 OK

### D9. causal D=64 边界补测 — 阈值 6144 确认
- bench_causal_boundary.py: 6144 int8 优1.4% (噪音内), 6656+ int8 优 3-9%
- 6144 处接近平衡, 6144 以下 direct 优 → 阈值合理

### D10. GQA 阈值标定 — 无需调整
- GQA 长序列 n/t≈1.0-1.03 (与 MHA 同), 当前阈值对 GQA 适用

### 结论
A~D 方向中: 仅 D8 (HND 阈值) 与 v_prefetch UB 修复落地为代码改动; C7/D9 为扫描
验证 (默认配置确认最优); A1/A2/B3/B4/C5/C6/C8 均经实验证实不实施或负收益。
再次印证"默认编译配置即最优"——剩余差距 (v_transpose 固有成本) 为架构性。

---

## 9. 剩余方向 1~3 深化实验 (本轮)

> 针对报告 §9.2 尚未处理的最高优先方向逐项实验。基线 commit `af8689b`
> (try.md §0 基线的后续, 已含 quant 大 block/mean v2/vt grid/HND 阈值)。

### 9.1 方向 1: D=64 int8 self (32 行 kernel) 的 PV LDS 缓存 —— 负优化, 不实施

**动机** (报告 §9.2 方向 1): 32 行 kernel (`attn_kernel_impl_32_t`, SDXL01/07/13
所用) 在 VT_GLOBAL=1 时 PV 的 v_frag 每 lane 每 dt 从全局 V_T 行读 32B, 与 VAE
优化前同款 L1/L2 冗余 (D=64/BN=32: 2子块 x2ct x4dt = 16 次 x32B x128 lanes =
64KB/迭代, 实际 tile 仅 4KB, 冗余 ~16 倍)。VAE (D=128) 的 LDS PV 缓存曾 -35%,
但 D=64 是否适用报告 §八"注意"标记为"尚未实验"。

**实现** (已回退): 给 `attn_kernel_impl_32_t` 加模板参数 `bool UseLdsPv`, v_tile
改 `[D][N]` 布局 (HeadDim x VTileStride=80, 64x80=10KB), 每迭代将 V_T 当前 tile
(32n x 64D) 拷入 LDS + 1 次 `__syncthreads()`, PV 的 v_frag 改从 LDS 行读。
host 端加 env `SAGEATTN_INT8_32_LDS` (-1=auto: kv>=4096 启用 / 0=关 / 1=开)。

**同进程 A/B** (bench_lds32_ab.py, 轮转, ratio = lds_on/lds_off):

| 用例 | lds_off ms | lds_on ms | ratio | 结论 |
|------|-----------|----------|-------|------|
| SDXL01 (4096) | 4.713 | 5.377 | **1.141** | 慢 14.1% |
| SDXL07 (6144) | 10.170 | 10.509 | **1.033** | 慢 3.3% |
| SDXL13 (9216) | 22.130 | 23.248 | **1.051** | 慢 5.1% |

**结论**: 全用例负优化 (+3~+14%)。归因: ① 32 行 kernel 仅 4 warps (128 threads),
每迭代 +1 次 `__syncthreads()` 的同步开销相对大 (impl_t/D=128 是 8 warps 且 VAE
序列极长); ② 3.9 曾为省 LDS/VGPR 将 v_tile 缩掉 (SDXL01/07/13 反超 triton 8-9%),
加回 10KB LDS 削弱该调度收益; ③ D=64 场景 V_T 数据量小 (每 head 6144x64x2B=768KB),
冗余读部分命中 L2, 省下的 L2 带宽有限, 而 D=64 的 PV 瓶颈是 WMMA 计算吞吐 (非 L2)。
→ 与 C8 (D=64 direct 短序列负优化) 共同覆盖了报告 §八"注意"的空白:
**LDS PV 缓存仅对 D=128 超长序列 (VAE, kv>=16384) 有效, D=64 int8 self 不适用**。
代码已回退 (git checkout), 实验脚本已删。

### 9.2 方向 2a: v_transpose ∥ mean_seq multi-stream 并行 —— 负优化, 不实施

**动机** (报告 §9.2 方向 2a): mean 与 v_transpose 无依赖, 理论上并行后 wall-time
从 sum(0.49+0.21) 降到 max(0.49,0.21) (Anima01), 省 ~0.21ms。

**实现** (已回退): core.py int8 路径加 `SAGEATTN_STREAM_PARALLEL=1` 开关, 用
第二个 stream 跑 mean_seq, 主 stream 跑 v_transpose, 两者 wait_stream 同步。

**同进程 A/B** (bench_stream_ab.py, 轮转, ratio = on/off):

| 用例 | seq_off ms | seq_on ms | ratio | 结论 |
|------|-----------|----------|-------|------|
| SDXL01 | 4.737 | 4.876 | 1.029 | 慢 2.9% |
| SDXL07 | 9.907 | 10.572 | 1.067 | 慢 6.7% |
| SDXL13 | 22.228 | 23.697 | 1.066 | 慢 6.6% |
| Anima01 | 17.602 | 17.966 | 1.021 | 慢 2.1% |
| AnimaVAE01 | 43.050 | 46.577 | 1.082 | 慢 8.2% |

**结论**: 全用例负优化 (+2~+8%)。归因: Radeon 780M 仅 12 CU, mean_seq (256 threads
grid) 与 v_transpose (256 threads grid) 并发时争抢 CU/内存带宽, 串行反而更优
(两个 kernel 各自都能打满可用带宽)。与报告 §9.2 方向 2a 的风险预期一致
("iGPU 12 CU 下并发 kernel 争抢, 收益需实测")。代码已回退, 实验脚本已删。

### 9.3 方向 2b: smooth_k=False 跳过 mean kernel —— 有收益, 精度/性能权衡

**动机** (报告 §9.2 方向 2b): 跳过 mean kernel 省 0.21ms@Anima01, 精度余量充足。

**精度验证** (check_smoothk.py, randn 数据, vs SDPA fp32 参考):

| 用例 | smooth_k=True | smooth_k=False | 说明 |
|------|---------------|----------------|------|
| SDXL01 | mae=0.00702 c=0.999930 | mae=0.00537 c=0.999930 | 不降反升 |
| SDXL07 | mae=0.01212 c=0.999934 | mae=0.01196 c=0.999934 | 不降反升 |
| SDXL13 | mae=0.00519 c=0.999934 | mae=0.00458 c=0.999933 | 不降反升 |
| Anima01 | mae=0.00488 c=0.999924 | mae=0.00439 c=0.999924 | 不降反升 |
| AnimaVAE01 | mae=0.00146 c=0.999925 | mae=0.00122 c=0.999925 | 不降反升 |

**非零均值 K 的精度影响** (K = randn + dc):

| K 偏置 dc | smooth_k=True cos | smooth_k=False cos | 说明 |
|-----------|-------------------|--------------------|------|
| 0 | 0.999934 | 0.999933 | 相当 |
| 3 | 0.999933 | 0.999860 | 略降 |
| 10 | 0.999933 | 0.999500 | 明显降 (仍 >0.99) |

**同进程 A/B 性能** (bench_smoothk_ab.py, 轮转, ratio = F/T):

| 用例 | smooth_k=T ms | smooth_k=F ms | ratio | 收益 |
|------|---------------|---------------|-------|------|
| SDXL01 | 4.785 | 4.617 | 0.965 | **-3.5%** |
| SDXL07 | 9.970 | 9.843 | 0.987 | -1.3% |
| SDXL13 | 22.196 | 22.082 | 0.995 | -0.5% |
| Anima01 | 17.510 | 17.277 | 0.987 | -1.3% |

**分析**: randn (零均值) 下减 mean 无收益 (mean≈0, 减去反而引入 mean 值自身的
fp16/bf16 舍入误差 → 精度略降); 真实模型 K 经 LayerNorm 后均值≈0, 但 RoPE 后
分布可能偏移。减 mean 的价值仅在 K 有明显 DC 偏置时体现 (dc=10 时 cos 0.9995
vs 0.99993, 仍远超 0.99/0.05 阈值)。收益端到端 -0.5~-3.5% (SDXL01 最显著,
因总时长最短)。**是否默认启用取决于精度/性能权衡, 待定**。

**决策**: 用户确认默认启用 smooth_k=False (追求性能, 精度余量充足)。已改 core.py
默认值 `kwargs.get("smooth_k", False)` (显式传 smooth_k=True 仍可覆盖, 供 K 有
明显 DC 偏置的场景)。

**实施后验证**:
- 正确性: check_correctness 8/8 OK; pytest `test_sageattn_rdna3.py` 36/36 passed
- 三方同进程 (bench_3way.py, smooth_k=False): SDXL01 n/fa=0.951 (反超 FA),
  SDXL07 n/fa=1.050, SDXL13 n/fa=1.031, Anima01 n/fa=1.025
- 完整 benchmark vs FA (benchmark_attn.py, native, smooth_k=False 快照
  `benchmark_attn-result-native-smoothk0.txt`), 与旧 native 快照 (smooth_k=True)
  的 ratio 对比 (跨 session FA 绝对时间有波动, 趋势可信):

| 用例 | 旧 native/FA (smooth_k=T) | 新 native/FA (smooth_k=F) | 改善 |
|------|--------------------------|--------------------------|------|
| SDXL01 | 1.03 | **0.987** | 反超 FA (1.3%) |
| SDXL07 | 1.05 | 1.035 | 差距收窄 |
| SDXL13 | 1.05 | 1.035 | 差距收窄 |
| Anima01 | 1.04 | 1.022 | 差距收窄 |
| SDXLVAE01 | 0.98 | 0.971 | 更快 |
| Anima03 | — | 0.974 | 快 2.6% |
| Anima05 | — | 0.953 | 快 4.7% |

---

## 10. 方向 2c / 3 (attn 主 kernel 深化与 vt 带宽) —— 分析结论: 不实施

**方向 2c (v_transpose 带宽再提升)**: 实测当前 vt 带宽 (bench_vt_ab.py) D=64
31-37GB/s、Anima01 (D=128) 34.6GB/s、VAE 26-27GB/s。报告 §八 已试过大 tile
(64n×64d/128n×32d)、uint2 显式向量化均更慢 (编译器已自动向量化)。写回已为
"每 thread 4 连续 half (8B), 8 thread 合并 64B cache line"。即使提到理论上限
~45GB/s, Anima01 vt 0.486→0.374ms 仅省 0.11ms≈0.6% (总 17ms), 投入产出比极低
→ 不实施。

**方向 3a (D=128 impl_t V_T tile 双缓冲)**: 报告 §3.8 已证明 VAE 的 LDS PV 优化后
瓶颈是 WMMA 吞吐 (10.9 TFLOP 硬件上限), 非 barrier 非 occupancy; "V 拷贝后
`__syncthreads`"的串行等待被 QK 计算 (长延迟 WMMA) 掩盖。双缓冲仅省 barrier 等待,
而 barrier 非瓶颈 → 预期收益 ≈0。且 §八 曾有"双缓冲失败"记录。→ 不实施。

**方向 3b (QK/PV 多累加器 ILP)**: 报告 §八 已试 "QK 2 累加器 +3%" (微利, 未达
显著), 32 行 kernel 已用 "2 子块共享 k_frag" 的 ILP 结构 → 无新空间。

**方向 3c (32 行 QK k_frag 双缓冲)**: 报告 §八 已试 "k_frag 提前加载到寄存器 →
0%" (LDS 读被 WMMA 掩盖), QK WMMA 内串行非瓶颈 → 不实施。

**本轮总规律印证**: 方向 1/2a 的结构性改动 (LDS 缓存/多 stream) 反而负优化,
方向 2c/3 的微优化预期收益 <1% —— 再次印证报告 §八"指令级微优化全部失败,
有效的优化是结构性的 (且已基本穷尽)"。native 相对 FA 的剩余差距 (SDXL07/13 3.5%、
Anima01 2.2%) 为 int8 路径的 v_transpose + quant 固有成本, 已达本硬件 (Radeon 780M
iGPU, 无 MFMA, WMMA 吞吐上限) 的极限。

---

## 11. 第九章方向 1: direct 路径消除 v_transpose —— 负优化, 不实施

**动机** (报告 §9.2 方向 1): direct 路径 (fp16/bf16 QK, 无量化) 无条件转置 V
(SDXL10 vt 0.12ms、SDXL16 0.17ms), 这是 direct self 慢 FA 7-11% 的主因。假设:
direct 的 QK 不量化、PV 不必用 V_T 行读, 可用 `SAGEATTN_VT_GLOBAL=0` 的 LDS 列读
(`load_fp16_col_frag`) 读 V 原布局, 跳过 v_transpose 后 direct 只剩 1 个 kernel。

**代码修改** (已回退):
- `direct_attn_kernel_impl_t` 加模板参数 `bool UseVT`, 函数体内所有
  `SAGEATTN_VT_GLOBAL` 判断替换为 `UseVT` (v_tile/v_prefetch 声明三元、V 初始加载
  循环、V prefetch、PV 段、V 写回 LDS 五处); launch wrapper
  `fp16/bf16_attn_kernel_wpe2_t` 透传 `UseVT`
- host 端 `fp16/bf16_attn_gfx11_t` 加 env `SAGEATTN_DIRECT_VT` (1=用 V_T 全局行读
  默认, 0=用 LDS 列读原始 V), 宏 `LAUNCH_*_K/T/BN/BM` 增加 UVT 运行时参数, 在
  `LAUNCH_*_T` 内 `if (UVT)` 展开为 `true/false` 两个编译期常量 launch
- core.py direct 分支: `SAGEATTN_DIRECT_VT=0` 时跳过 `ops.v_transpose` 与 padded
  V_T 分配, 直接传原始 v

**附带修复 (排查中发现)**: direct kernel 的 V 初始加载循环 (原 1580-1602) 在
`SAGEATTN_VT_GLOBAL=1` 时**缺少 `if (!SAGEATTN_VT_GLOBAL)` 守卫** —— v_tile 缩为
1 元素后循环仍写 `v_tile[n*VStride+d]` 越界 (UB, 碰巧没崩, 与 §8 C8 的 v_prefetch
UB 同类); 本实验给该循环加了 `if (!UseVT)` 守卫 (随方向 1 一起回退, 未单独落地)。

**正确性** (改动生效期间): pytest 36/36 (DIRECT_VT=0 与 =1 均过), check_correctness
8/8 OK —— LDS 列读路径输出正确 (cos≈1.0, mae 与 V_T 路径一致)。

**同进程 A/B** (bench_direct_vt_ab.py, 轮转, ratio = off/on, <1 为跳过转置更快):

| 用例 | vt_on ms | vt_off ms | ratio | 结论 |
|------|----------|-----------|-------|------|
| SDXL04 (D64 direct self 1024) | 0.760 | 1.009 | 1.327 | 慢 33% |
| SDXL10 (D64 direct self 1536) | 1.639 | 2.156 | 1.316 | 慢 32% |
| SDXL16 (D64 direct self 2304) | 3.405 | 4.319 | 1.268 | 慢 27% |
| SDXL02 (D64 cross 4096×77) | 0.207 | 0.195 | 0.942 | 快 5.8% |
| SDXL03 (D64 cross 4096×154) | 0.296 | 0.324 | 1.095 | 慢 9.5% |
| SDXL05 (D64 cross 1024×77) | 0.161 | 0.156 | 0.966 | 快 3.4% |
| SDXL06 (D64 cross 1024×154) | 0.205 | 0.234 | 1.142 | 慢 14% |
| Anima02 (D128 cross bf16 4096×512) | 2.607 | 3.403 | 1.305 | 慢 31% |
| Anima04 (D128 cross bf16 6144×512) | 3.829 | 5.310 | 1.387 | 慢 39% |
| Anima06 (D128 cross bf16 9216×512) | 5.559 | 7.819 | 1.407 | 慢 41% |

**结论**: 全用例绝大多数**负优化** (direct self +27~+33%、D=128 cross +31~+41%),
仅极短 cross (kv=77, SDXL02/05) 快 3-6% (在 ±5-7% 测量噪音边界, 且同 kv=154 的
SDXL03/06 反而慢 9-14%, 无稳定规律)。→ **回退, 不实施**。

**技术分析 (为什么负优化)**:
- LDS 列读 PV (`load_fp16_col_frag`) 每迭代每 lane 做 `DTiles × 16` 次 u16 标量 LDS
  读 (D=128 时 8×16=128 次/迭代), 带 bank 冲突 —— 这正是 §3.2 记录 V_T 方案要消除
  的原始瓶颈 ("128 次 u16 LDS 列读, 低效且 bank 冲突")
- UseVT=0 每迭代额外多 V 拷贝 (prefetch) + 1 次 `__syncthreads()` (V 写回 LDS 后),
  与 K 的软件流水线 barrier 叠加
- 省下的 vt 成本 (0.12-0.17ms) 远小于 LDS 列读带来的 attn kernel 退化 (SDXL10 attn
  约 +0.5ms, D=128 cross +1.5~2.3ms)
- 印证 §3.2 的核心设计判断: **V_T 行读是 direct 路径的刚需, v_transpose 是一次性
  支付的"划算"成本, 不能省**。报告 §9.2 方向 1 的假设 ("direct 短序列 DTiles 少、
  vt 占比高") 不成立——DTiles 少不抵 LDS 列读的 bank 冲突与 barrier 开销

**状态**: csrc 与 core.py 的方向 1 改动已 `git checkout`/精确回退; 实验脚本
`bench_direct_vt_ab.py` 保留供复现。报告 §9.2 方向 1 据此改写为"已证伪"。

---

## 12. direct self 短序列的 BM/BN 扫描 —— 确认默认已最优, 无一致规律

**动机**: direct self (SDXL10/16) 慢 FA 7-11% 全来自 v_transpose, 但 attn 主 kernel
的 BM 从未在报告中有过 direct self 的 A/B 记录 (报告 §5.3 只给结论 BM=64/BN=64,
§八 的 BM=128 失败是 int8/VAE 场景)。direct cross 用 BM=128 实测快 9-13% (同为
"减少 K/V 冗余读取"), 假设 direct self 短序列 BM=128 (K/V 重读减半) 可能同样受益。

**首轮扫描** (bench_direct_bm_ab.py, 5 配置 × 3 用例, 同进程轮转):

| 用例 | bm64_bn64(默认) | bm128_bn64 | bm32_bn64 | bm64_bn32 | bm64_bn128 |
|------|----------------|-----------|----------|----------|-----------|
| SDXL04 (1024) | 0.764 | 1.024 (+34%) | 0.809 (+6%) | 0.918 (+20%) | 0.799 (+5%) |
| SDXL10 (1536) | 1.697 | 1.644 (-3%) | 1.570 (-7.5%) | 1.584 (-6.7%) | 1.747 (+3%) |
| SDXL16 (2304) | 3.492 | 3.575 (+2%) | 3.525 (+1%) | 3.431 (-1.7%) | 3.356 (-3.9%) |

首轮 SDXL10 的 BM=32/BN=32 看似快 7%, 但 SDXL04 上 BM=32 反向慢 6%、SDXL16 慢 1%,
无跨用例一致规律 → 疑似热漂移。

**严格复测 SDXL10** (bench_direct_bm_confirm.py, warmup 500/rounds 10/iters 30):

| 配置 | 时间 | ratio |
|------|------|-------|
| bm64_bn64 (默认) | 1.551 | 1.000 |
| bm32_bn64 | 1.584 | 1.021 (+2.1%) |
| bm64_bn32 | 1.573 | 1.014 (+1.4%) |

严格复测证实首轮的 -7.5% 是热漂移噪音, BM=32/BN=32 实际慢 1-2%。**默认 BM=64/BN=64
即 direct self 最优, 无优化空间**。

**结论**: direct self 的 BM/BN 已是最优; 与报告 §八"occupancy 非瓶颈、BM/BN 无
一致规律"印证。direct self 慢 FA 7-11% 是 v_transpose 固定成本 (attn 主 kernel 已
持平/快 FA), 无法通过 BM/BN 调节消除。实验脚本 `bench_direct_bm_ab.py` 保留供复现。





