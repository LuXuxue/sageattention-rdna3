# gfx1035 (RDNA2) HIP Native 移植与优化工作日志

## 一、 概述
本文档记录将 SageAttention native 后端从 gfx1103 (RDNA3) 移植至 gfx1035 (RDNA2) 的技术方案、最终实现、正确性验证、性能基线与踩坑记录。
**当前最终状态**：
1. **架构定型**：最终采用 **V2 架构（BM=64, NR=2, NW=4, BN=16）** 作为核心 Kernel。早期尝试的 16-lane 协作设计（V3）因计算密度不足及 race condition 被放弃。
2. **性能基线**：D=128 长 self 路径已显著优于 Triton（快约 2.85 倍）；D=64 长 self 路径仍慢于 Triton 约 4 倍，为当前唯一性能瓶颈。
3. **工程约束**：当前 gfx10 与 gfx11 共用 `attn_gfx11.hip` 源文件，后续深度优化需建立独立的 gfx10 kernel 源。

---

## 二、 硬件环境与工具链基线
### 2.1 目标硬件规格（gfx1035）
| 项目 | 规格说明 |
| --- | --- |
| GPU 架构 | AMD Radeon Graphics（gfx1035，RDNA2，Rembrandt，消费级 iGPU） |
| 计算单元 | 6 CU |
| LDS 容量 | 64 KiB / CU |
| 显存 | 共享系统内存（Unified，约 14.4 GiB） |
| 时钟频率 | 2200 MHz |
| 矩阵/点积指令 | 支持 `V_DOT4_I32_I8` / `V_DOT2_F32_F16`；**无 WMMA / MFMA** |

### 2.2 工具链与基线确认
* **编译环境**：ROCm SDK（hipcc），通过 `--offload-arch` 实现单 wheel 同时包含 gfx1103 与 gfx1035 双 Target 构建。
* **对标基线**：Triton 后端在 gfx1035 上的最优配置（BM=128，BN=16，num_warps=4，num_stages=2）。

---

## 三、 架构差异与指令映射
RDNA2 (gfx1035) 与 RDNA3 (gfx1103) 在矩阵计算指令上存在根本差异，最终指令映射如下：

| 计算任务 | gfx1103 (RDNA3) | gfx1035 (RDNA2) | 备注 |
| --- | --- | --- | --- |
| int8 QK | WMMA i32 16×16×16_iu8 | `v_dot4c_i32_i8` (`__builtin_amdgcn_sdot4`) | SIMD 点积风格，每 lane 独立累加 |
| fp16/bf16 PV | WMMA f32 16×16×16_f16 | `v_dot2c_f32_f16` (`__builtin_amdgcn_fdot2`) | RDNA2 无张量核，fp16 矩阵乘降级为 V_DOT2 |
| 矩阵核 | 支持 WMMA | 无（消费级 RDNA2 缺失） | 彻底放弃 MFMA/WMMA 方案 |

*注：gfx1035 不支持 bf16 硬件指令，bf16 输入需在 LDS 暂存时转换为 fp16 处理。*

---

## 四、 核心工程实现
### 4.1 架构宏分发与双 Target 构建
使用 `#if defined(__GFX11__)` 包裹原有 gfx1103 WMMA kernel，`#if defined(__GFX10__)` 路由至 gfx1035 专属 kernel，保证 gfx1103 逻辑零改动。
**约束**：当前共用 `attn_gfx11.hip` 源文件，任何修改必须严格以宏隔离并进行双端回归验证。

### 4.2 最终 Kernel 架构（V2: BM=64, NR=2, BN=16）
经多轮演进验证，**V2 架构**为 gfx1035 上 D=64 路径的最优配置：
* **Block 布局**：1 block = 4 warps × 32 lanes = 128 线程。BM=64，BN=16。
* **线程级 ILP**：每线程处理 2 行（NR=2），通过多行独立点积链提升指令级并行度（ILP），有效隐藏 `sdot4` 延迟。
* **LDS 优化**：BN=16 显著降低单 tile LDS 占用（K tile ≈ 1 KB，V tile ≈ 2 KB），避免 occupancy 崩溃，同步延迟可被高占用率掩盖。
* **Q 预加载**：Q 预加载进寄存器，QK 内层循环仅从 LDS 读 K，避免 Q 的重复 LDS 流量。

### 4.3 Dispatch 策略
| 场景 | 配置 |
| --- | --- |
| D=64 (self / cross) | V2 Kernel (BM=64, NR=2, BN=16) |
| D=128 (self / cross) | V2 Kernel (避免 16-lane 设计导致的 LDS 翻倍与 occupancy 崩溃) |
| int8 / fp16 / bf16 | 各自独立分支分发，bf16 经 LDS 转 fp16 处理 |

### 4.4 关键实现要点
1. **Shuffle Mask**：`__shfl_xor_sync` 的 mask 必须为 64 位字面量（`0xFFFFFFFFFFFFFFFFULL`），32 位会触发编译期静态断言失败。
2. **同步屏障**：Q stage 写入 LDS 后必须 `__syncthreads()` 再预加载进寄存器，否则多 head 并发时会读到脏数据导致 NaN。
3. **边界守卫**：多行写回必须对每行独立做边界守卫（`valid0` / `valid1`），防止尾块越界污染相邻 block。
4. **GQA 映射**：GQA 头映射应为 `h / (q_heads / kv_heads)`，而非 `h % kv_heads`；int8 路径 q/k 恒按 HND 布局读取 stride。
5. **Softmax 缩放**：量化阶段已将 `sm_scale * log2e` 折入 score，Softmax 直接使用 `exp2(score - row_m)`，**切勿重复乘 `log2e`**。

---

## 五、 正确性验证
* **测试覆盖**：`test_sageattn_rdna3.py` 36/36 用例全部通过（涵盖 fp16/bf16、HND/NHD、causal、cross、GQA、smooth_k 等）。
* **严格断言**：在 `assert_close` 中强制加入 `torch.isnan(out).any()` 与 `torch.isinf(out).any()` 早返断言，彻底杜绝“输出含 NaN 但 cos/mae 仍通过”的漏报现象。
* **精度对齐**：native 后端与 Triton 后端输出 cos = 1.0，精确一致。

---

## 六、 最新性能基线
*测试环境：gfx1035 本机，轮转 3 rounds × 10 iters 取 median。*

| 用例 | Native V2 (ms) | Triton (ms) | 比率 (Native/Triton) | 状态评估 |
| --- | --- | --- | --- | --- |
| SDXL01 (D=64, int8, self, 4096) | 64.5 | 15.7 | **4.11×** | ⚠️ 瓶颈 |
| SDXL07 (D=64, int8, self, 6144) | 144.0 | 34.6 | **4.16×** | ⚠️ 瓶颈 |
| SDXL10 (D=64, fp16, self, 1536) | 32.3 | 4.93 | **6.55×** | ⚠️ 瓶颈 |
| SDXL05 (D=64, fp16, cross, 1024/77)| 2.60 | 0.66 | **3.94×** | ⚠️ 瓶颈 |
| SDXL06 (D=64, fp16, cross, 1024/154)| 4.30 | 0.91 | **4.72×** | ⚠️ 瓶颈 |
| **Anima01 (D=128, int8, self, 4096)**| **311.0** | **888.0** | **0.35× (快 2.85×)**| ✅ 优势 |

**结论**：D=128 路径已达成目标并具备显著优势；D=64 路径仍慢 Triton 约 4 倍，主要受限于软件流水线缺失与 LDS 读取粒度。

---

## 七、 优化演进与架构探索
| 版本 | 核心思路 | 结果与结论 |
| --- | --- | --- |
| **V1** (BM=32, BN=32) | 1 线程 1 行，LDS 协同暂存 | 正确性达标；但 `sdot4` 链串行、同步开销大，性能差。 |
| **V2** (BM=64, NR=2, BN=16) | 每线程持 2 行提升 ILP，BN=16 降低 LDS 占用 | **当前 D=64 最优配置**。比 V1 快 1.8×，VGPR/LDS 平衡良好。 |
| **V3** (16-lane 协作) | 16 lanes 协作归约单个 QK 值，消除单线程串行链 | **失败**。引发 race condition；且 per-thread ALU 利用率仅 1/16，计算密度太低，实测比 V2 慢 4×。 |
| **BM=128, NR=4** | 增大每线程行数至 4 行 | **失败**。VGPR 压力爆炸导致溢出，性能倒退。 |

---

## 八、 踩坑记录与已验证无效优化
### 8.1 正确性类踩坑
| 问题 | 现象 | 根因与修正 |
| --- | --- | --- |
| **宏参数 Stride 错误** | int8 路径输出 NaN / 垃圾值 | `L10_V2/V22/V3` 宏的 stride 参数序列中 `q_stride_n` 重复（应为 `k_stride_b`）。**修正**：严格核对参数序列。 |
| **NaN 漏报** | 测试显示 cos 通过，但实际输出含 NaN | NaN 在 `abs` 中传播不一定触发阈值。**修正**：引入 `isnan/isinf` 早返断言。 |
| **Softmax 缩放重复** | 输出放大约 11× | 量化阶段已折入 `log2e`，Softmax 中再次乘 `log2e` 属重复。**修正**：直接使用 `exp2`。 |
| **Q 预加载缺同步** | 多 head 时输出 NaN | Q stage 后未 `__syncthreads()` 即预读，读到其他线程未写完的 LDS。 |
| **写回边界守卫缺失** | 尾块越界污染相邻输出 | 多行写回需逐行独立 `valid` 守卫。 |
| **V_tile padding 不一致** | NaN / 越界写 | 按 padded stride 写入但按未 padded 尺寸读取。**修正**：废弃 padding 方案。 |

### 8.2 已验证的性能/架构负优化（全部回退）
| 尝试 | 思路 | 结论与根因 |
| --- | --- | --- |
| **16-lane 协作设计 (V3.5)** | 消除单线程串行链，16 lane 协作 1 row | **严重减速 (4×)**。计算密度太低，每 thread 仅持 1 个 q_reg，ALU 利用率极低。 |
| **2-stage K/V pipeline (D=128)** | 预取下一 KV tile 隐藏同步 | **严重减速 (LDS 溢出)**。LDS 占用翻倍导致 occupancy 崩溃，抵消流水线收益。 |
| **VT_GLOBAL (PV 直读全局)** | 省去 v_tile stage，依赖 L1 cache | **全面减速**。iGPU 无 L2 收益，L1 仅 32KB 且多 warp 竞争，**LDS 暂存是刚需**。 |
| **Streaming (无 LDS)** | 1 线程 1 行直读全局内存 | **严重减速**。iGPU 共享内存延迟高，K/V 被重复读取，带宽与延迟主导。 |
| **V_tile 布局改 [N][D]** | 改善 LDS bank 冲突 | **减速**。导致 PV 步骤访问模式变差。 |
| **BN=64 / BM=128** | 增大 tile 以摊销同步 / 增大 ILP | **减速**。BN 翻倍使 PV 链与 LDS 占用同步恶化；BM=128 导致 VGPR 溢出。 |

---

## 九、 瓶颈分析与后续优化方向
### 9.1 D=64 路径瓶颈分析
当前 D=64 路径慢 Triton 约 4 倍，核心归因于：
1. **软件流水线缺失**：Triton 采用 `num_stages=2` 软件流水线隐藏同步与内存延迟；在当前共用源 `attn_gfx11.hip` 中实现该机制需重写 LDS 布局，且极易引发 LDS 占用翻倍导致的 occupancy 崩溃。
2. **LDS 读取粒度不足**：Triton 使用宽向量读取，当前 native 实现多为逐元素或较小粒度读取。
3. **QK 串行链**：当前每 thread 仍串行执行多个 `sdot4`，缺乏有效的多路并发归约机制。

### 9.2 后续高价值方向（按优先级）
1. **建立独立 gfx10 kernel 源**：
   * 脱离 `attn_gfx11.hip` 共用源约束，解除修改空间限制。
   * 重新设计 LDS 布局，为引入 Triton 风格 2-stage 软件流水线创造条件。
   * 引入 `uint4` 等宽向量 K/V 读取，降低 LDS 读指令数。
2. **QK 串行链优化**：
   * 探索将 `sdot4` 链拆分为 2 路并发（累积到不同 acc 后归约），在减少 per-thread VGPR 压力的前提下提升 ALU 利用率。
3. **Persistent Kernel**：
   * 减少 kernel launch 开销（当前 SDXL01 grid 数达 640）。
4. **D=128 专项优化**：
   * 虽然 D=128 已快于 Triton，但 QK 归约链更长，reduce 开销占比更高，仍有进一步压榨性能的空间。

---

## 十、 会话进度更新（2026-08-30）

> 本节为持续推进 gfx1035 优化时动态追加的进度记录，避免测试死机/中断造成进度丢失。

### 10.1 BF16/FP16 int8 长序列 MaxErr 过大问题的根因与修复【已完成并验证】

**现象**：gfx1035 上 D=128 int8 路径（Anima01/03/05, AnimaVAE01 均为 BF16）MaxErr 过大：
| 用例 | 修复前 native | triton 参考 |
| --- | --- | --- |
| Anima01 (4096) | mae 0.27 / cos 0.975 | mae 0.0024 / cos 0.9999 |
| Anima03 (6144) | mae 0.47 / cos 0.969 | — |
| AnimaVAE01 (16384) | mae 0.20 / cos 0.964 | — |

**关键诊断**（用 Python 重建 native 的量化 QK + fp32 softmax/PV 与理想 P 对比）：
1. **量化本身不是根因**：`s_int8 * (q_scale * k_scale)` 与理想 P 的 cos=0.9999（量化 QK 正确）。
2. **问题在 V2(BN=32) kernel 的 online-softmax/PV 累积**：V1/V2/V22 三个布局（BM=32/64）在 D=128 且 kv_len≥2304 时全部出现 mae 0.2-0.34；V3(BN=16) 在同数据下 mae 0.003。
3. **改 p_tile 为 fp32 无效**：把 V2 的 p_tile 从 fp16 改为 fp32（并改标量 fp32 PV）后误差仍为 0.254 → **不是 fp16 概率截断问题**，而是 BN=32 + D=128 长序列下 online-softmax 的深层数值/结构问题（尚未彻底定位到指令级，保留为待办）。

**修复方案（已生效）**：在 `qk_int8_sv_bf16_attn_gfx11_t` 的 gfx10 分支中，D=128 int8 当 `kv_len>2048` 默认路由到 V3(BN=16)（该尺寸下 V3 精确：mae~0.003），否则用 V2(BN=32)。`SAGEATTN_GFX10_V3` 可覆盖（1=强制 V3，0=关闭 V3）。**仅改 gfx10 分支，gfx1103 零改动。**

**修复后验证**（gfx1035 实测）：Anima01/03/05 全部 cos 0.9999 / mae 0.003-0.004，对齐 triton。

**性能对比（修复后，round-robin 计时，含 quant+vt 全流程）**：
| 用例 | native (ms) | triton (ms) | 比率 (native/triton) |
| --- | --- | --- | --- |
| Anima01 (D128, 4096) | 538.9 | 888.5 | **0.61× (快 1.65×)** ✅ |
| Anima03 (D128, 6144) | 1231 | 2027 | **0.61× (快 1.65×)** ✅ |
| Anima05 (D128, 9216) | 2810 | triton 崩溃 | 快 ✅ |
| SDXL01 (D64, 4096) | 64.8 | 15.6 | 4.14× ⚠️ 仍为瓶颈 |
| SDXL07 (D64, 6144) | 143.6 | 34.3 | 4.19× ⚠️ 仍为瓶颈 |

**结论**：D=128 已修复且快于 triton（1.65×）；D=64 仍是唯一瓶颈（慢 4x，与报告 §6 一致）。注意：D=128 路由 V3 后比原 V2 的 311ms 慢，若后续定位到 V2 的 D=128 精度根因，可再切回 V2 提速到 2.85×。

### 10.2 V3 kernel 的已知边界 bug（待办）

**现象**：V3(BN=16) D=128 在 kv_len≤2048 时输出 NaN（seq 512/768/1024/1536/2048 均 NaN），kv_len≥3072 正常（mae 0.003-0.004）。**注意**：当前 pytest 的 D=64 用例在 int8 阈值以下的尺寸走 direct 路径，不会触发该 NaN，故测试全部通过。
**影响**：D=128 int8 小序列（kv≤2048）不能用 V3（已由 dispatch 保证走 V2，V2 在 kv≤2048 精确）。此为独立 bug，与 D=128 精度修复解耦，保留为待办。

### 10.3 pytest 增强【已完成】

- 新增 `TestSageAttnMaxErr`：覆盖 6 个用例（D=128/D=64 的 int8 BF16/FP16 + D=128/D=64 的 fp16 direct），判定条件与 `benchmark_attn.py` 一致：`max_err < 0.05`、`cos > 0.99`、无 NaN/Inf。对 triton 的 D=128 长序列（seq≥6144）提前 skip（iGPU 上 triton 长 D=128 会 python.dll 硬崩溃）。
- **验证**：`pytest --backend native` = 42 passed（原 36 + 新 6）；`--backend triton` 下 MaxErr 3 passed 3 skipped（无崩溃）。
- 使用小型触发尺寸（D=128 用 seq=2304 最小触发 V2 精度 bug 的尺寸；D=64 用 3072；direct 用 1024/2048），兼顾触发 bug + 快速 + 避免 fp32 SDPA 参考 OOM。

### 10.4 编译流程踩坑记录

- **hipify 一步是关键**：本项目的构建链是 `attn_gfx11.cu → (hipify) → attn_gfx11.hip → (hipcc 编译)`。`attn_gfx11.hip` 是已提交文件，ninja 默认 "skipped, already hipified" **不会随 .cu 改动重新生成**；修改 `.cu` 后必须确认日志出现 `attn_gfx11.cu -> attn_gfx11.hip [ok]`（而非 [skipped]），否则改动不生效。改 `mma_gfx10.h` 等头文件需 touch 或 clean 重建。
- 构建完成标志是 `copying ... _qattn_gfx11.pyd -> sageattention`，且 `(Get-Item pyd).LastWriteTime` 更新。`subprocess.CalledProcessError`/`clang-cl ... exit status 1` 是 torch cpp_extension 检查编译器版本的良性告警，非编译失败。
- 当前 setup.py 默认同时编译 `--offload-arch=gfx1103 --offload-arch=gfx1035` 双 target。

### 10.5 待办（按优先级）

1. **拆分 attn_gfx11.cu** → `attn_gfx110x.cu`(gfx11) / `attn_gfx103x.cu`(gfx10)，setup.py 按 GPU_ARCHS 判断只编译对应源，减少编译时间。gfx10 kernel 已在独立头文件（mma_gfx10.h, attn_gfx10_new.h）中，需将 host 调度/辅助函数（quant/mean/v_transpose + 3 个 API 入口）抽到共享源，避免重复符号。
2. **D=64 性能优化**（超 triton 4x 的瓶颈）：软件流水线、宽向量 LDS 读取、QK 链拆分（报告 §9.1）。
3. **定位并修复 V2 的 D=128 长序列 online-softmax 精度根因**（可恢复 V2 的 2.85× 优势）与 V3 的小序列 NaN bug。
4. 若时间允许，优化 D=128 的 QK 归约链与 persistent kernel 减少 launch 开销。