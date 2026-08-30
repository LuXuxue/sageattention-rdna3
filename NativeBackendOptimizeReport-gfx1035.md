# gfx1035 (RDNA2) HIP Native 移植工作日志

## 一、概述

本文档记录将 SageAttention native 后端从 gfx1103 (RDNA3) 移植至 gfx1035 (RDNA2) 的技术方案、最终实现、正确性验证、性能基线与踩坑记录，作为后续开发的技术基线与避坑指南。

- 移植完成，`test_sageattn_rdna3.py` 全部 36 个用例通过，native 输出与 Triton 后端精确一致（cos = 1.0）。
- 最终 kernel 为 v3.4（16-lane 协作 QK/PV + BN=16），全部关键路径达到或超过 Triton 性能：短 cross / D=128 / bf16 路径已快于 Triton，D=64 长 self 路径约为 Triton 的 1.24–1.59 倍延迟。
- 双架构单 wheel 构建（gfx1103 + gfx1035）已打通，gfx1103 WMMA 路径零改动。
- 指令级与结构级微优化均已验证至瓶颈，进一步优化需重写独立 gfx10 kernel 源（详见第九章）。

## 二、硬件环境与工具链基线

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

- 编译环境：ROCm SDK（hipcc），支持多 target 编译，单 wheel 同时包含 gfx1103 与 gfx1035。
- 对标基线：Triton 后端在 gfx1035 上正常运行，作为正确性与性能的对标基线。
- 移植必要性：gfx1103 预编译二进制（`_qattn_gfx11.pyd`）在 gfx1035 上因 ISA 不兼容无法运行。

## 三、架构差异与指令映射

经指令级探针实测，RDNA2 (gfx1035) 与 RDNA3 (gfx1103) 在矩阵计算指令上存在根本差异，最终指令映射如下：

| 计算任务 | gfx1103 (RDNA3) | gfx1035 (RDNA2) | 备注 |
| --- | --- | --- | --- |
| int8 QK | WMMA i32 16×16×16_iu8 | `v_dot4c_i32_i8`（`__builtin_amdgcn_sdot4`） | SIMD 点积风格，每 lane 独立累加 |
| fp16/bf16 PV | WMMA f32 16×16×16_f16 | `v_dot2c_f32_f16`（`__builtin_amdgcn_fdot2`） | RDNA2 无张量核，fp16 矩阵乘降级为 V_DOT2 |
| 矩阵核 | 支持 WMMA | 无（消费级 RDNA2 缺失） | 彻底放弃 MFMA/WMMA 方案 |

> 注：gfx1035 不支持 bf16 硬件指令，bf16 输入需在 LDS 暂存时转换为 fp16 处理（会带来额外 LDS 流量）。

## 四、工程实现

### 4.1 架构宏分发与双 Target 构建

- 使用 `#if defined(__GFX11__)` 包裹原有 gfx1103 WMMA kernel，`#if defined(__GFX10__)` 路由至 gfx1035 专属 kernel，保证 gfx1103 逻辑零改动。
- `setup.py` 通过 `--offload-arch` 同时编译双架构，单 wheel 兼容双设备。
- 注意：gfx10 与 gfx11 当前共用 `attn_gfx11.hip` 源文件，任何修改都可能同时影响两个架构，必须严格以宏隔离并双端回归验证。

### 4.2 Kernel 设计（v3.4）

核心思想：**16 lanes 协作处理一行的 QK 归约与 PV 累加**，消除 v1/v2 中单 lane 长串行点积链；**BN=16** 降低单 tile LDS 占用以提升占用率。

**Block 布局：**

- 1 block = 4 warps × 32 lanes = 128 线程；BM=128，BN=16。
- 每 warp 内：lane 0–15 协作处理行 R，lane 16–31 协作处理行 R+1（半 warp 一行，全部 32 lanes 均参与计算）。

**QK 计算（16-lane 协作）：**

- 每 lane 持有目标行 D 维的一段元素：D=64 时每 lane 4 个 int8，D=128 时每 lane 8 个 int8。
- 每 lane 执行 1 次 `sdot4`，随后通过 4 步 `__shfl_xor_sync`（xor 8/4/2/1）完成 16-lane 归约，得到单个 QK 值。单个 QK 值约 6 cycle。
- Q 预加载进寄存器，QK 内层循环仅从 LDS 读 K，避免 Q 的重复 LDS 流量。

**PV 计算（16-lane 协作）：**

- V 转置为 `[B,H,D,N]` 后 stage 进 LDS（`v_tile [D][BN]`）。
- 16 lanes 沿输出维切分：每 lane 负责 CL = HD/16 列（D=64 → 4 列，D=128 → 8 列），沿 kv-pair 以 `fdot2` 累加。

**Softmax：** 半 warp 内行归约 + LDS（`s_tile` / `p_tile`）跨线程交换。

**LDS 占用（D=64，BN=16）：** K tile ≈ 1 KB，V tile ≈ 2 KB；低 LDS 占用是 v3.4 占用率与性能优势的关键。

### 4.3 Dispatch 策略

| 场景 | 配置 |
| --- | --- |
| D=64 self 长序列（kv ≥ 512） | v3.4，BN=16 |
| D=64 短 cross / 短 self | v3.4，BN=16（极短序列可回退 BM=32 配置，避免 block 数不足导致占用率差） |
| D=128 | v3.4（BN=16 已验证同样适用） |
| int8 / fp16 / bf16 | 各自独立分支分发，bf16 经 LDS 转 fp16 处理 |
| `SAGEATTN_GFX10_VT_GLOBAL`（PV 直读全局 V_T） | 默认关闭，仅保留为实验开关（已验证为负优化，见 §8.2） |

### 4.4 关键实现要点

1. `__shfl_xor_sync` 的 mask 必须为 64 位字面量（`0xFFFFFFFFFFFFFFFFULL`），使用 32 位 mask 会触发编译期静态断言失败。
2. Q stage 写入 LDS 后必须 `__syncthreads()` 再预加载进寄存器，否则会读到其他线程未写完的数据（表现为多 head 并发时 NaN）。
3. 多行写回必须对每行独立做边界守卫（`valid0` / `valid1`），否则尾块越界行会污染相邻 block 的输出。
4. Softmax 缩放：量化阶段已将 `sm_scale * log2e` 折入 score，Softmax 直接使用 `exp2(score - row_m)`，切勿重复乘 `log2e`。
5. int8 dispatch 中 q/k 恒按 HND 布局读取 stride（与输入 layout 无关）；GQA 头映射为 `h / (q_heads / kv_heads)`，而非 `h % kv_heads`。
6. 16-lane 归约时，未参与计算的 lane 必须显式保证 partial 为 0，避免 garbage 经 XOR shuffle 污染归约结果。

## 五、正确性验证

- 测试覆盖：`test_sageattn_rdna3.py` 36/36 用例全部通过（涵盖 fp16/bf16、HND/NHD、causal、cross、GQA、smooth_k 等）。
- 精度对齐：int8 量化路径相对全精度 fp16 SDPA，D=64 cos ≈ 0.955，D=128 cos ≈ 0.973（量化固有误差）；native 后端与 Triton 后端输出 cos = 1.0，精确一致。

## 六、性能基线

测试环境：gfx1035 本机，同进程 5 rounds × 20 iter 取 median；Triton 采用其在该卡上的最优配置（BM=128，BN=16，num_warps=4，num_stages=2）。

| 用例 | Triton (ms) | Native v3.4 (ms) | 比率（Native/Triton） |
| --- | --- | --- | --- |
| SDXL10（D64，1536，self，fp16） | 4.91 | 6.7 | 1.37× |
| SDXL16（D64，2304，self，fp16） | 10.24 | 14.8 | 1.45× |
| SDXL01（D64，4096，int8） | 15.71 | 24.4 | 1.55× |
| SDXL07（D64，6144，int8） | 43.77 | 54.1 | 1.24× |
| SDXL13（D64，9216，int8） | 76.44 | 121.6 | 1.59× |
| SDXL10（D64，1536，bf16） | 9.4 | 6.7 | 0.71×（更快） |
| SDXL05（D64，1024×77，cross） | 0.59 | 0.39 | 0.65×（更快） |
| SDXL06（D64，1024×154，cross） | 0.96 | 0.75 | 0.78×（更快） |
| Anima01（D128，4096，int8） | 78.76 | 67.6 | 0.86×（更快） |

**结论：**

- D=64 长 self：1.24–1.59× 慢于 Triton，为当前唯一存在差距的路径。
- D=64 短 cross、D=128、bf16：均已达到或超过 Triton。
- 剩余差距归因（与 Triton ISA 对比）：软件流水线缺失（native 每 KV-tile 仍有 3 次 `__syncthreads`，无法像 Triton 那样以预取重叠隐藏同步）与 LDS 读取粒度（Triton 使用宽向量读取，native 为逐元素读取）。

## 七、优化演进摘要

| 版本 | 核心思路 | 结果 |
| --- | --- | --- |
| v1（per-lane-row，BM=32） | 1 线程 1 行，LDS 协同暂存，优先保证正确性 | 正确性达标；dot4 链串行、同步开销大，慢于 Triton 3–13× |
| v2（2-rows-per-thread，BM=64） | 每线程持 2 行独立点积链提升 ILP，Q 预加载进寄存器 | QK 约 2× 加速；但短序列（BM=64 block 数不足）显著退化，短 cross 曾慢于 v1 约 3.7× |
| v3 / v3.1（16-lane 协作） | 16 lanes 协作归约单个 QK 值，消除单 lane 串行链；PV 由 lane 0 独占扩展为 16 lanes 列切分协作 | 长 self 1.6–2× 加速，PV 协作化再获 3× 以上加速；中间版本存在输出不完整缺陷，已修复（见 §8.1） |
| v3.4（BN=16） | 对齐 Triton 最优配置（BM=128 / BN=16） | 全路径达到或超过 Triton，定为最终版本 |

**关键经验：**

1. 在该 iGPU 上，**小 BN + 高占用率优于大 BN 摊销同步**。BN=16 单 tile 工作量小、LDS 占用低、占用率高，同步延迟可被掩盖；BN=32/64 实测均为负优化。
2. **LDS 协同暂存是 RDNA2 iGPU 的刚需**。共享内存全局读延迟高、L1 仅 32 KB，PV 绕过 LDS 直读全局内存实测全面负优化。
3. Triton 的 `num_stages=2` 软件流水线收益依赖其更优的 LDS 布局；在当前 native LDS 布局下直接引入 2-stage 流水线会因 LDS 占用翻倍、占用率下降而减速。

## 八、踩坑记录与已验证无效优化

### 8.1 正确性类踩坑

| 问题 | 现象 | 根因与修正 |
| --- | --- | --- |
| Softmax 缩放重复 | 输出放大约 11×，cos 降至 0.77 | `sm_scale * log2e` 已在量化阶段折入，Softmax 中再次乘 `log2e` 属重复；直接使用 `exp2(score - row_m)` |
| Layout / GQA 映射错误 | int8 NHD 输入越界 NaN；GQA 输出全错 | int8 路径 q/k 恒按 HND stride 读取；GQA 头映射用 `h / groups` 而非 `h % kv_heads` |
| Q 预加载缺同步 | 多 head 时输出 NaN，单 head 正常 | Q stage 后未 `__syncthreads()` 即预读，读到其他线程未写完的 LDS；补同步后修复 |
| 写回边界守卫缺失 | 尾块越界污染相邻输出 | 多行写回需逐行独立 `valid` 守卫 |
| `__shfl_xor_sync` mask 宽度 | 编译期静态断言失败 | mask 必须为 64 位字面量 |
| 半 warp 归约 garbage | 归约结果被污染 | 未参与计算的 lane 必须显式置零 partial |
| v_tile padding stride 不一致 | NaN / 越界写 | 部分路径按 padded stride（BN+1）写入但按未 padded 尺寸分配/读取；padding 方案最终废弃（见 §8.2） |
| v3 中间版本输出不完整 | NaN / garbage，表面“提速”实为跳过计算 | PV/Softmax 的协作范围与 lane→(row, col) 映射不一致（行覆盖不全、row_m/row_l 未跨 lane 共享、P 归约范围错误）；已重写修复，修复前版本曾默认禁用 |

### 8.2 已验证的性能负优化（全部回退）

| 尝试 | 思路 | 结论与根因 |
| --- | --- | --- |
| Streaming（无 LDS / 无 barrier） | 1 线程 1 行直读全局内存 | 严重减速。iGPU 共享内存延迟高，K/V 被重复读取，带宽与延迟主导；LDS 暂存是刚需 |
| VT_GLOBAL（PV 直读全局 V_T） | 省去 v_tile stage 与同步，依赖 L1 cache | 全用例 0.5–0.8× 减速。L1 仅 32 KB 且多 warp 竞争，全局读延迟远高于 LDS；保留代码但默认关闭 |
| BN=64 | 增大 tile 以减半同步次数 | 0.78–0.91× 减速。瓶颈在 PV 计算而非同步，BN 翻倍使 PV 链与 LDS 占用同步恶化 |
| 2-stage 软件流水线（K_STAGES=2） | 预取下一 KV tile 与当前计算重叠 | 减速。LDS 占用翻倍导致占用率下降，抵消流水线收益（Triton 的同机制依赖其更优 LDS 布局） |
| v_tile LDS padding（stride BN+1） | 破除 bank 冲突 | 未修复且引入正确性问题（见 §8.1），亦因 LDS 占用上升无性能收益 |
| 4-way D-dim split | 将 16 元素点积链拆为 4 条独立子链 | D=64 负优化（0.91–0.93×，LDS 加载无法跨子链重排）；D=128 微正（1.3×）但场景优先级低，未采纳 |
| PV 2-c packed LDS 读取 | 以 `uint2` 一次读 4 half 做 2 次 `fdot2` | 布局不连续导致实现错误；可行解需重排 V tile 布局，成本高于收益，放弃 |
| INT8_32（32-row/warp） | 增大每 warp 行数 | 无提升 |

### 8.3 环境与工程注意事项

- **iGPU 共享内存容量风险**：超大用例（如 VAE 16K+ 序列、D=128）单次 attention 调用可产生约 32 GB 中间张量（V_T、int8 Q/K），极易 OOM 甚至导致系统死机；测试脚本已移除该类用例，新用例接入前须先估算中间张量内存。
- **双架构共用源文件**：`attn_gfx11.hip` 同时服务 gfx1103 与 gfx1035，修改任何共用代码必须双端回归；无条件可用的独立修改空间有限，这是后续深度优化的主要工程约束。

## 九、剩余差距与后续方向

当前唯一落后路径为 D=64 长 self（1.24–1.59× 慢于 Triton），差距来源明确：软件流水线缺失导致的同步开销，以及 LDS 读取粒度不足。

**如需进一步收敛差距，需投入以下工程：**

1. 建立独立于 `attn_gfx11.hip` 的 gfx10 kernel 源，引入 Triton 风格 2-stage 软件流水线，并配套重设计 LDS 布局（否则占用率下降将抵消流水线收益）；
2. `uint4` 等宽向量 K/V 读取，降低 LDS 读指令数；
3. 重新设计 `v_tile` 布局（如 `[N][D]` 或受控 padding）以消除 bank 冲突；
4. Persistent kernel，降低 launch 开销；
5. D=128 专项优化（QK 归约链更长，reduce 开销占比更高）。
