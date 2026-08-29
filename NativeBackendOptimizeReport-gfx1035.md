# gfx1035 (RDNA2) HIP Native 移植工作日志

本文档记录了将 sageattention native 后端从 gfx1103 (RDNA3) 移植至 gfx1035 (RDNA2) 的技术方案、实现细节、踩坑记录及性能分析，旨在为后续开发提供准确的技术基线与避坑指南。

## 一、 硬件环境与工具链基线

### 1.1 目标硬件规格 (gfx1035)
| 项目 | 规格说明 |
| --- | --- |
| **GPU 架构** | AMD Radeon Graphics (gfx1035, RDNA2, Rembrandt, 消费级 iGPU) |
| **计算单元** | 6 CU |
| **LDS 容量** | 64 KiB / CU |
| **显存** | 共享系统内存 (Unified=1, 约 14.4 GiB) |
| **时钟频率** | 2200 MHz |
| **矩阵/点积指令** | 支持 `V_DOT4_I32_I8` / `V_DOT2_F32_F16`；**无 WMMA** |

### 1.2 工具链与基线确认
*   **编译环境**：ROCm SDK (hipcc)，支持多 target 编译（可同时编译 gfx1103 与 gfx1035）。
*   **基线对标**：Triton 后端在 gfx1035 上正常运行，作为正确性与性能的对标基线。
*   **现状确认**：现有的 gfx1103 预编译二进制 (`_qattn_gfx11.pyd`) 在 gfx1035 上因 ISA 不兼容无法运行，确认了本次原生移植的必要性。

---

## 二、 架构差异与指令集实测

经过严格的指令级探针实测，明确了 RDNA2 (gfx1035) 与 RDNA3 (gfx1103) 在矩阵计算指令上的根本差异，并确立了最终的指令映射路线：

| 计算任务 | gfx1103 (RDNA3) 指令 | gfx1035 (RDNA2) 最终确认指令 | 备注说明 |
| --- | --- | --- | --- |
| **int8 QK** | WMMA i32 16x16x16_iu8 | `v_dot4c_i32_i8` (`__builtin_amdgcn_sdot4`) | SIMD 点积风格，每 lane 独立累加 |
| **fp16/bf16 PV** | WMMA f32 16x16x16_f16 | `v_dot2c_f32_f16` (`__builtin_amdgcn_fdot2`) | RDNA2 无张量核，fp16 矩阵乘降级为 V_DOT2 |
| **矩阵核** | 支持 WMMA | **无 (消费级 RDNA2 缺失)** | 彻底放弃 MFMA/WMMA 方案 |

*注：gfx1035 不支持 bf16 硬件指令，bf16 输入需在 LDS 暂存时转换为 fp16 处理。*

---

## 三、 移植策略与工程实现

### 3.1 架构宏分发与多 Target 构建
为保证 gfx1103 现有 WMMA 逻辑零改动，采用编译期架构宏分发策略：
*   使用 `#if defined(__GFX11__)` 包裹原有 WMMA kernel。
*   使用 `#if defined(__GFX10__)` 路由至 gfx1035 专属 kernel。
*   **构建系统改造**：`setup.py` 支持通过 `--offload-arch` 同时编译双架构，单 wheel 兼容双设备。

### 3.2 gfx1035 Kernel 核心设计
采用 **per-lane-row + LDS 共享布局**，优先保证正确性与内存访问效率：
*   **线程块配置**：128 线程/block，处理 32 行 × 4 列组 (BM=32, NW=4)。
*   **QK 计算**：使用 `v_dot4c_i32_i8` (int8) 或 `v_dot2c_f32_f16` (fp16)。每行 4 线程按列切分，无跨 lane shuffle。
*   **Softmax**：依赖 LDS 进行跨线程归约 (`s_tile` / `p_tile`)。
*   **PV 计算**：V 矩阵转置为 `[B,H,D,N]` 并拷入 LDS，每线程使用 `v_dot2c_f32_f16` 沿 kv-pair 累加输出列。
*   **寄存器压力控制**：重循环使用 `#pragma unroll 1`，严格控制每 lane 的 fp32 累加器数量，避免 LLVM 后端溢出。

---

## 四、 关键踩坑与修正记录 (避坑指南)

在移植过程中 encountered 多个导致编译崩溃或逻辑错误的问题，以下为根因分析与最终修正方案：

### 4.1 Softmax 缩放因子重复计算
*   **现象**：初期输出结果放大约 11 倍，cos 相似度骤降至 0.77。
*   **修正**：量化阶段已将 `sm_scale * log2e` 折入 score 中。Softmax 计算时直接使用 `exp2(score - row_m)` 即可，**切勿再次乘以 `log2e`**。

### 4.2 Tensor Layout 与 GQA 映射错误
*   **现象**：int8 路径在 NHD 输入时越界产生 NaN；GQA (Grouped-Query Attention) 用例输出完全错误。
*   **修正**：
    *   int8 dispatch 中，q/k 恒为 **HND 布局**（无论输入 layout），必须严格按 HND 读取 stride。
    *   GQA 头映射应使用 `h / (q_heads / kv_heads)`（即 `h / groups`），而非 `h % kv_heads`。

### 4.3 Streaming 变体在 iGPU 上性能劣化
*   **现象**：尝试去除 LDS 和 barrier 的 streaming 方案（1 线程 1 行，直接读全局内存），性能反而大幅下降（延迟增加数倍）。
*   **修正**：本机 iGPU 共享内存延迟高，无 LDS 暂存会导致 K/V 全局内存被重复读取，性能受内存带宽与延迟主导。**证明 LDS 协同暂存是 RDNA2 iGPU 的刚需**。

---

## 五、 正确性验证与性能现状

### 5.1 正确性验证
*   **测试覆盖**：`test_sageattn_rdna3.py` 36/36 用例全部通过（涵盖 fp16/bf16、HND/NHD、causal、cross、GQA、smooth_k 等）。
*   **精度对齐**：int8 精度与全精度 fp16 SDPA 对比，D64 cos 约 0.955，D128 cos 约 0.973。此误差为 int8 量化固有误差，native 后端与 Triton 后端输出 cos=1.0 精确一致。

### 5.2 性能现状与瓶颈分析
当前 per-lane-row 实现正确性达标，但性能落后 Triton 3-13x（硬件峰值利用率仅约 3-11%，而 Triton 可达 35-80%）。
**核心瓶颈**：
1.  **同步开销**：每 KV-tile 需 3 次 `__syncthreads()`，大量 barrier 导致流水线排空开销巨大。
2.  **ILP 不足**：点积指令依赖链串行，指令级并行度低。
3.  **占用率低**：块尺寸较小且 LDS 占用较大。
*注：指令级微优化（如 unroll 内层循环）仅带来 1.1-1.5x 的微弱提升，无法解决结构性瓶颈。*

---

## 六、 后续优化方向

鉴于指令级微优化已达瓶颈，后续开发需聚焦于**结构性重构**：

1.  **Tile-based 布局重构**：废弃 per-lane-row 方案，重构为 tile 布局。使每个 lane 持有 BM×BN 子块的 V_DOT4/V_DOT2 片段，以大幅提升 ILP 和寄存器利用率。
2.  **对齐 Triton 映射**：深入分析 Triton 在 gfx1035 上的 ISA，精确对齐其 lane 到输出元素的映射关系。
3.  **软件流水线**：引入类似 Triton `num_stages=2` 的软件流水线机制，提前取下一 KV tile 以隐藏全局内存/LDS 访问延迟。
4.  **LDS Bank 冲突消除**：在 LDS 布局中引入 Padding，消除当前 `v_tile [D][N]` 布局潜在的 Bank 冲突。