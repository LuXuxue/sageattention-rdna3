# SageAttention Native Backend 项目说明

本文档是 native（HIP）后端的项目技术说明，沉淀当前已定案的设计、分平台特性、已验证结论与踩坑点，供后续开发参照。

---

## 一、总体设计

### 1.1 平台特性对照

| 项目 | gfx110x（RDNA3） | gfx103x（RDNA2） |
| --- | --- | --- |
| 矩阵计算 | WMMA 16×16×16（fp16 / i8） | V_DOT4_I32_I8 / V_DOT2_F32_F16（无张量核） |
| int8 QK | WMMA i32（**fp16 的 2× 吞吐**） | `v_dot4c_i32_i8`（MAC 吞吐与 fdot2 持平） |
| fp16/bf16 PV | WMMA f32 | `v_dot2c_f32_f16` |
| bf16 | 原生支持 | **无硬件指令**，须 LDS 暂存时转 fp16 |
| LDS | 128 KiB/CU | 64 KiB/CU |
| CU 数（典型） | 12 CU（gfx1103） | 6 CU（gfx1035） |
| 验证状态 | 已优化并全量验证 | 已真机验证（gfx1035） |

> **gfx110x 的任何 WMMA 相关配置结论不可直接套用到 gfx103x**——两架构指令、LDS、CU 数差异大，参数选择需独立评估。

### 1.2 代码组织

```
csrc/
├── attn_gfx110x.cu / .h    # RDNA3：WMMA 主 kernel + dispatch + 公共辅助副本
├── attn_gfx103x.cu / .h    # RDNA2：V_DOT 主 kernel + dispatch + 公共辅助副本
├── pybind_gfx110x.cpp / pybind_gfx103x.cpp   # 分别产出 _qattn_gfx110x / _qattn_gfx103x pyd
├── mma_gfx11.h             # RDNA3 WMMA primitives
├── mma_gfx10.h             # RDNA2 V_DOT kernel（v1/v2/v2.2/v3）
├── attn_gfx10_new.h        # RDNA2 v3.5 16-lane 协作实验（默认不启用）
└── reduction_utils.cuh     # 公共 blockReduceMax / blockReduceMaxVec
```

两架构 .cu/.h 互不引用；公共辅助（mean / quant / v_transpose kernel + host 函数）各复制一份。`core.py` 按运行时设备 `gcnArchName` 懒加载对应 pyd 并注册 `torch.ops.sageattention`，同进程只加载一个 pyd。

### 1.3 共用核心设计

1. **转置 operand 布局**：QK 与 PV 都以转置形式输入（`S^T = K@Q^T`、`out^T = V^T@P^T`）。RDNA3 的动机是配合 WMMA 不对称 operand 布局并让 permlanex16 合并行归约；RDNA2 沿用同一计算顺序与输出转置写法。
2. **量化路径（int8 QK + fp16/bf16 PV）**：Q/K 量化为 int8（per-32-row / per-16-col group scale），PV 保持 fp16。输出 dtype 与输入一致。
3. **direct 路径（fp16/bf16 QK）**：跳过 quant/mean 辅助（固定 ~0.4ms），计算量小时占优。
4. **全局 V 转置 `SAGEATTN_VT_GLOBAL=1`（默认）**：V 预转置为 `V_T[B,H,D,N]`，消除 attn kernel 内 128 次列读；bf16→fp16 转换融合进 v_transpose。**V_T 的 n 维必须 padding 到 64 倍数**（防 v_frag_t 32B 直读越界 → NaN）。
5. **`smooth_k=False`（默认）**：跳过 mean kernel。randn 下减 mean 反而引入 fp16/bf16 舍入误差；仅当 K 有明显非零 DC 偏置时显式 `smooth_k=True`。
6. **quant 内核**：pass1 量化结果缓存到寄存器（不落 LDS），归约用 `blockReduceMaxVec`；**Threads=256**；BLK_Q=128/BLK_K=64。
7. **v_transpose grid 策略**：`total_tiles ≤ 4096 → grid=192`；`>4096 → clamp(total_tiles/10, 128, 1536)`。`SAGEATTN_VT_GRID` 可强制覆盖。**gfx103x 已同步此策略。**
8. **direct vs int8 分发**：按 headdim / kv_len / q_len / is_causal / tensor_layout 选择路径（§四）。

---

## 二、gfx110x（RDNA3）

### 2.1 指令与布局基础

- **WMMA 16×16×16 布局**：lane L 持有 C 输出 `C[2e+(L>>4)][L&15]`；A 按行、B 按列；每 lane 的 A/B 为 16 元素（v16h），C 为 8 元素（v8f）。转置后 `lane L 与 L^16` 持同一 softmax 行的偶/奇列，`v_permlanex16_b32` 一次合并整行归约。
- **permlanex16**：只支持 XOR-16；`__shfl_xor_sync` 替代反而慢 19%（6 VALU + 1 ds_bpermute，延迟 20–30 cycles）。
- **MFMA 不可用**：RDNA3（gfx1103）无 MFMA 指令，`+mai-insts` 导致 LLVM 后端段错误。
- 16B 向量写回：输出用 `uint4` 一次 8 个 fp16/bf16 连续写，要求 q strides 为 8 倍数。

### 2.2 当前 kernel 与参数

以 `attn_gfx110x.cu` 当前 dispatch 为准。模板形参顺序 `(HD, is_causal, BM, BN)`（block = BM/16×32 线程）。

| 路径 | 判定 | kernel / 参数 |
| --- | --- | --- |
| int8 D=64 self | `qo==kv` | **`wpe1_32`（每 warp 32 行）：BM=128/4 warps，BN=32**；`SAGEATTN_INT8_32=0` 回退 `wpe1` BM=128/BN=32 |
| int8 D=64 其余 | `kv<=77` | `wpe1` BM=64/BN=16 |
| int8 D=64 其余 | 其他 | `wpe1` BM=64/BN=32 |
| int8 D=128 self | `kv<6144` | `wpe2` BM=64/**BN=32** |
| int8 D=128 self | `kv>=6144` | `wpe2` BM=128/BN=64（8 warps） |
| int8 D=128 cross | — | `wpe2` BM=64/BN=32 |
| direct D=64 | `kv<=128` | `wpe2` BM=64/BN=16 |
| direct D=64 | self kv>128 | `wpe2` BM=64/BN=64 |
| direct D=64 | cross kv>128 | `wpe2` **BM=128/BN=32** |
| direct D=128 | 一律 | `wpe2` BM=64/BN=16 |
| VAE D=128 kv≥16384 | 附加 | BM=128 + LDS 缓存 V_T tile |

**配置要点**：
- D=128 int8 **BN=32 为默认**；BN=128 破坏正确性（cos≈0.62，已弃）。
- D=128 BM 自适应阈值默认 6144（`SAGEATTN_INT8_BM128_THR` 可调）。
- int8 V/OUT dtype 分离（方案 B）：bf16 输入 → V 预转 fp16 存 V_T、OUT 写 bf16。

### 2.3 已确认的性能边界

- **D=128 int8 attn ≈ 9.9–10 Top/s 恒定**，较 torch SDPA 快 1.3–1.7×。已接近 WMMA 执行单元吞吐 + softmax/score 寄存器管理的实际墙。
- **quant 带宽受限**（~90–100 GB/s），attn 计算密集型。
- **配置近局部最优**：D128 BM/BN/wpe 全扫描 ≤±2%；D64 direct BM 64→128 持平。
- 指令级微优化全部失败；**结构性优化（减冗余读、提 ILP、融合辅助 kernel、写回向量化）是仅有的有效路径**。

---

## 三、gfx103x（RDNA2）

### 3.1 指令基础

- 矩阵乘降级为 SIMD 点积链：int8 QK = `v_dot4c_i32_i8`，fp16 PV = `v_dot2c_f32_f16`。
- **int8 dot4 与 fp16 dot2 的 MAC 吞吐 ≈ 1:1**（实测 0.91–1.04×），int8 在 QK 上无吞吐优势。
- 无 bf16 硬件指令 → bf16 在 LDS 暂存时转 fp16。**triton bf16 路径会硬崩**（`LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16`），需 pre-skip。
- LDS 仅 64 KiB/CU，tile 尺寸选择受 occupancy 约束更紧。

### 3.2 kernel 变体

| 变体 | 说明 |
| --- | --- |
| v1 | BM=32（每 warp 1 行）—— D128 长 self 极慢，已被 v2 取代 |
| v2 | **BM=64、128 线程（4 warps）、每线程 NR 行独立点积链**、NW=4、BN=16/32 |
| v2.2 | PV 直接从全局 V_T 行读；长序列有效、短序列负，auto 仅 self 使用 |
| v3 | 16-lane 协作（半 warp 一行）：QK 归约正确，但 **PV 输出不完整**——direct 路径已默认禁用 |
| v4 | BM=64 BN=32，与 v2 结构相同；k_scale fix 后 D=128 默认使用此变体 |

### 3.3 当前默认路由

环境变量（默认值）：
- `SAGEATTN_GFX10_V2`（2=auto：self 且 qo≥512 用 v2，D64 direct 则 qo≥1024）
- `SAGEATTN_GFX10_VT_GLOBAL`（0=off；1=force v2.2；2=auto：self 且 qo≥1536）
- `SAGEATTN_GFX10_V3`（int8 默认 2=auto；direct 默认 0=禁用）

实际默认（源码 dispatch）：
- **int8 D=64**：self qo≥512 或 cross qo≥256 → v2 BN=16；其余 → v2.2→v2→v1。
- **int8 D=128**：默认 v4 BM=64 BN=32（k_scale fix 后精度与 BN=16 一致，快 13–17%）；kv≥6144 可切 BM=128。
- **direct fp16/bf16**：v3 默认禁用；auto 下 self qo≥1024 → v2（BN=32，BM=64），qo≥1536 且 vt_mode 开 → v2.2，其余 → v1。

### 3.4 gfx1035 实测基线

设备：gfx1035 (Rembrandt APU), 6 CUs, L2=2MB, 显存 ~14.7GB (共享)。理论峰值粗估：int8 dot4 ~6 TFLOPs, fp16 dot2 ~3 TFLOPs。正确性：pytest 42/42 通过。

| 用例 | path | 性能 | 备注 |
| --- | --- | --- | --- |
| D=64 int8 self 4096² | direct v2 | 0.44 T vs SDPA 0.45 T (0.97×) | 计算墙; 见下方**关键结论** |
| D=64 int8 self 6144² | direct v2 | 0.40 T vs SDPA 0.46 T (0.87×) | |
| D=64 int8 self 9216² | direct v2 | 0.38 T vs SDPA 0.42 T (0.89×) | |
| D=64 short self 1024² | direct v2 | 0.41 T vs SDPA 0.43 T (0.95×) | 阈值1016修正后从0.81×提升 |
| D=64 cross (kv=77/154) | direct v1 | **1.21–1.33×** SDPA | 短kv, 量化开销被并行分摊 |
| D=128 self/cross 4096² | direct v2 | 0.17 T vs SDPA 0.62–0.67 T (0.25–0.27×) | D128 硬件墙 |
| D=128 self 6144² | direct v2 | 0.18 T (881ms) | int8/v2.2 慢 1.8x/1.9x |
| D=128 self 9216² | direct v2 | 0.17 T (2006ms) | int8/v2.2 慢 1.8x/1.9x |
| D=128 self 16384² | direct v2 | 待测（启动极慢） | int8 v4 BM=128 崩溃（launch failure），int8 BM=64/v2.2 均更慢 |
| D=128 int8 4096² | int8 v4 BM=64 | 0.10 T (比 direct 慢 1.8×) | int8 对 D128 无优势（v2.2 PV from global V_T 更慢） |

**gfx1035 关键结论（本阶段实测修正）**：
- **D=64 int8 内核（v1/v2/v4）已真机验证正确且确定**：含 v4 BM=128 在 seq≥512（512/768/1015/1536/2048/4096）、NHD 与 HND、causal 与非 causal，cos>0.999 且重复运行逐位一致。早期"7 TFLOPS/head"数字来自不同测量口径。
- **曾误报 D64 int8 v4 BM=128 在 seq≥512 产出错误结果（cos≈0.03–0.1）**。经排查：那是已回退的 `__syncwarp()` 实验（QK→softmax 之间加同步反而破坏同波前 LDS 有序写读）造成的，base 内核无此 bug。**教训：RDNA2 同波前 LDS 写读天然有序，原"无同步"行为在实践上正确；不要想当然修"潜伏竞态"，须真机验证再改。**
- D=64 阈值调至 1016 使 1024+ 走 direct-v2 的性能收益仍成立（0.81×→0.95×）。
- D=128 int8 无吞吐优势（sdot4/fdot2 MAC 持平），direct 路径更快（0.17–0.18 T 即 6 CU 手写内核效率墙；int8 v4 BM=64 在 4096/6144/9216² 慢 1.8x, v2.2 慢 1.9x, v4 BM=128 在 16384² launch failure）。

---



### 3.4.1 gfx1035 本阶段优化更新（实测修正）

本阶段基于实测数据，对 gfx1035 D=64 调度做关键修正：

**D=64 self 路径调度**（实测 vs 旧报告）:
- 旧报告：`kv<1016` 走 int8（`v1`/`v2` 在 512-1023 病理性慢），`kv>=1016` 走 direct。
- 本阶段实测：当前 `v2 (BM=64)`、`v4 (BM=128)` 都无病态。所有长度（n=128~9216）int8 都比 direct 快 ~30x：
  | n | int8 v2 | direct |
  | --- | --- | --- |
  | 4096² | 128ms | 405ms |
  | 6144² | 283ms | (default) |
  | 9216² | 635ms | (default) |
- **新调度**：D=64 self 默认 `thr_d64=9999999`（即总是 int8）。删除 `_gfx103_d64_threshold()` 函数（已无用）。
- **跨 attn (q<kv)**：仍走 direct（避免 quant 开销）。

**v2 vs v4 内核对比**（D=64 int8，6144² HND）:
- `v2` (BM=64, BN=32, NW=4, NR=2, 128 threads)：**283ms**（当前默认）。
- `v4` (BM=128, BN=32, NW=4, NR=2, 256 threads)：540ms（被 `SAGEATTN_GFX10_V4=1` 强制）。
- 原因：v4 LDS 占用翻倍（`q_tile` 128*64=8KB vs 64*64=4KB）→ 占用率减半。已用 env 强制。

**v5/v6 实验**（未启用）:
- `attn_kernel_gfx10_i8_v5_t` (BM=128, BN=16, NW=8, NR=1, 128 threads)：单行/线程架构。SDXL D=64 HND 1024 实测 14ms（vs v2 8ms）反而更慢。原因是 1024-thread block layout 在 RDNA2 6 CU 上占用率不优。
- `attn_kernel_gfx10_i8_v6_d128_t` (BM=128, BN=32, NW=4, NR=2, 256 threads)：D=128 专用。**int8 路径工作正常，direct 路径在 NHD 时 cos=0.6 有 bug**（q_tile 直接从 global 加载的边界同步问题）。已暂时禁用 direct v6，回退到 v2。

**D=128 调度**（实测 vs 旧报告）:
- 实测：D=128 NHD 4096²，int8=697ms vs direct=386ms → direct 仍快 1.8x（与旧报告一致）。
- D=128 NHD 短序列（kv=512）：native 52ms vs Triton 3.3ms（15x 慢）。原因：Triton 在 `kv<=512` 走 fp16 路径（FP16_DIRECT_THRESHOLD=512），native 总是 direct。需新增 `SAGEATTN_DIRECT_THRESHOLD_D128_FP16=512` 实验。

### 3.4.2 gfx1035 后续优化方向（暂未实现）

| 方向 | 预期收益 | 风险/备注 |
| --- | --- | --- |
| **Triton-style 软件流水线 (num_stages=2 async copy)** | 高（预估 D=64 int8 4-8x 加速） | 需要 LDS 双缓冲（K+V ×2），重写主循环 |
| **1024-thread block layout (Triton BLOCK_M=128, BLOCK_N=16, NW=8)** | 高 | 寄存器压力增大 (CL=HD=64 fp32/thread) |
| **D=128 direct 短序列 fp16 路径** | 中（D=128 NHD 512: 52ms → ~6ms） | 调度加 `kv<=thr_d128_fp16` 分支 |
| **LDS swizzling (ds_swizzle_b32)** | 中（~10-20%） | 替换简单 +1 padding |

### 3.4.3 gfx1035 当前已验证结论

- pytest 70/70 通过
- D=64 int8 path cos > 0.99 for all test cases (HND/NHD, causal/non-causal, kv 128-9216)
- D=128 direct path cos > 0.99 for same range
- 重复运行确定（相同输入产生逐位一致输出）


## 四、direct/int8 分发

`core.py` 按 headdim/kv_len/q_len/is_causal/tensor_layout 选择路径。阈值经 E2E 交叉点实测确认最优。

### gfx1103

| headdim / 情形 | 路径 | 条件 |
| --- | --- | --- |
| D=64 HND self | int8 | kv > 2048 |
| D=64 NHD self | int8 | kv > 3072 |
| D=64 causal | direct | kv ≤ 6144 |
| D=64 cross | direct | kv ≤ 6144 |
| D=128 self / causal | direct | kv ≤ 2048 |
| D=128 cross | direct | kv ≤ 4096 |

### gfx1035

| headdim / 情形 | 路径 | 条件 |
| --- | --- | --- |
| D=64 self HND | int8 | N < 1016（**本阶段实测修正: 旧768 使769-1023走direct-v1(0.02T)而非int8(0.13T), 现统一1016**） |
| D=64 self NHD | int8 | N < 1016（**本阶段实测修正: 旧1024 使1024走int8错误, 现1016使1024走direct**） |
| D=64 cross | direct | kv ≤ 1016（短kv） |
| D=128 all | direct（优先） | — | **本阶段实测：4096/6144/9216² int8v4 慢 1.8x, v2.2 慢 1.9x, v4 BM=128 在 16384² launch failure** |

> **注意**：gfx1035 的阈值与 gfx1103 不同。**关键教训：`SAGEATTN_DIRECT_THRESHOLD_D64=0` 在 gfx103 上解析为 d64_default（HND=768/NHD=1016）而非0，因为 `int("0") or default` 中字符串"0"为真值——故环境变量"0"不能强制 int8。** 要真正强制 int8：D64 用大阈值 `"9999999"`（`use_direct=(kv>thr)` 恒假），D128 用负阈值 `"-1"`（`use_direct=(kv<=thr)` 恒假）。此前测试套件用 env=0 对长序列实为强制 direct 而非 int8，掩盖了 k_scale per-column / BN=32 的 int8 路径（现已改 9999999/-1 真强制并回归）。

环境变量覆盖（实验用，默认值即生产最优）：
- `SAGEATTN_DIRECT_THRESHOLD_D64`（HND=2048/NHD=3072）、`_D64_CAUSAL`（6144）、`_D64_CROSS`（6144）、`_D128`（2048）、`_D128_CROSS`（4096）
- `SAGEATTN_QUANT_BLK`：1=auto（128/64），128=强制 128/64，64=强制 64/32，0=旧 MIN_BLK 逻辑
- `SAGEATTN_INT8_32`（0=关闭 D64 self 每 warp 32 行 kernel）
- `SAGEATTN_INT8_BN128`（仅 D128 BM=64 分支）、`SAGEATTN_INT8_BM128` / `_BM128_THR`（BM 自适应，默认 6144）、`SAGEATTN_INT8_WPE`（1/2/4）
- `SAGEATTN_FP16_BM/FP16_BN`、`SAGEATTN_BF16_BM/BF16_BN`（direct 实验覆盖）
- `SAGEATTN_VT_GRID`（v_transpose，0=auto）
- gfx103x：`SAGEATTN_GFX10_V2` / `_VT_GLOBAL` / `_V3`
- `SAGEATTN_BACKEND`（triton/native，import 时读取一次）

---

## 五、测量方法论

所有性能结论必须遵循（否则无效）：
1. **同进程内交错轮转**：被测实现与参考实现在同一进程交替运行。
2. **充分预热**：≥20 次 warmup。
3. **取中位数**：≥50 次迭代取 median。
4. **iGPU 波动**：共享内存 iGPU 长时间运行因温度/时钟下降，小 case 跨 run 波动可达 ±25%；**跨会话绝对时间不可比**，只有同会话内比率口径成立。
5. **固定随机种子**：`torch.manual_seed(0)` 保证可复现。

正确性基准：与 FP32 SDPA 对比 cos / maxdiff（参考实现必须 `transpose(1,2)` 到与 kernel 输入一致的 HND/NHD 布局；早期多次因参考布局用错而误判）。

---

## 六、编译

```bash
GPU_ARCHS=gfx1103 pip install -e . --no-build-isolation          # 仅 RDNA3
GPU_ARCHS=gfx1035 pip install -e . --no-build-isolation          # 仅 RDNA2
GPU_ARCHS=gfx1103,gfx1035 pip install -e . --no-build-isolation  # 同时（或省略默认）
SAGEATTN_SKIP_BUILD=1 pip install -e . --no-build-isolation      # 仅装 Python 层
```

关键 HIP 编译选项：`-O3 -ffast-math -fgpu-flush-denormals-to-zero -DSAGEATTN_VT_GLOBAL=1`，并含 `-mllvm --lsr-drop-solution=1 / -enable-post-misched=1 / -amdgpu-early-inline-all=true / -amdgpu-function-calls=false / -amdgpu-max-memory-clause=32 / -amdgpu-vgpr-index-mode=1`。

---

## 七、踩坑点（后续开发必读）

### 7.1 gfx110x

1. **permlanex16 参数格式**：`v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]`，`%2`/`%3` 为 s/n 字面量。早年误判该指令不可用，实为内联汇编参数错误。
2. **MFMA 不可用**：`+mai-insts` 一开，LLVM AMDGPU 后端 WMMA codegen 直接段错误。
3. **V_T n 维必须 padding 到 64 倍数**：否则最后 kv-tile 的 v_frag_t 32B 直读越界 → NaN。
4. **16B 写回对齐**：q strides 须为 8 倍数，非 contiguous 输入触发 16B 写 UB。
5. **`blockReduceMax` 循环调用必须自同步**：D=64 quant 的 RATIO 循环同 block 多次调用该共享归约，读/写之间无 barrier 会跨调用复用 shared 而产生**非确定性精度差（maxdiff≈1.2e-2）**；函数末尾补 `__syncthreads()` 即修复。
6. **quant 的 Threads 必须与 dispatch `dim3 block()` 一致**：曾将 kernel 内 `constexpr Threads=512` 而 launch 仍 `block(256)`，导致 K/Q 各丢一半组、scale≈0、cos≈0.72 的确定性 bug。**保持 T=256**。
7. **wmma_* builtin 仅在 `__GFX11__` 下有效**：对其他架构返回零值实现（编译通过、运行失效）。
8. **stable torch library 单次注册**：同名 schema 重复注册报错；靠 core.py 懒加载保证同进程只 import 一个 pyd。
9. **`is_causal` 是运行时参数**：必须 host 端 if/else 生成编译期模板常量，不能直接当模板参数传。
10. **namespace 边界**：host/dispatch/公共函数必须全局作用域，否则 pybind 链接 LNK2001。
11. **`SAGEATTN_VT_GLOBAL=0` 不再维护**：删除/关闭该分支重建会硬崩溃；当前构建只支持 VT=1。

### 7.2 gfx103x

1. **无 WMMA/MFMA**：矩阵乘走 V_DOT 链，每 lane 独立累加。
2. **bf16 无硬件指令**：须 LDS 暂存转 fp16；增加 LDS 占用与转换开销。**triton bf16 路径硬崩**（不可捕获），需 pre-skip。
3. **v3 PV 输出不完整**：16-lane 协作方案 QK 归约正确，但 PV 每 lane 只写部分列 → 输出缺列。direct 路径已默认禁用。
4. **LDS 只有 64 KiB/CU**：BM=128 等会使 occupancy 崩溃，tile 选择须更保守。
5. **LDS bank conflict 在 RDNA2 上惩罚极大**：未 padding 时 s_tile 8-way bank conflict 导致 8× 延迟，padding 仅 +512 bytes 即获得 5–20× 提升（见 §八）。
6. **int8 dot4 与 fp16 dot2 MAC 吞吐持平**：D=128 int8 无吞吐优势，只增加量化开销；D=128 应优先 direct 路径。
7. **iGPU 热漂移严重**：跨会话绝对 ms 不可比；必须同进程交错 A/B。

### 7.3 通用

1. **`__shfl_xor_sync` 延迟高**：编译为 ~6 VALU + 1 ds_bpermute（20–30 cycles），能用 permlanex16 / lane-reduce 替代就替代。
2. **用 `exp2f`**：softmax 一律 `exp2f`（乘 log2e 预缩放），比 `expf` 快 ~1 cycle。
3. **`getENV` deprecation 警告**（`_CRT_INSECURE_DEPRECATE`）在 Windows 无害，hipcc 不报 error。
4. **架构检测**：`torch.cuda.get_device_properties().gcnArchName` 返回 `"gfx1103"` 等；core.py 优先字符串前缀判断。
5. **自己做实验用独立扩展**：A/B 对比时按参数化模块名单独编译，避免 torch.ops 注册冲突与缓存。
6. **iGPU 上并发 stream 无收益**：APU 共享内存带宽，vt‖quant 双 stream 重叠反而慢 7–14%。

---

## 八、已验证的失败路线（避免重复投入）

### 8.1 gfx110x

**A. 指令级微优化（全败或可忽略）**

| # | 尝试 | 结论 |
| --- | --- | --- |
| A1 | `__shfl_xor_sync` 替代 permlanex16 | 慢 19%（走 ds_bpermute） |
| A2 | 内联汇编 hand-craft permlanex16 / shfl | 无法复刻 exec mask 边界管理 |
| A3 | `expf` 替代 `exp2f`+log2e | exp2f 更快 1 cycle |
| A4 | MFMA / `+mai-insts` | RDNA3 无 MFMA，编译崩 |
| A5 | 减少 `__syncthreads` | 已最少（每 kv-tile 1 次） |
| A6 | `global_load_lds` K/V prefetch | 与现有 LDS 缓存等价，无收益 |
| A7 | occupancy 1→4 波/EU | 计算密集 kernel 无收益 |

**B. 数据布局 / 量化**

| # | 尝试 | 结论 |
| --- | --- | --- |
| B1 | fp8e5m2 存 V | 精度 cos<0.95，删除 |
| B2 | int4 QK | WMMA 不支持，软件模拟 3–5× 慢 |
| B3 | per-head scale | 无精度提升、quant 开销增 |
| B4 | smooth_k=True 设为默认 | 随机数据下慢 0.5–3.5%，保持 False |
| B5 | 在线/双 stream 重叠 vt‖quant | 共享带宽争抢，-7~-14% |
| B6 | quant T=256→512 | 中性；保持 256 |

**C. 结构优化（失败项）**

| # | 尝试 | 结论 |
| --- | --- | --- |
| C1 | v_transpose 融合进 attn | LDS 占用翻倍 occupancy 崩；V 全局流量放大 32–64× |
| C2 | LDS 缓存 V_T tile PV 用于 D=64 | D=64 v_frag 仅 8KB，浪费 occupancy；**仅 D=128 kv≥16384 有效** |
| C3 | wpe2/wpe4 流水线 | 收益 <1% |
| C4 | fp16 双子块（direct_32） | VGPR 溢出，0.34–0.56× |
| C5 | D128 双子块 | VGPR 溢出，12.2–12.7× 灾难 |
| C6 | V 片段跨子块复用 | 中性 |
| C7 | D128 换每 warp 双子块 kernel | 中性 |
| C8 | V 全局预取到寄存器 | +32 VGPR 伤 occupancy，劣化为主 |

**D. 调度 / 分发**

| # | 尝试 | 结论 |
| --- | --- | --- |
| D1 | 全路径统一 int8 | 短序列 quant 辅助 > int8 吞吐收益 |
| D2 | 全路径统一 direct | 长 self 慢 30–50% |
| D3 | 运行时 auto-tune | 开销 > 收益 |
| D4 | dispatch 融合（vt/quant+attn 单 op） | 中性偏慢（0.82–1.06×） |
| D5 | D128 BM/BN/wpe 全扫描 | 全平坦 ≤±2%；BN=128 破正确性（cos=0.62） |
| D6 | D64 int8 32w BK BN=32→64 | 回退 18–20%，BN=32 最优 |
| D7 | D64 direct BM=64→128 | 持平 |

**E. 已删除 / 不再维护**：fp8 存储 V；`SAGEATTN_VT_GLOBAL=0`；`attn_gfx10_new.h` v3.5。

### 8.2 gfx103x

| # | 尝试 | 结论 |
| --- | --- | --- |
| R1 | MFMA 路径 | RDNA2 无此指令 |
| R2 | v1（每 warp 1 行） | D128 长 self 极慢，被 v2 取代 |
| R3 | v3 16-lane 协作（fp16/bf16） | 只覆盖半行，**输出不完整** |
| R4 | v3 16-lane 协作（int8 PV） | 每 lane 仅写 4 列，**输出不完整** |
| R5 | BM=128 | LDS 占用翻倍，occupancy 崩 |
| R6 | 双 warp 协作 QK（NR=4） | LDS 翻倍，sdot4 延迟未有效隐藏 |
| R7 | D=128 int8 v4 BM=128 | 16384² 触发 `hipErrorLaunchFailure`（非 LDS OOM，路径本身不可用） |
| R8 | D=128 v2.2（PV from global V_T） | 4096–6144² 比 direct v2 慢 1.9x；内存访问模式放大 |

### 8.3 踩坑：早期错误结论（已纠正）

以下结论曾在开发过程中被误认为正确，后经真机验证推翻：

- **"D=128 BN=32 结构性不精确"**：最初发现 D=128 BN=32 精度差（cos≈0.97），误以为是 BN=32 的架构固有问题。**根因是 k_scale 粒度 bug**——所有 int8 kernel 在 BN-tile 开头读一次 k_scale，但 per_block_int8 的 k_scale 按 MIN_BLK_K=16 行分组，BN=32 的 tile 横跨两个 scale group，后 16 列被错误乘了前 16 列的 k_scale。修复 per-16 列读取后，D=128 BN=32 精度完全恢复（cos=0.9999）。**教训：精度问题先排查 scale/量化粒度，再怀疑架构差异。**

- **"D=64 BN=16 优于 BN=32"**：在 k_scale bug 修复前，D=64 BN=32 也有同样 bug 但测试未覆盖（42 tests 最大 N=2048 → 走 direct 路径）。修复后 D=64 BN=32 与 BN=16 精度一致，且 BN=32 性能更优。**教训：测试覆盖不足时，局部结论不可外推。**

- **"gfx103x v3 16-lane 协作可用"**：代码注释对 int8 v3 是否正确自相矛盾（一处说"BN=16 精度同 triton"，另一处说"int8 PV 输出不完整"）。**真机验证确认：v3 PV 输出确实不完整，direct 路径已默认禁用。** 代码注释矛盾时必须真机验证。

### 8.4 规律

- **指令级微优化全部失败**：编译器已充分调度。
- **结构性优化是唯一有效路径**：减冗余读（V 转置、LDS 缓存）、提 ILP（大 block、warp 内多子块）、融合辅助 kernel、写回向量化。
- **精确度方案均失败于精度或硬件支持**：fp8/int4/per-head scale 全灭；int8 per-group scale + fp16 PV 是当前最优点。
- **寄存器/占用是上限约束**：任何以"增加寄存器/LDS"换 ILP/延迟的方案都被 VGPR/occupancy 反噬。

---

## 九、剩余方向

### 9.1 用例图例（benchmark_attn.py）

| 用例 | 含义 |
| --- | --- |
| SDXL01/07/13 | D=64 int8 self 4096²/6144²/9216² |
| SDXL04/10/16 | D=64 direct self 1024²/1536²/2304² |
| SDXL02/05/08/11/14/17、03/06/09/12/15/18 | D=64 direct cross ×77 / ×154（text） |
| Anima01/03/05 | D=128 int8 self 4096²/6144²/9216²（bf16） |
| Anima02/04/06 | D=128 direct cross 4096×512/6144×512/9216×512 |
| SDXLVAE01 / AnimaVAE01 | D=128 int8 self 16384²（fp16 / bf16，VAE） |

### 9.2 剩余差距来源

1. **v_transpose 固有成本**（native 独有）：约 0.15–0.5ms/调用，短序列占比大；已做 grid-cap 优化，无法融合进 attn。
2. **int8 路径 quant 辅助累计**：quant+vt 共约 0.4–1.1ms；quant 已近带宽上限。
3. **短序列固定开销**（两后端共性）：kernel launch+同步，绝对差 0.03–0.12ms，占比大。

### 9.3 剩余方向

| 方向 | 平台 | 预期收益 | 风险 / 备注 |
| --- | --- | --- | --- |
| **D=64 长 self 专用内核重构** | gfx1035 | 高 | 现 v2 系列在 RDNA2 密度/占用约束下不足；新结构工程量大 |
| **gfx103x v3 裁决 + 路由修正** | gfx1035 | — | v3 PV 输出不完整已确认；需确认 int8 auto 路由是否仍选 v3 |
| **v_transpose grid 策略优化** | gfx1035 | 低–中 | CU 数不同，参数需重测 |
| int8 quant 再提速 | 两平台 | 低 | quant 带宽受限 |
| 消除 direct 的 v_transpose 固有成本 | 两平台 | 中 | attn 内转置失败于 LDS/带宽放大，无新思路 |
| per-stage 流水线 / 更激进 WPE | gfx110x | 低 | 收益 <1% |
| fp8 V 存储 | 两平台 | 不可行 | 精度不达标 + gfx1103 无 fp8 硬件 |
| bf16 硬件路径 | gfx103x | 不可行 | gfx1035 无 bf16 指令 |

**结论**：gfx110x 各主 kernel 已处于硬件实际墙（WMMA 吞吐 / 带宽双限），无安全增量；后续增量工作聚焦 gfx1035（先验证、再针对 D=64 长 self 重构）。
