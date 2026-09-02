# SageAttention Native Backend 优化与开发说明

本文档是 native（HIP）后端的项目技术说明，沉淀当前已定案的设计、分平台特性、已验证的结论与踩坑点，供后续开发参照。

**平台现状（必须区分）**：
- **gfx110x（RDNA3）**：WMMA 16×16×16 张量核实现，已完成系统性优化与全量验证，结论可信、性能数字即当前基线。
- **gfx103x（RDNA2，后续主要开发/验证平台）**：V_DOT4/V_DOT2 SIMD 实现，仅编译通过、**从未在 RDNA2 真机运行**，其分派/参数源自移植期设计且代码注释存在自相矛盾，相关结论一律标注"未验证"，须在 gfx1035 上重新评估（见 §三）。

两架构独立源文件、独立 kernel，共用 Python 层与 host 分发逻辑。文中性能结论遵守统一测量方法论（§五）；基线数字为当前代码在 gfx1103 上的实测（§9.2，并附 pytest 42/42 与全量正确性核验）。

---

## 一、总体设计

### 1.1 平台特性对照

| 项目 | gfx110x（RDNA3） | gfx103x（RDNA2） |
| --- | --- | --- |
| 矩阵计算 | WMMA 16×16×16（无 MFMA，`fp16` 16×16×16_f16、i8 16×16×16_iu8/s32） | V_DOT4_I32_I8 / V_DOT2_F32_F16（无张量核，每 lane 独立累加） |
| int8 QK | WMMA i32（**fp16 的 2× 吞吐**） | `v_dot4c_i32_i8` |
| fp16/bf16 PV | WMMA f32 | `v_dot2c_f32_f16` |
| bf16 | 原生支持 | **无硬件指令**，须 LDS 暂存时转 fp16 处理 |
| LDS | 128 KiB/CU | 64 KiB/CU |
| 验证状态 | 已优化并全量验证 | 仅编译通过，未真机运行 |

### 1.2 代码组织

```
csrc/
├── attn_gfx110x.cu / .h    # RDNA3：WMMA 主 kernel + dispatch + 公共辅助副本
├── attn_gfx103x.cu / .h    # RDNA2：V_DOT 主 kernel + dispatch + 公共辅助副本
├── pybind_gfx110x.cpp / pybind_gfx103x.cpp   # 分别产出 _qattn_gfx110x / _qattn_gfx103x pyd
├── mma_gfx11.h             # RDNA3 WMMA primitives（permlanex16、wmma_*、p_frag）
├── mma_gfx10.h             # RDNA2 V_DOT kernel（v1/v2/v2.2/v3）
├── attn_gfx10_new.h        # RDNA2 v3.5 16-lane 协作实验（默认不启用）
└── reduction_utils.cuh     # 公共 blockReduceMax / blockReduceMaxVec
```

两架构 .cu/.h 互不引用；公共辅助（mean / quant / v_transpose kernel + host 函数）各复制一份。`core.py` 不硬加载 pyd，首次调用时按运行时设备 `gcnArchName`（gfx11xx→`_qattn_gfx110x`，gfx10xx→`_qattn_gfx103x`）懒加载并注册 `torch.ops.sageattention`，同进程只加载匹配设备的一个 pyd，op 注册不冲突。

### 1.3 共用核心设计

1. **转置 operand 布局**：QK 与 PV 都以转置形式输入（`S^T = K@Q^T`、`out^T = V^T@P^T`）。RDNA3 的动机是配合 WMMA 不对称 operand 布局（A 按行、B 按列）并让 lane L 与 L^16 持有同一 softmax 行的偶/奇列（permlanex16 一次合并行归约，见 §2.1）；RDNA2 沿用同一计算顺序与输出转置写法。
2. **量化路径（int8 QK + fp16/bf16 PV）**：Q/K 量化为 int8（per-32-row / per-16-col group scale），PV 保持 fp16。输出 dtype 与输入一致。
3. **direct 路径（fp16/bf16 QK）**：跳过 quant/mean 辅助（固定 ~0.4ms），计算量小（causal / cross 短 q）时占优。
4. **全局 V 转置 `SAGEATTN_VT_GLOBAL=1`（默认）**：V 预转置为 `V_T[B,H,D,N]`，PV 从 V_T 行读（b128），消除 attn kernel 内 128 次列读；bf16→fp16 转换融合进 v_transpose，一次 kernel 完成。**V_T 的 n 维必须 padding 到 64 倍数**（防 v_frag_t 32B 直读越界 → NaN；padding 区填 0）。
5. **`smooth_k=False`（默认）**：跳过 mean kernel。randn 下减 mean 反而引入 mean 值自身的 fp16/bf16 舍入误差（详见 §8.1 B4）；仅当 K 有明显非零 DC 偏置时显式 `smooth_k=True`（mean kernel 32B 向量读，带宽 38→68–77GB/s）。
6. **quant 内核**：pass1 量化结果缓存到寄存器（`uint4 reg[PPT][2]`，不落 LDS），归约用多值向量归约 `blockReduceMaxVec`（barrier 数不随 RATIO 增长）；**Threads=256**；BLK_Q=128/BLK_K=64（D64 与 D128 一致）。
7. **v_transpose grid 策略（当前策略）**：`total_tiles ≤ 4096 → grid=192`（每 block 多 tile，省固定开销）；`>4096 → clamp(total_tiles/10, 128, 1536)`（原 grid=total 每 block 1 tile 在部分 GPU 状态下调度/同步开销主导，实测慢 40–60%；clamp 后各状态均不劣化）。`SAGEATTN_VT_GRID` 可强制覆盖。**注意：该策略已在 gfx110x 生效，gfx103x 源仍是旧 grid=total 逻辑，待同步（§3.4）**。
8. **direct vs int8 分发**：按 headdim / kv_len / q_len / is_causal / tensor_layout 选择路径（§四）。

---

## 二、gfx110x（RDNA3）

### 2.1 指令与布局基础

- **WMMA 16×16×16 布局**（probe 实测）：lane L 持有 C 输出 `C[2e+(L>>4)][L&15]`；A operand 按行提供、B operand 按列提供；每 lane 的 A/B operand 为 16 元素（v16h），C 为 8 元素（v8f）。转置 QK 后 `lane L 与 L^16` 持同一 softmax 行的偶/奇列，`v_permlanex16_b32`（XOR-16 跨半波交换，纯寄存器操作）一次合并整行归约。
- **permlanex16**：只支持 XOR-16，不能做半波内（XOR-8/4/2/1）通信；`__shfl_xor_sync` 替代反而慢 19%（6 VALU + 1 ds_bpermute，延迟 20–30 cycles）。
- **MFMA 不可用**：RDNA3（gfx1103）无 MFMA 指令，`+mai-insts` 导致 LLVM 后端段错误；WMMA 是唯一张量路径。
- 16B 向量写回：输出用 `uint4` 一次 8 个 fp16/bf16 连续写，要求 q strides 为 8 倍数（core.py 断言；非 contiguous 输入 16B 写会 UB）。

### 2.2 当前 kernel 与参数（当前基线，已核验）

以 `attn_gfx110x.cu` 当前 dispatch 为准。模板形参顺序一律 `(HD, is_causal, BM, BN)`（block = BM/16×32 线程）。

| 路径 | 判定 | kernel / 参数 |
| --- | --- | --- |
| int8 D=64 self | `qo==kv` | **`wpe1_32`（每 warp 32 行）：BM=128/4 warps，每 warp 2 子块共享 k_frag，BN=32**，默认启用；`SAGEATTN_INT8_32=0` 关闭后回退 `wpe1` BM=128/BN=32 |
| int8 D=64 其余 | `kv<=77` | `wpe1` BM=64/BN=16 |
| int8 D=64 其余 | 其他（int8 cross） | `wpe1` BM=64/BN=32 |
| int8 D=128 self | `kv<6144` | `wpe2` BM=64/**BN=32** |
| int8 D=128 self | `kv>=6144`（含 VAE ≥16384） | `wpe2` BM=128/BN=64（8 warps，K/V 重读减半） |
| int8 D=128 cross | — | `wpe2` BM=64/BN=32 |
| direct fp16/bf16 D=64 | `kv<=128` | `wpe2` BM=64/BN=16 |
| direct fp16/bf16 D=64 | self（kv>128） | `wpe2` BM=64/BN=64 |
| direct fp16/bf16 D=64 | cross（kv>128） | `wpe2` **BM=128/BN=32**（减少 K/V 冗余读，实测快 4–14%） |
| direct fp16/bf16 D=128 | 一律 | `wpe2` BM=64/BN=16 |
| VAE（D=128 kv≥16384） | 附加 | BM=128 基础上启用 **LDS 缓存 V_T tile**：PV 改从 LDS 行读，消除 8× L1/L2 冗余（128KB→16KB/迭代） |

**配置要点（均经同进程 A/B 实测确认）**：
- D=128 int8 的 **BN=32 为默认**：cross 与中短 self 优于 BN=64（score_cache 4×8 float/线程 拖累寄存器占用）；BN=128 破坏正确性（cos≈0.62，已弃）；`SAGEATTN_INT8_BN128`（16/32/64/128）可覆盖**但仅对 BM=64 分支生效**，BM=128 分支固定 BN=64。
- D=128 BM 自适应阈值默认 6144（`SAGEATTN_INT8_BM128_THR` 可调）：4096² 时 BM=64 略优，6144² 优 2.8%，9216² 优 3.7%，VAE 16384² 优 5.6%。
- D=128 int8 默认 wpe=2；D=64 默认 wpe=1（`SAGEATTN_INT8_WPE` 可扫，1/2/4 全平）。
- int8 路径 V/OUT dtype 分离（方案 B）：bf16 输入 → V 预转 fp16 存 V_T、OUT 由 kernel 直接写 bf16；fp16 输入 → V/OUT 均 fp16。

### 2.3 已确认的性能边界

- **D=128 int8 attn ≈ 9.9–10 Top/s 恒定**（4096²/6144²/16384² 三类 shape 相同），较同环境 torch SDPA 快 1.3–1.7×。三重验证（环境参数全扫描平坦、结构性重组中性、torch SDPA 对标）→ 该值已接近**WMMA 执行单元吞吐 + softmax/score 寄存器管理的实际墙**，非带宽/延迟/配置所致。
- **quant 带宽受限**（实测 ~90–100 GB/s，接近 iGPU 共享内存上限），attn 计算密集型。aux（vt 固有 + quant）+ attn 主核 = 全管道，主核占大头。
- **配置近局部最优**：D128 BM/BN/wpe 全扫描 ≤±2%；D64 direct BM 64→128、int8 D64 _32 的 BN 32→64、occupancy 提升等均无收益。
- 指令级微优化全部失败；**结构性优化（减冗余读、提升 ILP、融合辅助 kernel、写回向量化）是仅有的有效路径**。

---

## 三、gfx103x（RDNA2，后续开发平台）

> **重要**：本节的实现描述以当前 `attn_gfx103x.cu` / `mma_gfx10.h` 代码为准，但**全部默认值/auto 规则/性能取向来自移植期设计且未经真机验证**。开箱即用前必须全量重验（§3.4）。

### 3.1 指令基础

- 矩阵乘降级为 SIMD 点积链：int8 QK = `v_dot4c_i32_i8`，fp16 PV = `v_dot2c_f32_f16`。无 bf16 硬件指令 → bf16 在 LDS 暂存时转 fp16（代价：LDS 占用 + 转换开销）。
- LDS 仅 64 KiB/CU（RDNA3 的一半），tile 尺寸选择受 occupancy 约束更紧。

### 3.2 kernel 变体（与演进脉络）

| 变体 | 说明 |
| --- | --- |
| v1 | BM=32（每 warp 1 行、行内 4 lane 分列）—— D128 长 self 极慢，已被 v2 取代 |
| v2 | **BM=64、128 线程（4 warps）、每线程 NR 行独立点积链**（隐藏 sdot4 延迟）、NW=4、BN=16/32；后续主力结构 |
| v2.2 | PV 直接从全局 V_T 行读（v2 行布局前提）；长序列有效、短序列负，auto 仅 self 使用 |
| v3 | 16-lane 协作（半 warp 一行）：QK 归约正确，但 **PV 输出不完整**（fp16/bf16 只覆盖半行；int8 每 lane 仅写 4 列）——direct 路径默认禁用 |
| v3.5（`attn_gfx10_new.h`） | 16-lane 协作 BM=128 新排布，默认不启用、未被 dispatch 调用 |

### 3.3 当前默认路由（代码现状，未验证）

环境变量（默认值）：
- `SAGEATTN_GFX10_V2`（2=auto：self 且 qo≥512 用 v2，D64 direct 则 qo≥1024）
- `SAGEATTN_GFX10_VT_GLOBAL`（0=off；1=force v2.2；2=auto：self 且 qo≥1536）
- `SAGEATTN_GFX10_V3`（**int8 默认 2=auto**；**direct 默认 0=禁用**）

实际默认（源码 dispatch，粗读结论）：
- **int8 D=64**：self qo≥512 或 cross qo≥256 → v3（BN=16）；其余 → v2.2→v2→v1（BN 16/32）。
- **int8 D=128**：kv>2048 → v3（BN=16）；其余 → v2.2→v2→v1。
- **direct fp16/bf16**：v3 默认禁用；auto 下 self qo≥1024 → v2（BN=32，BM=64），qo≥1536 且 vt_mode 开 → v2.2，其余 → v1（BN=32）。

### 3.4 未决问题与 gfx1035 首轮工作清单

1. **v3 正确性裁决**：int8 路径在 auto 下默认选用 v3 且注释声称"BN=16 精度同 triton"；direct 路径注释声称"v3.4 BROKEN for fp16/bf16 / int8 PV 输出不完整"。**两处注释互相矛盾，必须真机验证**；若 int8 v3 PV 确实不完整，应将 int8 的 `SAGEATTN_GFX10_V3` 默认改为 0 并回退 v2，同时复核对应精度（MaxErr/cos，重点 Anima01/03/05、AnimaVAE01 等 D128 长序列）。
2. **全量功能/精度/性能基线**：pytest + SDPA 对标（含 causal、HND/NHD、bf16）建立 gfx1035 基线，再据 §5 方法论做性能回归对照。
3. **v_transpose grid 策略同步**：gfx103x 源码仍是 `grid=total`（>4096）旧逻辑，需按 §1.3 第 7 条对齐并在 gfx1035 上重测（CU 数不同，参数可能需重扫）。
4. **D=64 长 self 结构优化**：设计期结论为"计算密度不足 + 16-lane reduction + LDS 容量限制"慢于 triton（旧数值未经真机核对）；候选方向：LDS 排布重构、warp 间协作（1 warp QK / 3 warps PV）、专用 D=64 kernel。
5. **bf16 路径再评估**：无硬件 bf16 → fp16 转换开销 + LDS 占用，gfx1035 上实测必要性后决定是否对 D=64 提供 fp16-only 推荐路径。
6. `attn_gfx10_new.h`（v3.5）如不启用应从构建中剔除（占编译体积）。

---

## 四、direct/int8 分发

`core.py` 按 headdim/kv_len/q_len/is_causal/tensor_layout 选择路径。以下阈值经 E2E 交叉点实测确认最优。

| headdim / 情形 | 路径 | 条件 |
| --- | --- | --- |
| D=64 HND self | int8 | kv > 2048（HND 布局 stride 影响，平衡点更低） |
| D=64 NHD self | int8 | kv > 3072 |
| D=64 causal | direct | kv ≤ 6144（8192 时 int8 优 5.5%） |
| D=64 cross（q<kv） | direct | kv ≤ 6144 |
| D=128 self / causal | direct | kv ≤ 2048（2560 时 int8 优 5%） |
| D=128 cross（q<<kv） | direct | kv ≤ 4096 且满足 q/kv 判据（q≈kv/2 时 int8 转优） |

统一权衡：int8 以 quant+mean 固定辅助（~0.4ms）换 QK 2× 吞吐，计算量小时 direct 胜、长 self 时 int8 胜。

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
1. **同进程内交错轮转**：被测实现与参考实现在同一进程交替运行（避免跨进程初始化/散热/驱动状态差异）。
2. **充分预热**：≥20 次 warmup（前几轮受 kernel JIT/cache 冷启动影响）。
3. **取中位数**：≥50 次迭代取 median（避免尾部延迟/GC 干扰）。
4. **iGPU 波动**：共享内存 iGPU 长时间运行因温度/时钟下降，小 case 跨 run 波动可达 ±25%（同代码亦如此）；**跨会话绝对时间不可比**，只有同会话内比率口径成立。
5. **固定随机种子**：`torch.manual_seed(0)` 保证可复现。

正确性基准：与 FP32 SDPA 对比 cos / maxdiff（参考实现必须 `transpose(1,2)` 到与 kernel 输入一致的 HND/NHD 布局；早期多次因参考布局用错而误判）。

**踩坑**：用 `time.perf_counter()` 测 GPU kernel 不加 `torch.cuda.synchronize()` 会严重低估（异步执行）；计时前后必须同步。严格 perf 用独立子进程（每 backend 一个），避免 reload 干扰。

---

## 六、编译

```bash
GPU_ARCHS=gfx1103 pip install -e . --no-build-isolation          # 仅 RDNA3
GPU_ARCHS=gfx1035 pip install -e . --no-build-isolation          # 仅 RDNA2
GPU_ARCHS=gfx1103,gfx1035 pip install -e . --no-build-isolation  # 同时（或省略默认）
SAGEATTN_SKIP_BUILD=1 pip install -e . --no-build-isolation      # 仅装 Python 层
```

关键 HIP 编译选项（gfx110x，gfx103x 同源设置）：`-O3 -ffast-math -fgpu-flush-denormals-to-zero -DSAGEATTN_VT_GLOBAL=1`，并含 `-mllvm --lsr-drop-solution=1 / -enable-post-misched=1 / -amdgpu-early-inline-all=true / -amdgpu-function-calls=false / -amdgpu-max-memory-clause=32 / -amdgpu-vgpr-index-mode=1`。构建产物 `sageattention/_qattn_gfx110x.pyd`、`_qattn_gfx103x.pyd`。

---

## 七、易踩坑点（后续开发必读）

### 7.1 gfx110x

1. **permlanex16 参数格式**：`v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]`，`%2`/`%3` 为 s/n 字面量（如 `0x76543210` / `0xfedcba98`）。早年误判该指令不可用，实为内联汇编参数错误；参考 Triton 生成的 ISA 编码即可。
2. **MFMA 不可用**：`+mai-insts` 一开，LLVM AMDGPU 后端 WMMA codegen 直接段错误。
3. **V_T n 维必须 padding 到 64 倍数**：否则最后 kv-tile 的 v_frag_t 32B 直读越界读到未初始化内存 → NaN。
4. **16B 写回对齐**：q strides 须为 8 倍数（half 粒度），非 contiguous 输入触发 16B 写 UB；core.py 有断言。
5. **`blockReduceMax` 循环调用必须自同步**：D=64 quant 的 RATIO 循环同 block 多次调用该共享归约，读/写之间无 barrier 会跨调用复用 shared 而产生**非确定性精度差（maxdiff≈1.2e-2）**；函数末尾补 `__syncthreads()` 即修复（`reduction_utils.cuh`，一处修复 gfx110x/gfx103x 两源）。
6. **quant 的 Threads 必须与 dispatch `dim3 block()` 一致**：曾将 kernel 内 `constexpr Threads=512` 而 launch 仍 `block(256)`，导致 K/Q 各丢一半组、scale≈0、cos≈0.72 的"确定性 bug"（非竞态）；修正后 T=512 实测也只是中性。**保持 T=256**。
7. **wmma_* builtin 仅在 `__GFX11__` 下有效**：mma_gfx11.h 对其他架构返回零值实现（编译通过、运行失效）；gfx110x 必须 `--offload-arch=gfx11xx`。
8. **stable torch library 单次注册**：`STABLE_TORCH_LIBRARY` 同名 schema 重复注册报错；靠 core.py 懒加载保证同进程只 import 一个 pyd。
9. **`is_causal` 是运行时参数**：必须 host 端 if/else 生成编译期模板常量（`LAUNCH_*_T(HD, true/false, ...)`），不能直接当模板参数传。
10. **namespace 边界**：原 gfx11.cu 的匿名 namespace 在第 30 行起、~1892 行闭合；**host/dispatch/公共函数必须全局作用域**，否则 pybind 链接 LNK2001。拆分时 gfx10 分支块要整段删除（含 if 闭合 `}`），否则引用 `sageattn_gfx10::*` 编译失败。
11. **`SAGEATTN_VT_GLOBAL=0` 不再维护**：删除/关闭该分支重建会硬崩溃（device assert/illegal）；当前构建只支持 VT=1，勿再启用 VT=0 路径。

### 7.2 gfx103x

1. **无 WMMA/MFMA**：`+mai-insts` 无意义；矩阵乘走 V_DOT 链，每 lane 独立累加。
2. **bf16 无硬件指令**：须 LDS 暂存转 fp16；增加 LDS 占用与转换开销。
3. **v3 PV 输出不完整**：16-lane 协作方案 QK 归约正确，但 PV 每 lane 只写部分列 → 输出缺列。代码注释对 int8 v3 是否正确**自相矛盾**（见 §3.4）。direct 路径已默认禁用。
4. **LDS 只有 64 KiB/CU**：BM/BN 加大（BM=128 等）会使 occupancy 崩溃，tile 选择须更保守。
5. **公共辅助副本**：mean/quant/v_transpose kernel 在 gfx103x 源中与 gfx110x 各放一份（实现相同，分开编译符号不冲突）；改动两处同步。
6. **未同步项**：v_transpose grid cap 新策略、写回向量化等在 gfx103x 上的状态需核对待办清单（§3.4）。

### 7.3 通用

1. **`__shfl_xor_sync` 延迟高**：编译为 ~6 VALU + 1 ds_bpermute（20–30 cycles），能用 permlanex16 / lane-reduce 替代就替代。
2. **用 `exp2f`**：softmax 一律 `exp2f`（乘 log2e 预缩放），比 `expf` 快 ~1 cycle。
3. **`getenv` deprecation 警告**（`_CRT_INSECURE_DEPRECATE`）在 Windows 无害，hipcc 不报 error。
4. **架构检测**：`torch.cuda.get_device_properties().gcnArchName` 返回 `"gfx1103"` 等；core.py 优先字符串前缀判断，回退 major 整型。
5. **自己做实验用独立扩展**：需要 A/B 对比时按参数化模块名（如 `_qattn_<name>`）单独编译扩展，避免 torch.ops 注册冲突与缓存；用完删构建缓存。
6. **iGPU 上并发 stream 无收益**：APU 共享内存带宽，vt‖quant 双 stream 重叠反而慢 7–14%（带宽争抢），勿做跨 kernel 重叠。

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
| B4 | smooth_k=True 设为默认 | 随机数据下慢 0.5–3.5%（减 mean 引入舍入），保持 False |
| B5 | 在线/双 stream 重叠 vt‖quant | 共享带宽争抢，-7~-14% |
| B6 | quant T=256→512 | 先因 launch block 未同步改而"假 -42%"，修正后中性；保持 256 |

**C. 结构优化（失败项）**

| # | 尝试 | 结论 |
| --- | --- | --- |
| C1 | v_transpose 融合进 attn | LDS 占用翻倍 occupancy 崩；且 V 全局流量放大 32–64×，结构性不划算 |
| C2 | LDS 缓存 V_T tile PV 用于 D=64 | D=64 v_frag 仅 8KB，浪费 occupancy；**该优化仅 D=128 kv≥16384 有效** |
| C3 | wpe2/wpe4 流水线 | 收益 <1%，无意义 |
| C4 | fp16 双子块（direct_32） | q_frag v16h 8 VGPR/tile + out_acc/score_cache → D64 静态 ~192+、D128 ~400+ VGPR 溢出，0.34–0.56×；**fp16 不可照搬 int8 双子块** |
| C5 | D128 双子块（`_32` 泛型实例化） | out_acc+score_cache ~150+ VGPR 溢出，**12.2–12.7× 灾难**（cos 正确） |
| C6 | V 片段跨子块复用 + V-stage 重排 | 寄存器复用被全局 V 读带宽掩蔽，中性 |
| C7 | D128 换每 warp 双子块 kernel | 双子块 ILP 被 VGPR 溢出抵消，中性 |
| C8 | V 全局预取到寄存器 | +32 VGPR 伤 occupancy，-8~+6.5%，劣化为主 |

**D. 调度 / 分发**

| # | 尝试 | 结论 |
| --- | --- | --- |
| D1 | 全路径统一 int8 | 短序列 quant 辅助 > int8 吞吐收益 |
| D2 | 全路径统一 direct | 长 self 失去 i8 2× 吞吐，慢 30–50% |
| D3 | 运行时 auto-tune | 开销 > 收益；env 覆盖够了 |
| D4 | dispatch 融合（vt/quant+attn 单 op） | 中性偏慢（0.82–1.06×）——vt/quant 的 0.2–0.4ms 是真实 kernel 时间，无调度开销可省 |
| D5 | D128 BM/BN/wpe 全扫描 | 全平坦 ≤±2%；BN=128 破正确性（cos=0.62） |
| D6 | D64 int8 32w BK BN=32→64 | 回退 18–20%，BN=32 最优 |
| D7 | D64 direct BM=64→128（大 self） | 持平 |

**E. 已删除 / 不再维护**：fp8 存储 V（精度不达标，已删）；`SAGEATTN_VT_GLOBAL=0`（硬崩溃，勿启，见 §7.1）；`attn_gfx10_new.h` v3.5（未启用）。其余未采用方案（上述 C6–C8、融合 op 等）均未合入主代码。

### 8.2 gfx103x

| # | 尝试 | 结论 |
| --- | --- | --- |
| R1 | MFMA 路径 | 消费级 RDNA2 缺失，编译错误 |
| R2 | v1（每 warp 1 行） | D128 长 self 极慢，被 v2 取代 |
| R3 | v3 16-lane 协作（fp16/bf16） | 只覆盖半行，**输出不完整** |
| R4 | v3 16-lane 协作（int8 PV） | 每 lane 仅写 4 列，**输出不完整**（但 int8 dispatch 默认仍 auto 选用——待裁决，§3.4） |
| R5 | BM=128 | LDS 占用翻倍，occupancy 崩 |
| R6 | 双 warp 协作 QK（NR=4） | LDS 翻倍，sdot4 延迟未有效隐藏 |

### 8.3 规律

- **指令级微优化全部失败**（gfx110x A 组、gfx103x 同理）：编译器已充分调度。
- **结构性优化是唯一有效路径**：减冗余读（V 转置、LDS 缓存）、提 ILP（大 block、warp 内多子块）、融合辅助 kernel（bf16→fp16 进 v_transpose）、写回向量化。
- **精确度方案均失败于精度或硬件支持**：fp8/int4/per-head scale 全灭；int8 per-group scale + fp16 PV 是当前最优点。
- **寄存器/占用是上限约束**：任何以"增加寄存器/ LDS"换 ILP/延迟的方案（双子块、V 预取、T=512、BM=128）都被 VGPR/occupancy 反噬。

---

## 九、当前基线与剩余方向

### 9.1 用例图例（benchmark_attn.py）

| 用例 | 含义 |
| --- | --- |
| SDXL01/07/13 | D=64 int8 self 4096²/6144²/9216² |
| SDXL04/10/16 | D=64 direct self 1024²/1536²/2304² |
| SDXL02/05/08/11/14/17、03/06/09/12/15/18 | D=64 direct cross ×77 / ×154（text） |
| Anima01/03/05 | D=128 int8 self 4096²/6144²/9216²（bf16） |
| Anima02/04/06 | D=128 direct cross 4096×512/6144×512/9216×512 |
| SDXLVAE01 / AnimaVAE01 | D=128 int8 self 16384²（fp16 / bf16，VAE） |

### 9.2 基线（当前状态，gfx1103 实测）

基线 = 当前代码在 gfx1103 上的实测（默认编译、无 env、`smooth_k=False`；FlashAttn 未安装，无 FA 对比）。正确性：pytest 42/42 通过，且下表各用例均通过 SDPA 参考校验（maxerr≤0.005、cos≥0.9999）。比率 = SageAttn 相对 torch SDPA 的提速（>1 快；<1 意为慢于 SDPA）。计时为单会话 ROUNDS=2×ITER=15 交错轮转中位。

| 用例 | 路径 | e2e(ms) | /SDPA | 状态 |
| --- | --- | --- | --- | --- |
| SDXL01 | D64 int8 self 4096² | 4.581 | 1.54 | OK |
| SDXL02 | D64 direct cross ×77 | 0.359 | 1.69 | OK |
| SDXL03 | D64 direct cross ×154 | 0.503 | 1.53 | OK |
| SDXL04 | D64 direct self 1024² | 0.831 | 1.41 | OK |
| SDXL05 | D64 direct cross ×77 | 0.197 | 1.22 | OK |
| SDXL06 | D64 direct cross ×154 | 0.402 | 1.17 | OK |
| SDXL07 | D64 int8 self 6144² | 9.811 | 1.23 | OK |
| SDXL08 | D64 direct cross ×77 | 0.335 | 1.54 | OK |
| SDXL09 | D64 direct cross ×154 | 0.567 | 1.64 | OK |
| SDXL10 | D64 direct self 1536² | 2.291 | **0.79** | OK（但慢于 SDPA） |
| SDXL11 | D64 direct cross ×77 | 0.273 | 1.41 | OK |
| SDXL12 | D64 direct cross ×154 | 0.453 | 1.42 | OK |
| SDXL13 | D64 int8 self 9216² | 21.563 | 1.22 | OK |
| SDXL14 | D64 direct cross ×77 | 0.438 | 1.66 | OK |
| SDXL15 | D64 direct cross ×154 | 0.667 | 1.51 | OK |
| SDXL16 | D64 direct self 2304² | 3.284 | 1.06 | OK（勉强快于 SDPA） |
| SDXL17 | D64 direct cross ×77 | 0.370 | 1.21 | OK |
| SDXL18 | D64 direct cross ×154 | 0.544 | 1.52 | OK |
| Anima01 | D128 int8 self 4096² | 17.218 | 1.42 | OK |
| Anima02 | D128 direct cross 512 | 2.574 | 1.98 | OK |
| Anima03 | D128 int8 self 6144² | 37.606 | 1.19 | OK |
| Anima04 | D128 direct cross 512 | 3.769 | 1.96 | OK |
| Anima05 | D128 int8 self 9216² | 79.720 | 1.21 | OK |
| Anima06 | D128 direct cross 512 | 5.450 | 2.03 | OK |
| SDXLVAE01 | D128 int8 self 16384² fp16 | 60.795 | 1.29 | OK |
| AnimaVAE01 | D128 int8 self 16384² bf16 | 44.004 | 1.05 | OK（勉强快于 SDPA） |

> 注：除 SDXL10 外全部快于 torch SDPA（1.05–2.03×）。跨会话波动大（小用例 ±10–25%，SDPA 侧亦如此），判定以同会话交错对比为准。

**相对弱项（与 SDPA 对比）**：SDXL10（direct self 1536²，v_transpose 固有成本占比大，比 SDPA 慢 21%）；AnimaVAE01（1.05×）与 SDXL16（1.06×）仅略快。其余均显著快于 SDPA。

### 9.3 剩余差距来源

1. **v_transpose 固有成本**（native 独有）：约 0.15–0.5ms/调用，短序列占比大；已做 grid-cap 优化，无法融合进 attn（C1）。
2. **int8 路径 quant 辅助累计**：quant+vt 共约 0.4–1.1ms；quant 已近带宽上限，无安全削减手段。
3. **短序列固定开销**（两后端共性）：kernel launch+同步，绝对差 0.03–0.12ms，占比大。

### 9.4 剩余方向（含 gfx1035 优先项）

| 方向 | 平台 | 预期收益 | 风险 / 备注 |
| --- | --- | --- | --- |
| **D=64 长 self 专用内核重构** | gfx1035 | 高 | 现 v2 系列在 RDNA2 密度/占用约束下不足；新结构（LDS 排布、warp 协作）工程量大 |
| **gfx103x 全量真机验证 + v3 裁决** | gfx1035 | — | 第一优先级，见 §3.4 |
| **v_transpose grid 策略同步 + 重扫** | gfx1035 | 低–中 | gfx110x 的 clamp 参数需按 gfx1035 CU 数重测 |
| int8 quant 再提速 | gfx110x/gfx103x | 低 | T=512 已证无效；quant 带宽受限 |
| 消除 direct 的 v_transpose 固有成本 | 两平台 | 中 | attn 内转置失败于 LDS/带宽放大（C1），无新思路 |
| per-stage 流水线 / 更激进 WPE | gfx110x | 低 | C3/wpe 扫描证收益 <1% |
| fp8 V 存储 | 两平台 | 不可行 | 精度不达标 + gfx1103 无 fp8 硬件 |
| bf16 硬件路径 | gfx103x | 不可行 | gfx1035 无 bf16 指令 |

**结论**：gfx110x 各主 kernel 已处于硬件实际墙（WMMA 吞吐 / 带宽双限），无安全增量；后续增量工作聚焦 gfx1035（先验证、再针对 D=64 长 self 重构），且 **gfx110x 的任何"配置结论"（WMMA 相关）都不可直接套用到 gfx103x**。