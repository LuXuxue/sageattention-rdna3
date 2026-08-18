# SageAttention Native Backend 优化报告

> 本报告是项目优化的总记录：涵盖硬件基础、核心算法、实验过程与结论、分发逻辑、易踩坑点与剩余优化方向。

## 一、最终结论（当前最优状态）

Native backend 的核心优化链（按收益排序）：
1. **V 全局转置 PV（`SAGEATTN_VT_GLOBAL=1`）**：V 一次性转置为 `V_T [B,H,D,N]`，PV 的
   A=V^T 从 128 次 u16 LDS 列读变为 1 条行读 → direct 路径真实提升 17-36%
2. **V_T tile LDS 缓存 PV（VAE 超长 self-attn，新增）**：PV 的 v_frag 从 LDS 行读而非
   全局——消除每 lane 每 dt 一次 32B 全局行读的 L1/L2 带宽冗余（128 lanes×32 次/迭代
   = 128KB，实际 tile 仅 16KB）→ D=128 int8 超长序列 attn 提升 ~35%（SDXLVAE01
   94.7→62ms），与 triton 追平并反超
3. **每 warp 32 行 QK kernel**：D=64 int8 self 用 BM=128/4 warps，2 子块共享 k_frag
   （ILP）→ int8 self 从慢 7-13% 追平
4. **bf16→fp16 转换融合进 v_transpose**：省掉 `v.to(fp16)` 独立 kernel（Anima01 的 5%）
5. **D=128 int8 BN=32→64**：kv-tile 数减半（barrier/k_tile 填充减半）
6. **quant pass1→LDS 缓存**：pass2 从 LDS 读，省一半全局读流量（2.28→2.15ms）
7. **差异化 direct/int8 分发**（按 kv_len/q_len/is_causal 选择路径）
8. **写回向量化**（permlanex16 + 16B 连续写，全部 kernel 一致）

**当前 benchmark 状态（Radeon 780M）**：

| 类别 | native/triton | 说明 |
|------|--------------|------|
| Anima D=128 int8 self（4096/6144/9216） | 0.976/0.955/1.000 | 持平 triton（历史 0.949/0.939/0.965 vs FA） |
| Anima D=128 direct cross（kv=512） | 0.89-0.91 | **超 triton 9-11%** |
| SDXL direct cross（kv<=2048） | 0.95-1.0 | 持平/超 |
| SDXL01/07/13（D=64 int8 self, kv>=4096） | 0.92-1.03 | **01/07 反超 8-9%**（v_tile/v_prefetch 三元省 LDS/VGPR 收益） |
| SDXL10/16（D=64 direct, kv=1536/2304） | 1.07-1.13 | 慢 7-13%（架构性） |
| **SDXLVAE01/02（D=128 int8 self, 16384/24576）** | **0.92-0.97** | **超 triton 3-8%**（LDS PV 优化后；优化前慢 16-23%） |
| **AnimaVAE01/02（D=128 int8 self bf16, 16384/24576）** | **0.98-1.03** | 持平±3%（attn 主 kernel 已持平，差距=bf16 输入需 v_transpose 转换） |

> 注：VAE 用例的 03/04（36864/55296）因 iGPU 长时间运行死机已从 benchmark_attn.py 注释，
> 优化结论由 01/02（16384/24576）外推。

**关键结论**：
- i8 WMMA 在 RDNA3 为 fp16 的 2 倍吞吐（QK 占一半计算量）→ int8 在长 self-attn 占优；
  direct 省 quant+mean 辅助（固定 0.4-0.6ms）→ 在计算量小（causal/cross 短 q）时占优
- WMMA 硬件吞吐是 int8 的剩余上限（D=128 attn 达 ~10.9 TFLOP）
- **VAE 超长 self-attn 的瓶颈是 PV 的 WMMA + L1/L2 带宽冗余**（非 K/V 重读、非
  occupancy、非 LDS 容量）；LDS 缓存 V_T tile 是唯一有效结构优化（-35%）
- **默认编译配置（无 env）即为各用例最优配置**（BM=128/BN=128 等实验项均默认禁用）

---

## 二、硬件与指令基础

### 2.1 硬件规格（开发平台）

| 项目 | 规格 |
|------|------|
| GPU | AMD Radeon 780M (gfx1103, RDNA3 iGPU) |
| ROCm | 7.14（`rocm-sdk path --root` 定位工具链） |
| CU | 12 |
| LDS | 128 KiB / CU |
| VGPR | 512 KiB / CU（每 SIMD 256 VGPRs/lane 上限，warp 级 1024 槽） |
| L2 | 2 MiB（无 Infinity Cache） |
| 显存 | 共享系统内存（DDR5-5600, 64GB 双通道） |
| 内存带宽 | 理论 ~89.6 GB/s（128-bit @ 5600MT/s）；实测 copy 73 GB/s、读 79 GB/s |
| 矩阵指令 | WMMA 16×16×16（**无 MFMA**） |

**实测内存带宽**（benchmark_attn/bench_bandwidth.py）：
copy 73 GB/s、sum 79 GB/s、**torch transpose 仅 16.5 GB/s**。转置类操作是本平台
带宽瓶颈；native v_transpose 30 GB/s 已优于 torch 2 倍。

### 2.2 WMMA 布局（精确版，probe 实验验证）

```
C 输出:  lane L 持有 C[2e+(L>>4)][L&15], e=0..7   （行 = 2e+hw, 列 = L&15）
A operand: lane L 提供 A 行 L&15 的 16 列          （a[k] = A[L&15][k]）
B operand: lane L 提供 B 列 L&15 的 16 行          （b[k] = B[k][L&15]）
```

- **A 与 B 的 operand 布局不对称**（A 按行、B 按列）——这是转置 PV 设计的基础
- 每 lane 的 A/B operand 为 16 个元素（`v16h`），C 为 8 个元素（`v8f`）

**转置 QK 的用途**：交换 operand（A=k, B=q^T）后，WMMA 输出为 `S^T = K @ Q^T`，
`lane L 持有 S[L&15][2e+(L>>4)]` 恰好使 lane L 与 L^16 持有同一 softmax 行的偶/奇列，
permlanex16（XOR-16 交换）一次即可合并整行归约。

### 2.3 v_permlanex16_b32 指令

- 执行 XOR-16 跨半波交换（lane 0↔16, ..., 15↔31），纯寄存器操作
- **限制**：只能 XOR-16，不能做半波内（XOR-8/4/2/1）通信
- **踩坑**：早期误判该指令在 gfx1103 不可用（driver bug），实际是
  `__builtin_amdgcn_permlanex16` 参数格式错误；参考 Triton 生成的 ISA 编码即可
- **性能**：`__shfl_xor_sync` 替代反而慢 19%

### 2.4 ds_bpermute_b32 与 __shfl_xor

`__shfl_xor` 编译为约 6 VALU + 1 ds_bpermute（含 exec mask 边界检查），延迟 20-30 cycles。
内联汇编无法复制其 exec mask 管理逻辑。**`__shfl_xor` 是唯一可靠方式**，但本方案用
permlanex16 而非它。

### 2.5 MFMA

MFMA 在 RDNA3 (gfx1103) 不可用（编译错误/LLVM 崩溃），仅支持 WMMA 16×16×16。

---

## 三、核心算法与优化

### 3.1 转置布局算法

标准 flash-attention 的 QK WMMA（A=q, B=k）输出布局使归约需跨 16 lane 的 butterfly 通信。
**交换 operand（A=k, B=q）后**输出为转置矩阵 `S^T = K @ Q^T`，归约只需 1 次 permlanex16：
- max：`gm = max(row_m, local_mx)` 后 `gm = max(gm, permlanex16(gm))`
- sum：`row_l += partial_sm + permlanex16(partial_sm)`

**P fragment 寄存器内组装**（`assemble_p_frag`）：lane L 的 8 个 exp 值对应 P 行 (L&15)
的偶/奇列，与 lane^16 交换后获得完整 16 列，打包成 fp16 对按 WMMA B-operand 布局交错。

**转置 PV**：`out^T = V^T @ P^T`（A=V^T, B=P^T），输出布局自动回到
"out 行 (L&15) 的偶/奇 D 列"，写回按 `d = dt*16 + 2e + hw` 排列。

### 3.2 V 全局转置 PV（核心优化之一）

**动机**：原转置 PV 中 A = V^T 需要 V 的列读（跨行），经 LDS 中转后每迭代 128 次
u16 标量 LDS 读（低效且 bank 冲突）。

**解法**：V 一次性转置为 `V_T [B,H,D,N]`（n 连续），PV 的 A=V^T 变为行读（1 条 v16h）。

- **wmma 参数顺序必须是 `wmma(v_frag_t, p_frag, out_acc)`**（A=V^T, B=P^T, C=out^T）；
  若写成 `wmma(p_frag, v_frag_t)`（C=out）则 C 布局与写回转置，输出乱序（踩坑点 5）
- **vt_off 必须含 kb_base**：`((b*H+hkv)*D + dt*16+m_row)*N + (kb_base + ct*16)`，
  漏掉 kb_base 会只处理前 32 个 key（踩坑点 6）
- **V_T 的 n 维必须 padding 到 64 倍数**（见 3.6）
- 实现：`v_transpose` op（NHD/HND 均支持，LDS 分块转置，tile 32n×32d，pad 33 消
  bank 冲突；v_transpose 单独耗时 1.14ms@Anima01 ≈ 28-34GB/s，接近转置类带宽上限）
- core.py 在 int8 与 direct 路径均无条件转置 V；`setup.py` 默认编译 `-DSAGEATTN_VT_GLOBAL=1`

### 3.3 每 warp 32 行 QK kernel（核心优化之二，仅 D=64 int8 self）

**设计（`attn_kernel_impl_32_t`，BM=128, 4 warps）**：
- 每 warp 处理 32 行（2 个 16 行子块 sub0/sub1）
- **2 子块共享 k_frag 加载** → 每迭代产生 2 条独立 WMMA 依赖链
- softmax/PV/写回均 2 子块化（permlanex16 归约不变）
- **VGPR 压力**：2 组 per-q 状态，D=64 约 160-190 VGPR 可编译；**D=128 不可行**
  （q_frag[2][8]+out_acc[2][8]+score_cache ≈ 256+ VGPR → 溢出，实测慢 10-24x）
- D=64 self-attn 默认启用（`SAGEATTN_INT8_32=0` 回退 8 warps）

### 3.4 V/OUT dtype 分离（方案 B）

- **V 转 fp16 并入 v_transpose**（读 bf16 直接转 fp16 写 V_T，省 `v.to(fp16)` 独立
  kernel——实测后者 0.87ms@Anima01，即 5%）
- **输出由 kernel 直接写 bf16**（`OUT_DTYPE` 模板参数，o 复用，单次舍入精度更好）
- fp32 输入仍走 core 的 `v.to(fp16)`（v_transpose 不支持 fp32，场景少）

### 3.5 写回向量化

原写回为 2B 散布写（每 dt 8 次 × DTiles）。改为：
- permlanex16 交换（4 次/dt）获得完整行 16 列 → hw=0/1 lane 分工各写 16B 连续
- fp16 路径组装 half 位模式；bf16 路径从 float 单次舍入转 bf16 再组装
- 应用到全部 3 个 attn kernel（int8 impl_t / int8 32行 / direct）
- 实测无显著性能变化（写回被内存带宽/流水线掩盖），但消除 2B 散布写、统一代码

### 3.6 V_T padding（non_aligned 修复）

**问题**：kv_len 非 64 倍数时（如 1057），attn kernel 的 `v_frag_t` 32B 直读越界——
最后 kv-tile 读超出 V_T 的 n 维 → 未初始化内存 → NaN。k_tile 填充有掩码，v_frag_t
直读无掩码。

**修复**：V_T 的 n 维 padding 到 64 倍数（`padded_n = ceil(kv/64)*64`），三处规则一致：
- core.py：`v_t = torch.empty(B, H, D, padded_n)`
- v_transpose_kernel：写回用 `v_t_n = ceil(seq/64)*64` 寻址，padding 区写 0
- 3 个 attn kernel：`vt_off` 行 stride 用 `v_t_n`（kernel 内 `((kv_len+63)/64)*64` 计算）

**衍生 bug（review 发现）**：v_transpose 的 ntiles 若按 `ceil(seq/32)` 计算，只覆盖
32 对齐区，padding 区 `[ceil32, ceil64)` 未写 0（kv mod 64 ∈ [1,32] 时如 2049/2113
仍 NaN）。**必须按 `v_t_n/32` 计算 ntiles**（kernel 与 host 同步）。

### 3.7 quant LDS 缓存

**瓶颈定位**：pass1 读 32MB 仅 20GB/s（torch sum 同数据量 75GB/s）；非读粒度
（32B/thread 无改善）、非 blockReduce、非计算。大 MIN_BLK（64/128）反而 +37-45%
（每 block packs 多 → fmax 串行依赖链长；native 结构不适合大 block）。

**有效优化**：pass1 读入数据缓存到 LDS（`shared_data[512]` uint4 = 8KB，
恰好覆盖 D=128 Q 的 512 packs），pass2 从 LDS 读 → quant 2.28→2.15ms。
同时 pass1 读粒度 16B→32B（2 连续 pack 合并）。

### 3.8 V_T tile LDS 缓存 PV（VAE 超长 self-attn，2026-08 新增）

**背景**：benchmark_attn.py 新增 SDXLVAE/AnimaVAE 用例（D=128 int8 self，
kv=16384~55296, h=3/4）。同进程轮转显示 native 在 16384 用例比 triton 慢 52-72%
（SDXLVAE01 94.7 vs 62ms），24576+ 持平。分阶段 profiling 确认 attn 主 kernel 占
96-98%，辅助（mean/quant/v_transpose）仅 2-5%。

**瓶颈定位**（逐个排除）：
- BM=128/8 warps（K/V 重读减半）：±1% 无效 → 非 DRAM 带宽/重读受限
- BN=16/32/128、wpe=1/2/4、occupancy（v_tile 缩为 1 元素省 18KB LDS）：全部无效
  → 非 occupancy/LDS 容量受限，WMMA 计算本身接近吞吐上限
- **真正的结构性差距：PV 的 v_frag 每 lane 每 dt 从全局 V_T 行读 32B（b128）**。
  每迭代 4ct×8dt = 32 次/ lane，128 lanes × 32 × 32B = 128KB 全局读，而实际 tile
  数据仅 64n×128D×2B = 16KB —— **8 倍冗余，L1/L2 读带宽被 WMMA operand 布局放大**
- triton 的 V 走软件流水线（num_stages=2）缓存到 LDS，无此冗余 → 快

**解法**：V_T 当前 tile（64n×128D）每迭代一次拷入 LDS（布局 `[D][N]`，行 = D 维、
n 连续 16 half 行读），PV 的 v_frag 改从 LDS 行读：
- 全局读从 128KB/迭代（冗余）降到 16KB/迭代（每 thread 4 个 v16h 行读 32B）
- `VTileStride=80`（行间 8 bank 偏移，与 VStride 冲突模式相同；20KB/128 行）
- `__align__(32)` 保证 v16h 32B 对齐
- 每迭代 +1 次 `__syncthreads()`（V 拷贝后）

**结果**（同进程轮转）：SDXLVAE01 attn 94.7→62ms（**-35%**），SDXLVAE02 反超 triton
4%，AnimaVAE01 -36%（69.6→44.8ms）。LDS PV 优化后 BM=64 全面优于 BM=128（-6%，
v_tile 20KB + 8 warps barrier 开销），BM=128 仅保留 env 实验开关。

**踩坑**：初版 v_tile 大小沿用 `BLOCK_N*VStride`（64×144），而 [D][N] 布局需要
`HeadDim` 行 → 越界写 2 倍污染 LDS，4096+ 序列输出错误（err=0.13）；必须
`(SAGEATTN_VT_GLOBAL) ? (HeadDim*VTileStride) : (BLOCK_N*VStride)` 三元区分布局。

### 3.9 v_tile / v_prefetch 条件声明（省 LDS/VGPR）

`VT_GLOBAL=1` 时 LDS `v_tile` 与寄存器 `v_prefetch` 完全未使用（V 直接全局行读），
但声明无条件占用 18KB LDS + 8×uint4 VGPR。改为编译期常量三元
（`? 1 : N`，死分支被 DCE），三个 kernel（impl_t / impl_32_t / direct）一致：
- D=128（impl_t）：LDS PV 优化前无性能变化（occupancy 非瓶颈）；D=64 的 32 行
  kernel（SDXL01/07/13）实测反超 triton 8-9%（省 LDS/VGPR 的调度收益）

---

## 四、测量方法论

### 4.1 热漂移与频率抖动

- iGPU 冷启动后频率剧烈波动（同一 kernel 单次时间可跳变 20 倍）
- 需 ~600 次 kernel 调用预热才收敛；**跨会话/跨进程绝对时间不可比**
- 固定顺序交错轮转有轮内顺序效应（先测的后端更冷更慢）

### 4.2 修正后的测量方法

```
1. 每用例充分预热: 各后端交替跑 ≥600 次调用
2. 测量轮: 5-6 轮 × 每轮 20-30 iters
3. 轮间后端顺序旋转: 第 r 轮从 r 号后端开始（对称采样）
4. 各后端取中位数
```

**唯一可信对比 = 同进程内交错轮转的倍率。**

### 4.3 layout_code 陷阱

- `tensor_layout="NHD"` 必须传 **layout_code=0（kNHD）**；传 1（kHND）会让 kernel 按
  HND stride 解释 NHD 数据 → 输出错误（err≈3.0）但执行更快（假象）
- 任何"手动拆分 ops 调用"的验证必须与 core.py 逐参数一致（含 layout_code）
- **native 正确性验证必须显式设 `SAGEATTN_BACKEND=native`**：core.py 默认 `_BACKEND=triton`，
  不设 env 时 check 脚本测的是 triton 后端

---

## 五、分发逻辑（实测阈值）

### 5.1 direct/int8 选择（core.py）

**核心权衡**：direct（fp16 QK）省 quant+mean 辅助 kernel（固定 0.4-0.6ms）；
int8 用 i8 WMMA（RDNA3 上为 fp16 的 2 倍吞吐，优势与计算量成正比）。
计算量小（causal / cross 短 q）→ direct；计算量大（self 长序列）→ int8。

**阈值实测扫描**（benchmark_attn/bench_threshold*.py，同进程轮转 direct vs int8）：

| D | 场景 | 阈值（kv_len） | 关键数据 |
|---|------|---------------|----------|
| 64 | self 非 causal | 3072 | 3072 direct 优 1%、3456 int8 优 1.2%、4096 int8 优 3% |
| 64 | causal | 6144 | 4096/6144 direct 优 8%/2.7%、8192 int8 优 5.5% |
| 64 | cross (q<kv) | 6144 | q=3072/kv=4096 仍 direct 优 3% |
| 128 | self/causal | 2048 | 2048 direct 优 5%、2560 int8 优 5%、4096 int8 优 16% |
| 128 | cross q*2<kv | 4096 | q=512/1024 vs kv=4096 direct 优 25%/8%、q=2048(=kv/2) int8 优 5% |

**int8 量化路径的适用场景**（不仅限 self-attn）：
- D=64：self（kv>3072）、causal/cross（kv>6144）
- D=128：self/causal（kv>2048）、cross 且 q>=kv/2（kv>2048）
- benchmark 24 用例中，int8 实际只服务 self-attn（SDXL01/07/13 和 Anima01/03/05）

### 5.2 int8 路径 kernel 选择

```
D=64  self-attn -> 每 warp 32 行 kernel (BM=128, BN=32, 4 warps) [默认]
                  (SAGEATTN_INT8_32=0 回退 8 warps wpe1)
D=64  cross, kv<=77 -> (BM=64, BN=16) [wpe1]
D=64  cross, kv>77  -> (BM=64, BN=32) [wpe1]
D=128               -> (BM=64, BN=64) [wpe2]   (BN=64 实测优于 BN=32; BN=16 全面变差)
```

**VAE 超长 self-attn（kv≥16384）**：同样走 `(BM=64, BN=64) [wpe2]`，但 PV 走 3.8 的
LDS 缓存路径（`v_tile [D][N]` 布局，`VTileStride=80`）。LDS PV 优化后 BM=64 全面
优于 BM=128（-6%），BM=128/BN=128 仅保留 env 实验开关（默认禁用）。

### 5.3 fp16/bf16 direct 路径 kernel 选择

```
D=64  self -> (BM=64, BN=64)
D=64  cross, kv<=128 -> (BM=64, BN=32)
D=64  cross, kv>128  -> (BM=128, BN=32)   [BM=128 快 9-13%]
D=128                -> (BM=64, BN=16)    (BN=16 实测最优; BN=32/64 均更差)
```

### 5.4 环境变量汇总

| 变量 | 作用 | 默认 |
|------|------|------|
| `SAGEATTN_BACKEND` | 后端选择（native/triton） | triton |
| `SAGEATTN_INT8_32` | int8 self 用每 warp 32 行 kernel | 1（启用） |
| `SAGEATTN_INT8_WPE` | int8 实验：waves_per_eu（1/2/4） | 1 |
| `SAGEATTN_INT8_BN128` | D=128 int8 的 BN 覆盖（0 默认 64, 16/32/128） | 0（=64） |
| `SAGEATTN_INT8_BM128` | D=128 int8 强制 BM=128/8 warps（1 启用；实测 BM=64 更优） | 0（=64） |
| `SAGEATTN_QUANT_GPB` | quant 多 group 合并实验（1/2/4/8；实测无改善） | 1 |
| `SAGEATTN_FP16_BM` / `SAGEATTN_FP16_BN` | fp16 direct 实验配置覆盖 | 0 / 0 |
| `SAGEATTN_BF16_BM` / `SAGEATTN_BF16_BN` | bf16 direct 实验配置覆盖 | 0 / 0 |
| `SAGEATTN_DIRECT_THRESHOLD_D64` | D=64 self 的 direct/int8 阈值 | 3072 |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL` | D=64 causal 的阈值 | 6144 |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CROSS` | D=64 cross 的阈值 | 6144 |
| `SAGEATTN_DIRECT_THRESHOLD_D128` | D=128 self/causal 的阈值 | 2048 |
| `SAGEATTN_DIRECT_THRESHOLD_D128_CROSS` | D=128 cross 的阈值 | 4096 |

`SAGEATTN_VT_GLOBAL` 已由 `setup.py` 编译宏固定为 1（V_T 方案默认启用），无需设置。

---

## 六、编译

- `setup.py` 默认 `HIP_FLAGS` 含 **`-DSAGEATTN_VT_GLOBAL=1`**（V_T 方案）
- 构建：`pip install -e . --no-build-isolation`（Windows 需 MSVC + ROCm SDK 环境）
- 若需回退旧路径（无 V_T），改 setup.py 宏为 0 并关闭 core.py 转置
- 默认编译配置（无 env）即为各用例最优配置

---

## 七、易踩坑点（后续开发必读）

1. **测量必须同进程旋转轮转 + ≥600 次预热**；跨进程绝对时间不可比（见 §四）
2. **layout_code=0 是 NHD**；手动复刻 ops 调用传错 layout 会产生"错误但快"的假象
3. **native 正确性验证必须设 `SAGEATTN_BACKEND=native`**（默认是 triton）
4. **hipcc 对模板 `__device__` 函数体内的 `#if`/`if constexpr` 有解析 bug**：
   即使逻辑为空也会报虚假的 `attn_kernel_wpe1_t undeclared`。**解法：用运行时 `if (编译宏)`**
   （宏展开为常量，死分支被 DCE）；`#if` 只用于 host 函数与 namespace 顶层宏定义
5. **WMMA operand 顺序决定 C 布局**：`wmma(A=V^T, B=P^T)` 才得到 out^T（匹配写回）；
   交换成 `out = P @ V` 会转置输出。定位方法：corr(o,ref)≈0.005 但 corr(sorted)≈0.9997
6. **vt_off 必须含 kb_base**：漏掉会每迭代都读 V_T 前 32 个 n。
   定位方法：V[n][d]=n 时 o≈均匀和（15.3），V[n][d]=(n≥512) 时 o=0
7. **iGPU 频率抖动 vs 顺序效应**：A/B 对比必须轮转，否则结论可被污染
8. **V_T 的 n 维必须 padding 到 64 倍数**（三处规则一致：core.py 分配、v_transpose 写回、
   attn kernel 的 vt_off stride）；v_transpose 的 ntiles 必须按 `v_t_n/32` 计算，
   否则 padding 区 `[ceil32, ceil64)` 未写 0，kv mod 64 ∈ [1,32] 时仍 NaN
9. **LDS 用奇数 pad 消 bank 冲突时，行内偏移奇偶交替，8B/4B 向量访问必然未对齐**：
   要么 pad 取 4 的倍数（重做 bank 分析），要么全部用 16-bit 标量访问
10. **uint4/uint2 的字节数（16B/8B）与 half 数量（8/4）易混淆**：v_transpose 初版
    用 uint4 处理 4 half → 越界读+覆盖邻数据（cos 骤降至 0.92）
11. **16B 向量写回要求 o_off 8-half 对齐**：q 的 stride_b/n/h 均需为 8 倍数
    （core.py 有断言）；非 contiguous 输入会触发未对齐写（UB）
12. **"性能提升"必须先验证正确性再谈性能**：早期 V_T/32 行实现的部分"性能提升"
    来自错误输出（layout 错或只处理前 32 key）
13. **LDS 数组大小必须匹配实际布局**：V_T tile LDS 缓存（3.8）初版沿用
    `BLOCK_N*VStride`（64×144），而 `[D][N]` 布局需 `HeadDim` 行 → 越界写 2 倍污染
    相邻 LDS，短序列（2048）恰好无感、4096+ 输出 err=0.13。定位方法：先小序列
    （2048）对 reference 验证，再逐步放大；LDS 数组尺寸用三元表达式按布局区分
    （`(SAGEATTN_VT_GLOBAL) ? (HeadDim*VTileStride) : (BLOCK_N*VStride)`）

---

## 八、失败的优化尝试（避免重复）

| 优化 | 结果 |
|------|------|
| D=128 每 warp 32 行 kernel | VGPR 溢出（256 上限），慢 10-24x |
| MIN_BLK 调大（64/128, 对齐 triton 粒度） | quant +37-45%（fmax 串行链）；attn 退化 4% |
| BN 扫描（D=64 int8 16/32/64/128） | 大 BN 有害；BN=16/32 最优（D=128 则 BN=64 最优） |
| wpe1/2/4（occupancy） | 全部无改善或更差——occupancy 非瓶颈 |
| v_transpose 大 tile（64n×64d/128n×32d, 8-32 half/thread） | 均更慢（LDS bank 冲突或调度） |
| v_transpose uint2 显式向量化 | 更慢（编译器已自动优化） |
| quant 双累加器 / 跳过 blockReduce | 无帮助（瓶颈非依赖链/归约） |
| QK WMMA 依赖链拆分（2 累加器） | +3% |
| k_frag 提前加载到寄存器 | 0% |
| KStride+20（消 LDS bank 冲突） | +3% |
| softmax max/sum 树形归约 | +3% |
| K 直读 global（消 LDS+barrier） | +8%（gather 的 L2 利用率低） |
| QK 用 fp16 WMMA（i8 转 fp16） | +33%（转换开销超 WMMA 提速） |
| fast_exp2 位近似 / v_max3 / permlanex16→shfl / 双缓冲 | 全部失败 |
| BM=128/8 warps（VAE 超长 self, K/V 重读减半） | ±1% 无效 → 非 DRAM 带宽/重读受限 |
| BN=16/32/128（D=128 int8, VAE） | BN=16/32 慢 4-14%；BN=128 持平 → 保持 BN=64 |
| v_tile 缩为 1 元素（occupancy 4→6 blocks/CU） | 无性能变化 → occupancy 非瓶颈（省 LDS 仍有益, 3.9） |
| quant 多 group 合并（GPB=2/4/8 串行循环） | 无改善 → quant 瓶颈非 block 调度/DRAM 行切换 |
| triton PV_ACCUM fp16（VAE） | 无收益（ROCm triton 未受益） |
| triton BLOCK_M=128/64×8 warps（VAE） | autotune 未选中（BM=64/BN=32 更优） |
| **fp8e5m2 V_T 存储 + fp16 WMMA（FeatherOps 思路, 2026-08, 见八之附）** | **整体无提升**（13 用例仅 1 个快 7%，多数慢 0-24%） |

**规律**：指令级微优化全部失败；有效的优化是**结构性**的（V_T 消除 LDS 列读、
32 行共享 k_frag 提升 ILP、LDS 缓存省全局重读、bf16 融合省 kernel、差异化分发省辅助、
**V_T tile LDS 缓存消除 PV 冗余全局读**）。

---

## 八之附、fp8e5m2 存储方案实验总结（2026-08, ComfyUI-FeatherOps 思路移植）

**动机**：`ComfyUI-FeatherOps` 用 fp8e5m2 存 matmul 的 B 矩阵（LDS/全局），加载时
2 条 V_PERM_B32 快速 upcast 到 fp16（两者指数 bias 相同），省加载带宽/LDS 占用。
在 Strix Halo 上 fp16@fp8e5m2 matmul 达 43 TFLOPS vs Tensile fp16 36 TFLOPS（+20%）。

**移植方案**（详细实验过程见仓库 `fp8e5m2存储优化实验记录.md`）：
- V_T 以 fp8e5m2 存（v_transpose 融合量化, per-token scale），PV 的 v_frag 读 16B +
  perm upcast，P 列乘 scale（数学等价 out=(P·scale)@V_fp8）。direct 与 int8 路径共用。
- 正确性：fp8 路径 cos≈0.998（pytest 28/36 过, 8 个失败为 mae 0.36-0.51 超 0.35 阈值,
  2bit 尾数固有误差）；fp16 路径零破坏（36/36 过）。
- 性能（同进程轮转, 13 用例）：仅 SDXL10（direct self 1536）快 6.7%；VAE 慢 24%、
  Anima02 慢 32%、Anima01 慢 13%、短 kv cross 慢 8-19%。

**失败根因**（与 FeatherOps matmul 的本质差异）：
- FeatherOps 收益前提是**长 K-loop**（K=4096），加载指令占 K-loop 比例大；
  attention 的 QK/PV 的 **K 维只有 head_dim（64/128）**，WMMA 计算主导，
  v_frag 加载（32B→16B）占比小。
- fp8 引入的 perm upcast（8 条/v16h）+ P×scale（8 float mul）+ 额外 v_scale kernel
  （读 V 全量）无法被省下的加载带宽覆盖。
- LDS 中转路径（int8 D=128/VAE）更差：V_T tile 每迭代从 L2 读（2MB L2 覆盖，
  带宽本不稀缺），perm 是纯开销 → +13~24%。
- 唯一正收益用例的 L2 稀缺带宽场景收益不稳定（同形状 2304 反而慢 4.6%）。

**状态**：**代码已回退**（性能无改善，按项目决策仅保留文档记录；完整实验过程见
`fp8e5m2存储优化实验记录.md`，实验脚本与数据均可复现）。

---

## 九、剩余可优化方向

1. **SDXL10/16 仍有 7-13% 差距**（direct 路径，i8 WMMA 2x 吞吐是架构性差距）：
   - 若 RDNA3 有 i8→fp16 的批量转换指令，QK 可回 int8（当前逐元素转换开销 33%）
   - 或者直接参考 triton 的 int8 attn + 大 block quant 组合（triton 的 quant 0.23ms
     得益于 128 行大 block，但 native 大 block 会触发 fmax 串行链——需多累加器 ILP 方案）
2. **SDXL01/07/13 慢 3-6%**（int8 路径）：attn 已持平 triton，差距=辅助 kernel
   （quant 0.30 + v_transpose 0.23 + mean 0.09 = 0.62ms，短序列占比高）：
   - mean_seq 融合进 quant（跨 block 归约需原子或二次 kernel）
   - v_transpose 的 30GB/s 上限（转置类带宽瓶颈）——若 DDR5 带宽更高可重测
3. **quant 的 32MB 读仅 20GB/s**：与 torch sum 的 75GB/s 差 3.6 倍，根因未完全定位
   （非读粒度/归约/计算；2026-08 实测 block 级多 group 合并（GPB=2/4/8）无改善，
   排除 block 调度/DRAM 行切换假设；疑为 LDS 双缓冲/延迟隐藏不足，可试 pass1 与
   pass2 软件流水（pass1 预读下组数据到寄存器））
4. **AnimaVAE（bf16）仍慢 triton 2-3%**（VAE 优化后）：attn 主 kernel 已持平，
   差距=bf16 输入需 v_transpose 转 fp16（triton 直接读 bf16 原布局）：
   - V_T 保持 bf16（v_transpose 免转换）+ PV 用 bf16 WMMA（v_wmma_f32_16x16x16_bf16，
     吞吐与 fp16 相同）——省 v_transpose 转换开销
   - 或 v_transpose 与 attn 融合（分块转置+计算，消除 V_T 全局往返）——大工程
5. **GQA 场景**（h_q≠h_kv）：当前按 head 分组处理，kv_heads 共享可进一步优化
   （benchmark 未覆盖 GQA 性能；D=128 GQA 下 int8 优势更大，阈值需按 hq/hkv 比调整）
6. **D=128 direct 的 BM 探索**：当前 BM=64/BN=16，可测 BM=32/128（direct 路径
   未做 BM 全扫描）
7. **causal D=64 kv>8192 边界**：8192 时 int8 已优 5.5%，但 7168 附近未测，
   阈值 6144 可能略保守
8. **VAE 03/04（36864/55296）未实测**：因 iGPU 长时间运行死机已从 benchmark 注释；
   优化结论由 16384/24576 外推（LDS PV 收益在更大序列应更显著）
