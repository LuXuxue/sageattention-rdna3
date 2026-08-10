# SageAttention Native Backend 优化报告

> 本报告是项目优化的总记录：涵盖硬件基础、核心算法、实验过程与结论、分发逻辑、易踩坑点与剩余优化方向。

## 一、最终结论（当前最优状态）

Native backend 的核心优化链（按收益排序）：
1. **V 全局转置 PV（`SAGEATTN_VT_GLOBAL=1`）**：V 一次性转置为 `V_T [B,H,D,N]`，PV 的
   A=V^T 从 128 次 u16 LDS 列读变为 1 条行读 → direct 路径真实提升 17-36%
2. **每 warp 32 行 QK kernel**：D=64 int8 self 用 BM=128/4 warps，2 子块共享 k_frag
   （ILP）→ int8 self 从慢 7-13% 追平
3. **bf16→fp16 转换融合进 v_transpose**：省掉 `v.to(fp16)` 独立 kernel（Anima01 的 5%）
4. **D=128 int8 BN=32→64**：kv-tile 数减半（barrier/k_tile 填充减半）
5. **quant pass1→LDS 缓存**：pass2 从 LDS 读，省一半全局读流量（2.28→2.15ms）
6. **差异化 direct/int8 分发**（按 kv_len/q_len/is_causal 选择路径）
7. **写回向量化**（permlanex16 + 16B 连续写，全部 kernel 一致）

**当前 benchmark 状态（24 用例全 OK，Radeon 780M）**：

| 类别 | native/triton | native/FA | 说明 |
|------|--------------|-----------|------|
| Anima D=128 int8 self（4096/6144/9216） | 0.976/0.955/1.000 | 0.949/0.939/0.965 | **全面超 triton 和 FA** |
| Anima D=128 direct cross（kv=512） | 0.89-0.91 | 0.82-0.88 | **超 triton 9-11%** |
| SDXL direct cross（kv<=2048） | 0.95-1.0 | 0.86-1.2 | 持平/超 |
| SDXL01/07/13（D=64 int8 self, kv>=4096） | 1.03-1.06 | 1.00-1.07 | 慢 3-6%（辅助占比高） |
| SDXL10/16（D=64 direct, kv=1536/2304） | 1.07-1.13 | 1.07-1.13 | 慢 7-13%（架构性） |

**关键结论**：
- i8 WMMA 在 RDNA3 为 fp16 的 2 倍吞吐（QK 占一半计算量）→ int8 在长 self-attn 占优；
  direct 省 quant+mean 辅助（固定 0.4-0.6ms）→ 在计算量小（causal/cross 短 q）时占优
- WMMA 硬件吞吐是 int8 的剩余上限（D=128 attn 达 ~10.9 TFLOP）
- **默认编译配置（无 env）即为各用例最优配置**

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
| `SAGEATTN_INT8_BN128` | D=128 int8 的 BN 覆盖（0 默认 64, 16/32） | 0（=64） |
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

**规律**：指令级微优化全部失败；有效的优化是**结构性**的（V_T 消除 LDS 列读、
32 行共享 k_frag 提升 ILP、LDS 缓存省全局重读、bf16 融合省 kernel、差异化分发省辅助）。

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
   （非读粒度/归约/计算；疑为 6144 个小 block 的调度与 DRAM 行切换）——可尝试
   block 级循环合并多个 group（保持每 thread 少量 packs）
4. **GQA 场景**（h_q≠h_kv）：当前按 head 分组处理，kv_heads 共享可进一步优化
   （benchmark 未覆盖 GQA 性能；D=128 GQA 下 int8 优势更大，阈值需按 hq/hkv 比调整）
5. **D=128 direct 的 BM 探索**：当前 BM=64/BN=16，可测 BM=32/128（direct 路径
   未做 BM 全扫描）
6. **causal D=64 kv>8192 边界**：8192 时 int8 已优 5.5%，但 7168 附近未测，
   阈值 6144 可能略保守
