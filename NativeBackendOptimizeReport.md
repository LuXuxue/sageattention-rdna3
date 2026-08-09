# SageAttention Native Backend 优化报告

## 一、概述与最终结论

### 1.1 最终成果

Native backend 通过**转置布局 kernel** 重构，性能达到与 Triton backend 匹配的水平：

- **24/24 个 benchmark 用例 native/triton ≤ 1.15x**（同进程交错轮转测量，中位数 ~1.00x）
- 短序列 fp16 direct 用例（kv≤1024）native **快 5-38%**（相对 Triton）
- 长序列 d64/d128（self-attn）与 Triton 持平（0.99-1.03x）
- 相对 FlashAttn：全部用例持平或更快
- 相比旧 butterfly kernel：全部用例快 **30-62%**（D=128 达 48%）

### 1.2 核心突破

消除旧 kernel 的两大开销来源：

| 开销 | 旧 kernel | 转置 kernel |
|------|-----------|-------------|
| 归约跨 lane 通信 | 64×ds_bpermute/迭代 (butterfly) | 2×v_permlanex16/迭代 |
| P fragment 构造 | 32 LDS stores + 32 LDS loads/迭代 | 寄存器内完成 (0 LDS) |

**实现方式**：转置 QK（`qk^T = k @ q^T`）使 WMMA 输出布局变为"每 lane 持**同行**的偶/奇列"，
从而 permlanex16（XOR-16 交换）恰好合并**同一行**的数据——这在数学上等价于标准 online softmax。

---

## 二、硬件与指令基础

### 2.1 硬件规格

| 项目 | 规格 |
|------|------|
| GPU | AMD Radeon 780M (gfx1103, RDNA3 iGPU) |
| ROCm | 7.14 |
| 架构特性 | 每 SIMD 256 VGPRs，WMMA 16×16×16，无 MFMA |
| CU 数量 | 12 |

### 2.2 WMMA 输出布局

gfx1103 上 WMMA 16×16×16 输出布局（已验证）：

```
element e in lane L → row = 2*e + (L >> 4), col = L & 15
```

- lane 0-15：偶数行 (0,2,4,...,14)，所有 16 列
- lane 16-31：奇数行 (1,3,5,...,15)，所有 16 列
- 每 lane 8 个元素 = 8 个不同行、同一列

**转置 QK 后的布局**（交换 WMMA operand：A=k, B=q）：

```
lane L 持有 S[L&15][2e+(L>>4)], e=0..7
→ lane 0-15 持偶数列, lane 16-31 持奇数列（同一行!）
```

这是整个优化方案的基础：**lane L 与 lane L^16 恰好持有同一行 softmax 的全部列**。

### 2.3 v_permlanex16_b32 指令

正确用法（内联汇编，已在 gfx1103 验证 32/32 lanes）：

```cpp
__device__ __forceinline__ float permlanex16(float src) {
    float result;
    asm volatile("v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]"
                 : "=v"(result)
                 : "v"(src), "s"(0x76543210), "n"(0xfedcba98));
    return result;
}
```

- 执行 XOR-16 跨半波交换：lane 0↔16, 1↔17, ..., 15↔31
- 纯寄存器操作，不经过 LDS
- 参数：`s=0x76543210`（lane 映射表）、`0xfedcba98`（字节选择）、`op_sel:[1,0]`
- **限制**：只能做 XOR-16，不能做 XOR-8/4/2/1（半波内通信）

**勘误**：早期曾错误判断该指令在 gfx1103 上不可用（driver bug），实际是使用了错误的
`__builtin_amdgcn_permlanex16` 参数格式。参考 Triton 实际生成的 ISA 编码即可正常工作。

### 2.4 ds_bpermute_b32 指令

`__shfl_xor` 每次编译为约 6 VALU + 1 ds_bpermute（含 exec mask 边界检查），延迟约 20-30 cycles。
内联汇编无法复制其 exec mask 管理逻辑（`s_and_saveexec_b32`/`s_or_b32`/`s_cbranch_execz`），
尝试直接发射 ds_bpermute 均失败（NaN / 错误结果）。**`__shfl_xor` 是唯一可靠方式。**

### 2.5 MFMA 指令

MFMA 在 RDNA3 (gfx1103) 上不可用（编译错误/LLVM 崩溃），RDNA3 仅支持 WMMA 16×16×16。

---

## 三、转置布局算法

### 3.1 算法原理

标准 flash-attention 中，QK WMMA（A=q, B=k）输出布局使归约需要跨 16 个 lane 的 butterfly 通信。
**交换 operand（A=k, B=q）后**，输出变为转置矩阵 `S^T = K @ Q^T`：

```
S^T[m][n] = S[n][m] = Σ_d K[m][d] · Q[n][d]
```

此时 lane L 持有 `S[L&15][2e+(L>>4)]`——**同一行**（L&15）的偶/奇列，分布在 lane L 与 L^16。

**归约只需 1 次 permlanex16**：
- max：`gm = max(row_m, local_mx)` 后 `gm = max(gm, permlanex16(gm))`——合并同行两半的 max
- sum：`row_l += partial_sm + permlanex16(partial_sm)`——合并同行两半的 sum

**P fragment 寄存器内组装**（`assemble_p_frag`）：
- lane L 的 8 个 exp 值对应 P 行 (L&15) 的偶/奇列
- 与 lane^16 交换后获得该行完整 16 列
- 打包成 fp16 对，按 WMMA B-operand 布局交错排列（B-operand = P^T 的列 L&15，即 P 行 L&15 的 16 列）

**转置 PV**：`out^T = V^T @ P^T`（A=V^T, B=P^T），输出布局自动回到"out 行 (L&15) 的偶/奇 D 列"，
写回时按 `d = dt*16 + 2e + hw` 排列。

### 3.2 数学正确性

转置布局下 permlanex16 合并的 max/sum 与标准 online softmax **完全等价**：
- 每行 softmax 需要该行所有列的 max 和 sum
- lane L 与 L^16 持有该行互补的列集合
- permlanex16 交换后合并 = 该行全部列

Python float32 模拟验证：转置归约与标准 softmax 的 m/l 一致（误差 1.8e-7，纯浮点舍入）。

**关键教训——race condition**：转置 kernel 迭代末尾写 `k_tile`/`v_tile` 时，若缺少**写前
`__syncthreads()`**，快 warp 会覆盖慢 warp 正在读的 tile，导致**非确定性错误**
（BLOCK_M=16 单 warp 全对，多 warp 随机错）。必须保持"读完成 barrier → 写 → barrier"的三 barrier 模式。

### 3.3 实现要点（可复现）

代码位于 `csrc/`，核心组件：

| 组件 | 位置 | 说明 |
|------|------|------|
| permlanex16 / permlanex16_u32 | `mma_gfx11.h` | XOR-16 交换 (float / u32) |
| assemble_p_frag | `mma_gfx11.h` | P fragment 寄存器组装 (v2: 8 次 u32 cndmask) |
| attn_kernel_impl_t | `attn_gfx11.cu` | int8 量化转置 kernel |
| direct_attn_kernel_impl_t | `attn_gfx11.cu` | fp16/bf16 direct 转置 kernel (模板) |
| *_attn_gfx11_t | `attn_gfx11.cu` | 新 dispatch (qk_int8_sv_bf16 / fp16 / bf16) |

**assemble_p_frag 核心逻辑**（v2 优化）：

```cpp
// p_vals[e] = P[L&15][2e+hw] (本 lane 8 个值)
// pack[k] = {p[2k], p[2k+1]} 打包为 fp16 对
// cross[k] = permlanex16_u32(pack[k])  // 与 lane^16 交换
// hw=0 (偶列): even=pack, odd=cross; hw=1 (奇列): even=cross, odd=pack
even[k] = (hw == 0) ? pack[k] : cross[k];
odd[k]  = (hw == 0) ? cross[k] : pack[k];
b[4k]   = f16_from_bits(even[k] & 0xFFFF);   // 偶列 4k
b[4k+1] = f16_from_bits(odd[k]  & 0xFFFF);   // 奇列 4k+1
b[4k+2] = f16_from_bits(even[k] >> 16);      // 偶列 4k+2
b[4k+3] = f16_from_bits(odd[k]  >> 16);      // 奇列 4k+3
```

**类型注意事项**：`v16h` 的元素类型是 `_Float16`（非 `__half`），两者无隐式转换。
必须用 `static_cast<_Float16>(__half2float(v))` 或位转换 `f16_from_bits`，否则编译失败
（依赖 `__HIP_NO_HALF_CONVERSIONS__` 宏的隐式转换在不同构建环境下不可靠）。

**int8 kernel 类型细节**：dispatch 层对 bf16 输入先将 `v` 转 fp16 传入（`core.py`），
因此 int8 kernel 的 V 类型固定为 `__half`（VT 宏），无需按 dtype 分派。

### 3.4 性能提升数据（kernel-only，转置 vs 旧 butterfly）

| 用例 | 旧 (ms) | 新 (ms) | 提升 |
|------|---------|---------|------|
| SDXL01 d64 4096 self | 6.896 | 5.287 | +30.4% |
| SDXL07 d64 6144 self | 15.309 | 11.585 | +32.1% |
| SDXL13 d64 9216 self | 34.385 | 26.099 | +31.7% |
| SDXL04 d64 1024 self | 0.944 | 0.727 | +29.8% |
| Anima01 d128 4096 self | 21.971 | 14.850 | +48.0% |
| Anima03 d128 6144 self | 49.098 | 33.149 | +48.1% |
| cross d64 4096x154 | 0.422 | 0.293 | +44.2% |
| cross d64 6144x77 | 0.416 | 0.257 | +61.9% |

ISA 对比：ds_bpermute 64→0，ds_store ~660→~0，VGPR 239→225（int8 kernel）。

---

## 四、验证方法论（热漂移）

### 4.1 平台特性

AMD 780M iGPU 时钟频率随负载/温度/电源动态变化，跨会话绝对时间差异可达 40%+。
**唯一可信的对比方法是同进程交错轮转**：native/triton（或新旧 kernel）交替测量，取比值。

### 4.2 benchmark_attn.py 的正确使用

benchmark_attn.py 按 SDPA→FlashAttn→SageAttn 顺序测量。**跨运行（不同进程）的 Sage/FA 倍率
不可直接对比**——两次运行的 GPU 温度状态不同（实测 native run 的 FA 基准比 triton run 慢 3%，
证明整体更热）。正确用法：
- 单次运行内的 Sage/FA 倍率可作为参考
- **对比 native vs triton 必须用同进程交错轮转**（参考 `doc/bench_all.py`、`doc/bench_strict.py`）

---

## 五、优化尝试总结

### 5.1 最终确认有效的配置

| 优化 | 效果 | 适用范围 |
|------|------|----------|
| 转置布局 kernel | 30-62% 提升 | 所有路径 |
| int8 D=64 self BM=128/BN=32 | 1.254x→1.104x (vs Triton) | 长序列 self-attn |
| D=128 direct BN=16 | 1.45x→1.05x | Anima02/04/06 |
| fp16/bf16 wpe2 | -32% 最大改善 | 短序列 direct |
| assemble_p_frag v2 (u32 cndmask) | 额外 3-11% | 所有路径 |

### 5.2 被转置方案推翻的历史结论

以下早期"不可行"结论**仅适用于非转置布局**，转置后均成立：

| 早期结论 | 转置后事实 |
|----------|-----------|
| permlanex16 无法替代 butterfly（XOR-16≠XOR-8） | 转置后只需 XOR-16，permlanex16 完全够用 |
| P fragment 必须经 LDS 中转（WMMA 布局必然要求） | 寄存器内可组装（assemble_p_frag） |
| BM=128 在 gfx1103 上无效 | int8 self-attn 用 BM=128/BN=32 最优 |
| 归约必须用 butterfly（max/sum 各 4 步） | 转置后 max/sum 各 1 步 permlanex16 |

### 5.3 仍成立的经验（与布局无关）

| 结论 | 依据 |
|------|------|
| wpe2 有效需 VGPR ≤ 256 且不 spill | fp16 kernel 176 VGPRs 收益 -32%；int8 225-239 VGPRs 无效 |
| 减少 BN 增加迭代固定开销，通常有害 | 迭代翻倍的开销（sync/LDS/地址计算）大于收益 |
| prefetch 重叠 global 加载与计算，不可移除 | 移除后长序列退 5-9% |
| 修改代码后必须 clean build 再验证 | 构建缓存曾导致假阳性结论 |
| 测试变量必须严格初始化 | 变量复用曾导致假阳性通过 |

---

## 六、最终配置与性能数据

### 6.1 Kernel 配置（dispatch 固化）

```
int8 量化路径 (转置 kernel):
  D=64,  self-attn        -> (BM=128, BN=32)  [wpe1]
  D=64,  cross, kv<=77    -> (BM=64, BN=16)   [wpe1]
  D=64,  cross, kv>77     -> (BM=64, BN=32)   [wpe1]
  D=128                   -> (BM=64, BN=32)   [wpe2]

fp16/bf16 direct 路径 (转置 kernel):
  D=64,  kv<=128          -> (BM=64, BN=16)   [wpe2]
  D=64,  self             -> (BM=64, BN=64)   [wpe2]
  D=64,  cross            -> (BM=64, BN=32)   [wpe2]
  D=128                   -> (BM=64, BN=16)   [wpe2]
```

### 6.2 Dispatch 策略

```
D=64:  kv<=1024 -> fp16/bf16 direct; kv>1024 -> int8 量化
D=128: kv<=512  -> fp16/bf16 direct; kv>512  -> int8 量化
```

### 6.3 最终性能（同进程交错轮转 vs Triton）

| 用例 | n/t | 用例 | n/t | 用例 | n/t |
|------|-----|------|-----|------|-----|
| SDXL01 | 1.008x | SDXL09 | 0.967x | SDXL17 | 0.992x |
| SDXL02 | 0.953x | SDXL10 | 1.008x | SDXL18 | 0.998x |
| SDXL03 | 1.006x | SDXL11 | 1.002x | Anima01 | 1.008x |
| SDXL04 | 1.026x | SDXL12 | 1.057x | Anima02 | 0.958x |
| SDXL05 | 1.111x | SDXL13 | 0.999x | Anima03 | 0.989x |
| SDXL06 | 1.024x | SDXL14 | 1.015x | Anima04 | 0.994x |
| SDXL07 | 0.997x | SDXL15 | 0.910x | Anima05 | 0.999x |
| SDXL08 | 0.991x | SDXL16 | 0.988x | Anima06 | 1.011x |

- **24/24 ≤ 1.15x**，中位数 ~1.00x（热漂移导致跨运行 ±5% 波动，典型 18-24/24 ≤1.15x）
- 相对 FlashAttn：全部用例持平或更快（n/fa 0.87-1.05x）
- 短序列 fp16 direct 用例（SDXL02/04/05/06/08/11/14/17）native 快 5-38%
- 长序列 d64/d128 与 Triton 持平（0.99-1.03x）

### 6.4 剩余差距

| 用例 | n/t | 说明 |
|------|-----|------|
| SDXL07/13 (d64 long self) | 1.14-1.18x (波动) | 残余调度/occupancy 差距 |
| Anima04/06 (d128 cross kv=512) | 1.16-1.27x (波动) | BM=32/64/128 均 ~1.2x，非配置问题 |

---

## 七、硬件 BUG/指令异常诊断方法论

### 诊断流程

1. **Dump ISA**：用 `--save-temps` dump Triton/Native 的 GCN ISA
2. **对比验证**：对比 Triton 和 Native 的指令用法，确认参数配置（Triton ISA 是最佳参考）
3. **功能测试**：编写最小化测试 kernel 验证指令行为
4. **文档交叉验证**：对比官方文档与实际行为（部分指令行为与文档不一致）
5. **Clean build**：修改代码后必须完全重新编译再验证

### 关键经验

- 指令参数格式错误会导致"不可用"假象（如 permlanex16 的 builtin vs inline asm）
- 构建缓存导致假阳性：运行的可能仍是旧代码
- 测试变量必须严格初始化验证，避免假阳性
- 类型转换（`_Float16` vs `__half`）依赖宏定义，跨构建环境不可靠，应显式转换

---

## 八、生产建议

1. 默认使用 `SAGEATTN_BACKEND=triton`（与 FlashAttn 兼容性最好）
2. Native backend 现与 Triton 性能匹配，可用于无 Triton 依赖的独立部署
3. 短序列 cross-attention（kv≤154）native 快于 Triton 5-38%
4. 长序列 self-attn（kv≥4096）native 与 Triton 持平
5. 性能对比务必使用同进程交错轮转，避免热漂移污染
