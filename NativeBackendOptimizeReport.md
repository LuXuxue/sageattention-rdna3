# SageAttention Native Backend 优化报告

## 一、最终结论（当前最优状态）

Native backend 当前包含**三个核心优化**：转置布局 kernel（早期重构）、**V 全局转置 PV**、**每 warp 32 行 QK kernel**。

**当前 benchmark 状态（24 用例全部 OK，旋转轮转 vs Triton）：**

| 类别 | n/t（native/triton） | 说明 |
|------|---------------------|------|
| int8 长序列 self-attn（D=64） | **1.01-1.12**（多数 1.03） | 32 行/warp 后从 1.07-1.13 追平 |
| Anima d128 int8 self | 1.03 | 与 triton 持平 |
| direct 短序列（SDXL04/Anima cross） | **0.60-0.95** | **native 快 5-40%**（V_T 收益） |
| SDXL02-18 direct cross | 0.62-0.85 | native 快 15-38% |

**关键结论**：
- **V 全局转置 PV（`SAGEATTN_VT_GLOBAL=1`）**：direct 路径真实提升 17-36%（消除 v_frag 的 128 次 u16 LDS 列读）
- **每 warp 32 行 QK kernel**：int8 self 从慢 7-13% 追平到 1-3%（共享 k_frag 的 ILP）
- **occupancy（wpe2/4）与配置（BM/BN）均不是当前瓶颈**；WMMA 硬件吞吐是 int8 的剩余上限

---

## 二、硬件与指令基础

### 2.1 硬件规格

| 项目 | 规格 |
|------|------|
| GPU | AMD Radeon 780M (gfx1103, RDNA3 iGPU) |
| ROCm | 7.14（TheRock 提供工具链，`rocm-sdk path --root` 定位） |
| CU | 12 |
| LDS | 128 KiB / CU |
| VGPR | 512 KiB / CU（每 SIMD 256 VGPRs/lane 上限，warp 级 1024 槽） |
| L2 | 2 MiB（无 Infinity Cache） |
| 显存 | 共享系统内存（DDR5/LPDDR5，~76.8 GB/s） |
| 矩阵指令 | WMMA 16×16×16（**无 MFMA**） |

### 2.2 WMMA 布局（精确版，probe 实验验证）

用 `probe_wmma.cu`（已知输入 A/B 观察 C 元素映射）确定的精确布局：

```
C 输出:  lane L 持有 C[2e+(L>>4)][L&15], e=0..7   （行 = 2e+hw, 列 = L&15）
A operand: lane L 提供 A 行 L&15 的 16 列          （a[k] = A[L&15][k]）
B operand: lane L 提供 B 列 L&15 的 16 行          （b[k] = B[k][L&15]）
```

- **A 与 B 的 operand 布局不对称**（A 按行、B 按列）——这是转置 PV 设计的基础，也是 V_T 踩坑的根源
- 每 lane 的 A/B operand 为 16 个元素（`v16h`），C 为 8 个元素（`v8f`）

**转置 QK 的用途**：交换 operand（A=k, B=q^T）后，WMMA 输出为 `S^T = K @ Q^T`，其布局
`lane L 持有 S[L&15][2e+(L>>4)]` 恰好使 **lane L 与 L^16 持有同一 softmax 行的偶/奇列**，
从而 permlanex16（XOR-16 交换）一次即可合并整行归约。

### 2.3 v_permlanex16_b32 指令

```cpp
__device__ __forceinline__ float permlanex16(float src) {
    float result;
    asm volatile("v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]"
                 : "=v"(result)
                 : "v"(src), "s"(0x76543210), "n"(0xfedcba98));
    return result;
}
```

- 执行 XOR-16 跨半波交换（lane 0↔16, ..., 15↔31），纯寄存器操作
- **限制**：只能 XOR-16，不能做半波内（XOR-8/4/2/1）通信
- **踩坑**：早期误判该指令在 gfx1103 不可用（driver bug），实际是 `__builtin_amdgcn_permlanex16`
  参数格式错误；参考 Triton 生成的 ISA 编码（`v_permlanex16_b32` + `op_sel:[1,0]`）即可
- **性能验证**：`__shfl_xor_sync` 替代 permlanex16 反而慢 19%——v_permlanex16 本身是高效的

### 2.4 ds_bpermute_b32 与 __shfl_xor

`__shfl_xor` 编译为约 6 VALU + 1 ds_bpermute（含 exec mask 边界检查），延迟约 20-30 cycles。
内联汇编无法复制其 exec mask 管理逻辑（`s_and_saveexec_b32` 等），直接发射 ds_bpermute 失败。
**`__shfl_xor` 是唯一可靠方式**，但本方案用 permlanex16 而非它。

### 2.5 MFMA

MFMA 在 RDNA3 (gfx1103) 不可用（编译错误/LLVM 崩溃），仅支持 WMMA 16×16×16。

---

## 三、转置布局算法

### 3.1 算法原理

标准 flash-attention 的 QK WMMA（A=q, B=k）输出布局使归约需跨 16 个 lane 的 butterfly 通信。
**交换 operand（A=k, B=q）后**输出为转置矩阵 `S^T = K @ Q^T`：

```
S^T[m][n] = S[n][m] = Σ_d K[m][d] · Q[n][d]
```

lane L 持有 `S[L&15][2e+(L>>4)]`——同一行（L&15）的偶/奇列，分布在 lane L 与 L^16。

**归约只需 1 次 permlanex16**：
- max：`gm = max(row_m, local_mx)` 后 `gm = max(gm, permlanex16(gm))`
- sum：`row_l += partial_sm + permlanex16(partial_sm)`

**P fragment 寄存器内组装**（`assemble_p_frag`）：lane L 的 8 个 exp 值对应 P 行 (L&15) 的偶/奇列，
与 lane^16 交换后获得该行完整 16 列，打包成 fp16 对，按 WMMA B-operand 布局交错排列。

**转置 PV**：`out^T = V^T @ P^T`（A=V^T, B=P^T），输出布局自动回到
"out 行 (L&15) 的偶/奇 D 列"，写回按 `d = dt*16 + 2e + hw` 排列。

### 3.2 V 全局转置 PV（核心优化之一）

**动机**：原转置 PV 中 A = V^T 需要 **V 的列读**（跨行），经 LDS 中转后每迭代
`load_fp16_col_frag` 产生 **128 次 u16 标量 LDS 读**（2B/lane，低效且 bank 冲突）。

**解法**：把 V 一次性转置为 `V_T [B,H,D,N]`（n 维连续），使 PV 的 **A = V^T 变为行读**：

```
A = V^T:  lane L 提供 A 行 L&15 的 16 列
          = V^T[L&15][n] = V[n][L&15] = V_T 行 L&15 的 16 连续 n -> 行读 b128（1 条 v16h）
B = P^T:  lane L 提供 P 行 L&15 的 16 列 -> 复用现有 p_frag 布局（assemble_p_frag 不变）
C = out^T: 转置解释后 = out 行 L&15 的偶/奇列 -> 写回复用
```

- **wmma 参数顺序必须是 `wmma(v_frag_t, p_frag, out_acc)`**（A=V^T, B=P^T, C=out^T 匹配写回）；
  若写成 `wmma(p_frag, v_frag_t, ...)`（A=P, B=V, C=out）则 C 布局与写回转置，输出乱序（踩坑点 7）
- **vt_off 必须含 kb_base**：`((b*H+hkv)*D + dt*16+m_row)*N + (kb_base + ct*16)`，
  漏掉 kb_base 会导致每迭代都读 V_T 前 32 个 n（只处理前 32 个 key，踩坑点 8）
- V_T 偏移假定 contiguous `[B,H,D,N]`；int8 路径 kv_len 恒为 8 倍数 → v16h 32B 读对齐安全
- 实现：新增 `v_transpose` op（[B,N,H,D]→[B,H,D,N]，NHD/HND 均支持，一次性 kernel 开销 +0.14ms@4096）
- core.py 在 int8 与 direct 路径均无条件转置 V；`setup.py` 默认编译 `-DSAGEATTN_VT_GLOBAL=1`

### 3.3 每 warp 32 行 QK kernel（核心优化之二）

**动机**：int8 瓶颈分解显示 QK WMMA 占 38%（1.93ms），WMMA 延迟未被有效隐藏。

**设计（`attn_kernel_impl_32_t`，BM=128, 4 warps）**：
- 每 warp 处理 32 行（2 个 16 行子块 sub0/sub1）
- **2 子块共享 k_frag 加载** → 每迭代产生 2 条独立 WMMA 依赖链（`score_acc0`/`score_acc1`）
- softmax/PV/写回均 2 子块化（permlanex16 归约不变）
- VGPR 压力：2 组 per-q 状态，通过共享 k_frag 控制在 256 VGPR/lane 上限内（实测可编译不溢出）
- D=64 self-attn **默认启用**（`SAGEATTN_INT8_32=0` 可回退 8 warps）

**验证**：与 8 warps 版输出逐位一致（`torch.equal True`）；wpe4 变体无额外收益。

### 3.4 V/OUT dtype 分离（方案 B）

- **V 预先转 fp16**（`v.to(fp16)`，带宽受限的 elementwise kernel，远快于 kernel 内逐元素转换；
  实测 kernel 内 bf16→fp16 标量转换开销 1.6-5.5%）
- **输出由 kernel 直接写 bf16**（`OUT_DTYPE` 模板参数，o 复用省 `o.to(bf16)` 转换 kernel，
  且单次舍入精度更好）
- V_T 转置在 fp16 转换之后进行（V_T 恒为 fp16，kernel 的 `V_DTYPE` 分支不触发）

---

## 四、测量方法论

### 4.1 热漂移与频率抖动

- iGPU 冷启动后**频率剧烈波动**：同一 kernel 单次时间在 0.19ms ↔ 4.2ms 跳变（20 倍）
- 需 ~600 次 kernel 调用预热才收敛；**跨会话/跨进程绝对时间不可比**
- **固定顺序交错轮转有轮内顺序效应**（先测的后端更冷更慢）

### 4.2 修正后的测量方法

```
1. 每用例充分预热: 各后端交替跑 ≥600 次调用
2. 测量轮: 5-6 轮 × 每轮 20-30 iters
3. 轮间后端顺序旋转: 第 r 轮从 r 号后端开始（对称采样）
4. 各后端取中位数
```

**唯一可信对比 = 同进程内交错轮转的倍率。**

### 4.3 layout_code 陷阱（易踩坑）

- `tensor_layout="NHD"` 必须传 **layout_code=0（kNHD）**；传 1（kHND）会让 kernel 按
  HND stride 解释 NHD 数据 → **输出错误（err≈3.0）但执行更快（假象）**
- **任何"手动拆分 ops 调用"的验证必须与 core.py 逐参数一致**（含 layout_code）
- **native 正确性验证必须显式设 `SAGEATTN_BACKEND=native`**：core.py 默认 `_BACKEND=triton`，
  不设 env 时 check 脚本测的是 triton 后端（早期大量"正确性验证"因此无效）

---

## 五、实验过程与结论

### 5.1 早期配置探索

| 实验 | 结论 |
|------|------|
| BN 覆盖（16/32/64/128） | 大 BN 全面有害（VGPR 压力 → occupancy 下降 + 尾块浪费）；BN=16/32 最优 |
| BM 覆盖（64/128） | kv=154 cross 的 BM=128 快 9-13%（旋转轮转确认）；kv=77 大 q 与 self 无益 |
| wpe1/2/4（occupancy） | 全部无改善或更差（int8 8 warps 下 wpe4 更差）——**occupancy 非瓶颈** |
| v_tile LDS 转置布局 | 失败（转置写 8× 标量 + bank 冲突，净损失 +22%） |
| fast_exp2 位近似 / v_max3 / permlanex16→shfl / 依赖链拆分 / 双缓冲 | 全部失败（见 5.3） |

### 5.2 瓶颈分解（int8 attn kernel，SDXL01 5.04ms）

用运行时 if（非 `#if`）跳过段测得：

| 组成 | 时间 | 占比 |
|------|------|------|
| QK WMMA (i8) | 1.93ms | 38% |
| PV WMMA (fp16) | ~1.2ms | 24% |
| k_frag LDS 读 | 0.46ms | 9% |
| softmax | 0.49ms | 10% |
| v_frag_t global 读 | 0.14ms | 3% |
| barrier/写回/其他 | ~0.8ms | 16% |

**早期"softmax 2.37ms"结论是错误输出时代（只处理前 32 key）的假象**；正确版 softmax 仅 10%。

### 5.3 失败的优化尝试

| 优化 | 结果 |
|------|------|
| QK WMMA 依赖链拆分（2 累加器） | +3%（v8i 加法 + 寄存器抵消） |
| k_frag 提前加载到寄存器 | 0% |
| KStride+20（消 LDS bank 冲突） | +3% |
| softmax max/sum 树形归约 | +3% |
| K 直读 global（消 LDS+barrier） | +8%（gather 的 L2 利用率低） |
| QK 用 fp16 WMMA（i8 转 fp16） | +33%（转换开销超 WMMA 提速） |
| v_frag_t 常量（测读开销） | 仅省 0.14ms（非瓶颈） |
| 每 warp 32 行 + wpe4 | 无额外收益（wpe1 版有效，见 3.3） |

**规律**：指令级微优化（exp 近似/v_max3/shfl/树形化/依赖链拆分）全部失败；
有效的优化是**结构性**的（V_T 消除 LDS 列读、32 行共享 k_frag 提升 ILP）。

### 5.4 V_T 正确性修复过程（踩坑记录）

早期 V_T 实现有**两个正确性 bug**（当时"性能提升"是错误输出假象）：

1. **WMMA operand 顺序**：误用 `wmma(p_frag, v_frag_t)` 算 `out = P @ V`，其 C 布局
   （lane 持 out[2e+hw][L&15]）与写回代码（out[L&15][2e+hw]）**转置**。
   定位方法：corr(o,ref)=0.005 但 corr(sorted)=0.9997 → 值匹配但位置乱序。
   修正：`wmma(v_frag_t, p_frag)`（C=out^T 匹配写回）。
2. **vt_off 漏 kb_base**：v_frag_t 每迭代都读 V_T 前 32 个 n。
   定位方法：V[n][d]=n 时 o≈均匀(0..31) 和（15.3），V[n][d]=(n≥512) 时 o=0。
   修正：n 偏移加 `kb_base`。

**教训**：native 正确性必须用 `SAGEATTN_BACKEND=native` 独立验证（见 4.3）。

---

## 六、当前 kernel 结构与复现配置

### 6.1 编译

- `setup.py` 默认 `HIP_FLAGS` 含 **`-DSAGEATTN_VT_GLOBAL=1`**（V_T 方案）
- 构建：`pip install -e . --no-build-isolation`（Windows 需 MSVC + ROCm SDK 环境）
- 若需回退旧路径（无 V_T），改 setup.py 宏为 0 并关闭 core.py 转置

### 6.2 dispatch 规则（当前代码）

**Python 层路径选择**（`core.py`）：
```
head_dim==64: kv_len <= 1024 -> fp16/bf16 direct; kv_len > 1024 -> int8
head_dim==128: kv_len <= 512  -> fp16/bf16 direct; kv_len > 512  -> int8
V 在两条路径均无条件转置为 V_T [B,H,D,N]
```

**int8 路径**（`qk_int8_sv_bf16_attn_gfx11_t`）：
```
D=64  self-attn -> 每 warp 32 行 kernel (BM=128, BN=32, 4 warps) [默认]
                  (SAGEATTN_INT8_32=0 回退 8 warps wpe1)
D=64  cross, kv<=77 -> (BM=64, BN=16) [wpe1]
D=64  cross, kv>77  -> (BM=64, BN=32) [wpe1]
D=128               -> (BM=64, BN=32) [wpe2]
实验 env: SAGEATTN_INT8_WPE (1/2/4)
```

**fp16/bf16 direct 路径**：
```
D=64  self -> (BM=64, BN=64)
D=64  cross, kv<=128 -> (BM=64, BN=32)
D=64  cross, kv>128  -> (BM=128, BN=32)   [BM=128 快 9-13%]
D=128                -> (BM=64, BN=16)
实验 env: SAGEATTN_FP16_BM / SAGEATTN_FP16_BN（同理 BF16）
```

### 6.3 环境变量汇总

| 变量 | 作用 | 默认 |
|------|------|------|
| `SAGEATTN_BACKEND` | 后端选择（native/triton） | triton |
| `SAGEATTN_VT_GLOBAL` | 历史 env（现由编译宏固定，无需设置） | 1（编译时） |
| `SAGEATTN_INT8_32` | int8 self 用每 warp 32 行 kernel | 1（启用） |
| `SAGEATTN_INT8_WPE` | int8 实验配置（wpe1/2/4） | 1 |
| `SAGEATTN_FP16_BM` / `SAGEATTN_FP16_BN` | fp16 direct 实验配置 | 0 / 0 |

---

## 七、易踩坑点（后续开发必读）

1. **测量必须同进程旋转轮转 + ≥600 次预热**；跨进程绝对时间不可比（见 §四）
2. **layout_code=0 是 NHD**；手动复刻 ops 调用传错 layout 会产生"错误但快"的假象
3. **native 正确性验证必须设 `SAGEATTN_BACKEND=native`**（默认是 triton）
4. **hipcc 对模板 `__device__` 函数体内的 `#if`/`if constexpr` 有解析 bug**：
   即使逻辑为空也会报虚假的 `attn_kernel_wpe1_t undeclared`。**解法：用运行时 `if (编译宏)`**
   （宏展开为常量，死分支被 DCE）；`#if` 只用于 host 函数与 namespace 顶层宏定义
5. **WMMA operand 顺序决定 C 布局**：`wmma(A=V^T, B=P^T)` 才得到 out^T（匹配写回）；
   交换成 `out = P @ V` 会转置输出
6. **早期"性能提升"部分来自错误输出**（只处理前 32 key 或 layout 错误）——
   任何优化必须先用 native 正确性验证，再谈性能
7. **iGPU 频率抖动 vs 顺序效应**：A/B 对比必须轮转，否则结论可被污染（早期 BM=128 结论曾被部分推翻）
8. **int8 的 kv_len 必须是 8 倍数**（v16h 32B 读对齐）；v_transpose 输出必须 contiguous

---

## 八、剩余可优化方向

1. **SDXL10/16 仍有 7-12% 差距**（1536/2304 int8 self）——当前 32 行 kernel 对这两个 shape
   未完全追平，可针对其 q_len/kv_len 比例特调（BN 或子块调度）
2. **QK WMMA（i8）是最大单项（1.93ms）**：硬件吞吐限制；可能的突破方向
   - 参考 Triton 的 `waves_per_eu=4` + 4 warps 的**整体 block 组织**（当前 32 行 kernel 已接近，但 wpe4 无额外收益）
   - 探索 i8→fp16 的**批量转换指令**（当前逐元素转换开销 33%，若有向量化转换则 QK fp16 可行）
3. **softmax 虽仅 10%**：D=128（Anima）的 softmax 占比更高（每迭代 16 值 × 更多 dt），可单独优化
4. **k_frag LDS 读 0.46ms**：K 直读失败（L2 gather 差），可尝试**分块转置的 K 存储**
   （类似 V_T，使 K 行读连续）——需量化阶段输出 K_T
5. **D=128 的每 warp 32 行**：当前 32 行只用于 D=64 self；D=128（Anima）走 BM=64/8 warps，
   可评估 32 行变体
6. **写回与 barrier 共 0.8ms**：写回可尝试 permlanex16 交换后向量化（16 halfs 连续）；
   barrier 已随 K 双缓冲探索过（无收益），V_T 后仅剩 K 的 barrier
7. **GQA 场景**：当前 kernel 按 head 分组处理，kv_heads 共享可进一步优化（未在 24 用例覆盖）
