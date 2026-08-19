# try.md — fp8e5m2 存储优化实验记录（FeatherOps 思路移植）

> 本文件记录将 `ComfyUI-FeatherOps` 的 fp8e5m2 思路移植到 `sageattention-rdna3`
> native HIP backend 的完整过程：设计、实现、测试、性能数据、技术分析。
> 全程可追溯、可重现；如死机可从本文件恢复进度。

## 0. 背景与动机（2026-08-18）

### 0.1 ComfyUI-FeatherOps 的核心思路
- 仓库: `ComfyUI-FeatherOps`
- 机制（见其 README + `kernel/hip/hip_kernel.cu`）：
  1. matmul 的 B 矩阵（K-loop 重复加载的 operand）以 **fp8e5m2** 存 LDS/全局；
  2. LDS→VGPR 加载时用 **V_PERM_B32 快速 upcast 到 fp16**（`fp8e5m2x4_to_half2x2`，
     2 条指令转 4 字节），因为 fp8e5m2 与 fp16 的指数 bias 相同(15)，只需尾数零扩展；
  3. 收益：加载带宽减半、LDS 占用减半、K-loop 中 compute-load overlap 改善。
- 关键点：RDNA3 无原生 fp8 WMMA（只有 fp16/bf16/int8），故 fp8 只做存储格式，
  **计算仍用 fp16 WMMA**。

### 0.2 移植到 attention 的候选点
direct 路径（fp16/bf16 WMMA）与 int8 路径的 PV 部分：
- **QK**：k 以 fp8e5m2 存 → QK 误差风险大（softmax 输入，指数放大）
- **PV**：V_T 以 fp8e5m2 存 → 误差线性传递，对应 SageAttention 的 PV fp8 量化思路

### 0.3 精度模拟结论（torch 纯模拟）
随机数据 + 幅度不均分布（部分 token 幅度大 0.5~3.5x），fp32 matmul 模拟（真实 WMMA
为 fp32 累加，不能用 fp16 matmul——量化后 k 值域 ±57344 会溢出 fp16）：

| 方案 | cos | max_err | 判定 |
|------|-----|---------|------|
| QK int8 per-128tok（现有路径对照） | 0.9997 | 0.43-0.73 | 基准 |
| QK fp8e5m2 per-token | 0.995-0.996 | 1.9-2.5 | ❌ max_err 为 int8 的 4-5 倍 |
| QK fp8e5m2 per-16/32tok | 0.992-0.996 | 1.8-2.9 | ❌ 同上 |
| **PV fp8e5m2 per-token** | **0.9987** | 0.76-0.85 | ✅ cos 过 0.99，误差线性传递 |

**决策**：QK 不用 fp8（精度不可行，印证 SageAttention 用 int8 而非 fp8 做 QK 的原因）；
**PV 的 V 以 fp8e5m2 存储（per-token scale）** 为唯一可行方案。该方案同时适用于
direct 路径（短序列）与 int8 路径（VAE 超长 self-attn 的 PV 全局读带宽瓶颈，报告 3.8 节）。

### 0.4 技术可行性验证（perm upcast）
- `C:\Build\tmp_exp\perm_test*.cu`：独立 HIP 测试验证 FeatherOps 的
  `__builtin_amdgcn_perm(0u, p, 0x010c000cu)` 语义。
- **踩坑**：hipcc 默认 target 非 gfx1103 → "device kernel image is invalid"（所有 kernel
  静默失败输出 0）。必须 `--offload-arch=gfx1103`。
- 验证结果：`perm(0u, p, 0x010c000c)` → [b1:0:b0:0] 零扩展 fp16 对；`perm(0u, p, 0x030c020c)`
  → [b3:0:b2:0]。与 FeatherOps 语义一致。✅
- **fp16→fp8e5m2 截断法**：fp8e5m2 bits = fp16 bits >> 8（尾数截断）。
  注意 clamp：fp16 值 ≥57344（E=31）会进入 fp8 的 inf/NaN 编码区，需 clamp 到 ±57344
  （0x7E/0xFE），否则 inf×0=P 列时产生 NaN。

## 1. 设计方案

### 1.1 总览
```
V (fp16/bf16) --v_scale kernel--> v_scale [B,H,N] (fp16, per-token absmax/57344)
             --v_transpose_fp8--> V_T8 [B,H,D,N] (uint8 fp8e5m2)
attn kernel: PV 的 v_frag 从 V_T8 读 16B -> perm upcast v16h -> fp16 WMMA
             P fragment 每列乘 v_scale（out = P @ V = (P*scale) @ V_fp8）
```

### 1.2 粒度选择
- V 量化 scale 粒度：**per-token**（每 (B,H,n) 一个 scale，D 维归约）——P 列上乘 scale
  即可（线性等价），实现开销最小。
- 尝试过 per-(token, d-chunk)（每 16 个 d 一个 scale）：需要在每个 dt 分别修正 p_frag，
  PV 的 p_frag 组装要移入 dt 循环（开销大），且模拟实现中发现语义复杂，弃用。

### 1.3 接口设计
- attn kernel（3 个）新增参数 `const __half* v_scale`（nullptr = 原 fp16 V_T 路径；
  非空 = fp8 V_T 路径，运行时分支）。
- 新增 op：`v_scale(V) -> scale`、`v_transpose_fp8(V, v_t8, v_scale, layout)`。
- core.py：`SAGEATTN_V_FP8=1` env 开关（默认 0，实验性）。

## 2. 实现进度（全部完成；代码已按用户指示回退，仅文档保留）
- [x] 0.3 精度模拟
- [x] 0.4 perm upcast 验证
- [x] mma_gfx11.h fp8e5m2 helper（fp8e5m2x4_to_half2x2 / fp8e5m2x16_to_v16h / half_to_fp8e5m2+clamp）
- [x] v_scale kernel（per-token absmax/57344 → fp16 scale [B,H,N]）
- [x] v_transpose_fp8 kernel（V → V_T8 uint8, 写回时 __hdiv 量化）
- [x] direct kernel PV fp8 分支（v_scale!=nullptr：16B 全局读 + perm upcast + P×scale）
- [x] int8 kernel PV fp8 分支（impl_t：V 拷贝段 fp8→LDS upcast；impl_32_t：直接全局 fp8 读）
- [x] host dispatch + pybind（fp16/bf16/qk_int8 attn 加 v_scale 参数；新 op v_scale/v_transpose_fp8）
- [x] core.py 开关（SAGEATTN_V_FP8=1，direct/int8 两条路径）
- [x] 编译 + 正确性测试（fp16 路径 36/36 过；fp8 路径 28/36 过，8 个 mae 超阈值）
- [x] 性能对比（13 用例同进程轮转：仅 1 个快 7%，整体无提升 → 结论见 §6）
- [x] 文档更新（try.md / NativeBackendOptimizeReport.md / README.md）

## 3. 实现要点记录
- **V_T fp8 布局**：与 fp16 V_T 完全相同的 [B,H,D,N]（n padding 到 64 倍数），仅元素 1B。
  padding 区写 0x00；v_scale 只覆盖真实 n（kv_len），attn 的 P×scale 只读有效列。
- **数学等价**：out = P @ V = (P·scale) @ V_fp8，scale 乘在 P 列上（每 (b,hkv,n) 一个 fp16）。
  P 的 16 列 = n ∈ [kb_base+ct*BK, +16)；v2 优化后只在 p_vals（8 个）上按列
  base+2e+hw 乘 scale（读 sptr[2e+hw]），permlanex16 交换的同行列共享同一 scale。
- **clamp**：fp16→fp8 截断高字节，值 ≥57344（E=31）clamp 到 ±57344（0x7E/0xFE），
  否则 fp8 进入 inf/NaN 编码区，后续 P×scale 的 inf×0 = NaN。
- **attn kernel 运行时分支**：`if (v_scale != nullptr)` 双路径（v 指针按 fp16/uint8
  reinterpret），避免新增模板参数触发 hipcc 模板 #if bug（报告踩坑点 4）。
- **v_transpose fp8 量化**：__hdiv(hvals, scale)（1 条 fp16 除法），scale 防 0（clamp 1e-7）。
- **v_scale kernel**：每 thread 一个 (b,h,n) 行，head_dim 元素 absmax，grid-stride。
- **P×scale 优化（v2）**：初版在 assemble_p_frag 后乘 16 个 fp16；改为在 p_vals 上乘
  8 个 float（p_vals[e] 对应列 base+2e+hw，permlanex16 交换的同行列共享同一 scale，
  assemble 前乘等价且省一半指令）。

## 4. 正确性验证
### 4.1 快速对比（SAGEATTN_V_FP8=0 vs 1）
所有用例 fp16 路径 cos=1.0000（**原路径零破坏**，v_scale=nullptr 分支正确）：
| 用例 | fp16 cos | fp8 cos | fp8 max_err |
|------|----------|---------|-------------|
| D64 fp16 self 256 | 1.0000 | 0.9983 | 0.067 |
| D64 fp16 cross 1024x128 | 1.0000 | 0.9983 | 0.162 |
| D128 fp16 cross | 1.0000 | 0.9984 | 0.082 |
| D128 bf16 cross | 1.0000 | 0.9982 | 0.053 |
| D64 causal 512 | 1.0000 | 0.9983 | 0.482 |
| D64 int8-path 2048 | 1.0000 | 0.9982 | 0.029 |
| D128 int8-path 2048 | 1.0000 | 0.9984 | 0.024 |

### 4.2 pytest（test_sageattn_rdna3.py）
- SAGEATTN_V_FP8=0：**36 passed**（无回归）
- SAGEATTN_V_FP8=1：28 passed / 8 failed，全部失败为 **mae 超 0.35 阈值**
  （0.36~0.51，集中在 causal/sm_scale 用例），cos 全部通过。
  → fp8e5m2 2bit 尾数的固有量化误差，比 int8 路径（mae~0.3）大 1.5-2 倍。

## 5. 性能对比（同进程轮转, 700-900 预热, 5-10 轮, 中位数）
### 5.1 干净测量（3 遍用例轮转）
| 用例 | 路径 | fp16(ms) | fp8(ms) | fp8/fp16 |
|------|------|----------|---------|----------|
| SDXL10 (D64 direct self 1536) | direct | 2.279 | 2.126 | **0.933** |
| SDXL16 (D64 direct self 2304) | direct | 4.446 | 4.652 | 1.046 |
| SDXL07 (D64 direct self 6144) | direct | 13.132 | 13.516 | 1.029 |
| SDXL08 (cross 6144x77) | direct | 0.305 | 0.361 | 1.185 |
| SDXL05 (cross 1024x77) | direct | 0.168 | 0.182 | 1.082 |
| SDXL02 (cross 4096x77) | direct | 0.255 | 0.253 | 0.993 |
| Anima02 (D128 cross 512) | direct | 3.218 | 4.249 | 1.320 |
| Anima04 (D128 cross) | direct | 4.935 | 4.939 | 1.001 |
| Anima06 (D128 cross) | direct | 7.209 | 7.386 | 1.024 |
| D64 causal 6144 | direct | 13.090 | 13.491 | 1.031 |
| SDXL01 (D64 int8 self 4096) | int8 32w | 6.209 | 6.368 | 1.026 |
| Anima01 (D128 int8) | int8 LDS | 22.517 | 25.442 | 1.130 |
| SDXLVAE01 (int8 16384) | int8 LDS | 79.890 | 99.111 | **1.241** |

（单用例 10 轮复测：SDXL10 稳定 0.935-0.952；VAE 稳定 1.09-1.25，非噪声）

## 6. 最终结论：**未确认整体性能提升**

### 6.1 性能分析（为什么 FeatherOps 思路在 attention 中不成立）
1. **FeatherOps 的收益前提是长 K-loop**：fp16@fp8e5m2 matmul 中 K=4096，
   LDS→VGPR 加载指令占 K-loop 指令流比例大，fp8 存储 + 2 条 V_PERM upcast
   省加载带宽/指令的效果显著（Strix Halo 43 vs 36 TFLOPS）。
2. **attention 的 QK/PV 的 K 维 = head_dim（64/128）**：WMMA 16×16×16（~32 cycles）
   主导计算，v_frag 加载（32B→16B）占比小。fp8 引入的 8 条 V_PERM/v16h +
   P×scale 8 float mul + 额外 v_scale kernel（读 V 全量）**无法被省下的
   加载带宽覆盖** → 净负收益。
3. **LDS 中转路径（int8 D=128 / VAE）更差**：V_T tile 每迭代 16KB 从 L2 读
   （2MB L2 覆盖），带宽本不稀缺；fp8 拷贝段每迭代多 128×8=1024 条 perm，
   LDS 写不变 → +13~24% 纯开销。
4. **唯一正收益用例（SDXL10 -6.7%）**：直接全局行读 + V_T 3.9MB 超 L2 → 带宽
   真实稀缺，fp8 省带宽有效。但同形状 SDXL16（2304）却慢 4.6%，收益不稳定、
   不可预测 → 不能作为可靠优化。
5. **精度代价**：fp8e5m2 2bit 尾数 → mae 0.36-0.51（int8 的 1.5-2 倍），
   QK 用 fp8 更差（模拟 max_err 2.5，softmax 指数放大）。

### 6.2 与 ComfyUI-FeatherOps 的本质差异
| 维度 | FeatherOps matmul | attention QK/PV |
|------|-------------------|-----------------|
| K 长度 | 4096（大矩阵） | 64/128（head_dim） |
| 主导 | 加载带宽（K-loop） | WMMA 计算（短 K） |
| fp8 upcast 开销占比 | 小（相对加载） | 大（相对 WMMA） |
| 结论 | +20% | -5% ~ -24%（多数用例） |
