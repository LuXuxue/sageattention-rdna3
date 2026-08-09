import torch
import torch.nn.functional as F
import gc
import sys
import os
import statistics

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"


# ==========================================
# 1. 导入与兼容性处理
# ==========================================
try:
    from flash_attn import flash_attn_func
    HAS_FA = True
except ImportError:
    HAS_FA = False
    print("[Warning] flash_attn 未安装或未找到。")

try:
    from sageattention import sageattn
    HAS_SAGE = True
except ImportError:
    try:
        from sageattention.core import sageattn
        HAS_SAGE = True
    except ImportError:
        HAS_SAGE = False
        print("[Warning] sageattention 未安装或未找到。")

if not HAS_FA and not HAS_SAGE:
    print("错误: 至少需要安装 flash_attn或sageattention才能运行此脚本。")
    sys.exit(1)


# ==========================================
# 2. 核心计算函数
# ==========================================
def calculate_tflops(b, h_q, sq, sk, d, time_ms):
    """
    计算 Attention 的 TFLOPS。
    支持 GQA，计算量由 Query Heads 决定。
    """
    flops = 4 * b * h_q * sq * sk * d
    tflops = flops / (time_ms / 1000.0) / 1e12
    return tflops


def get_sdpa_reference(q, k, v, out_dtype):
    """
    使用 PyTorch 原生 SDPA，并基于 FP32 计算高精度 Reference。
    最终结果转换为当前测试用例指定的精度：FP16 或 BF16。
    """
    with torch.no_grad():
        # SDPA 期望输入形状为 [Batch, Heads, SeqLen, Dim]
        # 原生 SDPA 支持 GQA 广播，无需手动 repeat KV
        q_sdpa = q.float().permute(0, 2, 1, 3)
        k_sdpa = k.float().permute(0, 2, 1, 3)
        v_sdpa = v.float().permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)

        # 转回 [Batch, SeqLen, Heads, Dim]，并转换为目标精度
        out = out.permute(0, 2, 1, 3).to(out_dtype)
        return out


def calc_error(out, ref):
    """
    计算精度误差指标。
    统一转到 FP32 后计算，避免 BF16 / FP16 直接参与统计。
    """
    out = out.float()
    ref = ref.float()

    max_err = (out - ref).abs().max().item()
    mse = ((out - ref) ** 2).mean().item()
    cos_sim = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()

    return max_err, mse, cos_sim


def benchmark_single(func, q, k, v, warmup=10, iters=50):
    """
    单个函数计时（GPU Event），返回平均时间 (ms)。
    """
    with torch.no_grad():
        for _ in range(warmup):
            _ = func(q, k, v)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            _ = func(q, k, v)
        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / iters


def benchmark_round_robin(fns, q, k, v, warmup=10, iters=50, rounds=3):
    """
    交错轮转计时：多个函数交替测量 rounds 轮，返回各函数的中位时间。

    动机：测试平台（Radeon 780M iGPU）存在严重热漂移，
    顺序测量会使后测的函数吃亏。交错轮转让所有函数
    经历相近的 GPU 温度状态，取中位数消除离群值。
    """
    results = [[] for _ in fns]
    with torch.no_grad():
        for fn in fns:
            for _ in range(warmup):
                _ = fn(q, k, v)
        torch.cuda.synchronize()

        for _ in range(rounds):
            for i, fn in enumerate(fns):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    _ = fn(q, k, v)
                e.record()
                torch.cuda.synchronize()
                results[i].append(s.elapsed_time(e) / iters)

    return [statistics.median(r) for r in results]


# ==========================================
# 3. 定义测试用例
# Format:
# name, b, h_q, h_kv, sq, sk, d, dtype
#
# SDXL 模型使用 FP16
# Anima 模型使用 BF16
# ==========================================
test_cases = [
    # 1. SDXL，MHA: h_q == h_kv，FP16
    ("SDXL01", 1, 10, 10, 4096, 4096, 64, torch.float16),
    ("SDXL02", 1, 10, 10, 4096, 77, 64, torch.float16),
    ("SDXL03", 1, 10, 10, 4096, 154, 64, torch.float16),
    ("SDXL04", 1, 20, 20, 1024, 1024, 64, torch.float16),
    ("SDXL05", 1, 20, 20, 1024, 77, 64, torch.float16),
    ("SDXL06", 1, 20, 20, 1024, 154, 64, torch.float16),
    ("SDXL07", 1, 10, 10, 6144, 6144, 64, torch.float16),
    ("SDXL08", 1, 10, 10, 6144, 77, 64, torch.float16),
    ("SDXL09", 1, 10, 10, 6144, 154, 64, torch.float16),
    ("SDXL10", 1, 20, 20, 1536, 1536, 64, torch.float16),
    ("SDXL11", 1, 20, 20, 1536, 77, 64, torch.float16),
    ("SDXL12", 1, 20, 20, 1536, 154, 64, torch.float16),
    ("SDXL13", 1, 10, 10, 9216, 9216, 64, torch.float16),
    ("SDXL14", 1, 10, 10, 9216, 77, 64, torch.float16),
    ("SDXL15", 1, 10, 10, 9216, 154, 64, torch.float16),
    ("SDXL16", 1, 20, 20, 2304, 2304, 64, torch.float16),
    ("SDXL17", 1, 20, 20, 2304, 77, 64, torch.float16),
    ("SDXL18", 1, 20, 20, 2304, 154, 64, torch.float16),

    # 2. Anima，MHA: h_q == h_kv，BF16
    ("Anima01", 1, 16, 16, 4096, 4096, 128, torch.bfloat16),
    ("Anima02", 1, 16, 16, 4096, 512, 128, torch.bfloat16),
    ("Anima03", 1, 16, 16, 6144, 6144, 128, torch.bfloat16),
    ("Anima04", 1, 16, 16, 6144, 512, 128, torch.bfloat16),
    ("Anima05", 1, 16, 16, 9216, 9216, 128, torch.bfloat16),
    ("Anima06", 1, 16, 16, 9216, 512, 128, torch.bfloat16),

    # 3. Krea2，GQA: h_q = 48, h_kv = 12
    # 如启用，请根据实际模型精度修改最后一个 dtype 字段。
    # ("Krea01", 1, 48, 12, 4213, 4213, 128, torch.bfloat16),
    # ("Krea02", 1, 48, 12, 117, 117, 128, torch.bfloat16),
]


# ==========================================
# 4. 执行测试主循环
# ==========================================
def run_benchmarks():
    if not torch.cuda.is_available():
        print("错误: 本脚本需要 CUDA 设备。")
        sys.exit(1)

    device = "cuda"
    torch.manual_seed(0)

    # 交错轮转轮数（抗热漂移）
    ROUNDS = int(os.getenv("SAGEATTN_BENCH_ROUNDS", "3"))
    # 每轮迭代次数
    ITERS = int(os.getenv("SAGEATTN_BENCH_ITERS", "50"))

    for name, b, h_q, h_kv, sq, sk, d, dtype in test_cases:
        dtype_str = "FP16" if dtype == torch.float16 else "BF16"
        threshold = 0.05

        # 分离初始化 Q 与 K/V 的 Heads 数量以支持 GQA
        q = torch.randn(b, sq, h_q, d, device=device, dtype=dtype)
        k = torch.randn(b, sk, h_kv, d, device=device, dtype=dtype)
        v = torch.randn(b, sk, h_kv, d, device=device, dtype=dtype)

        try:
            ref_out = get_sdpa_reference(q, k, v, dtype)
        except RuntimeError as e:
            print(f"[{name}] OOM or RuntimeError during SDPA Reference calculation: {e}")
            del q, k, v
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- SDPA Baseline (仅测速，一次即可) ---
        def sdpa_func(q, k, v):
            q_p = q.permute(0, 2, 1, 3)
            k_p = k.permute(0, 2, 1, 3)
            v_p = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q_p, k_p, v_p)
            return out.permute(0, 2, 1, 3)

        try:
            sdpa_time = benchmark_single(sdpa_func, q, k, v)
            sdpa_tflops = calculate_tflops(b, h_q, sq, sk, d, sdpa_time)
            print(
                f"{name:<8} | {dtype_str:<9} | {'SDPA(Base)':<10} | "
                f"{sdpa_time:<8.3f} | {sdpa_tflops:<6.2f} | "
                f"{'Baseline':<8} | {'-':<8} | {'-':<10} | {'-':<8} | {'-':<6}"
            )
        except Exception as e:
            err_msg = str(e).replace("\n", " ")[:20]
            print(
                f"{name:<8} | {dtype_str:<9} | {'SDPA(Base)':<10} | "
                f"{'Error':<8} | {'-':<6} | {'-':<8} | {'-':<8} | "
                f"{'-':<10} | {'-':<8} | {err_msg:<6}"
            )
            del q, k, v, ref_out
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- 收集待测后端 (FA / Sage)，交错轮转 ---
        backends = []

        if HAS_FA:
            def fa_func(q, k, v):
                return flash_attn_func(q, k, v, causal=False)
            backends.append(("FlashAttn", fa_func))

        if HAS_SAGE:
            def sage_func(q, k, v):
                try:
                    return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
                except TypeError:
                    return sageattn(q, k, v)
            backends.append(("SageAttn", sage_func))

        # 至少一个后端才测（SDPA 已单独测过）
        if not backends:
            del q, k, v, ref_out
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- 交错轮转计时 (FA/Sage 交替, 抗热漂移) ---
        try:
            if len(backends) == 1:
                name_b, fn_b = backends[0]
                t_b = benchmark_single(fn_b, q, k, v)
                times = [t_b]
            else:
                fns = [b[1] for b in backends]
                times = benchmark_round_robin(
                    fns, q, k, v, warmup=10, iters=ITERS, rounds=ROUNDS
                )
        except Exception as e:
            err_msg = str(e).replace("\n", " ")[:20]
            for backend_name, _ in backends:
                print(
                    f"{name:<8} | {dtype_str:<9} | {backend_name:<10} | "
                    f"{'Error':<8} | {'-':<6} | {'-':<8} | {'-':<8} | "
                    f"{'-':<10} | {'-':<8} | {err_msg:<6}"
                )
            del q, k, v, ref_out
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- 输出各后端结果 + 精度检查 (vs SDPA 参考) ---
        for (backend_name, backend_fn), t in zip(backends, times):
            tflops = calculate_tflops(b, h_q, sq, sk, d, t)
            speedup_str = f"{sdpa_time / t:.2f}x"

            status = "-"
            max_err = mse_val = cos = 0.0
            try:
                out = backend_fn(q, k, v)
                max_err, mse_val, cos = calc_error(out, ref_out)
                status = "FAIL" if max_err > threshold else "OK"
            except Exception as e:
                status = "ERR"

            print(
                f"{name:<8} | {dtype_str:<9} | {backend_name:<10} | "
                f"{t:<8.3f} | {tflops:<6.2f} | "
                f"{speedup_str:<8} | {max_err:<8.6f} | "
                f"{mse_val:<10.8f} | {cos:<8.6f} | {status:<6}"
            )

        # 清理显存
        del q, k, v, ref_out
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(
        f"{'Shape':<8} | {'Precision':<9} | {'Backend':<10} | "
        f"{'Time':<8} | {'TFLOPS':<6} | {'Speedup':<8} | "
        f"{'MaxErr':<8} | {'MSE':<10} | {'CosSim':<8} | {'Status':<6}"
    )
    run_benchmarks()
