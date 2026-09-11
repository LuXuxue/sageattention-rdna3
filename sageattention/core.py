import torch
import os
from typing import Any, Optional, Tuple, Union

# Backend selection via environment variable:
#   SAGEATTN_BACKEND=triton - Triton autotune kernel
#   SAGEATTN_BACKEND=native - HIP native WMMA kernel
_BACKEND = os.getenv("SAGEATTN_BACKEND", "native").lower()

# 按当前 HIP 设备的 gfx arch 懒加载 _qattn_gfx110x (RDNA3) 或 _qattn_gfx103x (RDNA2)。
# 同进程只加载匹配设备的一个 pyd; 两扩展注册到同一 torch.ops.sageattention 库, 故 schema 必须一致。
_qattn_ops = None
GFX_NATIVE_ENABLED = False
GFX_ARCH_LOADED = None
_import_error = None

def _detect_gfx_arch():
    """Return 'gfx110x' (RDNA3) or 'gfx103x' (RDNA2) for current HIP device, or None."""
    try:
        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev)
        # 优先 gcnArchName (ROCm HIP 返回 "gfx1103" 等); 回退 prop.major 整型
        name = getattr(prop, 'gcnArchName', None) or getattr(prop, 'name', '')
        if isinstance(name, str) and name.startswith('gfx'):
            if name.startswith('gfx11'):
                return 'gfx110x'
            if name.startswith('gfx103'):
                return 'gfx103x'
            return None
        mj = getattr(prop, 'major', None)
        if mj == 11:
            return 'gfx110x'
        if mj == 10:
            return 'gfx103x'
        return None
    except Exception:
        return None


def _load_native_extension(arch):
    """Import the native pyd for the given arch ('gfx110x' or 'gfx103x')."""
    mod_name = f'_qattn_{arch}'
    import importlib
    mod = importlib.import_module(f'.{mod_name}', package=__package__ or 'sageattention')
    return mod, torch.ops.sageattention


def _get_native_ops():
    global _qattn_ops, GFX_NATIVE_ENABLED, GFX_ARCH_LOADED, _import_error
    if _qattn_ops is not None:
        return _qattn_ops
    arch = _detect_gfx_arch()
    if arch is None:
        raise RuntimeError(
            "sageattention native extension: cannot determine AMD gfx arch of current device. "
            "Supported: gfx110x (RDNA3), gfx103x (RDNA2). "
            "Use backend='triton' or set SAGEATTN_BACKEND=triton."
        )
    try:
        _load_native_extension(arch)  # import side effect registers torch.ops
        _qattn_ops = torch.ops.sageattention
        GFX_NATIVE_ENABLED = True
        GFX_ARCH_LOADED = arch
    except Exception as e:
        _import_error = e
        raise RuntimeError(
            f"sageattention native extension (_qattn_{arch}) is not available. "
            f"Build with GPU_ARCHS containing an {arch[:-1]}* arch, e.g.\n"
            "  GPU_ARCHS=gfx1103 pip install -e . --no-build-isolation\n"
            f"on a ROCm/HIP system.\nOriginal error: {e}"
        ) from e
    return _qattn_ops


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on CUDA/HIP device."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "Input tensors must be fp16, bf16, or fp32."
    )
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim in [64, 128], f"head_dim must be 64 or 128, got {headdim}."

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
        "Last dim of qkv must be contiguous."
    )

    if sm_scale is None:
        sm_scale = headdim ** -0.5

    if _BACKEND == "triton":
        from .triton_backend import sageattn as _triton_sageattn
        return _triton_sageattn(
            q, k, v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            **kwargs,
        )

    input_dtype = dtype
    if dtype == torch.float32:
        v = v.to(torch.float16)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        dtype = torch.float16

    ops = _get_native_ops()

    # 写回为 16B 向量写 (d 恒为 8 倍数), 故 q 的 stride_b/n/h 均须为 8 倍数 (o=empty_like(q) 继承 stride)。
    # 非 contiguous 的 q (permute/slice) 会导致未对齐 16B 写 (UB)。k/v 仅需 head_dim stride==1 (上面已断言)。
    assert q.stride(0) % 8 == 0 and q.stride(1) % 8 == 0 and q.stride(2) % 8 == 0, (
        "native backend requires q strides that are multiples of 8 halfs "
        "(16B aligned write-back). "
        f"Got strides={q.stride()}. Use contiguous tensors."
    )

    layout_code = 1 if tensor_layout == "HND" else 0

    # bm_sel 控制 direct kernel 的 BM (0=默认, 1=32, 2=128), 仅 gfx1103 使用。
    bm_sel = int(kwargs.get("bm_sel", 0) or os.getenv("SAGEATTN_BM_SEL", "0"))

    o = torch.empty_like(q)

    if tensor_layout == "HND":
        kv_len_actual = k.size(2)
        q_len = q.size(2)
    else:
        kv_len_actual = k.size(1)
        q_len = q.size(1)

    # gfx1035 (RDNA2) 恒走 int8 (direct 在所有形状下 4-65x 慢于 int8); gfx1103 保持 direct/int8 平衡。
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'gcnArchName', None)
    is_gfx103 = isinstance(arch, str) and arch.startswith('gfx103')
    if headdim == 64:
        if is_gfx103:
            use_direct = False
        else:
            d64_default = 2048 if tensor_layout == "HND" else 3072
            thr_d64 = int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64", str(d64_default)) or d64_default)
            if is_causal:
                use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL", "6144") or 6144))
            elif q_len < kv_len_actual:
                use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64_CROSS", "6144") or 6144))
            else:
                use_direct = (kv_len_actual <= thr_d64)
    else:
        if is_gfx103:
            use_direct = False
        else:
            thr_d128 = int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D128", "2048") or 2048)
            if q_len * 2 < kv_len_actual:
                use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D128_CROSS", "4096") or 4096))
            else:
                use_direct = (kv_len_actual <= thr_d128)

    if use_direct:
        # V 直接交给 v_transpose: bf16 由 kernel 内部转 fp16 (省独立 cast kernel)。
        # V_T [B,H,D,N] 的 n 维 padding 到 64 倍数并填 0, 防 attn kernel 的 v_frag_t 32B 直读越界。
        v_attn = v
        kv_heads_n = k.size(1) if tensor_layout == "HND" else k.size(2)
        padded_n = ((kv_len_actual + 63) // 64) * 64
        v_t = torch.empty(
            q.size(0), kv_heads_n, headdim, padded_n,
            device=q.device, dtype=torch.float16
        )
        ops.v_transpose(v_attn, v_t, layout_code)
        v_attn = v_t
        if input_dtype == torch.bfloat16:
            ops.bf16_attn_t(
                q, k, v_attn, o,
                layout_code, int(is_causal), sm_scale, bm_sel
            )
        else:
            ops.fp16_attn_t(
                q, k, v_attn, o,
                layout_code, int(is_causal), sm_scale, bm_sel
            )
    else:
        # int8 路径: gfx103x 直接读 native [B,H,N,D] V; gfx110x 内核读 V_T [B,H,D,N], 需全局转置。
        v_native = is_gfx103
        kv_heads_n = k.size(1) if tensor_layout == "HND" else k.size(2)
        if v_native:
            v_for_attn = v
            if tensor_layout == "NHD":
                v16 = v if v.dtype == torch.float16 else v.to(torch.float16)
                if v16.is_contiguous():
                    b_, n_, h_, d_ = v16.shape
                    v_for_attn = v16.as_strided((b_, h_, n_, d_), (n_ * h_ * d_, d_, h_ * d_, 1))
                else:
                    v_for_attn = v16.permute(0, 2, 1, 3).contiguous()
            elif v.dtype != torch.float16:
                v_for_attn = v.to(torch.float16)
            # v_scale 占位 (gfx103x kernel 读 native fp16 V, 不读取 v_scale; 仅满足 pybind 签名)
            v_scale_t = torch.empty(
                q.size(0), kv_heads_n, (kv_len_actual + 31) // 32,
                device=q.device, dtype=torch.float32
            )
        else:
            # gfx110x: bf16 输入由 v_transpose 内部转 fp16, 直接传 bf16 (省 .to(fp16) cast kernel)。
            padded_n = ((kv_len_actual + 63) // 64) * 64
            v_t = torch.empty(
                q.size(0), kv_heads_n, headdim, padded_n,
                device=q.device, dtype=torch.float16
            )
            ops.v_transpose(v, v_t, layout_code)
            v_for_attn = v_t
            # v_scale 占位 (kernel 不读取), 仅满足 pybind 签名
            v_scale_t = torch.empty(
                q.size(0), kv_heads_n, (kv_len_actual + 31) // 32,
                device=q.device, dtype=torch.float32
            )
        o_int8 = o

        # smooth_k: K 减 mean 后再量化, 默认 False。
        smooth_k = kwargs.get("smooth_k", False)
        if smooth_k:
            k_mean = ops.mean_seq(k, layout_code)
        else:
            k_mean = torch.empty(0, device=q.device, dtype=q.dtype)
        # v10 INQ (in-kernel Q int8 quant): 小 kv 时 prepass 的 Q 量化 round-trip 主导端到端时间,
        # 主 kernel 从 fp16/bf16 源在核内量化 Q, prepass 跳过 Q。仅 gfx103x 支持
        # (gfx110x 主 kernel 忽略 q_fp, 必须用 prepass 的 q_int8)。去掉 INQ 会恶化非因果小 kv 精度。
        q_skip_inq = False
        q_fp = q
        if is_gfx103:
            v10_on = True
            ipv_on = False
            inq_wanted = (kv_len_actual <= 1024)
            q_skip_inq = bool(
                headdim in (64, 128) and not is_causal and v10_on and not ipv_on and inq_wanted
                and q.is_contiguous()
            )
            if q_skip_inq and tensor_layout == "NHD":
                b_, s_, h_, d_ = q.shape
                q_fp = q.as_strided((b_, h_, s_, d_), (s_ * h_ * d_, d_, h_ * d_, 1))
        q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
            q, k, k_mean, layout_code, sm_scale, int(q_skip_inq)
        )
        # 签名含 v_scale 与 q_fp, 两扩展均按此签名重建 (schema 一致)。
        ops.qk_int8_sv_bf16_attn_t(
            q_int8, k_int8, v_for_attn, o_int8,
            q_scale, k_scale, v_scale_t,
            layout_code, int(is_causal), sm_scale,
            q_fp if is_gfx103 else torch.empty(0, device=q.device, dtype=q.dtype)
        )

    if input_dtype == torch.float32:
        o = o.to(torch.float32)

    if return_lse:
        # LSE 未由本 kernel 计算, 返回全零占位
        seq_dim = 2 if tensor_layout == "HND" else 1
        seq_len = q.size(seq_dim)
        lse = torch.zeros(
            (q.size(0), q.size(1 if tensor_layout == "HND" else 2), seq_len),
            dtype=torch.float32, device=q.device
        )
        return o, lse

    return o


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return sageattn(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        **kwargs,
    )