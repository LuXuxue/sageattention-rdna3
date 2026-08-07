import torch
import os
from typing import Any, Optional, Tuple, Union

# Backend selection via environment variable:
#   SAGEATTN_BACKEND=triton            - Triton autotune kernel (default, best perf)
#   SAGEATTN_BACKEND=native            - HIP native WMMA kernel
_BACKEND = os.getenv("SAGEATTN_BACKEND", "triton").lower()

_qattn_gfx11 = None
GFX11_NATIVE_ENABLED = False
_import_error = None

try:
    from . import _qattn_gfx11 as _ext
    _qattn_gfx11 = torch.ops.sageattention
    GFX11_NATIVE_ENABLED = True
except Exception as e:
    _import_error = e


def _get_native_ops():
    global _qattn_gfx11, GFX11_NATIVE_ENABLED
    if _qattn_gfx11 is None:
        try:
            from . import _qattn_gfx11 as _ext
            _qattn_gfx11 = torch.ops.sageattention
            GFX11_NATIVE_ENABLED = True
        except Exception as e:
            raise RuntimeError(
                "sageattention native extension (_qattn_gfx11) is not available. "
                "Please build and install the package with:\n"
                "  pip install -e . --no-build-isolation\n"
                "on a system with ROCm/HIP and a gfx11xx GPU.\n"
                f"Original error: {e}"
            ) from e
    return _qattn_gfx11


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

    layout_code = 1 if tensor_layout == "HND" else 0

    o = torch.empty_like(q)

    if tensor_layout == "HND":
        kv_len_actual = k.size(2)
    else:
        kv_len_actual = k.size(1)

    if headdim == 64:
        use_direct = (kv_len_actual <= 1024)
    else:
        use_direct = (kv_len_actual <= 512)

    if use_direct:
        if input_dtype == torch.bfloat16:
            ops.bf16_attn(
                q, k, v, o,
                layout_code, int(is_causal), sm_scale
            )
        else:
            ops.fp16_attn(
                q, k, v, o,
                layout_code, int(is_causal), sm_scale
            )
    else:
        if input_dtype == torch.bfloat16:
            v_for_attn = v.to(torch.float16)
            o_int8 = torch.empty(q.shape, dtype=torch.float16, device=q.device)
        else:
            v_for_attn = v
            o_int8 = o

        smooth_k = kwargs.get("smooth_k", True)
        if smooth_k:
            k_mean = ops.mean_seq(k, layout_code)
        else:
            k_mean = torch.empty(0, device=q.device, dtype=q.dtype)
        q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
            q, k, k_mean, layout_code, sm_scale
        )
        ops.qk_int8_sv_bf16_attn(
            q_int8, k_int8, v_for_attn, o_int8,
            q_scale, k_scale,
            layout_code, int(is_causal), sm_scale
        )

        if input_dtype == torch.bfloat16:
            o = o_int8.to(torch.bfloat16)

    if input_dtype == torch.float32:
        o = o.to(torch.float32)

    if return_lse:
        # LSE not computed by this kernel; return zeros as placeholder
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
