import torch
import os
from typing import Any, Optional, Tuple, Union

# Backend selection via environment variable:
#   SAGEATTN_BACKEND=triton            - Triton autotune kernel (default, best perf)
#   SAGEATTN_BACKEND=native            - HIP native WMMA kernel (transposed layout)
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

    # native 写回为 16B 向量写: o_off = b*stride_b + n*stride_n + h*stride_h + d, d 恒为 8 倍数,
    # 故 q 的 stride_b/n/h 均需为 8 倍数 (o=empty_like(q) 继承 stride)。
    # contiguous 输入恒满足; 非 contiguous 的 q (permute/slice) 会导致未对齐 16B 写 (UB)。
    # 注意: 该断言只约束 q (决定 o 的布局), k/v 仅需 head_dim stride==1 (上面已断言)。
    assert q.stride(0) % 8 == 0 and q.stride(1) % 8 == 0 and q.stride(2) % 8 == 0, (
        "native backend requires q strides that are multiples of 8 halfs "
        "(16B aligned write-back). "
        f"Got strides={q.stride()}. Use contiguous tensors."
    )

    layout_code = 1 if tensor_layout == "HND" else 0

    # 实验参数: bm_sel 控制 direct kernel 的 BM (0=默认, 1=32, 2=128)
    bm_sel = int(kwargs.get("bm_sel", 0) or os.getenv("SAGEATTN_BM_SEL", "0"))

    o = torch.empty_like(q)
    # 注: 写回为 16B 向量写, 要求 o 的 head_dim 维连续 (stride 1) 且 o_off 8-half 对齐;
    #     empty_like(q) 恒 contiguous 满足 (core 断言 q.stride(-1)==1)

    if tensor_layout == "HND":
        kv_len_actual = k.size(2)
        q_len = q.size(2)
    else:
        kv_len_actual = k.size(1)
        q_len = q.size(1)

    # direct/int8 分发: 本质是"省 quant+mean 辅助(固定 0.4-0.6ms)" vs "i8 WMMA 2x 吞吐(与计算量成正比)"的权衡
    # 计算量小(causal/cross 短 q)时 direct 胜; 计算量大(self 长序列)时 int8 胜。阈值扫描见 bench_threshold*.py:
    #   D=64 self 非 causal: kv=3072 direct 优1%, 3456 int8 优1.2%       -> 3072
    #   D=64 causal:          kv=4096/6144 direct 优8%/2.7%, 8192 int8 优5.5% -> 6144
    #   D=64 cross (q<kv):    q=3072/kv=4096 仍 direct 优3%              -> 6144
    #   D=128 self/causal:    kv=2048 direct 优5%, 2560 int8 优5%        -> 2048
    #   D=128 cross q<<kv:    q=512/1024 vs kv=4096 direct 优25%/8%, q=2048(=kv/2) int8 优5% -> q<kv/2 且 kv<=4096
    if headdim == 64:
        thr_d64 = int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64", "3072") or 3072)
        if is_causal:
            use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL", "6144") or 6144))
        elif q_len < kv_len_actual:
            # cross-attn: q 短时 direct 省辅助收益大 (q=3072/kv=4096 仍优 3%)
            use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64_CROSS", "6144") or 6144))
        else:
            use_direct = (kv_len_actual <= thr_d64)
    else:
        thr_d128 = int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D128", "2048") or 2048)
        if q_len * 2 < kv_len_actual:
            # cross 且 q 明显短: direct 优 (q=512/1024 vs kv=4096 优 8-25%; q=2048=kv/2 时 int8 优)
            use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D128_CROSS", "4096") or 4096))
        else:
            use_direct = (kv_len_actual <= thr_d128)

    if use_direct:
        # V 直接交给 v_transpose: bf16 输入由 kernel 内部转 fp16 (省 v.to(fp16) 独立 kernel)
        v_attn = v
        # V 全局转置 (V_T [B,H,D,N]) + out = P @ V: 需配套 -DSAGEATTN_VT_GLOBAL=1 编译 (setup.py 默认)
        # 注意: V_T 的 n 维 padding 到 64 的倍数 (防 attn kernel 的 v_frag_t 32B 直读越界,
        # kv_len 非 64 倍数时最后 kv-tile 越界读未初始化内存 -> NaN; padding 区由 v_transpose 填 0)
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
        # int8 路径, V/OUT dtype 分离 (方案B):
        #   V 转 fp16 并入 v_transpose (读 bf16 直接写 fp16 V_T, 一次 kernel 完成转置+转换,
        #   省掉 v.to(fp16) 独立 kernel 的一次额外全局读写往返)
        #   输出由 kernel 直接写 bf16 (o 复用, 省 o.to(bf16) 转换 kernel, 且单次舍入精度更好)
        v_for_attn = v
        # V 全局转置 (V_T [B,H,D,N]) + 无 LDS PV: 需配套 -DSAGEATTN_VT_GLOBAL=1 编译 (setup.py 默认)
        # n 维 padding 到 64 倍数, 防 v_frag_t 32B 直读越界 (见 direct 路径注释)
        kv_heads_n = k.size(1) if tensor_layout == "HND" else k.size(2)
        padded_n = ((kv_len_actual + 63) // 64) * 64
        v_t = torch.empty(
            q.size(0), kv_heads_n, headdim, padded_n,
            device=q.device, dtype=torch.float16
        )
        ops.v_transpose(v_for_attn, v_t, layout_code)
        v_for_attn = v_t
        o_int8 = o

        smooth_k = kwargs.get("smooth_k", True)
        if smooth_k:
            k_mean = ops.mean_seq(k, layout_code)
        else:
            k_mean = torch.empty(0, device=q.device, dtype=q.dtype)
        q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
            q, k, k_mean, layout_code, sm_scale
        )
        ops.qk_int8_sv_bf16_attn_t(
            q_int8, k_int8, v_for_attn, o_int8,
            q_scale, k_scale,
            layout_code, int(is_causal), sm_scale
        )

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
