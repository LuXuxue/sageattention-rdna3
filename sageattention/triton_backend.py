import torch
import math
import triton
import triton.language as tl
import os
import json
from typing import Any, Optional

DEFAULT_PV_ACCUM_DTYPE = os.getenv("SAGEATTN_DEFAULT_PV_ACCUM_DTYPE", "fp32").lower()
if DEFAULT_PV_ACCUM_DTYPE not in ("fp32", "fp16", "fp16+fp32"):
    DEFAULT_PV_ACCUM_DTYPE = "fp32"

arch = str(triton.runtime.driver.active.get_current_target().arch).strip()
env_config_json = os.environ.get('FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON')
if env_config_json:
    env_config = json.loads(env_config_json)
    configs = [
        triton.Config(
            {
                'BLOCK_M': env_config.get('BLOCK_M'),
                'BLOCK_N': env_config.get('BLOCK_N'),
                'waves_per_eu': env_config.get('waves_per_eu', None)
            },
            num_warps=env_config.get('num_warps'),
            num_stages=env_config.get('num_stages')
        )
    ]
    fp16_configs = configs
elif arch == "gfx1103":
    configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 0}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 4}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 0}, num_warps=4, num_stages=1),
    ]
    fp16_configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'waves_per_eu': 0}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 0}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 4}, num_warps=4, num_stages=1),
    ]
elif arch == "gfx1035":
    configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_warps=2, num_stages=2), #Anima
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=2), #SDXL
    ]
    fp16_configs = configs
elif arch.startswith("gfx"):
    configs = [
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'waves_per_eu': waves}, num_warps=nw, num_stages=ns)
        for bm in [128, 64, 32]
        for bn in [64, 32, 16]
        if bm > bn
        for waves in [0, 1, 2, 3, 4, 6]
        for nw in [2, 4, 8]
        for ns in [1, 2, 3, 4]
    ]
    fp16_configs = configs
elif arch == "75":
    configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_warps=2, num_stages=2), #Anima
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_warps=4, num_stages=2), #SDXL
    ]
    fp16_configs = configs
else:
    configs = [
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_warps=nw, num_stages=ns)
        for bm in [128, 64, 32]
        for bn in [64, 32, 16]
        if bm > bn
        for nw in [2, 4, 8]
        for ns in [1, 2, 3, 4]
    ]
    fp16_configs = configs

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scales_full, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                    start_m,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    offs_m: tl.constexpr, offs_n: tl.constexpr,
                    MIN_BLK_N: tl.constexpr,
                    PV_ACCUM_FP32: tl.constexpr,
                    IS_CAUSAL: tl.constexpr,
                    ):
    scale_offs = tl.arange(0, BLOCK_N // MIN_BLK_N)
    num_full_blocks = kv_len // BLOCK_N
    for start_n in range(0, num_full_blocks * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_ptrs, eviction_policy='evict_first')
        k_scales = tl.load(K_scale_ptr + scale_offs, eviction_policy='evict_first')
        k_scales = tl.reshape(k_scales, [BLOCK_N // MIN_BLK_N, 1])
        k_scales = tl.broadcast_to(k_scales, [BLOCK_N // MIN_BLK_N, MIN_BLK_N])
        k_scales_full = tl.reshape(k_scales, [BLOCK_N])
        qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        qk = qk * q_scales_full[:, None] * k_scales_full[None, :]
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        if IS_CAUSAL:
            all_masked = (m_ij <= -1.0e5)
            p = tl.where(all_masked[:, None], 0.0, p)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        if IS_CAUSAL:
            alpha = tl.where(all_masked, 0.0, alpha)
        l_i = l_i * alpha + l_ij
        v = tl.load(V_ptrs, eviction_policy='evict_first')
        v_dtype = v.dtype
        p = p.to(v_dtype)
        if PV_ACCUM_FP32:
            acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += (BLOCK_N // MIN_BLK_N)
        V_ptrs += BLOCK_N * stride_vn
    start_n = num_full_blocks * BLOCK_N
    if start_n < kv_len:
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask, other=0.0, eviction_policy='evict_first')
        k_scales = tl.load(K_scale_ptr + scale_offs, eviction_policy='evict_first')
        k_scales = tl.reshape(k_scales, [BLOCK_N // MIN_BLK_N, 1])
        k_scales = tl.broadcast_to(k_scales, [BLOCK_N // MIN_BLK_N, MIN_BLK_N])
        k_scales_full = tl.reshape(k_scales, [BLOCK_N])
        qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        qk = qk * q_scales_full[:, None] * k_scales_full[None, :]
        qk = tl.where(k_mask, qk, -1.0e6)
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        if IS_CAUSAL:
            all_masked = (m_ij <= -1.0e5)
            p = tl.where(all_masked[:, None], 0.0, p)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        if IS_CAUSAL:
            alpha = tl.where(all_masked, 0.0, alpha)
        l_i = l_i * alpha + l_ij
        v_mask = offs_n[:, None] < (kv_len - start_n)
        v = tl.load(V_ptrs, mask=v_mask, other=0.0, eviction_policy='evict_first')
        v_dtype = v.dtype
        p = p.to(v_dtype)
        if PV_ACCUM_FP32:
            acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
    return acc, l_i

@triton.autotune(
    list(configs),
    key=['qo_len', 'kv_len', 'H', 'HEAD_DIM', 'num_kv_groups']
)
@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,
              stride_oz, stride_oh, stride_on,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              MIN_BLK_M: tl.constexpr,
              MIN_BLK_N: tl.constexpr,
              PV_ACCUM_FP32: tl.constexpr,
              IS_CAUSAL: tl.constexpr,
              ):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, MIN_BLK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, MIN_BLK_N)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m * (BLOCK_M // MIN_BLK_M)
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len, other=0.0, eviction_policy='evict_last')
    q_scale_offs = tl.arange(0, BLOCK_M // MIN_BLK_M)
    q_scales = tl.load(Q_scale_ptr + q_scale_offs, eviction_policy='evict_last')
    q_scales = tl.reshape(q_scales, [BLOCK_M // MIN_BLK_M, 1])
    q_scales = tl.broadcast_to(q_scales, [BLOCK_M // MIN_BLK_M, MIN_BLK_M])
    q_scales_full = tl.reshape(q_scales, [BLOCK_M])
    acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scales_full, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                               start_m,
                               BLOCK_M, HEAD_DIM, BLOCK_N,
                               offs_m, offs_n, MIN_BLK_N,
                               PV_ACCUM_FP32,
                               IS_CAUSAL)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len), eviction_policy='evict_last')

@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, K_mean, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                stride_mz, stride_mh,
                                sm_scale,
                                C: tl.constexpr, BLK: tl.constexpr, MIN_BLK: tl.constexpr,
                                SMOOTH_K: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_k = tl.arange(0, C)
    RATIO: tl.constexpr = BLK // MIN_BLK
    if SMOOTH_K:
        km_ptrs = K_mean + off_b * stride_mz + off_h * stride_mh + offs_k
        km = tl.load(km_ptrs)
    for i in tl.static_range(RATIO):
        offs_min_blk = tl.arange(0, MIN_BLK)
        curr_offs_n = off_blk * BLK + i * MIN_BLK + offs_min_blk
        input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + curr_offs_n[:, None] * stride_in + offs_k[None, :]
        output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + curr_offs_n[:, None] * stride_on + offs_k[None, :]
        x = tl.load(input_ptrs, mask=curr_offs_n[:, None] < L, other=0.0)
        x = x.to(tl.float32)
        if SMOOTH_K:
            x = x - km[None, :]
        x *= sm_scale
        curr_scale = tl.max(tl.abs(x)) / 127.
        curr_scale = tl.maximum(curr_scale, 1e-12)
        rcp_scale = 1.0 / curr_scale
        x_int8 = x * rcp_scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        x_int8 = x_int8.to(tl.int8)
        tl.store(output_ptrs, x_int8, mask=curr_offs_n[:, None] < L)
        scale_idx = off_blk * RATIO + i
        max_scale_idx = tl.cdiv(L, MIN_BLK)
        scale_mask = scale_idx < max_scale_idx
        scale_ptr = Scale + off_b * stride_sz + off_h * stride_sh + scale_idx
        tl.store(scale_ptr, curr_scale, mask=scale_mask)

def per_block_int8(q, k, km, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    MIN_BLKQ = 32
    MIN_BLKK = 16
    q_scale = torch.empty((b, h_qo, (qo_len + MIN_BLKQ - 1) // MIN_BLKQ, 1), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + MIN_BLKK - 1) // MIN_BLKK, 1), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    SMOOTH_K = km is not None
    km_ptr = km if SMOOTH_K else q
    stride_mz, stride_mh = (km.stride(0), km.stride(1)) if SMOOTH_K else (0, 0)
    grid_q = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid_q](
        q, q_int8, q_scale, q, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        0, 0,
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ, MIN_BLK=MIN_BLKQ,
        SMOOTH_K=False
    )
    grid_k = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid_k](
        k, k_int8, k_scale, km_ptr, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        stride_mz, stride_mh,
        sm_scale=1.0,
        C=head_dim, BLK=BLKK, MIN_BLK=MIN_BLKK,
        SMOOTH_K=SMOOTH_K
    )
    return q_int8, q_scale, k_int8, k_scale

def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", output_dtype=torch.float16, pv_accum_dtype="fp32", is_causal=False):
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    grid = lambda META: (triton.cdiv(qo_len, META['BLOCK_M']), h_qo, b)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        HEAD_DIM=HEAD_DIM_K,
        MIN_BLK_M=32,
        MIN_BLK_N=16,
        PV_ACCUM_FP32=(pv_accum_dtype == "fp32"),
        IS_CAUSAL=is_causal,
    )
    return o

@triton.jit
def _attn_fwd_inner_fp16(acc, l_i, m_i, q, kv_len,
                         K_ptrs, V_ptrs, stride_kn, stride_vn,
                         start_m,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                         offs_m: tl.constexpr, offs_n: tl.constexpr,
                         PV_ACCUM_FP32: tl.constexpr,
                         IS_CAUSAL: tl.constexpr,
                         ):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    num_full_blocks = kv_len // BLOCK_N
    for start_n in range(0, num_full_blocks * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_ptrs, eviction_policy='evict_first')
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted)
        if IS_CAUSAL:
            all_masked = (m_ij <= -1.0e5)
            p = tl.where(all_masked[:, None], 0.0, p)
        l_ij = tl.sum(p, 1)
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        if IS_CAUSAL:
            alpha = tl.where(all_masked, 0.0, alpha)
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, eviction_policy='evict_first')
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        if PV_ACCUM_FP32:
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        else:
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float16)
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    start_n = num_full_blocks * BLOCK_N
    if start_n < kv_len:
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask, other=0.0, eviction_policy='evict_first')
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(k_mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted)
        if IS_CAUSAL:
            all_masked = (m_ij <= -1.0e5)
            p = tl.where(all_masked[:, None], 0.0, p)
        l_ij = tl.sum(p, 1)
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        if IS_CAUSAL:
            alpha = tl.where(all_masked, 0.0, alpha)
        acc = acc * alpha[:, None]
        v_mask = offs_n[:, None] < (kv_len - start_n)
        v = tl.load(V_ptrs, mask=v_mask, other=0.0, eviction_policy='evict_first')
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        if PV_ACCUM_FP32:
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        else:
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float16)
    return acc, l_i

@triton.autotune(
    list(fp16_configs),
    key=['qo_len', 'kv_len', 'H', 'HEAD_DIM', 'num_kv_groups']
)
@triton.jit
def _attn_fwd_fp16(Q, K, V, Out,
                   stride_qz, stride_qh, stride_qn,
                   stride_kz, stride_kh, stride_kn,
                   stride_vz, stride_vh, stride_vn,
                   stride_oz, stride_oh, stride_on,
                   qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
                   HEAD_DIM: tl.constexpr,
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   PV_ACCUM_FP32: tl.constexpr,
                   IS_CAUSAL: tl.constexpr,
                   SM_SCALE: tl.constexpr,
                   ):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len, other=0.0, eviction_policy='evict_last')
    q = (q * (SM_SCALE * RCP_LN2)).to(q.dtype)
    acc, l_i = _attn_fwd_inner_fp16(acc, l_i, m_i, q, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                     start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                     offs_m, offs_n,
                                     PV_ACCUM_FP32, IS_CAUSAL)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len), eviction_policy='evict_last')


def forward_fp16(q, k, v, sm_scale, tensor_layout="HND", output_dtype=torch.float16,
                 pv_accum_dtype="fp32", is_causal=False):
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    num_kv_groups = h_qo // h_kv
    grid = lambda META: (triton.cdiv(qo_len, META['BLOCK_M']), h_qo, b)
    _attn_fwd_fp16[grid](
        q, k, v, o,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        HEAD_DIM=head_dim,
        PV_ACCUM_FP32=(pv_accum_dtype == "fp32"),
        IS_CAUSAL=is_causal,
        SM_SCALE=sm_scale,
    )
    return o


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    headdim = q.size(-1)
    assert headdim in [64, 128], "headdim should be in [64, 128]."
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    seq_dim = 1 if tensor_layout == "NHD" else 2
    kv_len = k.size(seq_dim)
    if sm_scale is None:
        sm_scale = headdim ** -0.5
    if dtype == torch.float32:
        v = v.to(torch.float16)

    FP16_DIRECT_THRESHOLD = 512
    if kv_len <= FP16_DIRECT_THRESHOLD:
        if smooth_k:
            km = k.mean(dim=seq_dim, keepdim=True)
            k = k - km
        return forward_fp16(q, k, v, sm_scale, tensor_layout=tensor_layout,
                            output_dtype=dtype, pv_accum_dtype=pv_accum_dtype,
                            is_causal=is_causal)

    km = None
    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=False) 
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km, BLKQ=128, BLKK=64, sm_scale=sm_scale, tensor_layout=tensor_layout
    )
    o = forward(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, pv_accum_dtype=pv_accum_dtype, is_causal=is_causal)
    return o