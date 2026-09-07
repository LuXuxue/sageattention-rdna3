#pragma once

#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

// gfx103x (RDNA2) 转置布局 attention kernel 的 host 分发入口 (由 pybind 映射到 op)
Tensor qk_int8_sv_bf16_attn_gfx103x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor q_scale,
    Tensor k_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    Tensor q_fp);

Tensor fp16_attn_gfx103x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t bm_sel);

Tensor bf16_attn_gfx103x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t bm_sel);

std::vector<Tensor> quant_qk_int8_gfx103x(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale,
    int64_t skip_q);

Tensor mean_seq_gfx103x(Tensor input, int64_t tensor_layout);

// V [B,N,H,D] -> V_T [B,H,D,N] (contiguous), 供无 LDS PV 模式
Tensor v_transpose_gfx103x(Tensor value, Tensor value_t, int64_t tensor_layout);

// TEMPORARY: fetch the int8-PV debug dump (diag bit 5, env SAGEATTN_GFX10_IPV_DBG)
Tensor ipv_dbg_fetch_gfx103x(Tensor like);
