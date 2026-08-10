#pragma once

#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

// 转置布局 (Triton 风格) attention kernel: qk^T = k @ q^T + permlanex16 归约
Tensor qk_int8_sv_bf16_attn_gfx11_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor q_scale,
    Tensor k_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale);

Tensor fp16_attn_gfx11_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t bm_sel);

Tensor bf16_attn_gfx11_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t bm_sel);

std::vector<Tensor> quant_qk_int8_gfx11(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale);

Tensor mean_seq_gfx11(Tensor input, int64_t tensor_layout);

// V [B,N,H,D] -> V_T [B,H,D,N] (contiguous), 供无 LDS PV 模式
Tensor v_transpose_gfx11(Tensor value, Tensor value_t, int64_t tensor_layout);
