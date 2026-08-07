#pragma once

#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

Tensor qk_int8_sv_bf16_attn_gfx11(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor q_scale,
    Tensor k_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale);

Tensor fp16_attn_gfx11(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale);

Tensor bf16_attn_gfx11(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale);

std::vector<Tensor> quant_qk_int8_gfx11(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale);

Tensor mean_seq_gfx11(Tensor input, int64_t tensor_layout);
