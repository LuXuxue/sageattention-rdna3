#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "attn_gfx11.h"

PyMODINIT_FUNC PyInit__qattn_gfx11(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_gfx11",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

STABLE_TORCH_LIBRARY(sageattention, m) {
    m.def("qk_int8_sv_bf16_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor q_scale, Tensor k_scale, int tensor_layout, "
            "int is_causal, float sm_scale"
          ") -> Tensor");
    m.def("fp16_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "int tensor_layout, int is_causal, float sm_scale"
          ") -> Tensor");
    m.def("bf16_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "int tensor_layout, int is_causal, float sm_scale"
          ") -> Tensor");
    m.def("quant_qk_int8("
            "Tensor query, Tensor key, Tensor key_mean, int tensor_layout, "
            "float sm_scale"
          ") -> Tensor[]");
    m.def("mean_seq(Tensor input, int tensor_layout) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(sageattention, CUDA, m) {
    m.impl("qk_int8_sv_bf16_attn", TORCH_BOX(qk_int8_sv_bf16_attn_gfx11));
    m.impl("fp16_attn", TORCH_BOX(fp16_attn_gfx11));
    m.impl("bf16_attn", TORCH_BOX(bf16_attn_gfx11));
    m.impl("quant_qk_int8", TORCH_BOX(quant_qk_int8_gfx11));
    m.impl("mean_seq", TORCH_BOX(mean_seq_gfx11));
}
