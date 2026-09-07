"""SageAttention native HIP 内核测试 (gfx1035 / RDNA2 iGPU)。

覆盖当前默认 dispatch 下的全部数值路径:
  direct (V_T)        fp16/bf16 cross-attn      -> TestCrossAttn::test_direct
  int8 v10 INQ        HND/NHD/short self/cross  -> TestBasic::test_short (kv<=1024)
  int8 v10 non-INQ    HND/NHD/bf16 self long    -> TestBasic::test_int8_long, TestMaxErr
  int8 v10 causal     短序列 / 长序列 / q>kv     -> TestCausal
  (diagonal 早停)      NHD 布局                  -> TestCausal::test_nhd_causal
  GQA (kvh 映射)      非因果 / 因果               -> TestGQA
  smooth_k (mean)      int8 长序列               -> TestSmoothK
  边界/健壮            短序列/非对齐尾部/单头/确定性 -> TestEdgeCases
"""
import os

os.environ["SAGEATTN_BACKEND"] = "native"

import torch
import torch.nn.functional as F
import pytest


@pytest.fixture(autouse=True)
def check_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")


@pytest.fixture(autouse=True)
def seed_random():
    torch.manual_seed(0)


@pytest.fixture
def sageattn():
    """强制加载 native 扩展; 未编译则跳过。"""
    from sageattention import sageattn
    from sageattention.core import _get_native_ops

    try:
        _get_native_ops()
    except Exception as e:
        pytest.skip(f"native extension not built: {e}")
    return sageattn


def ref_hnd(q, k, v, is_causal=False, sm_scale=None):
    """HND [B,H,S,D] PyTorch SDPA 参考。"""
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale)


def ref_nhd(q, k, v, is_causal=False):
    """NHD [B,S,H,D] fp32 SDPA 参考 (高精度, 用于 MaxErr 回归)。"""
    with torch.no_grad():
        out = F.scaled_dot_product_attention(
            q.float().permute(0, 2, 1, 3),
            k.float().permute(0, 2, 1, 3),
            v.float().permute(0, 2, 1, 3),
            is_causal=is_causal,
        )
    return out.permute(0, 2, 1, 3).to(q.dtype)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def assert_close(out, ref, dtype):
    """统一精度断言: NaN/Inf 硬检查 + cos/mae 阈值 (int8 量化放宽)。"""
    assert not torch.isnan(out).any().item(), "output contains NaN"
    assert not torch.isinf(out).any().item(), "output contains Inf"
    cos = cosine_similarity(out, ref)
    mae = max_abs_error(out, ref)
    assert cos > (0.98 if dtype == torch.bfloat16 else 0.99), f"cosine {cos} too low"
    assert mae < 0.35, f"max abs error {mae} too high"


class TestBasic:
    """布局 (HND/NHD) x dtype (fp16/bf16) x 路径 (INQ 短序列 / v10 长序列)。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seq_len", [128, 256, 1024])
    def test_hnd_short(self, sageattn, head_dim, dtype, seq_len):
        """HND 短序列 (kv<=1024) -> int8 v10 INQ。1024 为 INQ 边界。"""
        b, h = 2, 8
        q = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        assert_close(out, ref_hnd(q, k, v), dtype)

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256])
    def test_nhd_short(self, sageattn, head_dim, seq_len):
        """NHD 短序列 -> INQ (NHD zero-copy as_strided Q/V 视图)。"""
        b, h = 2, 8
        q = torch.randn(b, seq_len, h, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(b, seq_len, h, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(b, seq_len, h, head_dim, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="NHD")
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        assert_close(out, ref_hnd(qh, kh, vh).transpose(1, 2), torch.float16)

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_int8_long(self, sageattn, head_dim, dtype):
        """kv>1024 self -> int8 v10 (非 INQ) 长序列路径。"""
        b, h, seq_len = 1, 8, 2048
        q = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        assert_close(out, ref_hnd(q, k, v), dtype)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_nhd_int8_long(self, sageattn, head_dim):
        """NHD 长序列 (kv>1024) -> NHD 布局 + 非 INQ 主内核 + prepass。"""
        b, h, seq_len = 1, 4, 2048
        q = torch.randn(b, seq_len, h, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(b, seq_len, h, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(b, seq_len, h, head_dim, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="NHD")
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        assert_close(out, ref_hnd(qh, kh, vh).transpose(1, 2), torch.float16)

    def test_sm_scale(self, sageattn):
        """自定义 sm_scale (默认 head_dim^-0.5)。"""
        b, h, seq_len, d = 1, 4, 256, 64
        q = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        v = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        sm_scale = 0.5
        out = sageattn(q, k, v, tensor_layout="HND", sm_scale=sm_scale)
        assert_close(out, ref_hnd(q, k, v, sm_scale=sm_scale), torch.float16)


class TestCrossAttn:
    """q_len != kv_len: direct (V_T) 短 kv, int8 v10 长 kv。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_direct(self, sageattn, head_dim, dtype):
        """cross、kv 短 -> fp16/bf16 direct (V_T) 内核。

        D64: kv<=6144; D128: q*2<kv 且 kv<=4096。选 kv 使其走 direct。
        """
        b, h = 1, 4
        q_len, kv_len = (512, 2048) if head_dim == 64 else (512, 1536)
        q = torch.randn(b, h, q_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, kv_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, kv_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        assert_close(out, ref_hnd(q, k, v), dtype)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_int8_long(self, sageattn, head_dim):
        """cross、kv 超长 -> int8 v10 (非 INQ)。D64 需 kv>6144, D128 需 kv>4096 且 q*2<kv。"""
        b, h, dtype = 1, 4, torch.float16
        q_len, kv_len = 512, 8192
        q = torch.randn(b, h, q_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, kv_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, kv_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        assert_close(out, ref_hnd(q, k, v), dtype)


class TestCausal:
    """Causal -> v10 causal + 块对角线早停 (kb_lim=min(kv,qo,m0+BM))。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    def test_causal_short(self, sageattn, head_dim, dtype, seq_len):
        """HND self 短序列 (对角早停 + 掩码全路径)。"""
        b, h = 2, 8
        q = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        assert_close(out, ref_hnd(q, k, v, is_causal=True), dtype)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_causal_long(self, sageattn, head_dim):
        """HND self 长序列 (kv>1024): 完整对角早停, 尾块无越界。"""
        b, h, seq_len, dtype = 1, 4, 2048, torch.float16
        q = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, seq_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        assert_close(out, ref_hnd(q, k, v, is_causal=True), dtype)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_causal_cross(self, sageattn, head_dim):
        """q>kv causal: kb_lim 被 qo_len 截断 (kb_lim=min(kv,qo,m0+BM))。"""
        b, h, dtype = 1, 4, torch.float16
        q_len, kv_len = 2048, 1024
        q = torch.randn(b, h, q_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, h, kv_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, h, kv_len, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        assert_close(out, ref_hnd(q, k, v, is_causal=True), dtype)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_nhd_causal(self, sageattn, head_dim):
        """NHD + causal (NHD 视图 + 对角早停组合)。"""
        b, h, seq_len, dtype = 1, 4, 256, torch.float16
        q = torch.randn(b, seq_len, h, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(b, seq_len, h, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(b, seq_len, h, head_dim, dtype=dtype, device="cuda")
        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        assert_close(out, ref_hnd(qh, kh, vh, is_causal=True).transpose(1, 2), dtype)


class TestGQA:
    """Grouped Query Attention: kvh = h/(q_heads/kv_heads) 映射。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("num_kv_groups", [2, 4])
    def test_gqa(self, sageattn, head_dim, num_kv_groups):
        b, q_heads, seq_len = 2, 8, 256
        kv_heads = q_heads // num_kv_groups
        q = torch.randn(b, q_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(b, kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(b, kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        k_exp = k.repeat_interleave(num_kv_groups, dim=1)
        v_exp = v.repeat_interleave(num_kv_groups, dim=1)
        assert_close(out, ref_hnd(q, k_exp, v_exp), torch.float16)

    def test_gqa_causal(self, sageattn):
        """GQA + causal (kvh 映射与对角早停组合)。"""
        b, q_heads, kv_heads, seq_len, d = 1, 8, 4, 256, 64
        q = torch.randn(b, q_heads, seq_len, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, kv_heads, seq_len, d, dtype=torch.float16, device="cuda")
        v = torch.randn(b, kv_heads, seq_len, d, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        k_exp = k.repeat_interleave(q_heads // kv_heads, dim=1)
        v_exp = v.repeat_interleave(q_heads // kv_heads, dim=1)
        assert_close(out, ref_hnd(q, k_exp, v_exp, is_causal=True), torch.float16)


class TestSmoothK:
    """smooth_k (K 减 mean) int8 路径开关。"""

    @pytest.mark.parametrize("smooth_k", [True, False])
    def test_effect(self, sageattn, smooth_k):
        b, h, seq_len, d = 2, 8, 2048, 64
        q = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        v = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND", smooth_k=smooth_k)
        assert_close(out, ref_hnd(q, k, v), torch.float16)


class TestEdgeCases:
    """鲁棒性: 短序列 / 非对齐尾部 / 单头 / 确定性。"""

    def test_short_seq(self, sageattn):
        b, h, seq_len, d = 1, 4, 32, 64
        q = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        v = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        assert_close(out, ref_hnd(q, k, v), torch.float16)

    def test_non_aligned_seq(self, sageattn):
        """非 BN/BM 对齐序列: 尾部 tile 掩码 (n>=kv_len 置零) 不产生 NaN。"""
        b, h, d = 1, 4, 64
        for seq_len in [33, 65, 100, 127, 129, 255, 1057]:
            q = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
            k = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
            v = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
            assert_close(sageattn(q, k, v, tensor_layout="HND"), ref_hnd(q, k, v), torch.float16)

    def test_single_head(self, sageattn):
        b, h, seq_len, d = 1, 1, 256, 64
        q = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        v = torch.randn(b, h, seq_len, d, dtype=torch.float16, device="cuda")
        out = sageattn(q, k, v, tensor_layout="HND")
        assert_close(out, ref_hnd(q, k, v), torch.float16)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_repeated_same_result(self, sageattn, head_dim):
        """int8 内核重复运行逐位一致 (软max LDS 无竞态, softmax LDS 写读同波前有序)。"""
        b, h, seq_len = 1, 4, 2048
        q = torch.randn(b, h, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(b, h, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(b, h, seq_len, head_dim, dtype=torch.float16, device="cuda")
        outs = [sageattn(q, k, v, tensor_layout="HND") for _ in range(3)]
        for o in outs[1:]:
            assert torch.equal(o, outs[0]), "int8 kernel non-deterministic"


class TestMaxErr:
    """int8 各路径 MaxErr 回归 (基准同 benchmark_attn.py: max_err < 0.05, cos > 0.99)。

    D128 的 BN32 在线 softmax 曾在 kv>2048 时精度劣化 (Anima/nVAE: mae 0.2-0.47);
    这些用例锁定 kv 跨 16 行 k_scale 分组边界的长序列量化精度。
    """

    # (name, b, h_q, h_kv, sq, d, dtype) — NHD 布局, self
    CASES = [
        ("D128_BF16_int8_2304", 1, 4, 4, 2304, 128, torch.bfloat16),
        ("D128_FP16_int8_2304", 1, 4, 4, 2304, 128, torch.float16),
        ("D64_BF16_int8_3072", 1, 4, 4, 3072, 64, torch.bfloat16),
        ("D64_FP16_int8_3072", 1, 4, 4, 3072, 64, torch.float16),
        ("D128_FP16_int8_1024", 1, 4, 4, 1024, 128, torch.float16),
        ("D64_FP16_int8_2048", 1, 4, 4, 2048, 64, torch.float16),
    ]

    @pytest.mark.parametrize("case_idx", range(len(CASES)))
    def test_maxerr(self, sageattn, case_idx):
        name, b, h_q, h_kv, sq, d, dtype = self.CASES[case_idx]
        q = torch.randn(b, sq, h_q, d, device="cuda", dtype=dtype)
        k = torch.randn(b, sq, h_kv, d, device="cuda", dtype=dtype)
        v = torch.randn(b, sq, h_kv, d, device="cuda", dtype=dtype)
        out = sageattn(q, k, v, tensor_layout="NHD")
        ref = ref_nhd(q, k, v)
        assert not torch.isnan(out).any().item(), f"{name}: NaN"
        assert not torch.isinf(out).any().item(), f"{name}: Inf"
        cos = cosine_similarity(out, ref)
        mae = max_abs_error(out, ref)
        assert cos > 0.99, f"{name}: cosine {cos} too low"
        assert mae < 0.05, f"{name}: max abs error {mae} too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])