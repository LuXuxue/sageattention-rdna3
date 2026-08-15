import torch
import torch.nn.functional as F
import pytest
import os
from typing import Optional

# 默认后端为 native (本项目核心); 用户可显式设 SAGEATTN_BACKEND=triton 覆盖
# setdefault 尊重已有环境变量, 与 core.py 的 triton 默认值解耦
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("SAGEATTN_BACKEND", "native")
_IS_TRITON_BACKEND = os.getenv("SAGEATTN_BACKEND").lower() == "triton"


@pytest.fixture(autouse=True)
def seed_random():
    """固定随机种子, 保证测试可复现 (避免随机数据导致的偶发失败)。"""
    torch.manual_seed(0)


def reference_attention(q, k, v, is_causal=False, sm_scale=None):
    """Reference implementation using PyTorch SDPA (期望 HND 布局 [B,H,S,D])。"""
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=sm_scale
    )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute max absolute error."""
    return (a.float() - b.float()).abs().max().item()


def assert_close(out, ref, dtype):
    """统一的精度断言。

    - fp16: 直接路径 (kv 短) 误差极小; int8 路径 (kv 长) 有量化误差
    - bf16: 精度本身较低, 阈值放宽
    """
    cos = cosine_similarity(out, ref)
    mae = max_abs_error(out, ref)
    if dtype == torch.bfloat16:
        assert cos > 0.98, f"bf16 cosine similarity {cos} too low"
    else:
        assert cos > 0.99, f"fp16 cosine similarity {cos} too low"
    # int8 量化引入的 per-element 误差, 随机数据下可达 ~0.3
    assert mae < 0.35, f"Max abs error {mae} too high"
    return cos, mae


@pytest.fixture(autouse=True)
def check_gpu():
    """Skip tests if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")


@pytest.fixture
def sageattn():
    """Import sageattn from the package."""
    try:
        from sageattention import sageattn
        if not _IS_TRITON_BACKEND:
            from sageattention.core import GFX11_NATIVE_ENABLED
            if not GFX11_NATIVE_ENABLED:
                pytest.skip(
                    "sageattention native extension (_qattn_gfx11) not built. "
                    "Run: pip install -e . --no-build-isolation on a ROCm/HIP system."
                )
        return sageattn
    except ImportError as e:
        pytest.skip(f"sageattention not installed: {e}")


class TestSageAttnBasic:
    """基本功能: HND/NHD 布局, fp16/bf16, 直接路径与 int8 路径。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    def test_hnd_fp16(self, sageattn, head_dim, seq_len):
        """HND 布局, fp16, 直接路径 (kv=seq<=1024 for D=64, <=512 for D=128)。"""
        batch, heads = 2, 8
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        assert_close(out, ref, torch.float16)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_hnd_fp16_int8_path(self, sageattn, head_dim):
        """HND 布局, fp16, 长序列 (触发 int8 量化路径: D=64 kv>1024, D=128 kv>512)。"""
        batch, heads = 1, 4
        # D=64: 2048>1024; D=128: 1024>512
        seq_len = 2048 if head_dim == 64 else 1024
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        assert_close(out, ref, torch.float16)

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256])
    def test_nhd_fp16(self, sageattn, head_dim, seq_len):
        """NHD 布局, fp16。"""
        batch, heads = 2, 8
        q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        # 转 HND 做参考
        q_hnd = q.transpose(1, 2)
        k_hnd = k.transpose(1, 2)
        v_hnd = v.transpose(1, 2)
        ref = reference_attention(q_hnd, k_hnd, v_hnd, is_causal=False).transpose(1, 2)
        assert_close(out, ref, torch.float16)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_bf16(self, sageattn, head_dim):
        """bf16 dtype。"""
        batch, heads, seq_len = 2, 8, 256
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        assert_close(out, ref, torch.bfloat16)

    def test_sm_scale(self, sageattn):
        """自定义 sm_scale (默认应为 head_dim^-0.5)。"""
        batch, heads, seq_len, head_dim = 1, 4, 256, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        sm_scale = 0.5
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=sm_scale)
        ref = reference_attention(q, k, v, is_causal=False, sm_scale=sm_scale)
        assert_close(out, ref, torch.float16)


class TestSageAttnCrossAttn:
    """Cross-attention: q_len != kv_len。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_cross_attn(self, sageattn, head_dim):
        """q_len 与 kv_len 不同 (kv 较短, 触发 fp16 direct 路径)。"""
        batch, heads = 1, 4
        q_len, kv_len = 1024, 128
        q = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        assert_close(out, ref, torch.float16)

    def test_cross_attn_int8(self, sageattn):
        """cross-attention: kv 长 (触发 int8 路径)。"""
        batch, heads, head_dim = 1, 4, 64
        q_len, kv_len = 512, 2048
        q = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        assert_close(out, ref, torch.float16)


class TestSageAttnCausal:
    """Causal attention tests。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    def test_causal(self, sageattn, head_dim, seq_len):
        """Causal attention (fp16 direct)。"""
        batch, heads = 2, 8
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        ref = reference_attention(q, k, v, is_causal=True)
        assert_close(out, ref, torch.float16)

    def test_causal_int8(self, sageattn):
        """Causal + 长序列 (int8 路径)。"""
        batch, heads, head_dim = 1, 4, 64
        seq_len = 2048
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        ref = reference_attention(q, k, v, is_causal=True)
        assert_close(out, ref, torch.float16)


class TestSageAttnGQA:
    """Grouped Query Attention tests。"""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("num_kv_groups", [2, 4])
    def test_gqa(self, sageattn, head_dim, num_kv_groups):
        """GQA: fewer KV heads than Q heads。"""
        batch, q_heads, seq_len = 2, 8, 256
        kv_heads = q_heads // num_kv_groups

        q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)

        # Reference: expand KV heads (相邻 group 映射)
        k_exp = k.repeat_interleave(num_kv_groups, dim=1)
        v_exp = v.repeat_interleave(num_kv_groups, dim=1)
        ref = reference_attention(q, k_exp, v_exp, is_causal=False)
        assert_close(out, ref, torch.float16)


class TestSageAttnSmoothK:
    """Smooth K quantization tests。"""

    @pytest.mark.parametrize("smooth_k", [True, False])
    def test_smooth_k_effect(self, sageattn, smooth_k):
        """With and without smooth_k (int8 路径)。"""
        batch, heads, seq_len, head_dim = 2, 8, 2048, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", smooth_k=smooth_k)
        ref = reference_attention(q, k, v, is_causal=False)
        assert_close(out, ref, torch.float16)


class TestSageAttnEdgeCases:
    """Edge case tests。"""

    def test_short_seq(self, sageattn):
        """Very short sequence。"""
        batch, heads, seq_len, head_dim = 1, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND")
        ref = reference_attention(q, k, v)
        assert_close(out, ref, torch.float16)

    def test_non_aligned_seq(self, sageattn):
        """Sequence length not aligned to block size。"""
        batch, heads, head_dim = 1, 4, 64
        for seq_len in [33, 65, 100, 127, 129, 255, 1057]:
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
            v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

            out = sageattn(q, k, v, tensor_layout="HND")
            ref = reference_attention(q, k, v)
            cos = cosine_similarity(out, ref)
            assert cos > 0.98, (
                f"Non-aligned seq_len={seq_len} cosine similarity {cos} too low"
            )

    def test_single_head(self, sageattn):
        """Single attention head。"""
        batch, heads, seq_len, head_dim = 1, 1, 256, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND")
        ref = reference_attention(q, k, v)
        assert_close(out, ref, torch.float16)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
