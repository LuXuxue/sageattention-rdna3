import torch
import torch.nn.functional as F
import pytest
import os
from typing import Optional

_IS_TRITON_BACKEND = os.getenv("SAGEATTN_BACKEND", "native").lower() == "triton"


def reference_attention(q, k, v, is_causal=False, sm_scale=None):
    """Reference implementation using PyTorch SDPA."""
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
    """Basic functionality tests."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    def test_hnd_fp16(self, sageattn, head_dim, seq_len):
        """Test HND layout with fp16."""
        batch, heads = 2, 8
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)

        cos_sim = cosine_similarity(out, ref)
        mae = max_abs_error(out, ref)
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low"
        # int8 quantization inherently introduces per-element error;
        # MAE up to ~0.3 is expected for head_dim=128 with random data.
        assert mae < 0.35, f"Max abs error {mae} too high"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_nhd_fp16(self, sageattn, head_dim):
        """Test NHD layout with fp16."""
        batch, heads, seq_len = 2, 8, 256
        q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        # For NHD, transpose to HND for reference
        q_hnd = q.transpose(1, 2)
        k_hnd = k.transpose(1, 2)
        v_hnd = v.transpose(1, 2)
        ref_hnd = reference_attention(q_hnd, k_hnd, v_hnd, is_causal=False)
        ref = ref_hnd.transpose(1, 2)

        cos_sim = cosine_similarity(out, ref)
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_bf16(self, sageattn, head_dim):
        """Test with bf16 dtype."""
        batch, heads, seq_len = 2, 8, 256
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)

        cos_sim = cosine_similarity(out, ref)
        assert cos_sim > 0.98, f"Cosine similarity {cos_sim} too low for bf16"


class TestSageAttnCausal:
    """Causal attention tests."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    def test_causal(self, sageattn, head_dim, seq_len):
        """Test causal attention."""
        batch, heads = 2, 8
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        ref = reference_attention(q, k, v, is_causal=True)

        cos_sim = cosine_similarity(out, ref)
        assert cos_sim > 0.99, f"Causal cosine similarity {cos_sim} too low"


class TestSageAttnGQA:
    """Grouped Query Attention tests."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("num_kv_groups", [2, 4])
    def test_gqa(self, sageattn, head_dim, num_kv_groups):
        """Test grouped query attention (fewer KV heads than Q heads)."""
        batch, q_heads, seq_len = 2, 8, 256
        kv_heads = q_heads // num_kv_groups

        q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)

        # Reference: expand KV heads
        k_exp = k.repeat_interleave(num_kv_groups, dim=1)
        v_exp = v.repeat_interleave(num_kv_groups, dim=1)
        ref = reference_attention(q, k_exp, v_exp, is_causal=False)

        cos_sim = cosine_similarity(out, ref)
        assert cos_sim > 0.99, f"GQA cosine similarity {cos_sim} too low"


class TestSageAttnSmoothK:
    """Smooth K quantization tests."""

    @pytest.mark.parametrize("smooth_k", [True, False])
    def test_smooth_k_effect(self, sageattn, smooth_k):
        """Test with and without smooth_k."""
        batch, heads, seq_len, head_dim = 2, 8, 256, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", smooth_k=smooth_k)
        ref = reference_attention(q, k, v, is_causal=False)

        cos_sim = cosine_similarity(out, ref)
        # smooth_k should generally give better accuracy
        threshold = 0.99 if smooth_k else 0.98
        assert cos_sim > threshold, (
            f"Cosine similarity {cos_sim} too low (smooth_k={smooth_k})"
        )


class TestSageAttnEdgeCases:
    """Edge case tests."""

    def test_short_seq(self, sageattn):
        """Test with very short sequence."""
        batch, heads, seq_len, head_dim = 1, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND")
        ref = reference_attention(q, k, v)

        cos_sim = cosine_similarity(out, ref)
        assert cos_sim > 0.98, f"Short seq cosine similarity {cos_sim} too low"

    def test_non_aligned_seq(self, sageattn):
        """Test with sequence length not aligned to block size."""
        batch, heads, head_dim = 1, 4, 64
        for seq_len in [33, 65, 100, 127, 129, 255]:
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
            v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

            out = sageattn(q, k, v, tensor_layout="HND")
            ref = reference_attention(q, k, v)

            cos_sim = cosine_similarity(out, ref)
            assert cos_sim > 0.98, (
                f"Non-aligned seq_len={seq_len} cosine similarity {cos_sim} too low"
            )

    def test_single_head(self, sageattn):
        """Test with single attention head."""
        batch, heads, seq_len, head_dim = 1, 1, 256, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND")
        ref = reference_attention(q, k, v)

        cos_sim = cosine_similarity(out, ref)
        assert cos_sim > 0.99, f"Single head cosine similarity {cos_sim} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
