import torch
import torch.nn.functional as F
import pytest
import os
from typing import Optional

# 后端显式指定：所有测试通过 conftest.py 的 pytest --backend 选项显式选择 native/triton。
# _selected_backend (conftest.py) 在导入 sageattention 之前设置 SAGEATTN_BACKEND，
# 避免 core.py 在 import 时读环境变量的歧义。
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


@pytest.fixture(autouse=True)
def seed_random():
    """固定随机种子, 保证测试可复现 (避免随机数据导致的偶发失败)。"""
    torch.manual_seed(0)


def reference_attention(q, k, v, is_causal=False, sm_scale=None):
    """Reference implementation using PyTorch SDPA (期望 HND 布局 [B,H,S,D])。"""
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=sm_scale
    )


def reference_fp32_nhd(q, k, v, is_causal=False):
    """High-precision fp32 SDPA reference for NHD input [B,S,H,D]. 输出转回原 dtype。"""
    with torch.no_grad():
        out = F.scaled_dot_product_attention(
            q.float().permute(0, 2, 1, 3),
            k.float().permute(0, 2, 1, 3),
            v.float().permute(0, 2, 1, 3),
            is_causal=is_causal,
        )
        return out.permute(0, 2, 1, 3).to(q.dtype)


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
    - NaN/Inf 检查：assert_close 本身不强制要求 finite，但包装器会检查
    """
    # NaN/Inf 早期检测：kernel bug 可能产生 NaN 而不违反 cos/mae 阈值（NaN 在比较中被忽略）
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    assert not has_nan, f"Output contains NaN values — kernel likely produced garbage (NaN passes cos/mae comparisons)."
    assert not has_inf, f"Output contains Inf values — kernel likely overflowed."

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
def sageattn(backend):
    """Import sageattn from the package."""
    try:
        from sageattention import sageattn
        if backend == "native":
            from sageattention.core import _get_native_ops
            try:
                _get_native_ops()
            except Exception as e:
                pytest.skip(
                    f"sageattention native extension not built: {e}. "
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


class TestSageAttnInt8KScale:
    """k_scale per-column (per-16-row group) 回归测试。

    所有 int8 kernel 的 QK 曾只按 BN-tile 读一次 k_scale (per-32), 而 k_scale 实际按
    MIN_BLK_K=16 行分组。BN=32 的 tile 横跨两个 scale group, 导致每个 tile 的后 16 列
    被错误乘了前 16 列的 scale → cos≈0.96-0.97。
    该 bug 曾因测试覆盖不全而漏检:
      - D=64: 测试最大 N=2048 (旧阈值下走 direct, 不触 int8), int8 只在 >2048 才进
        (而未测); 现在默认 int8 用于短序列 (N<=768), 需显式强制覆盖。
      - D=128 BN=32: 旧默认走 v3 BN=16 (准确), BN=32 只经 env 可达, 故同 bug 从未被
        默认测试覆盖; 现在 v4 BM=64 BN=32 是默认, 必须回归。

    测试策略: 用 monkeypatch 强制 int8 路径, 覆盖各布局/长短序列,
    确保 BN=32 (kv>77) 与 k_scale 16 行分组边界均被验证。

    gfx103 env 陷阱: D64 用 `use_direct=(kv>thr)`, D128 用 `use_direct=(kv<=thr)`,
    且 `int("0") or default` 因字符串"0"为真值返回 default。故 D64 的 env="0" 解析为
    default (HND=768/NHD=1016) -> 仅短序列走 int8, 长序列落回 direct (掩盖 int8 路径);
    D128 的 env="0" 解析为 9999999 -> 恒走 direct (根本不触 int8)。要真正强制 int8:
    D64 用大阈值 "9999999" (kv 恒 < 之), D128 用负阈值 "-1" (kv 恒 > 之)。
    """

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [256, 512, 1024, 2048])
    def test_int8_forced_hnd(self, sageattn, monkeypatch, head_dim, seq_len):
        """HND + 真正强制 int8 (D64=9999999/D128=-1), 短到中长序列。修复前 cos≈0.96 (k_scale bug)。"""
        # 强制 int8 路径 (无论 D 的默认阈值如何)
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D64", "9999999")
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D128", "-1")
        batch, heads = 1, 4
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        cos = cosine_similarity(out, ref)
        # 修复前 cos≈0.96; 修复后 >0.9995。用 0.99 判定 (含 int8 量化误差余量)。
        assert cos > 0.99, f"HND D={head_dim} N={seq_len} int8 cosine similarity {cos} too low (k_scale per-column bug?)"

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [512, 1024])
    def test_int8_forced_nhd(self, sageattn, monkeypatch, head_dim, seq_len):
        """NHD + 强制 int8, 验证 k_scale per-column 在 NHD 布局同样正确。"""
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D64", "9999999")
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D128", "-1")
        batch, heads = 1, 4
        q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        q_hnd = q.transpose(1, 2)
        k_hnd = k.transpose(1, 2)
        v_hnd = v.transpose(1, 2)
        ref = reference_attention(q_hnd, k_hnd, v_hnd, is_causal=False).transpose(1, 2)
        cos = cosine_similarity(out, ref)
        assert cos > 0.99, f"NHD D={head_dim} N={seq_len} int8 cosine similarity {cos} too low (k_scale per-column bug?)"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_int8_forced_long(self, sageattn, monkeypatch, head_dim):
        """真正强制 int8 + 长序列: D=64 N=4096 走 v4 BM=128, D=128 N=4096 走 v4 BM=64 BN=32。
        (历史: D64 env="0" 实为 direct, 此测试曾从未触 int8 长路径; 现 D64="9999999"/D128="-1"
        强制 int8, 覆盖 kv>2048 的 k_scale 16 行分组与 BN=32 边界。)
        修复前 (k_scale per-column bug) cos≈0.96-0.97; 修复后 >0.999。"""
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D64", "9999999")
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D128", "-1")
        batch, heads, seq_len = 1, 4, 4096
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        cos = cosine_similarity(out, ref)
        assert cos > 0.99, f"HND D={head_dim} N=4096 long int8 cosine similarity {cos} too low (k_scale per-column bug?)"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_int8_forced_bf16(self, sageattn, monkeypatch, head_dim):
        """强制 int8 + bf16, 验证 k_scale per-column 在 bf16 输入 dtype 同样正确。"""
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D64", "9999999")
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D128", "-1")
        batch, heads, seq_len = 1, 4, 1024
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        cos = cosine_similarity(out, ref)
        # bf16 精度较低, 放宽到 0.98 (与 assert_close 一致)
        assert cos > 0.98, f"HND D={head_dim} N=1024 bf16 int8 cosine similarity {cos} too low"


class TestSageAttnD64Dispatch:
    """gfx1035 D=64 self-attention 分发边界回归。

    实测 (NHD 与 HND) D=64 self 的 int8/direct 交叉点均在 ~1016:
      - kv<1016: int8 内核胜 (dot4 吞吐, 短序列量化开销被并行分摊)
      - kv>=1024: direct-v2 内核胜 (v2 需 >=1024 行才有足够 block; v1 在 512-1023 病理性慢)
    历史 bug: 旧默认阈值 1024 使 1024² 走 int8 (应走 direct v2), 且使 <1024 走 v1 (病理性慢)。
    本阶段修正: HND 阈值曾误设 768, 使 769-1023 走 direct-v1 (0.02T) 而 int8 6x 更快 (0.13T);
    已统一为 1016。本测试验证边界尺寸在默认分发下正确性不受损, 并验证 769-1023 路由到 int8。

    注意 (本阶段实测修正): 曾疑 D64 int8 内核 v4 BM=128 在 seq>=512 产出错误结果
    (cos≈0.03-0.1), 从而 512-1015 默认分发走"错误 int8"。经真机验证: 当前代码该范围
    int8 输出正确且确定 (cos>0.999), 疑似误报源是已回退的 __syncwarp() 实验, 而非 base 内核。
    故 int8 在 512-1015 路径可放心使用; 512/768/960 用例即覆盖此范围。
    """

    @pytest.mark.parametrize("seq_len", [768, 960, 1024, 2048, 4096])
    def test_d64_self_default_dispatch(self, sageattn, seq_len):
        """D=64 self 默认分发 (不强制): 边界尺寸走对应最优路径且结果正确。"""
        batch, heads, head_dim = 1, 4, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = reference_attention(q, k, v, is_causal=False)
        cos = cosine_similarity(out, ref)
        assert cos > 0.99, f"HND D=64 N={seq_len} default dispatch cosine {cos} too low"
        del q, k, v, out, ref
        torch.cuda.empty_cache()

    def test_d64_self_gfx103_threshold_is_1016(self):
        """gfx1035 D=64 self 默认阈值必须为 1016 (HND 与 NHD 统一)。

        回归 (本阶段实测修正): 旧 HND 阈值 768 使 769-1023 走 direct-v1 (0.02T, 病理性慢,
        需 >=1024 行才有 block), 而 int8 在该范围 6x 更快 (0.13T)。直接锁定阈值常量,
        防止再次误调。dispatch 用 `use_direct=(kv>thr)`: thr=1016 保证 kv<1016 走 int8,
        kv>=1024 走 direct-v2, 两布局一致。
        """
        from sageattention.core import _gfx103_d64_threshold
        for layout in ("HND", "NHD"):
            assert _gfx103_d64_threshold(layout) == 1016, (
                f"gfx1035 D64 {layout} threshold must be 1016 (int8 below, direct-v2 at 1024+)"
            )

    @pytest.mark.parametrize("seq_len", [768, 960, 1024, 2048])
    def test_d64_self_nhd_default_dispatch(self, sageattn, seq_len):
        """NHD 布局 D=64 self 默认分发: 验证 int8/direct 阈值边界 (1016) 附近正确。"""
        batch, heads, head_dim = 1, 4, 64
        q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda")

        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        q_hnd = q.transpose(1, 2)
        k_hnd = k.transpose(1, 2)
        v_hnd = v.transpose(1, 2)
        ref = reference_attention(q_hnd, k_hnd, v_hnd, is_causal=False).transpose(1, 2)
        cos = cosine_similarity(out, ref)
        assert cos > 0.99, f"NHD D=64 N={seq_len} default dispatch cosine {cos} too low"
        del q, k, v, out, ref
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_int8_repeated_same_result(self, sageattn, monkeypatch, head_dim):
        """int8 内核重复运行结果确定性: 验证无共享内存竞态。

        历史: 曾有人给 v4 的 QK 写 s_tile 与 softmax 读 s_tile 之间加 __syncwarp() 想修"潜在
        竞态", 反而破坏 int8 内核 (seq>=512 变错)。RDNA2 同波前 LDS 写读天然有序, 原"无同步"
        行为在实践上正确, 故 syncwarp 已回退。此测试 (D64=9999999/D128=-1 真正强制 int8)
        验证重复运行逐位一致 (确定性), 防止任何会引入竞态的改动回退。
        """
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D64", "9999999")
        monkeypatch.setenv("SAGEATTN_DIRECT_THRESHOLD_D128", "-1")
        batch, heads, seq_len = 1, 4, 2048
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

        outs = [sageattn(q, k, v, tensor_layout="HND", is_causal=False) for _ in range(3)]
        for i in range(1, len(outs)):
            assert torch.equal(outs[i], outs[0]), (
                f"D={head_dim} N=2048 int8 repeated run {i} differs from run 0 "
                "(softmax LDS race? non-deterministic)"
            )
        del q, k, v
        torch.cuda.empty_cache()


class TestSageAttnMaxErr:
    """int8 / fp16-direct 路径 MaxErr 回归测试。

    覆盖 gfx1035 各主要路径:
      - D=128 int8 (BF16/FP16): 修复前 V2(BN=32) 在 kv>2048 时 online-softmax 精度损失
        (Anima01/03/05, AnimaVAE01: mae 0.2-0.47, cos<0.98)。
        修复后路由 V3(BN=16), mae 应 < 0.05 (对齐 triton)。
      - D=64 int8 (BF16/FP16): 长序列, 保持低 MaxErr。
      - D=128/D=64 fp16 direct: 短序列 (非 int8), 精度应更高。
    所有用例与 benchmark_attn.py 相同判定: max_err < 0.05 才通过。
    """

    # 每个路径用小型但能触发对应分支的用例 (兼顾触发 bug + 测试快速 + 避免 fp32 SDPA 参考 OOM)。
    # int8 路径: D=128 需 kv>2048 (seq=2304 为最小触发 V2 精度 bug 尺寸); D=64 需 kv>3072 (seq=3072)。
    # direct 路径: D=128 kv<=2048 (seq=1024); D=64 kv<=3072 (seq=2048)。
    # (name, b, h_q, h_kv, sq, d, dtype)
    CASES = [
        ("D128_BF16_int8", 1, 4, 4, 2304, 128, torch.bfloat16),
        ("D128_FP16_int8", 1, 4, 4, 2304, 128, torch.float16),
        ("D64_BF16_int8", 1, 4, 4, 3072, 64, torch.bfloat16),
        ("D64_FP16_int8", 1, 4, 4, 3072, 64, torch.float16),
        ("D128_FP16_direct", 1, 4, 4, 1024, 128, torch.float16),
        ("D64_FP16_direct", 1, 4, 4, 2048, 64, torch.float16),
    ]

    @pytest.mark.parametrize("case_idx", range(len(CASES)))
    def test_int8_maxerr(self, sageattn, backend, case_idx):
        name, b, h_q, h_kv, sq, d, dtype = self.CASES[case_idx]
        # 对 triton backend 跳过超长 D=128 用例: iGPU 上 triton 长序列(D=128, seq>=6144)
        # 不稳定, 会触发 python.dll 硬崩溃/OOM (非可捕获异常, 必须提前 skip)。
        # native 后端不受影响 (这些正是本次要回归的用例)。
        if backend == "triton" and d == 128 and sq >= 6144:
            pytest.skip(f"triton D=128 long-seq (seq={sq}) is unstable on iGPU (crash/OOM)")
        q = torch.randn(b, sq, h_q, d, device="cuda", dtype=dtype)
        k = torch.randn(b, sq, h_kv, d, device="cuda", dtype=dtype)
        v = torch.randn(b, sq, h_kv, d, device="cuda", dtype=dtype)

        ref = reference_fp32_nhd(q, k, v, is_causal=False)
        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        assert not has_nan, f"{name}: output contains NaN"
        assert not has_inf, f"{name}: output contains Inf"

        cos = cosine_similarity(out, ref)
        mae = max_abs_error(out, ref)
        # 判定条件与 benchmark_attn.py 一致: max_err < 0.05 才通过。
        # 修复前 D128 int8 长序列 mae 0.2-0.47; 修复后 (V3/BN=16) 与 triton 同精度 mae<0.01。
        assert cos > 0.99, f"{name}: cosine similarity {cos} too low"
        assert mae < 0.05, f"{name}: max abs error {mae} too high (must be <0.05 like benchmark)"
        del q, k, v, out, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--backend", "native"])