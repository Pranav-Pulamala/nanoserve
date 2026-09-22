"""Compare Triton RoPE with the existing PyTorch RoPE."""

import importlib.util

import pytest
import torch

from tokserve.engine.rope import apply_rope
from tokserve.kernels.rope import triton_apply_rope

CUDA_TRITON_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)


def test_triton_rope_rejects_cpu_inputs() -> None:
    query = torch.ones(1, 2, 3, 8)
    key = torch.ones(1, 1, 3, 8)
    positions = torch.arange(3)

    with pytest.raises(ValueError, match="CUDA tensors"):
        triton_apply_rope(query, key, positions, theta=10_000.0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize(
    ("batch_size", "query_heads", "kv_heads", "length", "head_dim"),
    [(1, 4, 2, 3, 8), (2, 6, 2, 5, 16)],
)
@pytest.mark.parametrize("start_position", [0, 11])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_rope_matches_pytorch_at_absolute_positions(
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    length: int,
    head_dim: int,
    start_position: int,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        pytest.skip("native CUDA bfloat16 is unavailable")

    torch.manual_seed(7)
    query = torch.randn(
        batch_size, query_heads, length, head_dim, device="cuda", dtype=dtype
    )
    key = torch.randn(
        batch_size, kv_heads, length, head_dim, device="cuda", dtype=dtype
    )
    positions = torch.arange(
        start_position,
        start_position + length,
        device="cuda",
        dtype=torch.int64,
    )

    expected_query, expected_key = apply_rope(query, key, positions, theta=10_000.0)
    actual_query, actual_key = triton_apply_rope(query, key, positions, theta=10_000.0)

    tolerance = {
        torch.float32: 2e-5,
        torch.float16: 2e-3,
        torch.bfloat16: 1e-2,
    }[dtype]
    torch.testing.assert_close(
        actual_query, expected_query, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(actual_key, expected_key, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_triton_rope_accepts_noncontiguous_inputs() -> None:
    query = torch.randn(1, 3, 4, 8, device="cuda").transpose(1, 2)
    key = torch.randn(1, 3, 2, 8, device="cuda").transpose(1, 2)
    positions = torch.tensor([5, 6, 7], device="cuda")

    expected_query, expected_key = apply_rope(query, key, positions, theta=10_000.0)
    actual_query, actual_key = triton_apply_rope(query, key, positions, theta=10_000.0)

    torch.testing.assert_close(actual_query, expected_query, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_key, expected_key, rtol=2e-5, atol=2e-5)
