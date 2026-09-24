"""CUDA prefill validation for Triton tiled attention."""

import importlib.util

import pytest
import torch
from torch.testing import assert_close

from tokserve.engine.attention import (
    causal_attention,
    repeat_key_value,
)
from tokserve.kernels.tiled_attention import (
    BLOCK_M,
    BLOCK_N,
    triton_tiled_attention,
)

CUDA_TRITON_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not CUDA_TRITON_AVAILABLE,
        reason="CUDA and Triton are required",
    ),
]


def pytorch_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run TokServe's ordinary causal PyTorch GQA."""

    groups = query.shape[1] // key.shape[1]
    repeated_key = repeat_key_value(key, num_groups=groups)
    repeated_value = repeat_key_value(value, num_groups=groups)
    output, _ = causal_attention(
        query,
        repeated_key,
        repeated_value,
    )
    return output


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return justified comparison tolerances for one dtype."""

    if dtype == torch.float32:
        return 1e-4, 1e-5

    if dtype == torch.float16:
        return 2e-2, 2e-2

    return 4e-2, 4e-2


@pytest.mark.parametrize(
    "sequence_length",
    [
        1,
        BLOCK_M - 1,
        BLOCK_M,
        BLOCK_M + 1,
        2 * BLOCK_N + 3,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
def test_prefill_matches_pytorch_across_tile_boundaries(
    sequence_length: int,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("native CUDA bfloat16 is unavailable")

    torch.manual_seed(500)
    query = torch.randn(
        1,
        4,
        sequence_length,
        16,
        device="cuda",
        dtype=dtype,
    )
    key = torch.randn(
        1,
        2,
        sequence_length,
        16,
        device="cuda",
        dtype=dtype,
    )
    value = torch.randn(
        1,
        2,
        sequence_length,
        16,
        device="cuda",
        dtype=dtype,
    )

    expected = pytorch_gqa(query, key, value)
    actual = triton_tiled_attention(query, key, value)
    rtol, atol = tolerance(dtype)

    assert actual.shape == query.shape
    assert actual.dtype == dtype
    assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("num_query_heads", "num_key_value_heads"),
    [
        (2, 2),
        (4, 2),
        (8, 2),
        (8, 1),
    ],
)
def test_prefill_supports_multiple_gqa_ratios(
    num_query_heads: int,
    num_key_value_heads: int,
) -> None:
    torch.manual_seed(501)
    sequence_length = BLOCK_M + 5
    query = torch.randn(
        1,
        num_query_heads,
        sequence_length,
        16,
        device="cuda",
    )
    key = torch.randn(
        1,
        num_key_value_heads,
        sequence_length,
        16,
        device="cuda",
    )
    value = torch.randn(
        1,
        num_key_value_heads,
        sequence_length,
        16,
        device="cuda",
    )

    expected = pytorch_gqa(query, key, value)
    actual = triton_tiled_attention(query, key, value)

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_future_key_value_cannot_change_earlier_prefill_outputs() -> None:
    """Catch causal-mask errors spanning multiple K/V tiles."""

    torch.manual_seed(502)
    sequence_length = 2 * BLOCK_N + 1
    query = torch.randn(
        1,
        4,
        sequence_length,
        16,
        device="cuda",
    )
    key = torch.randn(
        1,
        2,
        sequence_length,
        16,
        device="cuda",
    )
    value = torch.randn(
        1,
        2,
        sequence_length,
        16,
        device="cuda",
    )

    original = triton_tiled_attention(query, key, value)

    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, :, -1, :] = 10_000.0
    changed_value[:, :, -1, :] = -10_000.0

    changed = triton_tiled_attention(
        query,
        changed_key,
        changed_value,
    )

    assert_close(
        changed[:, :, :-1, :],
        original[:, :, :-1, :],
        rtol=0,
        atol=0,
    )


def test_partial_final_key_tile_does_not_read_invalid_values() -> None:
    """Ensure values beyond Tk cannot influence the output."""

    torch.manual_seed(503)
    sequence_length = BLOCK_N + 3
    query = torch.randn(
        1,
        4,
        sequence_length,
        16,
        device="cuda",
    )
    key = torch.randn(
        1,
        2,
        sequence_length,
        16,
        device="cuda",
    )
    value = torch.randn(
        1,
        2,
        sequence_length,
        16,
        device="cuda",
    )

    expected = pytorch_gqa(query, key, value)
    actual = triton_tiled_attention(query, key, value)

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)
