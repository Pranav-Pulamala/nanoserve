"""CUDA correctness tests for the first Triton tiled-attention kernel."""

import importlib.util

import pytest
import torch
from torch.testing import assert_close

from tokserve.engine.attention import causal_attention
from tokserve.kernels.tiled_attention import triton_tiled_attention

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


@pytest.mark.parametrize(
    ("query_length", "key_length", "query_offset", "head_dim"),
    [
        (1, 1, 0, 8),
        (7, 7, 0, 8),
        (16, 16, 0, 16),
        (17, 17, 0, 16),
        (1, 35, 34, 8),
        (3, 19, 16, 8),
    ],
)
def test_triton_tiled_attention_matches_pytorch(
    query_length: int,
    key_length: int,
    query_offset: int,
    head_dim: int,
) -> None:
    torch.manual_seed(400)
    query = torch.randn(
        2,
        2,
        query_length,
        head_dim,
        device="cuda",
    )
    key = torch.randn(
        2,
        2,
        key_length,
        head_dim,
        device="cuda",
    )
    value = torch.randn(
        2,
        2,
        key_length,
        head_dim,
        device="cuda",
    )

    expected, _ = causal_attention(
        query,
        key,
        value,
        query_position_offset=query_offset,
    )
    actual = triton_tiled_attention(
        query,
        key,
        value,
        query_position_offset=query_offset,
    )

    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    assert actual.is_cuda
    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_kernel_handles_noncontiguous_input_views() -> None:
    torch.manual_seed(401)
    query_source = torch.randn(1, 2, 9, 16, device="cuda")
    key_source = torch.randn(1, 2, 9, 16, device="cuda")
    value_source = torch.randn(1, 2, 9, 16, device="cuda")

    query = query_source[..., ::2]
    key = key_source[..., ::2]
    value = value_source[..., ::2]

    assert not query.is_contiguous()

    expected, _ = causal_attention(query, key, value)
    actual = triton_tiled_attention(query, key, value)

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("num_query_heads", "num_key_value_heads"),
    [
        (2, 2),
        (4, 2),
        (8, 2),
        (8, 1),
    ],
)
def test_triton_tiled_attention_supports_gqa(
    num_query_heads: int,
    num_key_value_heads: int,
) -> None:
    torch.manual_seed(402)
    query = torch.randn(
        1,
        num_query_heads,
        19,
        8,
        device="cuda",
    )
    key = torch.randn(
        1,
        num_key_value_heads,
        19,
        8,
        device="cuda",
    )
    value = torch.randn(
        1,
        num_key_value_heads,
        19,
        8,
        device="cuda",
    )

    groups = num_query_heads // num_key_value_heads
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    expected, _ = causal_attention(
        query,
        expanded_key,
        expanded_value,
    )
    actual = triton_tiled_attention(query, key, value)

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_gqa_mapping_uses_head_distinct_values() -> None:
    query = torch.zeros(1, 4, 1, 8, device="cuda")
    key = torch.zeros(1, 2, 1, 8, device="cuda")
    value = torch.empty(1, 2, 1, 8, device="cuda")
    value[:, 0, :, :] = 10.0
    value[:, 1, :, :] = 20.0

    actual = triton_tiled_attention(query, key, value)

    assert_close(actual[:, 0], torch.full_like(actual[:, 0], 10.0))
    assert_close(actual[:, 1], torch.full_like(actual[:, 1], 10.0))
    assert_close(actual[:, 2], torch.full_like(actual[:, 2], 20.0))
    assert_close(actual[:, 3], torch.full_like(actual[:, 3], 20.0))


def test_invalid_gqa_ratio_is_rejected() -> None:
    query = torch.randn(1, 3, 7, 8, device="cuda")
    key = torch.randn(1, 2, 7, 8, device="cuda")
    value = torch.randn(1, 2, 7, 8, device="cuda")

    with pytest.raises(
        ValueError,
        match="query head count must be divisible",
    ):
        triton_tiled_attention(query, key, value)
