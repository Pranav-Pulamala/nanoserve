"""Tests that pin the PyTorch attention contract used by tiled kernels."""

from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from tokserve.engine.attention import (
    causal_attention,
    repeat_key_value,
)


def independent_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GQA attention independently from TokServe's implementation."""

    batch_size, num_query_heads, query_length, head_dim = query.shape
    key_batch_size, num_key_value_heads, key_length, key_head_dim = key.shape

    assert batch_size == key_batch_size
    assert head_dim == key_head_dim
    assert value.shape == key.shape
    assert num_query_heads % num_key_value_heads == 0

    num_groups = num_query_heads // num_key_value_heads
    kv_head_indices = torch.arange(num_query_heads, device=query.device) // num_groups
    expanded_key = key.index_select(1, kv_head_indices)
    expanded_value = value.index_select(1, kv_head_indices)

    scores = torch.matmul(
        query,
        expanded_key.transpose(-1, -2),
    ) / sqrt(head_dim)

    query_positions = torch.arange(
        query_position_offset,
        query_position_offset + query_length,
        device=query.device,
    )
    key_positions = torch.arange(
        key_length,
        device=query.device,
    )
    allowed = key_positions[None, :] <= query_positions[:, None]
    scores = scores.masked_fill(
        ~allowed[None, None, :, :],
        float("-inf"),
    )

    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, expanded_value)
    return output, weights


@pytest.mark.parametrize(
    ("num_query_heads", "num_key_value_heads"),
    [
        (2, 2),
        (4, 2),
        (8, 2),
    ],
)
def test_gqa_contract_matches_independent_reference(
    num_query_heads: int,
    num_key_value_heads: int,
) -> None:
    """Pin ordinary MHA and multiple valid GQA grouping ratios."""

    torch.manual_seed(100)
    batch_size = 2
    query_length = 5
    key_length = 5
    head_dim = 4

    query = torch.randn(
        batch_size,
        num_query_heads,
        query_length,
        head_dim,
    )
    key = torch.randn(
        batch_size,
        num_key_value_heads,
        key_length,
        head_dim,
    )
    value = torch.randn(
        batch_size,
        num_key_value_heads,
        key_length,
        head_dim,
    )

    groups = num_query_heads // num_key_value_heads
    repeated_key = repeat_key_value(key, num_groups=groups)
    repeated_value = repeat_key_value(value, num_groups=groups)

    actual_output, actual_weights = causal_attention(
        query,
        repeated_key,
        repeated_value,
    )
    expected_output, expected_weights = independent_attention(
        query,
        key,
        value,
        query_position_offset=0,
    )

    assert actual_output.shape == (
        batch_size,
        num_query_heads,
        query_length,
        head_dim,
    )
    assert actual_weights.shape == (
        batch_size,
        num_query_heads,
        query_length,
        key_length,
    )
    assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)


def test_head_distinct_values_pin_gqa_mapping() -> None:
    """Ensure consecutive query-head groups share the intended KV head."""

    query = torch.zeros(1, 4, 1, 2)
    key = torch.zeros(1, 2, 1, 2)
    value = torch.tensor(
        [
            [
                [[10.0, 11.0]],
                [[20.0, 21.0]],
            ]
        ]
    )

    repeated_key = repeat_key_value(key, num_groups=2)
    repeated_value = repeat_key_value(value, num_groups=2)
    output, _ = causal_attention(
        query,
        repeated_key,
        repeated_value,
    )

    assert_close(output[0, 0], value[0, 0])
    assert_close(output[0, 1], value[0, 0])
    assert_close(output[0, 2], value[0, 1])
    assert_close(output[0, 3], value[0, 1])


def test_single_decode_query_uses_absolute_position() -> None:
    """A decode query at absolute position four can use five K/V tokens."""

    torch.manual_seed(101)
    query = torch.randn(1, 4, 1, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)

    repeated_key = repeat_key_value(key, num_groups=2)
    repeated_value = repeat_key_value(value, num_groups=2)

    actual_output, actual_weights = causal_attention(
        query,
        repeated_key,
        repeated_value,
        query_position_offset=4,
    )
    expected_output, expected_weights = independent_attention(
        query,
        key,
        value,
        query_position_offset=4,
    )

    assert actual_weights.shape == (1, 4, 1, 5)
    assert torch.all(actual_weights > 0)
    assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)


def test_multi_token_continuation_uses_absolute_causal_positions() -> None:
    """Continuation queries use history length as their absolute offset."""

    torch.manual_seed(102)
    query = torch.randn(1, 4, 3, 4)
    key = torch.randn(1, 2, 7, 4)
    value = torch.randn(1, 2, 7, 4)

    repeated_key = repeat_key_value(key, num_groups=2)
    repeated_value = repeat_key_value(value, num_groups=2)

    actual_output, actual_weights = causal_attention(
        query,
        repeated_key,
        repeated_value,
        query_position_offset=4,
    )
    expected_output, expected_weights = independent_attention(
        query,
        key,
        value,
        query_position_offset=4,
    )

    assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)

    assert torch.all(actual_weights[:, :, 0, 5:] == 0)
    assert torch.all(actual_weights[:, :, 1, 6:] == 0)
    assert torch.all(actual_weights[:, :, 2, :7] > 0)


def test_awkward_prefill_length_matches_reference() -> None:
    """Pin causal behavior for a non-power-of-two sequence length."""

    torch.manual_seed(103)
    query = torch.randn(1, 4, 7, 4)
    key = torch.randn(1, 2, 7, 4)
    value = torch.randn(1, 2, 7, 4)

    repeated_key = repeat_key_value(key, num_groups=2)
    repeated_value = repeat_key_value(value, num_groups=2)

    actual_output, actual_weights = causal_attention(
        query,
        repeated_key,
        repeated_value,
    )
    expected_output, expected_weights = independent_attention(
        query,
        key,
        value,
        query_position_offset=0,
    )

    assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)

    forbidden = torch.triu(
        torch.ones(7, 7, dtype=torch.bool),
        diagonal=1,
    )
    assert torch.all(actual_weights[:, :, forbidden] == 0)


def test_future_key_value_changes_cannot_affect_earlier_queries() -> None:
    """Pin causal isolation independently of projection and RoPE code."""

    torch.manual_seed(104)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)

    original_output, _ = causal_attention(query, key, value)

    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, :, 4, :] = 10_000.0
    changed_value[:, :, 4, :] = -10_000.0

    changed_output, _ = causal_attention(
        query,
        changed_key,
        changed_value,
    )

    assert_close(
        changed_output[:, :, :4, :],
        original_output[:, :, :4, :],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_attention_preserves_supported_cpu_dtype(
    dtype: torch.dtype,
) -> None:
    """Pin the reference path's output and weight dtype."""

    query = torch.ones(1, 2, 3, 4, dtype=dtype)
    key = torch.ones(1, 2, 3, 4, dtype=dtype)
    value = torch.ones(1, 2, 3, 4, dtype=dtype)

    output, weights = causal_attention(query, key, value)

    assert output.dtype == dtype
    assert weights.dtype == dtype
