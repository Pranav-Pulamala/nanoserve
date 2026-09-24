"""Correctness tests for the educational tiled-attention reference."""

import pytest
import torch
from torch.testing import assert_close

from tokserve.engine.attention import (
    causal_attention,
    repeat_key_value,
)
from tokserve.engine.tiled_attention import tiled_attention_reference


def ordinary_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int = 0,
) -> torch.Tensor:
    """Run TokServe's ordinary PyTorch GQA reference."""

    num_groups = query.shape[1] // key.shape[1]
    repeated_key = repeat_key_value(key, num_groups=num_groups)
    repeated_value = repeat_key_value(value, num_groups=num_groups)
    output, _ = causal_attention(
        query,
        repeated_key,
        repeated_value,
        query_position_offset=query_position_offset,
    )
    return output


@pytest.mark.parametrize(
    (
        "batch_size",
        "num_query_heads",
        "num_key_value_heads",
        "query_length",
        "key_length",
        "head_dim",
        "query_offset",
        "query_block_size",
        "key_block_size",
    ),
    [
        (1, 2, 2, 8, 8, 4, 0, 4, 4),
        (1, 4, 2, 7, 7, 4, 0, 3, 3),
        (2, 8, 2, 9, 9, 8, 0, 4, 5),
        (1, 4, 2, 1, 11, 8, 10, 4, 4),
        (1, 4, 2, 3, 11, 4, 8, 2, 5),
    ],
)
def test_tiled_attention_matches_ordinary_attention(
    batch_size: int,
    num_query_heads: int,
    num_key_value_heads: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    query_offset: int,
    query_block_size: int,
    key_block_size: int,
) -> None:
    torch.manual_seed(200)
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

    expected = ordinary_attention(
        query,
        key,
        value,
        query_position_offset=query_offset,
    )
    actual = tiled_attention_reference(
        query,
        key,
        value,
        query_position_offset=query_offset,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )

    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_tiled_attention_is_stable_for_extreme_scores() -> None:
    query = torch.tensor([[[[1000.0, 0.0], [-1000.0, 0.0], [500.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]]]])
    value = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]])

    expected = ordinary_attention(query, key, value)
    actual = tiled_attention_reference(
        query,
        key,
        value,
        query_block_size=2,
        key_block_size=1,
    )

    assert torch.isfinite(actual).all()
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_tiled_attention_respects_causal_tile_boundaries() -> None:
    torch.manual_seed(201)
    query = torch.randn(1, 2, 9, 4)
    key = torch.randn(1, 2, 9, 4)
    value = torch.randn(1, 2, 9, 4)

    original = tiled_attention_reference(
        query,
        key,
        value,
        query_block_size=4,
        key_block_size=4,
    )

    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, :, 8, :] = 1000.0
    changed_value[:, :, 8, :] = -1000.0

    changed = tiled_attention_reference(
        query,
        changed_key,
        changed_value,
        query_block_size=4,
        key_block_size=4,
    )

    assert_close(
        changed[:, :, :8, :],
        original[:, :, :8, :],
        rtol=0,
        atol=0,
    )


def test_tiled_attention_uses_correct_gqa_head_mapping() -> None:
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

    actual = tiled_attention_reference(
        query,
        key,
        value,
        query_block_size=1,
        key_block_size=1,
    )

    assert_close(actual[0, 0], value[0, 0])
    assert_close(actual[0, 1], value[0, 0])
    assert_close(actual[0, 2], value[0, 1])
    assert_close(actual[0, 3], value[0, 1])


@pytest.mark.parametrize(
    ("query_block_size", "key_block_size"),
    [
        (1, 1),
        (2, 3),
        (4, 4),
        (16, 16),
    ],
)
def test_result_is_independent_of_tile_sizes(
    query_block_size: int,
    key_block_size: int,
) -> None:
    torch.manual_seed(202)
    query = torch.randn(1, 4, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 8)

    expected = ordinary_attention(query, key, value)
    actual = tiled_attention_reference(
        query,
        key,
        value,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_tiled_attention_rejects_invalid_gqa_ratio() -> None:
    query = torch.ones(1, 3, 2, 4)
    key = torch.ones(1, 2, 2, 4)
    value = torch.ones(1, 2, 2, 4)

    with pytest.raises(
        ValueError,
        match="query head count must be divisible",
    ):
        tiled_attention_reference(query, key, value)
