import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tokserve.reference.attention import (
    merge_heads,
    multi_head_attention,
    split_heads,
)


def test_split_and_merge_heads_round_trip() -> None:
    inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    split = split_heads(inputs, num_heads=2)
    merged = merge_heads(split)

    assert split.shape == (2, 2, 3, 2)
    assert merged.shape == inputs.shape
    assert_array_equal(merged, inputs)


def test_split_heads_rejects_invalid_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        split_heads(np.ones((1, 2, 5)), num_heads=2)


def test_single_token_identity_multi_head_attention() -> None:
    inputs = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    identity = np.eye(4)

    output, weights = multi_head_attention(
        inputs,
        identity,
        identity,
        identity,
        identity,
        num_heads=2,
        causal=True,
    )

    assert output.shape == (1, 1, 4)
    assert weights.shape == (1, 2, 1, 1)
    assert_allclose(output, inputs)
    assert_allclose(weights, np.ones((1, 2, 1, 1)))


def test_multi_head_attention_is_deterministic() -> None:
    inputs = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    identity = np.eye(2)

    first, _ = multi_head_attention(
        inputs,
        identity,
        identity,
        identity,
        identity,
        num_heads=2,
    )
    second, _ = multi_head_attention(
        inputs,
        identity,
        identity,
        identity,
        identity,
        num_heads=2,
    )

    assert_allclose(first, second)
