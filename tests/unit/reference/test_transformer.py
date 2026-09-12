import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nanoserve.reference.transformer import (
    TransformerBlockWeights,
    transformer_block,
)


def make_weights(hidden_size: int, feed_forward_size: int) -> TransformerBlockWeights:
    return TransformerBlockWeights(
        norm1_weight=np.ones(hidden_size),
        norm1_bias=np.zeros(hidden_size),
        query_weight=np.eye(hidden_size),
        key_weight=np.eye(hidden_size),
        value_weight=np.zeros((hidden_size, hidden_size)),
        attention_output_weight=np.eye(hidden_size),
        norm2_weight=np.ones(hidden_size),
        norm2_bias=np.zeros(hidden_size),
        feed_forward_up_weight=np.zeros((feed_forward_size, hidden_size)),
        feed_forward_down_weight=np.zeros((hidden_size, feed_forward_size)),
    )


def test_zero_sublayers_preserve_residual_input() -> None:
    inputs = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    original = inputs.copy()

    result = transformer_block(
        inputs,
        make_weights(hidden_size=2, feed_forward_size=4),
        num_heads=1,
    )

    assert_allclose(result, inputs)
    assert_array_equal(inputs, original)


def test_transformer_block_output_shape() -> None:
    inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    result = transformer_block(
        inputs,
        make_weights(hidden_size=4, feed_forward_size=8),
        num_heads=2,
    )

    assert result.shape == inputs.shape


def test_transformer_block_is_deterministic() -> None:
    inputs = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    weights = make_weights(hidden_size=2, feed_forward_size=4)

    first = transformer_block(inputs, weights, num_heads=2)
    second = transformer_block(inputs, weights, num_heads=2)

    assert_allclose(first, second)


def test_transformer_block_rejects_nonbatched_input() -> None:
    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        transformer_block(
            np.ones((2, 3)),
            make_weights(hidden_size=3, feed_forward_size=6),
            num_heads=1,
        )
