import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tokserve.reference.llama.attention import (
    grouped_query_attention,
    repeat_key_value,
    reshape_projection,
)
from tokserve.reference.llama.config import LlamaConfig


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )


def test_reshape_projection_has_expected_shape() -> None:
    projection = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    result = reshape_projection(
        projection,
        num_heads=2,
        head_dim=2,
    )

    assert result.shape == (2, 2, 3, 2)
    assert_array_equal(result[0, 0, 0], np.array([0.0, 1.0]))
    assert_array_equal(result[0, 1, 0], np.array([2.0, 3.0]))


def test_repeat_key_value_makes_grouping_visible() -> None:
    key_value = np.array(
        [
            [
                [[[1.0]], [[2.0]]],
                [[[3.0]], [[4.0]]],
            ]
        ]
    ).reshape(1, 2, 2, 1)

    result = repeat_key_value(key_value, num_groups=2)

    assert result.shape == (1, 4, 2, 1)
    assert_array_equal(result[:, 0], key_value[:, 0])
    assert_array_equal(result[:, 1], key_value[:, 0])
    assert_array_equal(result[:, 2], key_value[:, 1])
    assert_array_equal(result[:, 3], key_value[:, 1])


def test_single_token_grouped_query_attention() -> None:
    config = make_config()
    inputs = np.array([[[1.0, 2.0, 3.0, 4.0]]])

    output, weights = grouped_query_attention(
        inputs,
        query_weight=np.eye(4),
        key_weight=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        value_weight=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        output_weight=np.eye(4),
        positions=np.array([0]),
        config=config,
    )

    expected = np.array([[[1.0, 2.0, 1.0, 2.0]]])

    assert output.shape == (1, 1, 4)
    assert weights.shape == (1, 2, 1, 1)
    assert_allclose(output, expected)
    assert_allclose(weights, np.ones((1, 2, 1, 1)))


def test_causal_output_cannot_depend_on_future_token() -> None:
    config = make_config()
    first_inputs = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    second_inputs = first_inputs.copy()
    second_inputs[0, 1] = np.array([100.0, 100.0, 100.0, 100.0])

    query_weight = np.eye(4)
    key_value_weight = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    first_output, _ = grouped_query_attention(
        first_inputs,
        query_weight,
        key_value_weight,
        key_value_weight,
        np.eye(4),
        np.array([0, 1]),
        config,
    )
    second_output, _ = grouped_query_attention(
        second_inputs,
        query_weight,
        key_value_weight,
        key_value_weight,
        np.eye(4),
        np.array([0, 1]),
        config,
    )

    assert_allclose(first_output[:, 0], second_output[:, 0])


def test_grouped_query_attention_rejects_invalid_weight_shape() -> None:
    config = make_config()

    with pytest.raises(ValueError, match="key_weight"):
        grouped_query_attention(
            np.ones((1, 2, 4)),
            query_weight=np.eye(4),
            key_weight=np.ones((4, 4)),
            value_weight=np.ones((2, 4)),
            output_weight=np.eye(4),
            positions=np.array([0, 1]),
            config=config,
        )


def test_repeat_key_value_rejects_nonpositive_group_count() -> None:
    with pytest.raises(ValueError, match="num_groups must be positive"):
        repeat_key_value(
            np.ones((1, 1, 2, 2)),
            num_groups=0,
        )
