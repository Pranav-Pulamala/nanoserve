import numpy as np
import pytest
from numpy.testing import assert_allclose

from nanoserve.reference.attention import scaled_dot_product_attention


def test_attention_has_expected_shapes_and_values() -> None:
    query = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = query.copy()
    value = np.array([[[[10.0, 0.0], [0.0, 20.0]]]])

    output, weights = scaled_dot_product_attention(query, key, value)

    scale = np.sqrt(2.0)
    first_weight = np.exp(1.0 / scale) / (np.exp(1.0 / scale) + 1.0)
    expected_weights = np.array(
        [[[[first_weight, 1.0 - first_weight], [1.0 - first_weight, first_weight]]]]
    )
    expected_output = expected_weights @ value

    assert output.shape == (1, 1, 2, 2)
    assert weights.shape == (1, 1, 2, 2)
    assert_allclose(weights, expected_weights)
    assert_allclose(output, expected_output)


def test_attention_weights_sum_to_one() -> None:
    values = np.ones((2, 3, 4, 5))

    _, weights = scaled_dot_product_attention(values, values, values)

    assert_allclose(weights.sum(axis=-1), np.ones((2, 3, 4)))


def test_causal_attention_cannot_use_future_values() -> None:
    query = np.ones((1, 1, 3, 1))
    key = np.ones((1, 1, 3, 1))
    value = np.array([[[[1.0], [10.0], [100.0]]]])

    output, weights = scaled_dot_product_attention(
        query,
        key,
        value,
        causal=True,
    )

    assert_allclose(output[0, 0, 0], np.array([1.0]))
    assert weights[0, 0, 0, 1] == 0.0
    assert weights[0, 0, 0, 2] == 0.0


def test_attention_rejects_mismatched_head_dimensions() -> None:
    with pytest.raises(ValueError, match="head dimensions"):
        scaled_dot_product_attention(
            np.ones((1, 1, 2, 3)),
            np.ones((1, 1, 2, 4)),
            np.ones((1, 1, 2, 3)),
        )
