import numpy as np
import pytest
from numpy.testing import assert_allclose

from tokserve.reference.layers import feed_forward, gelu, layer_norm


def test_layer_norm_has_zero_mean_and_unit_variance() -> None:
    inputs = np.array([[1.0, 2.0, 3.0]])
    result = layer_norm(
        inputs,
        np.ones(3),
        np.zeros(3),
        epsilon=1e-12,
    )

    assert_allclose(result.mean(axis=-1), np.zeros(1), atol=1e-12)
    assert_allclose(result.var(axis=-1), np.ones(1), atol=1e-11)


def test_layer_norm_applies_weight_and_bias() -> None:
    inputs = np.array([[1.0, 3.0]])
    weight = np.array([2.0, 3.0])
    bias = np.array([0.5, -0.5])

    result = layer_norm(inputs, weight, bias, epsilon=1e-12)

    assert_allclose(result, np.array([[-1.5, 2.5]]), atol=1e-11)


def test_layer_norm_rejects_invalid_parameter_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        layer_norm(np.ones((2, 3)), np.ones(2), np.zeros(3))


def test_gelu_zero_and_symmetry_relation() -> None:
    values = np.array([-2.0, 0.0, 2.0])
    result = gelu(values)

    assert result[1] == 0.0
    assert_allclose(result[2] - result[0], 2.0)


def test_feed_forward_shape_and_values() -> None:
    inputs = np.array([[[1.0, 0.0]]])
    up_weight = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    down_weight = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    result = feed_forward(inputs, up_weight, down_weight)

    expected_hidden = gelu(np.array([[[1.0, 0.0, 1.0]]]))
    expected = expected_hidden @ down_weight.T

    assert result.shape == (1, 1, 2)
    assert_allclose(result, expected)
