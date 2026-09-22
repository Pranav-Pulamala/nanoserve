import numpy as np
import pytest
from numpy.testing import assert_allclose

from tokserve.reference.layers import layer_norm
from tokserve.reference.llama.layers import rms_norm, silu, swiglu


def test_rms_norm_matches_small_manual_example() -> None:
    inputs = np.array([[3.0, 4.0]])
    weight = np.array([1.0, 2.0])

    result = rms_norm(inputs, weight, epsilon=1e-12)

    root_mean_square = np.sqrt((9.0 + 16.0) / 2.0)
    expected = np.array(
        [
            [
                3.0 / root_mean_square,
                8.0 / root_mean_square,
            ]
        ]
    )

    assert result.shape == inputs.shape
    assert_allclose(result, expected, atol=1e-12)


def test_rms_norm_does_not_subtract_the_mean() -> None:
    inputs = np.array([[1.0, 2.0, 3.0]])
    weight = np.ones(3)

    rms_result = rms_norm(inputs, weight)
    layer_result = layer_norm(inputs, weight, np.zeros(3))

    assert not np.allclose(rms_result, layer_result)
    assert not np.allclose(rms_result.mean(axis=-1), np.zeros(1))


def test_rms_norm_rejects_invalid_weight_shape() -> None:
    with pytest.raises(ValueError, match=r"weight must have shape \(D,\)"):
        rms_norm(np.ones((2, 3)), np.ones(2))


def test_rms_norm_rejects_nonpositive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        rms_norm(np.ones((2, 3)), np.ones(3), epsilon=0.0)


def test_silu_matches_expected_values() -> None:
    inputs = np.array([-1.0, 0.0, 1.0])

    result = silu(inputs)

    expected = inputs / (1.0 + np.exp(-inputs))
    assert_allclose(result, expected)
    assert result[1] == 0.0


def test_swiglu_matches_deterministic_example() -> None:
    inputs = np.array([[[1.0, 2.0]]])
    gate_weight = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    up_weight = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, -1.0],
        ]
    )
    down_weight = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    result = swiglu(
        inputs,
        gate_weight,
        up_weight,
        down_weight,
    )

    gate_projection = inputs @ gate_weight.T
    up_projection = inputs @ up_weight.T
    expected = silu(gate_projection) * up_projection
    expected = expected @ down_weight.T

    assert result.shape == inputs.shape
    assert_allclose(result, expected)


def test_swiglu_rejects_mismatched_gate_and_up_weights() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        swiglu(
            np.ones((1, 2)),
            np.ones((3, 2)),
            np.ones((4, 2)),
            np.ones((2, 3)),
        )


def test_swiglu_rejects_invalid_down_weight_shape() -> None:
    with pytest.raises(ValueError, match=r"down_weight must have shape \(D, I\)"):
        swiglu(
            np.ones((1, 2)),
            np.ones((3, 2)),
            np.ones((3, 2)),
            np.ones((3, 2)),
        )
