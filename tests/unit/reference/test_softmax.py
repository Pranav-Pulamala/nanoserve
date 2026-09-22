import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tokserve.reference.ops import apply_causal_mask, causal_mask, softmax


def test_softmax_probabilities_sum_to_one() -> None:
    values = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    result = softmax(values)

    assert result.shape == values.shape
    assert_allclose(result.sum(axis=-1), np.ones(2))


def test_softmax_is_stable_for_large_values() -> None:
    result = softmax(np.array([1000.0, 1001.0]))

    assert np.all(np.isfinite(result))
    assert_allclose(result, np.array([0.26894142, 0.73105858]), rtol=1e-7)


def test_softmax_supports_an_explicit_axis() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = softmax(values, axis=0)

    assert_allclose(result.sum(axis=0), np.ones(2))


def test_softmax_rejects_invalid_axis() -> None:
    with pytest.raises(ValueError, match="axis is out of range"):
        softmax(np.ones((2, 2)), axis=2)


def test_causal_mask_has_expected_pattern() -> None:
    expected = np.array(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    assert_array_equal(causal_mask(3), expected)


def test_apply_causal_mask_hides_future_scores() -> None:
    result = apply_causal_mask(np.zeros((1, 1, 3, 3)))

    assert result.shape == (1, 1, 3, 3)
    assert np.isneginf(result[0, 0, 0, 1])
    assert result[0, 0, 2, 2] == 0.0
