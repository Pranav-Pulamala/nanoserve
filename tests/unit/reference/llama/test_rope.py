import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tokserve.reference.llama.rope import (
    apply_rope,
    inverse_frequencies,
    rotary_cos_sin,
    rotate_half,
)


def test_inverse_frequencies_have_expected_values() -> None:
    result = inverse_frequencies(4, theta=100.0)

    assert result.shape == (2,)
    assert_allclose(result, np.array([1.0, 0.1]))


def test_rotate_half_has_expected_values() -> None:
    inputs = np.array([1.0, 2.0, 3.0, 4.0])

    result = rotate_half(inputs)

    assert_array_equal(result, np.array([-3.0, -4.0, 1.0, 2.0]))


def test_position_zero_does_not_change_query_or_key() -> None:
    query = np.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = np.array([[[[4.0, 3.0, 2.0, 1.0]]]])

    rotated_query, rotated_key = apply_rope(
        query,
        key,
        np.array([0]),
    )

    assert_allclose(rotated_query, query)
    assert_allclose(rotated_key, key)


def test_rope_matches_known_position_one_rotation() -> None:
    query = np.array([[[[1.0, 0.0]]]])
    key = np.array([[[[0.0, 1.0]]]])

    rotated_query, rotated_key = apply_rope(
        query,
        key,
        np.array([1]),
    )

    expected_query = np.array([[[[np.cos(1.0), np.sin(1.0)]]]])
    expected_key = np.array([[[[-np.sin(1.0), np.cos(1.0)]]]])

    assert_allclose(rotated_query, expected_query)
    assert_allclose(rotated_key, expected_key)


def test_rope_preserves_vector_norms() -> None:
    query = np.arange(16, dtype=np.float64).reshape(1, 2, 2, 4)
    key = np.arange(8, dtype=np.float64).reshape(1, 1, 2, 4)
    positions = np.array([0, 1])

    rotated_query, rotated_key = apply_rope(
        query,
        key,
        positions,
    )

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert_allclose(
        np.linalg.norm(rotated_query, axis=-1),
        np.linalg.norm(query, axis=-1),
    )
    assert_allclose(
        np.linalg.norm(rotated_key, axis=-1),
        np.linalg.norm(key, axis=-1),
    )


def test_rotary_cos_sin_support_multiple_positions() -> None:
    cosine, sine = rotary_cos_sin(
        np.array([0, 1, 2]),
        head_dim=4,
        theta=100.0,
    )

    assert cosine.shape == (3, 4)
    assert sine.shape == (3, 4)
    assert_allclose(cosine[0], np.ones(4))
    assert_allclose(sine[0], np.zeros(4))


def test_rope_rejects_odd_head_dimension() -> None:
    with pytest.raises(ValueError, match="positive even integer"):
        inverse_frequencies(3)


def test_rope_rejects_incorrect_position_shape() -> None:
    query = np.ones((1, 2, 3, 4))
    key = np.ones((1, 1, 3, 4))

    with pytest.raises(ValueError, match=r"positions must have shape \(T,\)"):
        apply_rope(
            query,
            key,
            np.array([0, 1]),
        )
