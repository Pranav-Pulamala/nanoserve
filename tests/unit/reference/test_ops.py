import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tokserve.reference.ops import embedding_lookup, linear


def test_embedding_lookup_with_one_dimensional_ids() -> None:
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    token_ids = np.array([2, 0])

    result = embedding_lookup(token_ids, embeddings)

    assert result.shape == (2, 2)
    assert_array_equal(result, np.array([[5.0, 6.0], [1.0, 2.0]]))


def test_embedding_lookup_with_batched_ids() -> None:
    embeddings = np.arange(12, dtype=np.float64).reshape(4, 3)
    token_ids = np.array([[0, 2], [3, 1]])

    result = embedding_lookup(token_ids, embeddings)

    assert result.shape == (2, 2, 3)
    assert_array_equal(result[1, 0], embeddings[3])


def test_embedding_lookup_rejects_invalid_ids() -> None:
    embeddings = np.ones((3, 2))

    with pytest.raises(ValueError, match="out-of-range"):
        embedding_lookup(np.array([3]), embeddings)


def test_linear_without_bias() -> None:
    inputs = np.array([[1.0, 2.0]])
    weight = np.array([[3.0, 4.0], [5.0, 6.0]])

    result = linear(inputs, weight)

    assert result.shape == (1, 2)
    assert_allclose(result, np.array([[11.0, 17.0]]))


def test_linear_with_bias() -> None:
    inputs = np.array([[1.0, 2.0]])
    weight = np.array([[3.0, 4.0], [5.0, 6.0]])
    bias = np.array([0.5, -0.5])

    result = linear(inputs, weight, bias)

    assert_allclose(result, np.array([[11.5, 16.5]]))


def test_linear_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        linear(np.ones((2, 3)), np.ones((4, 2)))
