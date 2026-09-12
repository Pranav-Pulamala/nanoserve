"""Fundamental NumPy tensor operations."""

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]


def embedding_lookup(
    token_ids: IntArray,
    embedding_matrix: FloatArray,
) -> FloatArray:
    """Select embeddings.

    Shapes:
        token_ids: any integer shape
        embedding_matrix: (V, D)
        output: token_ids.shape + (D,)
    """

    if not np.issubdtype(token_ids.dtype, np.integer):
        raise TypeError("token_ids must contain integers")

    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must have shape (V, D)")

    if np.any(token_ids < 0) or np.any(token_ids >= embedding_matrix.shape[0]):
        raise ValueError("token_ids contain an out-of-range value")

    return np.asarray(embedding_matrix[token_ids], dtype=np.float64)


def linear(
    inputs: FloatArray,
    weight: FloatArray,
    bias: FloatArray | None = None,
) -> FloatArray:
    """Apply Y = X W-transpose + b along the final input dimension.

    Shapes:
        inputs: (..., Din)
        weight: (Dout, Din)
        bias: (Dout,), optional
        output: (..., Dout)
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    if weight.ndim != 2:
        raise ValueError("weight must have shape (Dout, Din)")

    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("input and weight dimensions are incompatible")

    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("bias must have shape (Dout,)")

    output = inputs @ weight.T

    if bias is not None:
        output = output + bias

    return np.asarray(output, dtype=np.float64)


def softmax(inputs: FloatArray, axis: int = -1) -> FloatArray:
    """Compute numerically stable softmax along an axis."""

    if inputs.ndim == 0:
        raise ValueError("inputs must have at least one dimension")

    if axis < -inputs.ndim or axis >= inputs.ndim:
        raise ValueError("axis is out of range for inputs")

    shifted = inputs - np.max(inputs, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=axis, keepdims=True)
    return np.asarray(exponentials / denominator, dtype=np.float64)


def causal_mask(sequence_length: int) -> NDArray[np.bool_]:
    """Return a lower-triangular attention mask with shape (T, T)."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    return np.tril(np.ones((sequence_length, sequence_length), dtype=np.bool_))


def apply_causal_mask(scores: FloatArray) -> FloatArray:
    """Replace future-position scores with negative infinity.

    Shape:
        scores: (..., T, T)
    """

    if scores.ndim < 2 or scores.shape[-2] != scores.shape[-1]:
        raise ValueError("scores must end with square (T, T) dimensions")

    mask = causal_mask(scores.shape[-1])
    return np.asarray(np.where(mask, scores, -np.inf), dtype=np.float64)
