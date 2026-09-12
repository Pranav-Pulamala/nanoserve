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
