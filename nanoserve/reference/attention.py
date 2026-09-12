"""NumPy reference attention operations."""

from math import sqrt

import numpy as np

from nanoserve.reference.ops import (
    FloatArray,
    apply_causal_mask,
    softmax,
)


def scaled_dot_product_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    causal: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Compute scaled dot-product attention.

    Shapes:
        query: (..., Tq, Dh)
        key: (..., Tk, Dh)
        value: (..., Tk, Dv)
        output: (..., Tq, Dv)
        weights: (..., Tq, Tk)
    """

    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must have at least two dimensions")

    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key, and value batch dimensions must match")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    if query.shape[-1] < 1:
        raise ValueError("head dimension must be positive")

    scores = query @ np.swapaxes(key, -1, -2)
    scores = scores / sqrt(query.shape[-1])

    if causal:
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("causal attention requires equal sequence lengths")
        scores = apply_causal_mask(scores)

    weights = softmax(np.asarray(scores, dtype=np.float64), axis=-1)
    output = weights @ value

    return (
        np.asarray(output, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
