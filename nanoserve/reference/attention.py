"""NumPy reference attention operations."""

from math import sqrt

import numpy as np

from nanoserve.reference.ops import (
    FloatArray,
    apply_causal_mask,
    linear,
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


def split_heads(inputs: FloatArray, num_heads: int) -> FloatArray:
    """Split (B, T, D) into (B, H, T, Dh)."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (B, T, D)")

    if num_heads < 1:
        raise ValueError("num_heads must be positive")

    batch_size, sequence_length, hidden_size = inputs.shape

    if hidden_size % num_heads != 0:
        raise ValueError("hidden size must be divisible by num_heads")

    head_size = hidden_size // num_heads
    reshaped = inputs.reshape(batch_size, sequence_length, num_heads, head_size)
    return np.asarray(reshaped.transpose(0, 2, 1, 3), dtype=np.float64)


def merge_heads(inputs: FloatArray) -> FloatArray:
    """Merge (B, H, T, Dh) into (B, T, D)."""

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (B, H, T, Dh)")

    batch_size, num_heads, sequence_length, head_size = inputs.shape
    transposed = inputs.transpose(0, 2, 1, 3)
    return np.asarray(
        transposed.reshape(
            batch_size,
            sequence_length,
            num_heads * head_size,
        ),
        dtype=np.float64,
    )


def multi_head_attention(
    inputs: FloatArray,
    query_weight: FloatArray,
    key_weight: FloatArray,
    value_weight: FloatArray,
    output_weight: FloatArray,
    *,
    num_heads: int,
    causal: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Apply projected multi-head self-attention to (B, T, D) inputs."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (B, T, D)")

    hidden_size = inputs.shape[-1]
    expected_weight_shape = (hidden_size, hidden_size)

    for name, weight in (
        ("query_weight", query_weight),
        ("key_weight", key_weight),
        ("value_weight", value_weight),
        ("output_weight", output_weight),
    ):
        if weight.shape != expected_weight_shape:
            raise ValueError(f"{name} must have shape (D, D)")

    query = split_heads(linear(inputs, query_weight), num_heads)
    key = split_heads(linear(inputs, key_weight), num_heads)
    value = split_heads(linear(inputs, value_weight), num_heads)

    attended, attention_weights = scaled_dot_product_attention(
        query,
        key,
        value,
        causal=causal,
    )
    merged = merge_heads(attended)
    output = linear(merged, output_weight)

    return output, attention_weights
