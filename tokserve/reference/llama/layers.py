"""Llama-specific normalization and feed-forward operations."""

import numpy as np

from tokserve.reference.ops import FloatArray, linear


def rms_norm(
    inputs: FloatArray,
    weight: FloatArray,
    *,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Apply RMSNorm along the final input dimension.

    Shapes:
        inputs: (..., D)
        weight: (D,)
        output: (..., D)
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    if weight.shape != (inputs.shape[-1],):
        raise ValueError("weight must have shape (D,)")

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    mean_square = np.mean(inputs**2, axis=-1, keepdims=True)
    normalized = inputs / np.sqrt(mean_square + epsilon)
    return np.asarray(normalized * weight, dtype=np.float64)


def silu(inputs: FloatArray) -> FloatArray:
    """Apply the SiLU activation elementwise."""

    return np.asarray(
        inputs / (1.0 + np.exp(-inputs)),
        dtype=np.float64,
    )


def swiglu(
    inputs: FloatArray,
    gate_weight: FloatArray,
    up_weight: FloatArray,
    down_weight: FloatArray,
) -> FloatArray:
    """Apply a bias-free Llama SwiGLU feed-forward network.

    Shapes:
        inputs: (..., D)
        gate_weight: (I, D)
        up_weight: (I, D)
        down_weight: (D, I)
        output: (..., D)
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    if gate_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("gate_weight and up_weight must have shape (I, D)")

    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have matching shapes")

    intermediate_size, hidden_size = gate_weight.shape

    if inputs.shape[-1] != hidden_size:
        raise ValueError("input hidden size is incompatible with projection weights")

    if down_weight.shape != (hidden_size, intermediate_size):
        raise ValueError("down_weight must have shape (D, I)")

    gate = silu(linear(inputs, gate_weight))
    up = linear(inputs, up_weight)
    hidden = gate * up
    return linear(hidden, down_weight)
