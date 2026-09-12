"""Classic transformer support layers implemented with NumPy."""

from math import sqrt

import numpy as np

from nanoserve.reference.ops import FloatArray, linear


def layer_norm(
    inputs: FloatArray,
    weight: FloatArray,
    bias: FloatArray,
    *,
    epsilon: float = 1e-5,
) -> FloatArray:
    """Normalize the final dimension, then apply scale and bias."""

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    hidden_size = inputs.shape[-1]

    if weight.shape != (hidden_size,) or bias.shape != (hidden_size,):
        raise ValueError("weight and bias must have shape (D,)")

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    mean = np.mean(inputs, axis=-1, keepdims=True)
    variance = np.mean((inputs - mean) ** 2, axis=-1, keepdims=True)
    normalized = (inputs - mean) / np.sqrt(variance + epsilon)
    return np.asarray(normalized * weight + bias, dtype=np.float64)


def gelu(inputs: FloatArray) -> FloatArray:
    """Apply the tanh approximation of GELU elementwise."""

    coefficient = sqrt(2.0 / np.pi)
    activated = (
        0.5 * inputs * (1.0 + np.tanh(coefficient * (inputs + 0.044715 * inputs**3)))
    )
    return np.asarray(activated, dtype=np.float64)


def feed_forward(
    inputs: FloatArray,
    up_weight: FloatArray,
    down_weight: FloatArray,
    *,
    up_bias: FloatArray | None = None,
    down_bias: FloatArray | None = None,
) -> FloatArray:
    """Apply linear up-projection, GELU, and linear down-projection."""

    hidden = linear(inputs, up_weight, up_bias)
    activated = gelu(hidden)
    return linear(activated, down_weight, down_bias)
