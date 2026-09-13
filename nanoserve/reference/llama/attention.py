"""Llama grouped-query attention implemented with NumPy."""

import numpy as np

from nanoserve.reference.attention import (
    merge_heads,
    scaled_dot_product_attention,
)
from nanoserve.reference.llama.config import LlamaConfig
from nanoserve.reference.llama.rope import apply_rope
from nanoserve.reference.ops import FloatArray, IntArray, linear


def reshape_projection(
    projection: FloatArray,
    *,
    num_heads: int,
    head_dim: int,
) -> FloatArray:
    """Reshape (B, T, H * Dh) into (B, H, T, Dh)."""

    if projection.ndim != 3:
        raise ValueError("projection must have shape (B, T, H * Dh)")

    batch_size, sequence_length, projected_size = projection.shape

    if projected_size != num_heads * head_dim:
        raise ValueError("projection size must equal num_heads * head_dim")

    reshaped = projection.reshape(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
    )
    return np.asarray(
        reshaped.transpose(0, 2, 1, 3),
        dtype=np.float64,
    )


def repeat_key_value(
    inputs: FloatArray,
    *,
    num_groups: int,
) -> FloatArray:
    """Repeat K/V heads from (B, Hkv, T, Dh) to (B, Hq, T, Dh)."""

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (B, Hkv, T, Dh)")

    if num_groups < 1:
        raise ValueError("num_groups must be positive")

    return np.asarray(
        np.repeat(inputs, num_groups, axis=1),
        dtype=np.float64,
    )


def grouped_query_attention(
    inputs: FloatArray,
    query_weight: FloatArray,
    key_weight: FloatArray,
    value_weight: FloatArray,
    output_weight: FloatArray,
    positions: IntArray,
    config: LlamaConfig,
) -> tuple[FloatArray, FloatArray]:
    """Apply causal Llama grouped-query self-attention.

    Shapes:
        inputs: (B, T, D)
        query_weight: (Hq * Dh, D), normally (D, D)
        key_weight: (Hkv * Dh, D)
        value_weight: (Hkv * Dh, D)
        output_weight: (D, D)
        positions: (T,)
        output: (B, T, D)
        attention weights: (B, Hq, T, T)
    """

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (B, T, D)")

    hidden_size = config.hidden_size
    key_value_size = config.num_key_value_heads * config.head_dim

    if inputs.shape[-1] != hidden_size:
        raise ValueError("inputs must use config.hidden_size")

    expected_shapes = {
        "query_weight": (hidden_size, hidden_size),
        "key_weight": (key_value_size, hidden_size),
        "value_weight": (key_value_size, hidden_size),
        "output_weight": (hidden_size, hidden_size),
    }
    supplied_weights = {
        "query_weight": query_weight,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "output_weight": output_weight,
    }

    for name, expected_shape in expected_shapes.items():
        if supplied_weights[name].shape != expected_shape:
            raise ValueError(f"{name} has an incompatible shape")

    if positions.shape != (inputs.shape[1],):
        raise ValueError("positions must have shape (T,)")

    query = reshape_projection(
        linear(inputs, query_weight),
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
    )
    key = reshape_projection(
        linear(inputs, key_weight),
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    value = reshape_projection(
        linear(inputs, value_weight),
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )

    query, key = apply_rope(
        query,
        key,
        positions,
        theta=config.rope_theta,
    )

    repeated_key = repeat_key_value(
        key,
        num_groups=config.num_key_value_groups,
    )
    repeated_value = repeat_key_value(
        value,
        num_groups=config.num_key_value_groups,
    )

    attended, attention_weights = scaled_dot_product_attention(
        query,
        repeated_key,
        repeated_value,
        causal=True,
    )
    merged = merge_heads(attended)
    output = linear(merged, output_weight)

    return output, attention_weights
