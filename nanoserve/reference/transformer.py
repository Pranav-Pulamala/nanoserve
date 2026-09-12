"""Composable classic transformer reference implementations."""

from dataclasses import dataclass

import numpy as np

from nanoserve.reference.attention import multi_head_attention
from nanoserve.reference.layers import feed_forward, layer_norm
from nanoserve.reference.ops import FloatArray


@dataclass(frozen=True)
class TransformerBlockWeights:
    """Weights for one classic pre-norm transformer block."""

    norm1_weight: FloatArray
    norm1_bias: FloatArray
    query_weight: FloatArray
    key_weight: FloatArray
    value_weight: FloatArray
    attention_output_weight: FloatArray
    norm2_weight: FloatArray
    norm2_bias: FloatArray
    feed_forward_up_weight: FloatArray
    feed_forward_down_weight: FloatArray


def transformer_block(
    inputs: FloatArray,
    weights: TransformerBlockWeights,
    *,
    num_heads: int,
    causal: bool = True,
    epsilon: float = 1e-5,
) -> FloatArray:
    """Apply a classic pre-norm transformer block to (B, T, D)."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (B, T, D)")

    normalized_attention_input = layer_norm(
        inputs,
        weights.norm1_weight,
        weights.norm1_bias,
        epsilon=epsilon,
    )
    attention_output, _ = multi_head_attention(
        normalized_attention_input,
        weights.query_weight,
        weights.key_weight,
        weights.value_weight,
        weights.attention_output_weight,
        num_heads=num_heads,
        causal=causal,
    )
    attention_residual = inputs + attention_output

    normalized_feed_forward_input = layer_norm(
        attention_residual,
        weights.norm2_weight,
        weights.norm2_bias,
        epsilon=epsilon,
    )
    feed_forward_output = feed_forward(
        normalized_feed_forward_input,
        weights.feed_forward_up_weight,
        weights.feed_forward_down_weight,
    )

    return np.asarray(
        attention_residual + feed_forward_output,
        dtype=np.float64,
    )
