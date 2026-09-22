"""Composable classic transformer reference implementations."""

from dataclasses import dataclass

import numpy as np

from tokserve.reference.attention import multi_head_attention
from tokserve.reference.layers import feed_forward, layer_norm
from tokserve.reference.ops import FloatArray, IntArray, embedding_lookup, linear


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


@dataclass(frozen=True)
class TinyTransformerConfig:
    """Shape configuration for the tiny reference transformer."""

    vocab_size: int
    hidden_size: int
    num_heads: int
    feed_forward_size: int
    num_layers: int
    max_sequence_length: int


@dataclass(frozen=True)
class TinyTransformerWeights:
    """Weights for the tiny reference transformer."""

    token_embeddings: FloatArray
    blocks: tuple[TransformerBlockWeights, ...]
    final_norm_weight: FloatArray
    final_norm_bias: FloatArray
    output_weight: FloatArray


def tiny_transformer(
    token_ids: IntArray,
    config: TinyTransformerConfig,
    weights: TinyTransformerWeights,
    *,
    epsilon: float = 1e-5,
) -> FloatArray:
    """Map token IDs shaped (B, T) to logits shaped (B, T, V)."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (B, T)")

    if config.vocab_size < 1:
        raise ValueError("vocab_size must be positive")

    if config.hidden_size < 1:
        raise ValueError("hidden_size must be positive")

    if config.num_layers < 1:
        raise ValueError("num_layers must be positive")

    if token_ids.shape[1] > config.max_sequence_length:
        raise ValueError("sequence length exceeds max_sequence_length")

    if weights.token_embeddings.shape != (
        config.vocab_size,
        config.hidden_size,
    ):
        raise ValueError("token_embeddings have an incompatible shape")

    if len(weights.blocks) != config.num_layers:
        raise ValueError("number of block weights must equal num_layers")

    if weights.output_weight.shape != (
        config.vocab_size,
        config.hidden_size,
    ):
        raise ValueError("output_weight has an incompatible shape")

    hidden = embedding_lookup(token_ids, weights.token_embeddings)

    for block_weights in weights.blocks:
        hidden = transformer_block(
            hidden,
            block_weights,
            num_heads=config.num_heads,
            causal=True,
            epsilon=epsilon,
        )

    hidden = layer_norm(
        hidden,
        weights.final_norm_weight,
        weights.final_norm_bias,
        epsilon=epsilon,
    )
    return linear(hidden, weights.output_weight)
