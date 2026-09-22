"""Llama decoder components implemented with NumPy."""

from dataclasses import dataclass

import numpy as np

from tokserve.reference.llama.attention import grouped_query_attention
from tokserve.reference.llama.config import LlamaConfig
from tokserve.reference.llama.layers import rms_norm, swiglu
from tokserve.reference.ops import (
    FloatArray,
    IntArray,
    embedding_lookup,
    linear,
)


@dataclass(frozen=True)
class LlamaBlockWeights:
    """Weights for one Llama decoder block."""

    input_norm_weight: FloatArray
    query_weight: FloatArray
    key_weight: FloatArray
    value_weight: FloatArray
    attention_output_weight: FloatArray
    post_attention_norm_weight: FloatArray
    gate_weight: FloatArray
    up_weight: FloatArray
    down_weight: FloatArray


def llama_block(
    inputs: FloatArray,
    positions: IntArray,
    weights: LlamaBlockWeights,
    config: LlamaConfig,
) -> FloatArray:
    """Apply one pre-norm Llama decoder block.

    Shapes:
        inputs: (B, T, D)
        positions: (T,)
        output: (B, T, D)
    """

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (B, T, D)")

    if inputs.shape[-1] != config.hidden_size:
        raise ValueError("inputs must use config.hidden_size")

    if positions.shape != (inputs.shape[1],):
        raise ValueError("positions must have shape (T,)")

    normalized_attention_input = rms_norm(
        inputs,
        weights.input_norm_weight,
        epsilon=config.rms_norm_eps,
    )
    attention_output, _ = grouped_query_attention(
        normalized_attention_input,
        weights.query_weight,
        weights.key_weight,
        weights.value_weight,
        weights.attention_output_weight,
        positions,
        config,
    )
    attention_residual = inputs + attention_output

    normalized_mlp_input = rms_norm(
        attention_residual,
        weights.post_attention_norm_weight,
        epsilon=config.rms_norm_eps,
    )
    mlp_output = swiglu(
        normalized_mlp_input,
        weights.gate_weight,
        weights.up_weight,
        weights.down_weight,
    )

    return np.asarray(
        attention_residual + mlp_output,
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LlamaModelWeights:
    """Weights for the complete tiny NumPy Llama model."""

    token_embeddings: FloatArray
    blocks: tuple[LlamaBlockWeights, ...]
    final_norm_weight: FloatArray
    lm_head_weight: FloatArray


def llama_forward(
    token_ids: IntArray,
    weights: LlamaModelWeights,
    config: LlamaConfig,
) -> FloatArray:
    """Map token IDs shaped (B, T) to logits shaped (B, T, V)."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (B, T)")

    sequence_length = token_ids.shape[1]

    if sequence_length < 1:
        raise ValueError("sequence length must be positive")

    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence length exceeds max_position_embeddings")

    if weights.token_embeddings.shape != (
        config.vocab_size,
        config.hidden_size,
    ):
        raise ValueError("token_embeddings have an incompatible shape")

    if len(weights.blocks) != config.num_hidden_layers:
        raise ValueError("number of block weights must equal num_hidden_layers")

    if weights.final_norm_weight.shape != (config.hidden_size,):
        raise ValueError("final_norm_weight must have shape (D,)")

    if weights.lm_head_weight.shape != (
        config.vocab_size,
        config.hidden_size,
    ):
        raise ValueError("lm_head_weight has an incompatible shape")

    hidden = embedding_lookup(token_ids, weights.token_embeddings)
    positions = np.arange(sequence_length, dtype=np.int64)

    for block_weights in weights.blocks:
        hidden = llama_block(
            hidden,
            positions,
            block_weights,
            config,
        )

    hidden = rms_norm(
        hidden,
        weights.final_norm_weight,
        epsilon=config.rms_norm_eps,
    )
    return linear(hidden, weights.lm_head_weight)
