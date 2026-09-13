"""Llama decoder components implemented with NumPy."""

from dataclasses import dataclass

import numpy as np

from nanoserve.reference.llama.attention import grouped_query_attention
from nanoserve.reference.llama.config import LlamaConfig
from nanoserve.reference.llama.layers import rms_norm, swiglu
from nanoserve.reference.ops import FloatArray, IntArray


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
