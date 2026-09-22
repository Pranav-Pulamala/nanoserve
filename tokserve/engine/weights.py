"""Weight transfer from NumPy reference containers to PyTorch modules."""

import numpy as np
import torch

from tokserve.engine.model import LlamaDecoderBlock, LlamaModel
from tokserve.reference.llama.model import (
    LlamaBlockWeights,
    LlamaModelWeights,
)
from tokserve.reference.ops import FloatArray


def copy_parameter(
    parameter: torch.Tensor,
    values: FloatArray,
    name: str,
) -> None:
    """Copy a NumPy array into a parameter without sharing storage."""

    if tuple(parameter.shape) != values.shape:
        raise ValueError(
            f"{name} has shape {values.shape}, expected {tuple(parameter.shape)}"
        )

    tensor = torch.tensor(
        np.array(values, copy=True),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    parameter.copy_(tensor)


def load_block_weights(
    block: LlamaDecoderBlock,
    weights: LlamaBlockWeights,
) -> None:
    """Load one NumPy decoder-block weight container."""

    with torch.no_grad():
        copy_parameter(
            block.input_norm.weight,
            weights.input_norm_weight,
            name="input_norm_weight",
        )
        copy_parameter(
            block.self_attention.q_proj.weight,
            weights.query_weight,
            name="query_weight",
        )
        copy_parameter(
            block.self_attention.k_proj.weight,
            weights.key_weight,
            name="key_weight",
        )
        copy_parameter(
            block.self_attention.v_proj.weight,
            weights.value_weight,
            name="value_weight",
        )
        copy_parameter(
            block.self_attention.o_proj.weight,
            weights.attention_output_weight,
            name="attention_output_weight",
        )
        copy_parameter(
            block.post_attention_norm.weight,
            weights.post_attention_norm_weight,
            name="post_attention_norm_weight",
        )
        copy_parameter(
            block.mlp.gate_proj.weight,
            weights.gate_weight,
            name="gate_weight",
        )
        copy_parameter(
            block.mlp.up_proj.weight,
            weights.up_weight,
            name="up_weight",
        )
        copy_parameter(
            block.mlp.down_proj.weight,
            weights.down_weight,
            name="down_weight",
        )


def load_model_weights(
    model: LlamaModel,
    weights: LlamaModelWeights,
) -> None:
    """Load complete NumPy Llama weights into a PyTorch model."""

    if len(model.layers) != len(weights.blocks):
        raise ValueError("model and weight layer counts must match")

    with torch.no_grad():
        copy_parameter(
            model.embed_tokens.weight,
            weights.token_embeddings,
            name="token_embeddings",
        )

        for layer_index, (block, block_weights) in enumerate(
            zip(model.layers, weights.blocks, strict=True)
        ):
            if not isinstance(block, LlamaDecoderBlock):
                raise TypeError(f"layer {layer_index} is not a LlamaDecoderBlock")
            load_block_weights(block, block_weights)

        copy_parameter(
            model.final_norm.weight,
            weights.final_norm_weight,
            name="final_norm_weight",
        )
        copy_parameter(
            model.lm_head.weight,
            weights.lm_head_weight,
            name="lm_head_weight",
        )
