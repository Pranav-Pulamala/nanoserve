"""Bridge from a tiny Hugging Face Llama model to NumPy weights."""

from typing import Any

import numpy as np
import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM

from nanoserve.reference.llama.config import LlamaConfig
from nanoserve.reference.llama.model import (
    LlamaBlockWeights,
    LlamaModelWeights,
)
from nanoserve.reference.ops import FloatArray

REFERENCE_SEED = 17


def to_hugging_face_config(config: LlamaConfig) -> HFLlamaConfig:
    """Convert the minimal nanoserve configuration to Hugging Face."""

    return HFLlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def create_hugging_face_model(
    config: LlamaConfig,
    *,
    seed: int = REFERENCE_SEED,
) -> LlamaForCausalLM:
    """Create a deterministic tiny Llama model without downloading weights."""

    torch.manual_seed(seed)
    hugging_face_config = to_hugging_face_config(config)
    hugging_face_config._attn_implementation = "eager"

    model = LlamaForCausalLM(hugging_face_config)
    model.eval()
    return model


def tensor_to_numpy(tensor: Any) -> FloatArray:
    """Copy a Torch tensor into a NumPy float64 array."""

    return np.asarray(
        tensor.detach().cpu().numpy(),
        dtype=np.float64,
    ).copy()


def map_hugging_face_weights(
    model: LlamaForCausalLM,
    config: LlamaConfig,
) -> LlamaModelWeights:
    """Map Hugging Face Llama parameters into nanoserve containers."""

    if model.config.num_hidden_layers != config.num_hidden_layers:
        raise ValueError("model and config layer counts must match")

    if model.config.hidden_size != config.hidden_size:
        raise ValueError("model and config hidden sizes must match")

    block_weights: list[LlamaBlockWeights] = []

    for layer in model.model.layers:
        block_weights.append(
            LlamaBlockWeights(
                input_norm_weight=tensor_to_numpy(layer.input_layernorm.weight),
                query_weight=tensor_to_numpy(layer.self_attn.q_proj.weight),
                key_weight=tensor_to_numpy(layer.self_attn.k_proj.weight),
                value_weight=tensor_to_numpy(layer.self_attn.v_proj.weight),
                attention_output_weight=tensor_to_numpy(layer.self_attn.o_proj.weight),
                post_attention_norm_weight=tensor_to_numpy(
                    layer.post_attention_layernorm.weight
                ),
                gate_weight=tensor_to_numpy(layer.mlp.gate_proj.weight),
                up_weight=tensor_to_numpy(layer.mlp.up_proj.weight),
                down_weight=tensor_to_numpy(layer.mlp.down_proj.weight),
            )
        )

    return LlamaModelWeights(
        token_embeddings=tensor_to_numpy(model.model.embed_tokens.weight),
        blocks=tuple(block_weights),
        final_norm_weight=tensor_to_numpy(model.model.norm.weight),
        lm_head_weight=tensor_to_numpy(model.lm_head.weight),
    )
