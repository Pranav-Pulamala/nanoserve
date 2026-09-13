import inspect

import numpy as np
from numpy.testing import assert_array_equal

from nanoserve.reference.llama.config import LlamaConfig
from nanoserve.reference.llama.hf_bridge import (
    REFERENCE_SEED,
    create_hugging_face_model,
    map_hugging_face_weights,
    to_hugging_face_config,
)
from nanoserve.reference.llama.model import llama_forward


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def test_hugging_face_config_matches_nanoserve_config() -> None:
    config = make_config()

    hugging_face_config = to_hugging_face_config(config)

    assert hugging_face_config.vocab_size == config.vocab_size
    assert hugging_face_config.hidden_size == config.hidden_size
    assert hugging_face_config.intermediate_size == config.intermediate_size
    assert hugging_face_config.num_hidden_layers == config.num_hidden_layers
    assert hugging_face_config.num_attention_heads == config.num_attention_heads
    assert hugging_face_config.num_key_value_heads == config.num_key_value_heads
    assert hugging_face_config.max_position_embeddings == (
        config.max_position_embeddings
    )
    assert hugging_face_config.rms_norm_eps == config.rms_norm_eps
    assert hugging_face_config.rope_theta == config.rope_theta
    assert hugging_face_config.tie_word_embeddings is False
    assert hugging_face_config.use_cache is False


def test_hugging_face_model_initialization_is_deterministic() -> None:
    config = make_config()

    first = create_hugging_face_model(config, seed=REFERENCE_SEED)
    second = create_hugging_face_model(config, seed=REFERENCE_SEED)

    assert_array_equal(
        first.model.embed_tokens.weight.detach().cpu().numpy(),
        second.model.embed_tokens.weight.detach().cpu().numpy(),
    )


def test_weight_mapping_produces_expected_shapes() -> None:
    config = make_config()
    model = create_hugging_face_model(config)
    weights = map_hugging_face_weights(model, config)
    key_value_size = config.num_key_value_heads * config.head_dim

    assert weights.token_embeddings.shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert weights.final_norm_weight.shape == (config.hidden_size,)
    assert weights.lm_head_weight.shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert len(weights.blocks) == config.num_hidden_layers

    for block in weights.blocks:
        assert block.input_norm_weight.shape == (config.hidden_size,)
        assert block.query_weight.shape == (
            config.hidden_size,
            config.hidden_size,
        )
        assert block.key_weight.shape == (
            key_value_size,
            config.hidden_size,
        )
        assert block.value_weight.shape == (
            key_value_size,
            config.hidden_size,
        )
        assert block.attention_output_weight.shape == (
            config.hidden_size,
            config.hidden_size,
        )
        assert block.post_attention_norm_weight.shape == (config.hidden_size,)
        assert block.gate_weight.shape == (
            config.intermediate_size,
            config.hidden_size,
        )
        assert block.up_weight.shape == (
            config.intermediate_size,
            config.hidden_size,
        )
        assert block.down_weight.shape == (
            config.hidden_size,
            config.intermediate_size,
        )


def test_mapped_weights_are_independent_numpy_arrays() -> None:
    config = make_config()
    model = create_hugging_face_model(config)
    weights = map_hugging_face_weights(model, config)

    assert isinstance(weights.token_embeddings, np.ndarray)
    assert weights.token_embeddings.dtype == np.float64

    original_value = model.model.embed_tokens.weight[0, 0].item()
    weights.token_embeddings[0, 0] += 1.0

    assert model.model.embed_tokens.weight[0, 0].item() == original_value


def test_bridge_does_not_download_or_generate() -> None:
    create_source = inspect.getsource(create_hugging_face_model)
    forward_source = inspect.getsource(llama_forward)

    assert "from_pretrained" not in create_source
    assert ".generate(" not in create_source
    assert "transformers" not in forward_source
    assert "torch" not in forward_source
