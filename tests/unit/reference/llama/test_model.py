import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tokserve.reference.llama.config import LlamaConfig
from tokserve.reference.llama.model import LlamaBlockWeights, llama_block


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )


def make_zero_sublayer_weights(config: LlamaConfig) -> LlamaBlockWeights:
    key_value_size = config.num_key_value_heads * config.head_dim

    return LlamaBlockWeights(
        input_norm_weight=np.ones(config.hidden_size),
        query_weight=np.eye(config.hidden_size),
        key_weight=np.zeros((key_value_size, config.hidden_size)),
        value_weight=np.zeros((key_value_size, config.hidden_size)),
        attention_output_weight=np.eye(config.hidden_size),
        post_attention_norm_weight=np.ones(config.hidden_size),
        gate_weight=np.zeros((config.intermediate_size, config.hidden_size)),
        up_weight=np.zeros((config.intermediate_size, config.hidden_size)),
        down_weight=np.zeros((config.hidden_size, config.intermediate_size)),
    )


def test_zero_sublayers_preserve_residual_input() -> None:
    config = make_config()
    inputs = np.arange(8, dtype=np.float64).reshape(1, 2, 4)
    original = inputs.copy()

    result = llama_block(
        inputs,
        np.array([0, 1]),
        make_zero_sublayer_weights(config),
        config,
    )

    assert_allclose(result, inputs)
    assert_array_equal(inputs, original)


def test_llama_block_preserves_input_shape() -> None:
    config = make_config()
    inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    result = llama_block(
        inputs,
        np.array([0, 1, 2]),
        make_zero_sublayer_weights(config),
        config,
    )

    assert result.shape == (2, 3, 4)


def test_llama_block_is_deterministic() -> None:
    config = make_config()
    inputs = np.array([[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]])
    weights = make_zero_sublayer_weights(config)
    positions = np.array([0, 1])

    first = llama_block(inputs, positions, weights, config)
    second = llama_block(inputs, positions, weights, config)

    assert_allclose(first, second)


def test_llama_block_rejects_wrong_hidden_size() -> None:
    config = make_config()

    with pytest.raises(
        ValueError,
        match="inputs must use config.hidden_size",
    ):
        llama_block(
            np.ones((1, 2, 3)),
            np.array([0, 1]),
            make_zero_sublayer_weights(config),
            config,
        )


def test_llama_block_rejects_wrong_position_shape() -> None:
    config = make_config()

    with pytest.raises(ValueError, match=r"positions must have shape \(T,\)"):
        llama_block(
            np.ones((1, 2, 4)),
            np.array([0]),
            make_zero_sublayer_weights(config),
            config,
        )
