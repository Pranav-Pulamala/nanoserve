import numpy as np
import pytest
from numpy.testing import assert_allclose

from tokserve.reference.llama.config import LlamaConfig
from tokserve.reference.llama.layers import rms_norm
from tokserve.reference.llama.model import (
    LlamaBlockWeights,
    LlamaModelWeights,
    llama_forward,
)
from tokserve.reference.ops import linear


def make_config(*, num_hidden_layers: int = 2) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=6,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4,
    )


def make_block_weights(config: LlamaConfig) -> LlamaBlockWeights:
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


def make_model_weights(config: LlamaConfig) -> LlamaModelWeights:
    token_embeddings = np.arange(
        config.vocab_size * config.hidden_size,
        dtype=np.float64,
    ).reshape(config.vocab_size, config.hidden_size)
    lm_head_weight = np.arange(
        config.vocab_size * config.hidden_size,
        dtype=np.float64,
    ).reshape(config.vocab_size, config.hidden_size)

    return LlamaModelWeights(
        token_embeddings=token_embeddings,
        blocks=tuple(
            make_block_weights(config) for _ in range(config.num_hidden_layers)
        ),
        final_norm_weight=np.ones(config.hidden_size),
        lm_head_weight=lm_head_weight,
    )


def test_llama_forward_produces_expected_logit_shape() -> None:
    config = make_config()
    weights = make_model_weights(config)
    token_ids = np.array([[0, 1, 2], [3, 4, 5]])

    logits = llama_forward(token_ids, weights, config)

    assert logits.shape == (2, 3, config.vocab_size)
    assert np.all(np.isfinite(logits))


def test_llama_forward_composes_multiple_layers() -> None:
    config = make_config(num_hidden_layers=2)
    weights = make_model_weights(config)
    token_ids = np.array([[1, 2]])

    logits = llama_forward(token_ids, weights, config)

    embedded = weights.token_embeddings[token_ids]
    normalized = rms_norm(
        embedded,
        weights.final_norm_weight,
        epsilon=config.rms_norm_eps,
    )
    expected = linear(normalized, weights.lm_head_weight)

    assert len(weights.blocks) == 2
    assert_allclose(logits, expected)


def test_llama_forward_is_deterministic() -> None:
    config = make_config()
    weights = make_model_weights(config)
    token_ids = np.array([[1, 2, 3]])

    first = llama_forward(token_ids, weights, config)
    second = llama_forward(token_ids, weights, config)

    assert_allclose(first, second)


def test_llama_forward_rejects_excessive_sequence_length() -> None:
    config = make_config()
    weights = make_model_weights(config)

    with pytest.raises(
        ValueError,
        match="max_position_embeddings",
    ):
        llama_forward(
            np.array([[0, 1, 2, 3, 4]]),
            weights,
            config,
        )


def test_llama_forward_rejects_invalid_token_ids() -> None:
    config = make_config()
    weights = make_model_weights(config)

    with pytest.raises(ValueError, match="out-of-range"):
        llama_forward(
            np.array([[config.vocab_size]]),
            weights,
            config,
        )


def test_llama_forward_rejects_wrong_layer_count() -> None:
    config = make_config(num_hidden_layers=2)
    valid_weights = make_model_weights(config)
    invalid_weights = LlamaModelWeights(
        token_embeddings=valid_weights.token_embeddings,
        blocks=(make_block_weights(config),),
        final_norm_weight=valid_weights.final_norm_weight,
        lm_head_weight=valid_weights.lm_head_weight,
    )

    with pytest.raises(
        ValueError,
        match="number of block weights",
    ):
        llama_forward(
            np.array([[0, 1]]),
            invalid_weights,
            config,
        )
