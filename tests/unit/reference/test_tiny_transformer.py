import numpy as np
import pytest
from numpy.testing import assert_allclose

from tokserve.reference.transformer import (
    TinyTransformerConfig,
    TinyTransformerWeights,
    TransformerBlockWeights,
    tiny_transformer,
)


def make_model() -> tuple[TinyTransformerConfig, TinyTransformerWeights]:
    config = TinyTransformerConfig(
        vocab_size=3,
        hidden_size=2,
        num_heads=1,
        feed_forward_size=4,
        num_layers=1,
        max_sequence_length=4,
    )
    block = TransformerBlockWeights(
        norm1_weight=np.ones(2),
        norm1_bias=np.zeros(2),
        query_weight=np.eye(2),
        key_weight=np.eye(2),
        value_weight=np.zeros((2, 2)),
        attention_output_weight=np.eye(2),
        norm2_weight=np.ones(2),
        norm2_bias=np.zeros(2),
        feed_forward_up_weight=np.zeros((4, 2)),
        feed_forward_down_weight=np.zeros((2, 4)),
    )
    weights = TinyTransformerWeights(
        token_embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        blocks=(block,),
        final_norm_weight=np.ones(2),
        final_norm_bias=np.zeros(2),
        output_weight=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
    return config, weights


def test_tiny_transformer_produces_expected_logits() -> None:
    config, weights = make_model()

    logits = tiny_transformer(
        np.array([[0, 1]]),
        config,
        weights,
    )

    normalized_value = 0.5 / np.sqrt(0.25 + 1e-5)
    expected = np.array(
        [
            [
                [normalized_value, -normalized_value, 0.0],
                [-normalized_value, normalized_value, 0.0],
            ]
        ]
    )

    assert logits.shape == (1, 2, 3)
    assert_allclose(logits, expected)


def test_tiny_transformer_supports_batches() -> None:
    config, weights = make_model()

    logits = tiny_transformer(
        np.array([[0, 1], [1, 2]]),
        config,
        weights,
    )

    assert logits.shape == (2, 2, 3)


def test_tiny_transformer_is_deterministic() -> None:
    config, weights = make_model()
    token_ids = np.array([[0, 1]])

    first = tiny_transformer(token_ids, config, weights)
    second = tiny_transformer(token_ids, config, weights)

    assert_allclose(first, second)


def test_tiny_transformer_rejects_excessive_sequence_length() -> None:
    config, weights = make_model()

    with pytest.raises(ValueError, match="max_sequence_length"):
        tiny_transformer(
            np.array([[0, 1, 2, 0, 1]]),
            config,
            weights,
        )
