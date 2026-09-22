import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close

from tokserve.engine.model import LlamaModel
from tokserve.engine.weights import load_model_weights
from tokserve.reference.llama.config import LlamaConfig
from tokserve.reference.llama.model import (
    LlamaBlockWeights,
    LlamaModelWeights,
    llama_block,
    llama_forward,
)

RTOL = 1e-5
ATOL = 1e-6


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )


def make_block_weights(
    config: LlamaConfig,
    generator: np.random.Generator,
) -> LlamaBlockWeights:
    key_value_size = config.num_key_value_heads * config.head_dim

    def random_array(shape: tuple[int, ...]) -> np.ndarray:
        return generator.normal(size=shape).astype(np.float64) / 8.0

    return LlamaBlockWeights(
        input_norm_weight=random_array((config.hidden_size,)) + 1.0,
        query_weight=random_array((config.hidden_size, config.hidden_size)),
        key_weight=random_array((key_value_size, config.hidden_size)),
        value_weight=random_array((key_value_size, config.hidden_size)),
        attention_output_weight=random_array((config.hidden_size, config.hidden_size)),
        post_attention_norm_weight=(random_array((config.hidden_size,)) + 1.0),
        gate_weight=random_array((config.intermediate_size, config.hidden_size)),
        up_weight=random_array((config.intermediate_size, config.hidden_size)),
        down_weight=random_array((config.hidden_size, config.intermediate_size)),
    )


def make_weights(config: LlamaConfig) -> LlamaModelWeights:
    generator = np.random.default_rng(17)

    return LlamaModelWeights(
        token_embeddings=generator.normal(
            size=(config.vocab_size, config.hidden_size)
        ).astype(np.float64),
        blocks=tuple(
            make_block_weights(config, generator)
            for _ in range(config.num_hidden_layers)
        ),
        final_norm_weight=(
            generator.normal(size=config.hidden_size).astype(np.float64) + 1.0
        ),
        lm_head_weight=generator.normal(
            size=(config.vocab_size, config.hidden_size)
        ).astype(np.float64),
    )


def test_loaded_model_logits_match_numpy() -> None:
    config = make_config()
    weights = make_weights(config)
    model = LlamaModel(config).eval()
    load_model_weights(model, weights)

    token_ids = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )

    torch_logits = model(torch.tensor(token_ids))
    numpy_logits = llama_forward(token_ids, weights, config)

    assert torch_logits.shape == numpy_logits.shape
    assert_allclose(
        torch_logits.numpy(),
        numpy_logits,
        rtol=RTOL,
        atol=ATOL,
    )


def test_first_decoder_block_matches_numpy() -> None:
    config = make_config()
    weights = make_weights(config)
    model = LlamaModel(config).eval()
    load_model_weights(model, weights)

    token_ids = np.array([[1, 2, 3]], dtype=np.int64)
    positions = np.arange(3, dtype=np.int64)
    numpy_hidden = weights.token_embeddings[token_ids]

    torch_hidden = model.embed_tokens(torch.tensor(token_ids))
    torch_block_output = model.layers[0](
        torch_hidden,
        torch.tensor(positions),
    )
    numpy_block_output = llama_block(
        numpy_hidden,
        positions,
        weights.blocks[0],
        config,
    )

    assert_allclose(
        torch_block_output.detach().numpy(),
        numpy_block_output,
        rtol=RTOL,
        atol=ATOL,
    )


def test_weight_transfer_copies_values() -> None:
    config = make_config()
    weights = make_weights(config)
    model = LlamaModel(config).eval()
    load_model_weights(model, weights)

    expected = model.embed_tokens.weight.detach().clone()
    weights.token_embeddings[0, 0] += 100.0

    assert_close(model.embed_tokens.weight, expected)


def test_transfer_preserves_model_dtype_and_device() -> None:
    config = make_config()
    weights = make_weights(config)
    model = LlamaModel(config).to(
        device="cpu",
        dtype=torch.float32,
    )

    load_model_weights(model, weights)

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_transfer_rejects_wrong_layer_count() -> None:
    config = make_config()
    valid_weights = make_weights(config)
    invalid_weights = LlamaModelWeights(
        token_embeddings=valid_weights.token_embeddings,
        blocks=(valid_weights.blocks[0],),
        final_norm_weight=valid_weights.final_norm_weight,
        lm_head_weight=valid_weights.lm_head_weight,
    )
    model = LlamaModel(config)

    with pytest.raises(ValueError, match="layer counts must match"):
        load_model_weights(model, invalid_weights)


def test_transfer_rejects_incompatible_parameter_shape() -> None:
    config = make_config()
    valid_weights = make_weights(config)
    invalid_weights = LlamaModelWeights(
        token_embeddings=np.ones((3, 3)),
        blocks=valid_weights.blocks,
        final_norm_weight=valid_weights.final_norm_weight,
        lm_head_weight=valid_weights.lm_head_weight,
    )
    model = LlamaModel(config)

    with pytest.raises(ValueError, match="token_embeddings has shape"):
        load_model_weights(model, invalid_weights)
