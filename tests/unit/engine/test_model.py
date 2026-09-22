import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close

from tokserve.engine.model import LlamaDecoderBlock
from tokserve.reference.llama.config import LlamaConfig
from tokserve.reference.llama.model import (
    LlamaBlockWeights,
)
from tokserve.reference.llama.model import (
    llama_block as numpy_llama_block,
)

RTOL = 1e-5
ATOL = 1e-6


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


def make_numpy_weights(config: LlamaConfig) -> LlamaBlockWeights:
    generator = np.random.default_rng(17)
    key_value_size = config.num_key_value_heads * config.head_dim

    return LlamaBlockWeights(
        input_norm_weight=generator.normal(size=config.hidden_size),
        query_weight=generator.normal(size=(config.hidden_size, config.hidden_size))
        / 4.0,
        key_weight=generator.normal(size=(key_value_size, config.hidden_size)) / 4.0,
        value_weight=generator.normal(size=(key_value_size, config.hidden_size)) / 4.0,
        attention_output_weight=generator.normal(
            size=(config.hidden_size, config.hidden_size)
        )
        / 4.0,
        post_attention_norm_weight=generator.normal(size=config.hidden_size),
        gate_weight=generator.normal(
            size=(config.intermediate_size, config.hidden_size)
        )
        / 4.0,
        up_weight=generator.normal(size=(config.intermediate_size, config.hidden_size))
        / 4.0,
        down_weight=generator.normal(
            size=(config.hidden_size, config.intermediate_size)
        )
        / 4.0,
    )


def load_weights(
    module: LlamaDecoderBlock,
    weights: LlamaBlockWeights,
) -> None:
    with torch.no_grad():
        module.input_norm.weight.copy_(torch.tensor(weights.input_norm_weight))
        module.self_attention.q_proj.weight.copy_(torch.tensor(weights.query_weight))
        module.self_attention.k_proj.weight.copy_(torch.tensor(weights.key_weight))
        module.self_attention.v_proj.weight.copy_(torch.tensor(weights.value_weight))
        module.self_attention.o_proj.weight.copy_(
            torch.tensor(weights.attention_output_weight)
        )
        module.post_attention_norm.weight.copy_(
            torch.tensor(weights.post_attention_norm_weight)
        )
        module.mlp.gate_proj.weight.copy_(torch.tensor(weights.gate_weight))
        module.mlp.up_proj.weight.copy_(torch.tensor(weights.up_weight))
        module.mlp.down_proj.weight.copy_(torch.tensor(weights.down_weight))


def zero_sublayer_weights(module: LlamaDecoderBlock) -> None:
    with torch.no_grad():
        module.self_attention.v_proj.weight.zero_()
        module.mlp.gate_proj.weight.zero_()
        module.mlp.up_proj.weight.zero_()
        module.mlp.down_proj.weight.zero_()


def test_decoder_block_matches_numpy_reference() -> None:
    config = make_config()
    weights = make_numpy_weights(config)
    module = LlamaDecoderBlock(config)
    load_weights(module, weights)

    inputs = np.array(
        [[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]],
        dtype=np.float64,
    )
    positions = np.array([0, 1], dtype=np.int64)

    torch_output = module(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(positions),
    )
    numpy_output = numpy_llama_block(
        inputs,
        positions,
        weights,
        config,
    )

    assert torch_output.shape == (1, 2, 4)
    assert_allclose(
        torch_output.detach().numpy(),
        numpy_output,
        rtol=RTOL,
        atol=ATOL,
    )


def test_zero_sublayers_preserve_residual_input() -> None:
    module = LlamaDecoderBlock(make_config())
    zero_sublayer_weights(module)
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    original = inputs.clone()

    output = module(inputs, torch.tensor([0, 1]))

    assert_close(output, inputs)
    assert_close(inputs, original)


def test_decoder_block_is_deterministic() -> None:
    module = LlamaDecoderBlock(make_config())
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    positions = torch.tensor([0, 1])

    first = module(inputs, positions)
    second = module(inputs, positions)

    assert_close(first, second)


def test_decoder_block_parameters_are_registered() -> None:
    module = LlamaDecoderBlock(make_config())
    parameter_names = set(dict(module.named_parameters()))

    assert parameter_names == {
        "input_norm.weight",
        "self_attention.q_proj.weight",
        "self_attention.k_proj.weight",
        "self_attention.v_proj.weight",
        "self_attention.o_proj.weight",
        "post_attention_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }


def test_decoder_block_preserves_device_and_dtype() -> None:
    module = LlamaDecoderBlock(make_config())
    inputs = torch.ones(1, 2, 4, dtype=torch.float32)
    positions = torch.tensor([0, 1], device=inputs.device)

    output = module(inputs, positions)

    assert output.device == inputs.device
    assert output.dtype == inputs.dtype


def test_decoder_block_rejects_wrong_hidden_size() -> None:
    module = LlamaDecoderBlock(make_config())

    with pytest.raises(ValueError, match="configured hidden_size"):
        module(
            torch.ones(1, 2, 3),
            torch.tensor([0, 1]),
        )
