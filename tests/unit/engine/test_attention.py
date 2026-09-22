import inspect

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close

from tokserve.engine.attention import (
    GroupedQueryAttention,
    repeat_key_value,
    reshape_projection,
)
from tokserve.reference.llama.attention import (
    grouped_query_attention as numpy_grouped_query_attention,
)
from tokserve.reference.llama.config import LlamaConfig

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


def load_weights(
    module: GroupedQueryAttention,
    query_weight: np.ndarray,
    key_weight: np.ndarray,
    value_weight: np.ndarray,
    output_weight: np.ndarray,
) -> None:
    with torch.no_grad():
        module.q_proj.weight.copy_(torch.tensor(query_weight))
        module.k_proj.weight.copy_(torch.tensor(key_weight))
        module.v_proj.weight.copy_(torch.tensor(value_weight))
        module.o_proj.weight.copy_(torch.tensor(output_weight))


def test_reshape_projection_has_expected_shape() -> None:
    projection = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    result = reshape_projection(
        projection,
        num_heads=2,
        head_dim=2,
    )

    assert result.shape == (2, 2, 3, 2)


def test_repeat_key_value_repeats_each_head_as_a_group() -> None:
    inputs = torch.tensor([[[[1.0]], [[2.0]]]]).reshape(1, 2, 1, 1)

    result = repeat_key_value(inputs, num_groups=2)

    assert result.shape == (1, 4, 1, 1)
    assert_close(result[:, 0], inputs[:, 0])
    assert_close(result[:, 1], inputs[:, 0])
    assert_close(result[:, 2], inputs[:, 1])
    assert_close(result[:, 3], inputs[:, 1])


def test_grouped_query_attention_matches_numpy() -> None:
    config = make_config()
    inputs = np.array(
        [[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]],
        dtype=np.float64,
    )
    query_weight = np.arange(16, dtype=np.float64).reshape(4, 4) / 16.0
    key_weight = np.arange(8, dtype=np.float64).reshape(2, 4) / 8.0
    value_weight = np.arange(8, dtype=np.float64).reshape(2, 4) / 16.0
    output_weight = np.eye(4)
    positions = np.array([0, 1], dtype=np.int64)

    module = GroupedQueryAttention(config)
    load_weights(
        module,
        query_weight,
        key_weight,
        value_weight,
        output_weight,
    )

    torch_output, torch_weights = module(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(positions),
    )
    numpy_output, numpy_weights = numpy_grouped_query_attention(
        inputs,
        query_weight,
        key_weight,
        value_weight,
        output_weight,
        positions,
        config,
    )

    assert torch_output.shape == (1, 2, 4)
    assert torch_weights.shape == (1, 2, 2, 2)
    assert_allclose(
        torch_output.detach().numpy(),
        numpy_output,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        torch_weights.detach().numpy(),
        numpy_weights,
        rtol=RTOL,
        atol=ATOL,
    )


def test_attention_is_causal() -> None:
    config = make_config()
    module = GroupedQueryAttention(config)

    first = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    changed = first.clone()
    changed[:, 1] = 100.0
    positions = torch.tensor([0, 1])

    first_output, first_weights = module(first, positions)
    changed_output, _ = module(changed, positions)

    assert_close(first_output[:, 0], changed_output[:, 0])
    assert first_weights[0, 0, 0, 1] == 0.0


def test_attention_preserves_device_and_dtype() -> None:
    config = make_config()
    module = GroupedQueryAttention(config)
    inputs = torch.ones(1, 2, 4, dtype=torch.float32)
    positions = torch.tensor([0, 1], device=inputs.device)

    output, weights = module(inputs, positions)

    assert output.device == inputs.device
    assert weights.device == inputs.device
    assert output.dtype == inputs.dtype
    assert weights.dtype == inputs.dtype


def test_attention_registers_projection_parameters() -> None:
    module = GroupedQueryAttention(make_config())

    assert set(dict(module.named_parameters())) == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
    }


def test_attention_rejects_position_device_mismatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    module = GroupedQueryAttention(make_config()).to("cuda")
    inputs = torch.ones(1, 2, 4, device="cuda")
    positions = torch.tensor([0, 1], device="cpu")

    with pytest.raises(ValueError, match="same device"):
        module(inputs, positions)


def test_attention_math_is_explicit() -> None:
    source = inspect.getsource(
        __import__(
            "tokserve.engine.attention",
            fromlist=["causal_attention"],
        ).causal_attention
    )

    assert "matmul" in source
    assert "softmax" in source
    assert "scaled_dot_product_attention" not in source
