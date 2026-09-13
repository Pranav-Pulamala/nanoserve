from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.testing import assert_allclose, assert_array_equal

from nanoserve.reference.llama.attention import grouped_query_attention
from nanoserve.reference.llama.config import LlamaConfig
from nanoserve.reference.llama.hf_bridge import (
    create_hugging_face_model,
    map_hugging_face_weights,
)
from nanoserve.reference.llama.layers import rms_norm, swiglu
from nanoserve.reference.llama.model import llama_block, llama_forward
from nanoserve.reference.ops import FloatArray

RTOL = 1e-4
ATOL = 1e-5


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def to_numpy(tensor: torch.Tensor) -> FloatArray:
    return np.asarray(
        tensor.detach().cpu().numpy(),
        dtype=np.float64,
    )


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def capture_output(
    storage: dict[str, torch.Tensor],
    name: str,
) -> Callable[[Any, tuple[Any, ...], Any], None]:
    def hook(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        storage[name] = tensor_from_output(output).detach().cpu()

    return hook


def test_layer_components_match_hugging_face() -> None:
    config = make_config()
    hugging_face_model = create_hugging_face_model(config)
    weights = map_hugging_face_weights(hugging_face_model, config)
    block_weights = weights.blocks[0]

    token_ids = np.array([[1, 5, 9]], dtype=np.int64)
    torch_token_ids = torch.tensor(token_ids, dtype=torch.long)
    positions = np.arange(token_ids.shape[1], dtype=np.int64)

    captured: dict[str, torch.Tensor] = {}
    layer = hugging_face_model.model.layers[0]

    handles = [
        layer.input_layernorm.register_forward_hook(
            capture_output(captured, "input_norm")
        ),
        layer.self_attn.register_forward_hook(capture_output(captured, "attention")),
        layer.mlp.register_forward_hook(capture_output(captured, "mlp")),
        layer.register_forward_hook(capture_output(captured, "block")),
    ]

    try:
        with torch.no_grad():
            hugging_face_model(
                input_ids=torch_token_ids,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden = weights.token_embeddings[token_ids]
    normalized_attention_input = rms_norm(
        hidden,
        block_weights.input_norm_weight,
        epsilon=config.rms_norm_eps,
    )
    attention_output, _ = grouped_query_attention(
        normalized_attention_input,
        block_weights.query_weight,
        block_weights.key_weight,
        block_weights.value_weight,
        block_weights.attention_output_weight,
        positions,
        config,
    )
    attention_residual = hidden + attention_output
    normalized_mlp_input = rms_norm(
        attention_residual,
        block_weights.post_attention_norm_weight,
        epsilon=config.rms_norm_eps,
    )
    mlp_output = swiglu(
        normalized_mlp_input,
        block_weights.gate_weight,
        block_weights.up_weight,
        block_weights.down_weight,
    )
    block_output = llama_block(
        hidden,
        positions,
        block_weights,
        config,
    )

    assert_allclose(
        normalized_attention_input,
        to_numpy(captured["input_norm"]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        attention_output,
        to_numpy(captured["attention"]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        mlp_output,
        to_numpy(captured["mlp"]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        block_output,
        to_numpy(captured["block"]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_final_logits_match_hugging_face() -> None:
    config = make_config()
    hugging_face_model = create_hugging_face_model(config)
    weights = map_hugging_face_weights(hugging_face_model, config)

    token_ids = np.array(
        [
            [1, 5, 9],
            [2, 7, 3],
        ],
        dtype=np.int64,
    )

    numpy_logits = llama_forward(token_ids, weights, config)

    with torch.no_grad():
        hugging_face_logits = hugging_face_model(
            input_ids=torch.tensor(token_ids, dtype=torch.long),
            use_cache=False,
        ).logits

    expected_logits = to_numpy(hugging_face_logits)

    assert numpy_logits.shape == (
        2,
        3,
        config.vocab_size,
    )
    assert numpy_logits.shape == expected_logits.shape
    assert np.all(np.isfinite(numpy_logits))
    assert np.all(np.isfinite(expected_logits))
    assert_allclose(
        numpy_logits,
        expected_logits,
        rtol=RTOL,
        atol=ATOL,
    )


def test_parity_execution_is_deterministic() -> None:
    config = make_config()
    hugging_face_model = create_hugging_face_model(config)
    weights = map_hugging_face_weights(hugging_face_model, config)
    token_ids = np.array([[1, 5, 9]], dtype=np.int64)
    torch_token_ids = torch.tensor(token_ids, dtype=torch.long)

    first_numpy = llama_forward(token_ids, weights, config)
    second_numpy = llama_forward(token_ids, weights, config)

    with torch.no_grad():
        first_hugging_face = hugging_face_model(
            input_ids=torch_token_ids,
            use_cache=False,
        ).logits
        second_hugging_face = hugging_face_model(
            input_ids=torch_token_ids,
            use_cache=False,
        ).logits

    assert_array_equal(first_numpy, second_numpy)
    assert_array_equal(
        first_hugging_face.detach().cpu().numpy(),
        second_hugging_face.detach().cpu().numpy(),
    )


def test_causal_logits_do_not_depend_on_future_tokens() -> None:
    config = make_config()
    hugging_face_model = create_hugging_face_model(config)
    weights = map_hugging_face_weights(hugging_face_model, config)

    first_tokens = np.array([[1, 5, 9]], dtype=np.int64)
    changed_future_tokens = np.array([[1, 8, 4]], dtype=np.int64)

    first_numpy = llama_forward(first_tokens, weights, config)
    changed_numpy = llama_forward(
        changed_future_tokens,
        weights,
        config,
    )

    with torch.no_grad():
        first_hugging_face = hugging_face_model(
            input_ids=torch.tensor(first_tokens, dtype=torch.long),
            use_cache=False,
        ).logits
        changed_hugging_face = hugging_face_model(
            input_ids=torch.tensor(
                changed_future_tokens,
                dtype=torch.long,
            ),
            use_cache=False,
        ).logits

    assert_allclose(
        first_numpy[:, 0],
        changed_numpy[:, 0],
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        to_numpy(first_hugging_face[:, 0]),
        to_numpy(changed_hugging_face[:, 0]),
        rtol=RTOL,
        atol=ATOL,
    )
