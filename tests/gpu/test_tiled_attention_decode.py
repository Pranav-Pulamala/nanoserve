"""CUDA cached-decode validation for Triton tiled attention."""

import importlib.util

import pytest
import torch
from torch.testing import assert_close

from tokserve.engine.attention import GroupedQueryAttention
from tokserve.engine.cache import LayerKVCache
from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.kernels.tiled_attention import triton_tiled_attention
from tokserve.reference.llama.config import LlamaConfig

CUDA_TRITON_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not CUDA_TRITON_AVAILABLE,
        reason="CUDA and Triton are required",
    ),
]


def tiny_config() -> LlamaConfig:
    """Return a deterministic grouped-query model configuration."""

    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def make_models() -> tuple[LlamaModel, LlamaModel]:
    """Create weight-identical PyTorch and Triton models."""

    torch.manual_seed(600)
    reference = LlamaModel(tiny_config()).to("cuda").eval()
    optimized = LlamaModel(tiny_config(), backend="triton").to("cuda").eval()
    optimized.load_state_dict(reference.state_dict())
    return reference, optimized


def create_layer_cache(config: LlamaConfig) -> LayerKVCache:
    """Create one CUDA cache for direct attention testing."""

    return LayerKVCache(
        batch_size=1,
        num_key_value_heads=config.num_key_value_heads,
        max_sequence_length=config.max_position_embeddings,
        head_dim=config.head_dim,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )


@pytest.mark.parametrize("history_length", [1, 7, 16, 17, 35])
def test_raw_single_token_decode_matches_pytorch(
    history_length: int,
) -> None:
    torch.manual_seed(601)
    query = torch.randn(1, 4, 1, 16, device="cuda")
    key = torch.randn(
        1,
        2,
        history_length + 1,
        16,
        device="cuda",
    )
    value = torch.randn(
        1,
        2,
        history_length + 1,
        16,
        device="cuda",
    )

    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = (
        torch.matmul(
            query,
            expanded_key.transpose(-1, -2),
        )
        / 4.0
    )
    expected = torch.matmul(
        torch.softmax(scores, dim=-1),
        expanded_value,
    )

    actual = triton_tiled_attention(
        query,
        key,
        value,
        query_position_offset=history_length,
    )

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_contiguous_prefill_and_decode_match_pytorch_model() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7, 2, 5]], device="cuda")
    next_token = torch.tensor([[9]], device="cuda")

    reference_prefill = prefill(reference, prompt)
    optimized_prefill = prefill(optimized, prompt)

    assert_close(
        optimized_prefill.logits,
        reference_prefill.logits,
        rtol=1e-4,
        atol=1e-5,
    )

    reference_logits = decode(
        reference,
        next_token,
        reference_prefill.cache,
    )
    optimized_logits = decode(
        optimized,
        next_token,
        optimized_prefill.cache,
    )

    assert_close(
        optimized_logits,
        reference_logits,
        rtol=1e-4,
        atol=1e-5,
    )
    assert optimized_prefill.cache.current_length == 6


def test_multiple_contiguous_decode_steps_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 2, 3]], device="cuda")
    continuation = (4, 5, 6, 7, 8)

    reference_result = prefill(reference, prompt)
    optimized_result = prefill(optimized, prompt)

    for expected_length, token_id in enumerate(continuation, start=4):
        token = torch.tensor([[token_id]], device="cuda")
        reference_logits = decode(
            reference,
            token,
            reference_result.cache,
        )
        optimized_logits = decode(
            optimized,
            token,
            optimized_result.cache,
        )

        assert_close(
            optimized_logits,
            reference_logits,
            rtol=1e-4,
            atol=1e-5,
        )
        assert optimized_result.cache.current_length == expected_length


def test_triton_attention_preserves_hkv_cache_and_omits_weights() -> None:
    torch.manual_seed(602)
    config = tiny_config()
    attention = GroupedQueryAttention(
        config,
        backend="triton",
    ).to("cuda")
    cache = create_layer_cache(config)
    prompt = torch.randn(
        1,
        5,
        config.hidden_size,
        device="cuda",
    )

    _, weights = attention(prompt, cache=cache)

    assert weights.numel() == 0
    assert cache.keys.shape == (
        1,
        config.num_key_value_heads,
        5,
        config.head_dim,
    )
    assert cache.values.shape == cache.keys.shape


def test_decode_projects_only_the_new_token() -> None:
    _, optimized = make_models()
    attention = optimized.layers[0].self_attention
    projected_lengths: list[int] = []

    def record_projection_length(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths.append(inputs[0].shape[1])

    handle = attention.k_proj.register_forward_hook(record_projection_length)

    try:
        result = prefill(
            optimized,
            torch.tensor([[1, 2, 3, 4]], device="cuda"),
        )
        decode(
            optimized,
            torch.tensor([[5]], device="cuda"),
            result.cache,
        )
        decode(
            optimized,
            torch.tensor([[6]], device="cuda"),
            result.cache,
        )
    finally:
        handle.remove()

    assert projected_lengths == [4, 1, 1]
