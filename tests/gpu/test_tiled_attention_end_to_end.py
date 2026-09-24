"""End-to-end validation of TokServe tiled-attention execution."""

import importlib.util
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import tokserve.engine.attention as attention_module
from tokserve.engine.attention import (
    causal_attention,
    repeat_key_value,
)
from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.inference import decode_paged, prefill_paged
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.engine.tiled_attention import tiled_attention_reference
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.generate import generate
from tokserve.generation.paged_generate import generate_with_paged_cache
from tokserve.generation.types import GenerationConfig
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
    """Return a deterministic GQA configuration."""

    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def make_models() -> tuple[LlamaModel, LlamaModel]:
    """Create weight-identical PyTorch and Triton models."""

    torch.manual_seed(1000)
    reference = LlamaModel(tiny_config()).to("cuda").eval()
    optimized = LlamaModel(tiny_config(), backend="triton").to("cuda").eval()
    optimized.load_state_dict(reference.state_dict())
    return reference, optimized


def make_manager(model: LlamaModel) -> PagedKVCacheManager:
    """Create a paged pool compatible with one model."""

    parameter = model.embed_tokens.weight
    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=16,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def test_component_parity_across_all_three_paths() -> None:
    """Compare ordinary, educational tiled, and Triton attention."""

    torch.manual_seed(1001)
    query = torch.randn(1, 4, 19, 16, device="cuda")
    key = torch.randn(1, 2, 19, 16, device="cuda")
    value = torch.randn(1, 2, 19, 16, device="cuda")

    repeated_key = repeat_key_value(key, num_groups=2)
    repeated_value = repeat_key_value(value, num_groups=2)
    ordinary, _ = causal_attention(
        query,
        repeated_key,
        repeated_value,
    )
    educational = tiled_attention_reference(
        query,
        key,
        value,
        query_block_size=7,
        key_block_size=5,
    )
    optimized = triton_tiled_attention(query, key, value)

    assert_close(educational, ordinary, rtol=1e-5, atol=1e-6)
    assert_close(optimized, ordinary, rtol=1e-4, atol=1e-5)


def test_full_model_logits_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7, 2, 9]], device="cuda")

    expected = reference(prompt)
    actual = optimized(prompt)

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_contiguous_prefill_and_multiple_decode_steps_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 2, 3]], device="cuda")
    continuation = (4, 5, 6, 7)

    reference_result = prefill(reference, prompt)
    optimized_result = prefill(optimized, prompt)

    assert_close(
        optimized_result.logits,
        reference_result.logits,
        rtol=1e-4,
        atol=1e-5,
    )

    for expected_length, token_id in enumerate(continuation, start=4):
        token = torch.tensor([[token_id]], device="cuda")
        expected = decode(reference, token, reference_result.cache)
        actual = decode(optimized, token, optimized_result.cache)

        assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        assert optimized_result.cache.current_length == expected_length


def test_paged_prefill_and_multiple_decode_steps_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 2, 3]], device="cuda")
    continuation = (4, 5, 6, 7)

    reference_result = prefill_paged(
        reference,
        prompt,
        make_manager(reference),
        sequence_id="reference",
    )
    optimized_result = prefill_paged(
        optimized,
        prompt,
        make_manager(optimized),
        sequence_id="optimized",
    )

    assert_close(
        optimized_result.logits,
        reference_result.logits,
        rtol=1e-4,
        atol=1e-5,
    )

    for expected_length, token_id in enumerate(continuation, start=4):
        token = torch.tensor([[token_id]], device="cuda")
        expected = decode_paged(
            reference,
            token,
            reference_result.cache,
        )
        actual = decode_paged(
            optimized,
            token,
            optimized_result.cache,
        )

        assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        assert optimized_result.cache.current_length == expected_length


def test_greedy_generation_matches_for_every_execution_mode() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7]], device="cuda")
    config = GenerationConfig(max_new_tokens=4)

    reference_uncached = generate(reference, prompt, config)
    optimized_uncached = generate(optimized, prompt, config)
    reference_cached = generate_with_cache(reference, prompt, config)
    optimized_cached = generate_with_cache(optimized, prompt, config)

    reference_paged = generate_with_paged_cache(
        reference,
        prompt,
        config,
        make_manager(reference),
        sequence_id="reference-generation",
    )
    optimized_paged = generate_with_paged_cache(
        optimized,
        prompt,
        config,
        make_manager(optimized),
        sequence_id="optimized-generation",
    )

    expected = reference_uncached.token_ids
    assert torch.equal(optimized_uncached.token_ids, expected)
    assert torch.equal(reference_cached.token_ids, expected)
    assert torch.equal(optimized_cached.token_ids, expected)
    assert torch.equal(reference_paged.token_ids, expected)
    assert torch.equal(optimized_paged.token_ids, expected)


def test_future_prompt_token_cannot_change_earlier_logits() -> None:
    """Prove end-to-end causal isolation through the Triton model."""

    _, optimized = make_models()
    original = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    changed = original.clone()
    changed[:, -1] = 17

    original_logits = optimized(original)
    changed_logits = optimized(changed)

    assert_close(
        changed_logits[:, :-1, :],
        original_logits[:, :-1, :],
        rtol=0,
        atol=0,
    )


def test_custom_contiguous_and_paged_kernels_are_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail if the Triton model silently falls back to PyTorch attention."""

    _, optimized = make_models()
    contiguous_calls = 0
    paged_calls = 0

    original_contiguous: Callable[..., torch.Tensor] = (
        attention_module.triton_tiled_attention
    )
    original_paged: Callable[..., torch.Tensor] = (
        attention_module.triton_paged_attention
    )

    def counted_contiguous(
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        nonlocal contiguous_calls
        contiguous_calls += 1
        return original_contiguous(*args, **kwargs)

    def counted_paged(
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        nonlocal paged_calls
        paged_calls += 1
        return original_paged(*args, **kwargs)

    monkeypatch.setattr(
        attention_module,
        "triton_tiled_attention",
        counted_contiguous,
    )
    monkeypatch.setattr(
        attention_module,
        "triton_paged_attention",
        counted_paged,
    )

    prompt = torch.tensor([[1, 2, 3]], device="cuda")
    optimized(prompt)
    prefill_paged(
        optimized,
        prompt,
        make_manager(optimized),
        sequence_id="invocation-proof",
    )

    assert contiguous_calls == optimized.config.num_hidden_layers
    assert paged_calls == optimized.config.num_hidden_layers
