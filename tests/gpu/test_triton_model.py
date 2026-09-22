"""End-to-end Llama parity between PyTorch and Triton backends."""

import importlib.util

import pytest
import torch

from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.inference import decode_paged, prefill_paged
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
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


def make_models() -> tuple[LlamaModel, LlamaModel]:
    torch.manual_seed(91)
    reference = LlamaModel(tiny_config()).to("cuda").eval()
    optimized = LlamaModel(tiny_config(), backend="triton").to("cuda").eval()
    optimized.load_state_dict(reference.state_dict())
    return reference, optimized


def make_manager(model: LlamaModel) -> PagedKVCacheManager:
    parameter = model.embed_tokens.weight
    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=8,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_full_forward_logits_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7]], device="cuda")

    assert_logits_close(optimized(prompt), reference(prompt))


def test_contiguous_prefill_and_decode_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7]], device="cuda")
    next_token = torch.tensor([[9]], device="cuda")

    reference_prefill = prefill(reference, prompt)
    optimized_prefill = prefill(optimized, prompt)
    assert_logits_close(optimized_prefill.logits, reference_prefill.logits)

    reference_decode = decode(reference, next_token, reference_prefill.cache)
    optimized_decode = decode(optimized, next_token, optimized_prefill.cache)
    assert_logits_close(optimized_decode, reference_decode)
    assert optimized_prefill.cache.current_length == 4


def test_paged_prefill_and_decode_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7]], device="cuda")
    next_token = torch.tensor([[9]], device="cuda")

    reference_result = prefill_paged(
        reference, prompt, make_manager(reference), sequence_id="torch"
    )
    optimized_result = prefill_paged(
        optimized, prompt, make_manager(optimized), sequence_id="triton"
    )
    assert_logits_close(optimized_result.logits, reference_result.logits)

    reference_decode = decode_paged(reference, next_token, reference_result.cache)
    optimized_decode = decode_paged(optimized, next_token, optimized_result.cache)
    assert_logits_close(optimized_decode, reference_decode)
    assert optimized_result.cache.current_length == 4


def test_greedy_generation_tokens_match() -> None:
    reference, optimized = make_models()
    prompt = torch.tensor([[1, 4, 7]], device="cuda")
    config = GenerationConfig(max_new_tokens=3)

    reference_result = generate(reference, prompt, config)
    optimized_result = generate(optimized, prompt, config)

    torch.testing.assert_close(
        optimized_result.token_ids,
        reference_result.token_ids,
        rtol=0,
        atol=0,
    )
