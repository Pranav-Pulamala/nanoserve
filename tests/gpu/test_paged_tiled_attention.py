"""CUDA validation for direct paged Triton attention."""

import importlib.util

import pytest
import torch
from torch.testing import assert_close

from tokserve.engine.attention import causal_attention, repeat_key_value
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.inference import decode_paged, prefill_paged
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.engine.paged.storage import PagedKVStorage
from tokserve.kernels.paged_attention import triton_paged_attention
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


def test_noncontiguous_physical_pages_match_contiguous_reference() -> None:
    torch.manual_seed(700)
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=4,
        num_key_value_heads=2,
        block_size=2,
        head_dim=8,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )
    block_ids = (3, 1, 2)
    key_length = 5
    query = torch.randn(1, 4, 1, 8, device="cuda")
    logical_key = torch.randn(1, 2, key_length, 8, device="cuda")
    logical_value = torch.randn(1, 2, key_length, 8, device="cuda")

    storage.key_storage.fill_(50_000.0)
    storage.value_storage.fill_(-50_000.0)

    for position in range(key_length):
        logical_block = position // storage.block_size
        offset = position % storage.block_size
        block_id = block_ids[logical_block]
        storage.write_token(
            layer_index=0,
            block_id=block_id,
            offset=offset,
            keys=logical_key[0, :, position, :],
            values=logical_value[0, :, position, :],
        )

    repeated_key = repeat_key_value(logical_key, num_groups=2)
    repeated_value = repeat_key_value(logical_value, num_groups=2)
    expected, _ = causal_attention(
        query,
        repeated_key,
        repeated_value,
        query_position_offset=key_length - 1,
    )
    actual = triton_paged_attention(
        query,
        storage,
        block_ids,
        layer_index=0,
        key_length=key_length,
        query_position_offset=key_length - 1,
    )

    assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def tiny_config() -> LlamaConfig:
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


def make_manager(model: LlamaModel) -> PagedKVCacheManager:
    parameter = model.embed_tokens.weight
    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=12,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def test_paged_prefill_and_multiple_decode_steps_match() -> None:
    torch.manual_seed(701)
    reference = LlamaModel(tiny_config()).to("cuda").eval()
    optimized = LlamaModel(tiny_config(), backend="triton").to("cuda").eval()
    optimized.load_state_dict(reference.state_dict())

    reference_result = prefill_paged(
        reference,
        torch.tensor([[1, 2, 3]], device="cuda"),
        make_manager(reference),
        sequence_id="reference",
    )
    optimized_result = prefill_paged(
        optimized,
        torch.tensor([[1, 2, 3]], device="cuda"),
        make_manager(optimized),
        sequence_id="optimized",
    )

    assert_close(
        optimized_result.logits,
        reference_result.logits,
        rtol=1e-4,
        atol=1e-5,
    )

    for token_id in (4, 5, 6, 7):
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

    assert optimized_result.cache.current_length == 7
