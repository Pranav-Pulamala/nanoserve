import pytest
import torch

from tokserve.engine.attention import GroupedQueryAttention
from tokserve.engine.cache import LayerKVCache
from tokserve.engine.paged.allocator import PhysicalBlockAllocator
from tokserve.engine.paged.sequence_cache import SequencePagedKVCache
from tokserve.engine.paged.storage import PagedKVStorage
from tokserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small GQA configuration with Hq != Hkv."""

    return LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def create_paged_cache(
    config: LlamaConfig,
    *,
    block_size: int = 2,
    num_blocks: int = 8,
) -> SequencePagedKVCache:
    """Create a one-layer paged cache for attention tests."""

    allocator = PhysicalBlockAllocator(num_blocks)
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=num_blocks,
        num_key_value_heads=config.num_key_value_heads,
        block_size=block_size,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return SequencePagedKVCache(
        allocator=allocator,
        storage=storage,
        max_sequence_length=config.max_position_embeddings,
    )


def run_paged_attention(
    attention: GroupedQueryAttention,
    inputs: torch.Tensor,
    paged_cache: SequencePagedKVCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve, execute, and finish one one-layer paged attention call."""

    paged_cache.begin_append(inputs.shape[1])
    output, weights = attention(
        inputs,
        paged_cache=paged_cache,
        layer_index=0,
    )
    paged_cache.finish_append()
    return output, weights


def create_contiguous_cache(config: LlamaConfig) -> LayerKVCache:
    """Create a matching contiguous attention cache."""

    return LayerKVCache(
        batch_size=1,
        num_key_value_heads=config.num_key_value_heads,
        max_sequence_length=config.max_position_embeddings,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_paged_prefill_matches_uncached_attention() -> None:
    torch.manual_seed(60)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    inputs = torch.randn(1, 5, config.hidden_size)
    paged_cache = create_paged_cache(config, block_size=2)

    expected_output, expected_weights = attention(inputs)
    actual_output, actual_weights = run_paged_attention(
        attention,
        inputs,
        paged_cache,
    )

    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        rtol=1e-5,
        atol=1e-6,
    )
    assert paged_cache.current_length == 5
    assert paged_cache.num_blocks == 3
    assert paged_cache.storage.num_key_value_heads == 2


def test_paged_decode_within_block_matches_contiguous_cache() -> None:
    torch.manual_seed(61)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    prompt = torch.randn(1, 2, config.hidden_size)
    next_token = torch.randn(1, 1, config.hidden_size)
    paged_cache = create_paged_cache(config, block_size=4)
    contiguous_cache = create_contiguous_cache(config)

    attention(prompt, cache=contiguous_cache)
    run_paged_attention(attention, prompt, paged_cache)

    expected_output, expected_weights = attention(
        next_token,
        cache=contiguous_cache,
    )
    actual_output, actual_weights = run_paged_attention(
        attention,
        next_token,
        paged_cache,
    )

    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        rtol=1e-5,
        atol=1e-6,
    )
    assert paged_cache.num_blocks == 1
    assert paged_cache.current_length == 3


def test_paged_decode_at_block_boundary_matches_contiguous_cache() -> None:
    torch.manual_seed(62)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    prompt = torch.randn(1, 2, config.hidden_size)
    next_token = torch.randn(1, 1, config.hidden_size)
    paged_cache = create_paged_cache(config, block_size=2)
    contiguous_cache = create_contiguous_cache(config)

    attention(prompt, cache=contiguous_cache)
    run_paged_attention(attention, prompt, paged_cache)

    expected_output, _ = attention(next_token, cache=contiguous_cache)
    actual_output, actual_weights = run_paged_attention(
        attention,
        next_token,
        paged_cache,
    )

    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=1e-5,
        atol=1e-6,
    )
    assert actual_weights.shape == (
        1,
        config.num_attention_heads,
        1,
        3,
    )
    assert paged_cache.num_blocks == 2
    assert paged_cache.current_length == 3


def test_multiple_paged_blocks_match_full_sequence_attention() -> None:
    torch.manual_seed(63)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    inputs = torch.randn(1, 6, config.hidden_size)
    paged_cache = create_paged_cache(config, block_size=2)

    full_output, _ = attention(inputs)
    run_paged_attention(attention, inputs[:, :3, :], paged_cache)

    for position in range(3, 6):
        actual_output, _ = run_paged_attention(
            attention,
            inputs[:, position : position + 1, :],
            paged_cache,
        )
        torch.testing.assert_close(
            actual_output,
            full_output[:, position : position + 1, :],
            rtol=1e-5,
            atol=1e-6,
        )

    assert paged_cache.current_length == 6
    assert paged_cache.num_blocks == 3


def test_paged_decode_projects_only_new_token() -> None:
    torch.manual_seed(64)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    paged_cache = create_paged_cache(config, block_size=2)
    projected_lengths: list[int] = []

    def record_key_projection(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths.append(inputs[0].shape[1])

    handle = attention.k_proj.register_forward_hook(record_key_projection)

    try:
        run_paged_attention(
            attention,
            torch.randn(1, 3, config.hidden_size),
            paged_cache,
        )
        run_paged_attention(
            attention,
            torch.randn(1, 1, config.hidden_size),
            paged_cache,
        )
    finally:
        handle.remove()

    assert projected_lengths == [3, 1]


def test_paged_attention_requires_layer_index() -> None:
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    paged_cache = create_paged_cache(config)
    paged_cache.begin_append(1)

    with pytest.raises(ValueError, match="layer_index is required"):
        attention(
            torch.randn(1, 1, config.hidden_size),
            paged_cache=paged_cache,
        )


def test_contiguous_and_paged_cache_cannot_be_combined() -> None:
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    paged_cache = create_paged_cache(config)
    contiguous_cache = create_contiguous_cache(config)

    with pytest.raises(ValueError, match="choose either contiguous or paged"):
        attention(
            torch.randn(1, 1, config.hidden_size),
            cache=contiguous_cache,
            paged_cache=paged_cache,
            layer_index=0,
        )
