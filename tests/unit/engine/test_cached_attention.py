import torch

from nanoserve.engine.attention import (
    GroupedQueryAttention,
    causal_attention,
)
from nanoserve.engine.cache import LayerKVCache
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small grouped-query configuration."""

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


def create_cache(
    config: LlamaConfig,
    *,
    max_sequence_length: int = 16,
) -> LayerKVCache:
    """Create one CPU layer cache matching the tiny attention module."""

    return LayerKVCache(
        batch_size=1,
        num_key_value_heads=config.num_key_value_heads,
        max_sequence_length=max_sequence_length,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_unequal_length_decode_query_attends_to_all_cached_keys() -> None:
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 4, 2)
    value = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]])

    output, weights = causal_attention(
        query,
        key,
        value,
        query_position_offset=3,
    )

    torch.testing.assert_close(
        weights,
        torch.full((1, 1, 1, 4), 0.25),
    )
    torch.testing.assert_close(
        output,
        torch.tensor([[[[2.5, 25.0]]]]),
    )


def test_multi_token_continuation_uses_absolute_causal_mask() -> None:
    query = torch.zeros(1, 1, 2, 2)
    key = torch.zeros(1, 1, 4, 2)
    value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)

    _, weights = causal_attention(
        query,
        key,
        value,
        query_position_offset=2,
    )

    assert weights.shape == (1, 1, 2, 4)
    assert weights[0, 0, 0, 3] == 0.0
    assert weights[0, 0, 0, :3].count_nonzero() == 3
    assert weights[0, 0, 1, :].count_nonzero() == 4


def test_prefill_populates_cache_in_kv_head_form() -> None:
    torch.manual_seed(10)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    cache = create_cache(config)
    inputs = torch.randn(1, 3, config.hidden_size)

    output, weights = attention(inputs, cache=cache)

    assert output.shape == (1, 3, config.hidden_size)
    assert weights.shape == (
        1,
        config.num_attention_heads,
        3,
        3,
    )
    assert cache.current_length == 3
    assert cache.keys.shape == (
        1,
        config.num_key_value_heads,
        3,
        config.head_dim,
    )
    assert cache.values.shape == (
        1,
        config.num_key_value_heads,
        3,
        config.head_dim,
    )


def test_one_token_decode_appends_to_cache() -> None:
    torch.manual_seed(11)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    cache = create_cache(config)
    prompt = torch.randn(1, 3, config.hidden_size)
    next_token = torch.randn(1, 1, config.hidden_size)

    attention(prompt, cache=cache)
    output, weights = attention(next_token, cache=cache)

    assert cache.current_length == 4
    assert output.shape == (1, 1, config.hidden_size)
    assert weights.shape == (
        1,
        config.num_attention_heads,
        1,
        4,
    )


def test_decode_preserves_previous_cached_values() -> None:
    torch.manual_seed(12)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    cache = create_cache(config)
    prompt = torch.randn(1, 3, config.hidden_size)

    attention(prompt, cache=cache)
    original_keys = cache.keys.clone()
    original_values = cache.values.clone()

    attention(
        torch.randn(1, 1, config.hidden_size),
        cache=cache,
    )

    torch.testing.assert_close(cache.keys[:, :, :3, :], original_keys)
    torch.testing.assert_close(cache.values[:, :, :3, :], original_values)


def test_cached_final_token_matches_full_sequence_attention() -> None:
    torch.manual_seed(13)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    complete_inputs = torch.randn(1, 4, config.hidden_size)

    full_output, _ = attention(complete_inputs)

    cache = create_cache(config)
    attention(complete_inputs[:, :3, :], cache=cache)
    cached_output, cached_weights = attention(
        complete_inputs[:, 3:, :],
        cache=cache,
    )

    torch.testing.assert_close(
        cached_output,
        full_output[:, 3:, :],
        rtol=1e-5,
        atol=1e-6,
    )
    assert cached_weights.shape == (
        1,
        config.num_attention_heads,
        1,
        4,
    )


def test_multiple_cached_tokens_match_full_sequence_final_outputs() -> None:
    torch.manual_seed(14)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    complete_inputs = torch.randn(1, 5, config.hidden_size)

    full_output, _ = attention(complete_inputs)

    cache = create_cache(config)
    attention(complete_inputs[:, :3, :], cache=cache)

    first_decode, _ = attention(
        complete_inputs[:, 3:4, :],
        cache=cache,
    )
    second_decode, _ = attention(
        complete_inputs[:, 4:5, :],
        cache=cache,
    )

    torch.testing.assert_close(
        first_decode,
        full_output[:, 3:4, :],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        second_decode,
        full_output[:, 4:5, :],
        rtol=1e-5,
        atol=1e-6,
    )
    assert cache.current_length == 5


def test_cache_capacity_is_enforced_by_attention() -> None:
    torch.manual_seed(15)
    config = tiny_config()
    attention = GroupedQueryAttention(config)
    cache = create_cache(config, max_sequence_length=3)

    attention(
        torch.randn(1, 3, config.hidden_size),
        cache=cache,
    )

    try:
        attention(
            torch.randn(1, 1, config.hidden_size),
            cache=cache,
        )
    except ValueError as error:
        assert str(error) == "KV cache capacity exceeded"
    else:
        raise AssertionError("attention should reject cache overflow")
