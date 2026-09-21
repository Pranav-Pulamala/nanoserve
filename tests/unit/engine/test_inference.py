import pytest
import torch

from nanoserve.engine.cache import KVCache
from nanoserve.engine.inference import (
    create_kv_cache,
    decode,
    prefill,
)
from nanoserve.engine.model import LlamaModel
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small deterministic model configuration."""

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


def test_create_kv_cache_matches_model_configuration() -> None:
    model = LlamaModel(tiny_config())

    cache = create_kv_cache(
        model,
        batch_size=2,
        max_sequence_length=12,
    )

    assert len(cache) == model.config.num_hidden_layers
    assert cache.batch_size == 2
    assert cache.num_key_value_heads == model.config.num_key_value_heads
    assert cache.max_sequence_length == 12
    assert cache.head_dim == model.config.head_dim
    assert cache.device == model.embed_tokens.weight.device
    assert cache.dtype == model.embed_tokens.weight.dtype
    assert cache.current_length == 0


def test_prefill_populates_cache_to_prompt_length() -> None:
    torch.manual_seed(30)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)

    result = prefill(model, prompt_ids)

    assert result.logits.shape == (1, 4, model.config.vocab_size)
    assert result.cache.current_length == 4
    assert all(layer.current_length == 4 for layer in result.cache.layers)


def test_prefill_logits_match_uncached_model() -> None:
    torch.manual_seed(31)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)

    expected = model(prompt_ids)
    result = prefill(model, prompt_ids)

    torch.testing.assert_close(
        result.logits,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_decode_consumes_one_token_and_grows_cache() -> None:
    torch.manual_seed(32)
    model = LlamaModel(tiny_config())
    result = prefill(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
    )

    logits = decode(
        model,
        torch.tensor([[4]], dtype=torch.int64),
        result.cache,
    )

    assert logits.shape == (1, 1, model.config.vocab_size)
    assert result.cache.current_length == 4


def test_repeated_decode_grows_cache_one_position_at_a_time() -> None:
    torch.manual_seed(33)
    model = LlamaModel(tiny_config())
    result = prefill(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
    )

    decode(model, torch.tensor([[4]]), result.cache)
    assert result.cache.current_length == 4

    decode(model, torch.tensor([[5]]), result.cache)
    assert result.cache.current_length == 5

    decode(model, torch.tensor([[6]]), result.cache)
    assert result.cache.current_length == 6


def test_decode_logits_match_uncached_full_sequence() -> None:
    torch.manual_seed(34)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    next_token = torch.tensor([[9]], dtype=torch.int64)

    result = prefill(model, prompt_ids)
    actual = decode(model, next_token, result.cache)

    complete_ids = torch.cat((prompt_ids, next_token), dim=1)
    expected = model(complete_ids)[:, -1:, :]

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_multiple_decode_logits_match_uncached_reference() -> None:
    torch.manual_seed(35)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 2]], dtype=torch.int64)
    first_token = torch.tensor([[3]], dtype=torch.int64)
    second_token = torch.tensor([[4]], dtype=torch.int64)
    result = prefill(model, prompt_ids)

    first_actual = decode(model, first_token, result.cache)
    first_complete = torch.cat((prompt_ids, first_token), dim=1)
    first_expected = model(first_complete)[:, -1:, :]

    second_actual = decode(model, second_token, result.cache)
    second_complete = torch.cat(
        (prompt_ids, first_token, second_token),
        dim=1,
    )
    second_expected = model(second_complete)[:, -1:, :]

    torch.testing.assert_close(
        first_actual,
        first_expected,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        second_actual,
        second_expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_decode_rejects_more_than_one_token() -> None:
    model = LlamaModel(tiny_config())
    result = prefill(model, torch.tensor([[1, 2]]))

    with pytest.raises(
        ValueError,
        match="decode requires exactly one token",
    ):
        decode(
            model,
            torch.tensor([[3, 4]]),
            result.cache,
        )


def test_decode_rejects_empty_cache() -> None:
    model = LlamaModel(tiny_config())
    cache = create_kv_cache(model, batch_size=1)

    with pytest.raises(
        ValueError,
        match="decode requires a populated cache",
    ):
        decode(model, torch.tensor([[1]]), cache)


def test_decode_rejects_batch_mismatch() -> None:
    model = LlamaModel(tiny_config())
    result = prefill(
        model,
        torch.tensor([[1, 2], [3, 4]]),
    )

    with pytest.raises(
        ValueError,
        match="token_ids batch size must match",
    ):
        decode(model, torch.tensor([[5]]), result.cache)


def test_prefill_rejects_prompt_larger_than_cache_capacity() -> None:
    model = LlamaModel(tiny_config())

    with pytest.raises(
        ValueError,
        match="prompt length exceeds cache capacity",
    ):
        prefill(
            model,
            torch.tensor([[1, 2, 3, 4]]),
            max_sequence_length=3,
        )


def test_decode_rejects_capacity_overflow() -> None:
    model = LlamaModel(tiny_config())
    result = prefill(
        model,
        torch.tensor([[1, 2, 3]]),
        max_sequence_length=3,
    )

    with pytest.raises(
        ValueError,
        match="KV cache capacity exceeded",
    ):
        decode(model, torch.tensor([[4]]), result.cache)


def test_cache_creation_rejects_capacity_above_model_limit() -> None:
    model = LlamaModel(tiny_config())

    with pytest.raises(
        ValueError,
        match="cannot exceed max_position_embeddings",
    ):
        create_kv_cache(
            model,
            batch_size=1,
            max_sequence_length=17,
        )


def test_prefill_and_decode_disable_gradients() -> None:
    model = LlamaModel(tiny_config())
    gradient_states: list[bool] = []

    def record_gradient_state(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        gradient_states.append(torch.is_grad_enabled())

    handle = model.embed_tokens.register_forward_hook(record_gradient_state)

    try:
        result = prefill(model, torch.tensor([[1, 2]]))
        decode(model, torch.tensor([[3]]), result.cache)
    finally:
        handle.remove()

    assert gradient_states == [False, False]


def test_cache_dtype_matches_float64_model() -> None:
    model = LlamaModel(tiny_config()).to(dtype=torch.float64)

    result = prefill(model, torch.tensor([[1, 2]]))

    assert result.cache.dtype == torch.float64
    assert all(layer.keys.dtype == torch.float64 for layer in result.cache.layers)


def test_manually_incompatible_cache_fails_clearly() -> None:
    model = LlamaModel(tiny_config())
    cache = KVCache(
        num_layers=1,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        max_sequence_length=8,
        head_dim=model.config.head_dim,
        device=model.embed_tokens.weight.device,
        dtype=model.embed_tokens.weight.dtype,
    )
    cache[0].append(
        torch.zeros(
            1,
            model.config.num_key_value_heads,
            1,
            model.config.head_dim,
        ),
        torch.zeros(
            1,
            model.config.num_key_value_heads,
            1,
            model.config.head_dim,
        ),
    )

    with pytest.raises(
        ValueError,
        match="cache layer count must match",
    ):
        decode(model, torch.tensor([[2]]), cache)
