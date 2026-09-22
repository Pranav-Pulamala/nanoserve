import pytest
import torch

from tokserve.engine.cache import KVCache
from tokserve.engine.model import LlamaModel
from tokserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a deterministic tiny Llama configuration."""

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


def create_cache(
    model: LlamaModel,
    *,
    batch_size: int = 1,
    num_layers: int | None = None,
    max_sequence_length: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> KVCache:
    """Create a cache matching a model unless an override is supplied."""

    config = model.config
    parameter = model.embed_tokens.weight

    return KVCache(
        num_layers=(config.num_hidden_layers if num_layers is None else num_layers),
        batch_size=batch_size,
        num_key_value_heads=(
            config.num_key_value_heads
            if num_key_value_heads is None
            else num_key_value_heads
        ),
        max_sequence_length=(
            config.max_position_embeddings
            if max_sequence_length is None
            else max_sequence_length
        ),
        head_dim=config.head_dim if head_dim is None else head_dim,
        device=parameter.device if device is None else device,
        dtype=parameter.dtype if dtype is None else dtype,
    )


def test_uncached_forward_behavior_remains_available() -> None:
    torch.manual_seed(20)
    model = LlamaModel(tiny_config())
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    logits = model(token_ids)

    assert logits.shape == (1, 3, 32)


def test_prefill_populates_every_layer_cache() -> None:
    torch.manual_seed(21)
    model = LlamaModel(tiny_config())
    cache = create_cache(model)
    prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)

    logits = model(prompt_ids, cache=cache)

    assert logits.shape == (1, 4, 32)
    assert cache.current_length == 4
    assert all(layer.current_length == 4 for layer in cache.layers)
    assert all(
        layer.keys.shape
        == (
            1,
            model.config.num_key_value_heads,
            4,
            model.config.head_dim,
        )
        for layer in cache.layers
    )


def test_each_decoder_layer_has_independent_cache_storage() -> None:
    torch.manual_seed(22)
    model = LlamaModel(tiny_config())
    cache = create_cache(model)
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    model(prompt_ids, cache=cache)

    assert cache[0] is not cache[1]
    assert cache[0].keys.data_ptr() != cache[1].keys.data_ptr()
    assert cache[0].values.data_ptr() != cache[1].values.data_ptr()


def test_prefill_logits_match_uncached_full_forward() -> None:
    torch.manual_seed(23)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)

    expected = model(prompt_ids)

    cache = create_cache(model)
    actual = model(prompt_ids, cache=cache)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_one_token_cached_decode_matches_full_sequence() -> None:
    torch.manual_seed(24)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    next_token = torch.tensor([[9]], dtype=torch.int64)
    complete_ids = torch.cat((prompt_ids, next_token), dim=1)

    expected = model(complete_ids)[:, -1:, :]

    cache = create_cache(model)
    model(prompt_ids, cache=cache)
    actual = model(next_token, cache=cache)

    assert cache.current_length == 4
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_multiple_cached_decode_steps_match_full_sequence() -> None:
    torch.manual_seed(25)
    model = LlamaModel(tiny_config())
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    first_token = torch.tensor([[4]], dtype=torch.int64)
    second_token = torch.tensor([[5]], dtype=torch.int64)

    cache = create_cache(model)
    model(prompt_ids, cache=cache)

    first_actual = model(first_token, cache=cache)
    first_complete = torch.cat((prompt_ids, first_token), dim=1)
    first_expected = model(first_complete)[:, -1:, :]

    second_actual = model(second_token, cache=cache)
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
    assert cache.current_length == 5


def test_cache_tensors_preserve_model_device_and_dtype() -> None:
    torch.manual_seed(26)
    model = LlamaModel(tiny_config()).to(dtype=torch.float64)
    cache = create_cache(model)
    prompt_ids = torch.tensor([[1, 2]], dtype=torch.int64)

    model(prompt_ids, cache=cache)

    for layer in cache.layers:
        assert layer.keys.device == model.embed_tokens.weight.device
        assert layer.values.device == model.embed_tokens.weight.device
        assert layer.keys.dtype == torch.float64
        assert layer.values.dtype == torch.float64


def test_model_rejects_wrong_cache_layer_count() -> None:
    model = LlamaModel(tiny_config())
    cache = create_cache(model, num_layers=1)

    with pytest.raises(ValueError, match="cache layer count must match"):
        model(torch.tensor([[1, 2]]), cache=cache)


def test_model_rejects_wrong_cache_batch_size() -> None:
    model = LlamaModel(tiny_config())
    cache = create_cache(model, batch_size=2)

    with pytest.raises(ValueError, match="cache batch size must match"):
        model(torch.tensor([[1, 2]]), cache=cache)


def test_model_rejects_wrong_cache_kv_head_count() -> None:
    model = LlamaModel(tiny_config())
    cache = create_cache(model, num_key_value_heads=1)

    with pytest.raises(ValueError, match="cache KV head count must match"):
        model(torch.tensor([[1, 2]]), cache=cache)


def test_model_rejects_wrong_cache_head_dimension() -> None:
    model = LlamaModel(tiny_config())
    cache = create_cache(model, head_dim=2)

    with pytest.raises(ValueError, match="cache head_dim must match"):
        model(torch.tensor([[1, 2]]), cache=cache)


def test_model_rejects_cache_capacity_larger_than_model_limit() -> None:
    model = LlamaModel(tiny_config())
    cache = create_cache(model, max_sequence_length=17)

    with pytest.raises(
        ValueError,
        match="cache capacity cannot exceed max_position_embeddings",
    ):
        model(torch.tensor([[1, 2]]), cache=cache)


def test_model_enforces_total_cached_sequence_limit() -> None:
    model = LlamaModel(tiny_config())
    cache = create_cache(model, max_sequence_length=4)

    model(torch.tensor([[1, 2, 3, 4]]), cache=cache)

    with pytest.raises(
        ValueError,
        match="sequence length exceeds max_position_embeddings|capacity exceeded",
    ):
        model(torch.tensor([[5]]), cache=cache)
