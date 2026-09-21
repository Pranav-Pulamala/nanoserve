import pytest
import torch

from nanoserve.engine.inference import decode, prefill
from nanoserve.engine.model import LlamaModel
from nanoserve.engine.paged.inference import decode_paged, prefill_paged
from nanoserve.engine.paged.manager import PagedKVCacheManager
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small deterministic Llama configuration."""

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


def create_manager(
    model: LlamaModel,
    *,
    block_size: int = 2,
    num_blocks: int = 8,
) -> PagedKVCacheManager:
    """Create a physical pool compatible with the model."""

    parameter = model.embed_tokens.weight

    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=num_blocks,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=block_size,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def test_paged_prefill_populates_every_layer() -> None:
    torch.manual_seed(70)
    model = LlamaModel(tiny_config())
    manager = create_manager(model, block_size=2)
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)

    result = prefill_paged(model, prompt, manager, sequence_id="A")

    assert result.logits.shape == (1, 5, model.config.vocab_size)
    assert result.cache.current_length == 5
    assert result.cache.num_blocks == 3
    assert manager.allocator.allocated_count == 3

    for layer_index in range(model.config.num_hidden_layers):
        keys, values = result.cache.read_layer(layer_index)
        assert keys.shape == (
            1,
            model.config.num_key_value_heads,
            5,
            model.config.head_dim,
        )
        assert values.shape == keys.shape


def test_paged_prefill_logits_match_uncached_and_contiguous() -> None:
    torch.manual_seed(71)
    model = LlamaModel(tiny_config())
    prompt = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)
    manager = create_manager(model)

    uncached_logits = model(prompt)
    contiguous_result = prefill(model, prompt)
    paged_result = prefill_paged(
        model,
        prompt,
        manager,
        sequence_id="A",
    )

    torch.testing.assert_close(
        paged_result.logits,
        uncached_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        paged_result.logits,
        contiguous_result.logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_one_token_paged_decode_matches_other_paths() -> None:
    torch.manual_seed(72)
    model = LlamaModel(tiny_config())
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    next_token = torch.tensor([[9]], dtype=torch.int64)
    manager = create_manager(model, block_size=2)

    contiguous_result = prefill(model, prompt)
    paged_result = prefill_paged(
        model,
        prompt,
        manager,
        sequence_id="A",
    )

    contiguous_logits = decode(model, next_token, contiguous_result.cache)
    paged_logits = decode_paged(model, next_token, paged_result.cache)
    complete = torch.cat((prompt, next_token), dim=1)
    full_logits = model(complete)[:, -1:, :]

    torch.testing.assert_close(
        paged_logits,
        contiguous_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        paged_logits,
        full_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert paged_result.cache.current_length == 4
    assert paged_result.cache.num_blocks == 2


def test_boundary_decode_allocates_exactly_one_new_block() -> None:
    torch.manual_seed(73)
    model = LlamaModel(tiny_config())
    manager = create_manager(model, block_size=2)
    result = prefill_paged(
        model,
        torch.tensor([[1, 2]], dtype=torch.int64),
        manager,
        sequence_id="A",
    )

    assert result.cache.num_blocks == 1
    free_before = manager.allocator.free_count

    decode_paged(
        model,
        torch.tensor([[3]], dtype=torch.int64),
        result.cache,
    )

    assert result.cache.current_length == 3
    assert result.cache.num_blocks == 2
    assert manager.allocator.free_count == free_before - 1


def test_within_block_decode_does_not_allocate_new_block() -> None:
    torch.manual_seed(74)
    model = LlamaModel(tiny_config())
    manager = create_manager(model, block_size=4)
    result = prefill_paged(
        model,
        torch.tensor([[1, 2]], dtype=torch.int64),
        manager,
        sequence_id="A",
    )

    free_before = manager.allocator.free_count

    decode_paged(
        model,
        torch.tensor([[3]], dtype=torch.int64),
        result.cache,
    )

    assert result.cache.current_length == 3
    assert result.cache.num_blocks == 1
    assert manager.allocator.free_count == free_before


def test_repeated_paged_decode_matches_full_sequence() -> None:
    torch.manual_seed(75)
    model = LlamaModel(tiny_config())
    manager = create_manager(model, block_size=2)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    result = prefill_paged(
        model,
        prompt,
        manager,
        sequence_id="A",
    )
    complete = prompt.clone()

    for token_id in (4, 5, 6):
        new_token = torch.tensor([[token_id]], dtype=torch.int64)
        actual = decode_paged(model, new_token, result.cache)
        complete = torch.cat((complete, new_token), dim=1)
        expected = model(complete)[:, -1:, :]

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    assert result.cache.current_length == 6
    assert result.cache.num_blocks == 3


def test_paged_layers_have_independent_kv_values() -> None:
    torch.manual_seed(76)
    model = LlamaModel(tiny_config())
    manager = create_manager(model)
    result = prefill_paged(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
        manager,
        sequence_id="A",
    )

    first_keys, first_values = result.cache.read_layer(0)
    second_keys, second_values = result.cache.read_layer(1)

    assert not torch.equal(first_keys, second_keys)
    assert not torch.equal(first_values, second_values)


def test_existing_uncached_and_contiguous_modes_still_work() -> None:
    torch.manual_seed(77)
    model = LlamaModel(tiny_config())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    uncached = model(prompt)
    contiguous = prefill(model, prompt)

    torch.testing.assert_close(
        contiguous.logits,
        uncached,
        rtol=1e-5,
        atol=1e-6,
    )


def test_paged_decode_rejects_multiple_tokens() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model)
    result = prefill_paged(
        model,
        torch.tensor([[1, 2]], dtype=torch.int64),
        manager,
        sequence_id="A",
    )

    with pytest.raises(
        ValueError,
        match="paged decode requires token_ids shaped",
    ):
        decode_paged(
            model,
            torch.tensor([[3, 4]], dtype=torch.int64),
            result.cache,
        )


def test_incompatible_pool_fails_before_registering_sequence() -> None:
    model = LlamaModel(tiny_config())
    manager = PagedKVCacheManager(
        num_layers=1,
        num_blocks=4,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        device=model.embed_tokens.weight.device,
        dtype=model.embed_tokens.weight.dtype,
    )

    with pytest.raises(
        ValueError,
        match="paged pool layer count must match",
    ):
        prefill_paged(
            model,
            torch.tensor([[1, 2]], dtype=torch.int64),
            manager,
            sequence_id="A",
        )

    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0


def test_failed_prefill_releases_sequence_blocks() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model, num_blocks=1, block_size=2)

    with pytest.raises(
        RuntimeError,
        match="not enough free physical KV blocks remain",
    ):
        prefill_paged(
            model,
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            manager,
            sequence_id="A",
        )

    assert manager.active_sequence_ids == ()
    assert manager.allocator.free_count == 1


def test_paged_cache_dtype_matches_float64_model() -> None:
    model = LlamaModel(tiny_config()).to(dtype=torch.float64)
    manager = create_manager(model)
    result = prefill_paged(
        model,
        torch.tensor([[1, 2]], dtype=torch.int64),
        manager,
        sequence_id="A",
    )

    assert result.cache.storage.dtype == torch.float64
    keys, values = result.cache.read_layer(0)
    assert keys.dtype == torch.float64
    assert values.dtype == torch.float64
