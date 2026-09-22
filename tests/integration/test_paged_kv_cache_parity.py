import pytest
import torch

from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.allocator import BlockExhaustedError
from tokserve.engine.paged.inference import decode_paged, prefill_paged
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.engine.paged.sequence_cache import SequencePagedKVCache
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.generate import generate
from tokserve.generation.paged_generate import generate_with_paged_cache
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a deterministic, CPU-friendly model configuration."""

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
    num_blocks: int = 8,
    block_size: int = 2,
) -> PagedKVCacheManager:
    """Create one physical pool compatible with the model."""

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


def append_distinct_values(
    cache: SequencePagedKVCache,
    token_values: list[float],
) -> None:
    """Write distinct values to every layer of a sequence cache."""

    cache.begin_append(len(token_values))

    for layer_index in range(cache.storage.num_layers):
        keys = torch.tensor(
            token_values,
            dtype=cache.storage.dtype,
            device=cache.storage.device,
        ).view(1, -1, 1)
        keys = keys.expand(
            cache.storage.num_key_value_heads,
            -1,
            cache.storage.head_dim,
        )
        keys = keys + layer_index * 100.0
        values = keys + 1000.0
        cache.write_layer(layer_index, keys, values)

    cache.finish_append()


def test_pool_exhaustion_preserves_existing_sequences() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model, num_blocks=2, block_size=2)
    sequence_a = manager.create_sequence("A", max_sequence_length=8)
    sequence_b = manager.create_sequence("B", max_sequence_length=8)

    append_distinct_values(sequence_a, [1.0, 2.0])
    append_distinct_values(sequence_b, [3.0, 4.0])

    a_keys_before, _ = sequence_a.read_layer(0)
    b_keys_before, _ = sequence_b.read_layer(0)
    a_keys_before = a_keys_before.clone()
    b_keys_before = b_keys_before.clone()

    with pytest.raises(BlockExhaustedError):
        sequence_a.begin_append(1)

    assert sequence_a.current_length == 2
    assert sequence_b.current_length == 2
    assert sequence_a.block_ids == (0,)
    assert sequence_b.block_ids == (1,)
    assert manager.allocator.free_count == 0
    assert manager.allocator.allocated_blocks == frozenset({0, 1})

    a_keys_after, _ = sequence_a.read_layer(0)
    b_keys_after, _ = sequence_b.read_layer(0)
    torch.testing.assert_close(a_keys_after, a_keys_before)
    torch.testing.assert_close(b_keys_after, b_keys_before)


def test_nonadjacent_physical_blocks_reconstruct_logical_order() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model, num_blocks=4, block_size=2)
    temporary = manager.create_sequence("temporary", max_sequence_length=8)
    target = manager.create_sequence("target", max_sequence_length=8)
    other = manager.create_sequence("other", max_sequence_length=8)

    append_distinct_values(temporary, [50.0, 51.0])
    append_distinct_values(target, [1.0, 2.0])
    append_distinct_values(other, [60.0, 61.0])

    assert temporary.block_ids == (0,)
    assert target.block_ids == (1,)
    assert other.block_ids == (2,)

    other_keys_before, _ = other.read_layer(0)
    other_keys_before = other_keys_before.clone()
    manager.release_sequence("temporary")

    append_distinct_values(target, [3.0, 4.0])

    assert target.block_ids == (1, 0)

    target_keys, _ = target.read_layer(0)
    assert target_keys.shape == (1, 2, 4, model.config.head_dim)

    for position, number in enumerate((1.0, 2.0, 3.0, 4.0)):
        torch.testing.assert_close(
            target_keys[0, :, position, :],
            torch.full(
                (model.config.num_key_value_heads, model.config.head_dim),
                number,
            ),
        )

    other_keys_after, _ = other.read_layer(0)
    torch.testing.assert_close(other_keys_after, other_keys_before)


def test_released_block_reuse_does_not_expose_stale_tokens() -> None:
    model = LlamaModel(tiny_config())
    manager = create_manager(model, num_blocks=1, block_size=4)
    sequence_a = manager.create_sequence("A", max_sequence_length=4)
    append_distinct_values(sequence_a, [1.0, 2.0, 3.0, 4.0])
    old_block_id = sequence_a.block_ids[0]

    manager.release_sequence("A")

    sequence_b = manager.create_sequence("B", max_sequence_length=4)
    empty_keys, empty_values = sequence_b.read_layer(0)

    assert empty_keys.shape[2] == 0
    assert empty_values.shape[2] == 0

    append_distinct_values(sequence_b, [9.0])

    assert sequence_b.block_ids == (old_block_id,)
    assert sequence_b.current_length == 1

    keys, values = sequence_b.read_layer(0)
    assert keys.shape[2] == 1
    assert values.shape[2] == 1
    torch.testing.assert_close(
        keys,
        torch.full((1, 2, 1, model.config.head_dim), 9.0),
    )
    torch.testing.assert_close(
        values,
        torch.full((1, 2, 1, model.config.head_dim), 1009.0),
    )


def test_prefill_and_multiple_decode_steps_match_both_baselines() -> None:
    torch.manual_seed(90)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model, block_size=2)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    continuation = (4, 5, 6)

    contiguous_result = prefill(model, prompt)
    paged_result = prefill_paged(
        model,
        prompt,
        manager,
        sequence_id="parity",
    )
    uncached_prompt_logits = model(prompt)

    torch.testing.assert_close(
        paged_result.logits,
        contiguous_result.logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        paged_result.logits,
        uncached_prompt_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert paged_result.cache.current_length == 3
    assert paged_result.cache.num_blocks == 2

    complete_ids = prompt.clone()

    for expected_length, token_id in enumerate(continuation, start=4):
        new_token = torch.tensor([[token_id]], dtype=torch.int64)
        contiguous_logits = decode(
            model,
            new_token,
            contiguous_result.cache,
        )
        paged_logits = decode_paged(
            model,
            new_token,
            paged_result.cache,
        )
        complete_ids = torch.cat((complete_ids, new_token), dim=1)
        uncached_logits = model(complete_ids)[:, -1:, :]

        torch.testing.assert_close(
            paged_logits,
            contiguous_logits,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            paged_logits,
            uncached_logits,
            rtol=1e-5,
            atol=1e-6,
        )
        assert paged_result.cache.current_length == expected_length

    assert paged_result.cache.num_blocks == 3


def test_greedy_generation_matches_uncached_and_contiguous() -> None:
    torch.manual_seed(91)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model)
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.int64)
    config = GenerationConfig(max_new_tokens=5)

    uncached = generate(model, prompt, config)
    contiguous = generate_with_cache(model, prompt, config)
    paged = generate_with_paged_cache(
        model,
        prompt,
        config,
        manager,
        sequence_id="greedy-parity",
    )

    assert torch.equal(paged.token_ids, uncached.token_ids)
    assert torch.equal(paged.token_ids, contiguous.token_ids)
    assert paged.stop_reason == uncached.stop_reason
    assert paged.stop_reason == contiguous.stop_reason
    assert manager.active_sequence_ids == ()
    assert manager.allocator.free_count == manager.allocator.num_blocks


def test_paged_decode_projects_only_new_tokens() -> None:
    torch.manual_seed(92)
    model = LlamaModel(tiny_config())
    model.eval()
    manager = create_manager(model)
    attention = model.layers[0].self_attention
    projected_lengths: dict[str, list[int]] = {
        "query": [],
        "key": [],
        "value": [],
    }

    def record_query(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths["query"].append(inputs[0].shape[1])

    def record_key(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths["key"].append(inputs[0].shape[1])

    def record_value(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_lengths["value"].append(inputs[0].shape[1])

    handles = [
        attention.q_proj.register_forward_hook(record_query),
        attention.k_proj.register_forward_hook(record_key),
        attention.v_proj.register_forward_hook(record_value),
    ]

    try:
        result = prefill_paged(
            model,
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            manager,
            sequence_id="projection-proof",
        )
        decode_paged(
            model,
            torch.tensor([[4]], dtype=torch.int64),
            result.cache,
        )
        decode_paged(
            model,
            torch.tensor([[5]], dtype=torch.int64),
            result.cache,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert projected_lengths == {
        "query": [3, 1, 1],
        "key": [3, 1, 1],
        "value": [3, 1, 1],
    }


def test_cpu_pool_preserves_device_dtype_and_hkv_shape() -> None:
    torch.manual_seed(93)
    model = LlamaModel(tiny_config()).to(dtype=torch.float64)
    manager = create_manager(model, block_size=2)
    result = prefill_paged(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
        manager,
        sequence_id="device-dtype",
    )

    assert manager.storage.device.type == "cpu"
    assert manager.storage.dtype == torch.float64
    assert manager.storage.storage_shape == (
        model.config.num_hidden_layers,
        manager.allocator.num_blocks,
        model.config.num_key_value_heads,
        2,
        model.config.head_dim,
    )

    for layer_index in range(model.config.num_hidden_layers):
        keys, values = result.cache.read_layer(layer_index)
        assert keys.device.type == "cpu"
        assert values.device.type == "cpu"
        assert keys.dtype == torch.float64
        assert values.dtype == torch.float64


@pytest.mark.gpu
def test_paged_prefill_and_decode_on_mps_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    torch.manual_seed(94)
    device = torch.device("mps")
    model = LlamaModel(tiny_config()).to(device)
    model.eval()
    manager = create_manager(model, block_size=2)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64, device=device)
    next_token = torch.tensor([[4]], dtype=torch.int64, device=device)

    result = prefill_paged(
        model,
        prompt,
        manager,
        sequence_id="mps",
    )
    logits = decode_paged(model, next_token, result.cache)

    assert result.cache.storage.device.type == "mps"
    assert logits.device.type == "mps"
    assert result.cache.current_length == 4

    complete = torch.cat((prompt, next_token), dim=1)
    expected = model(complete)[:, -1:, :].cpu()
    actual = logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
