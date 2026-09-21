import pytest
import torch

from nanoserve.engine.paged.allocator import (
    BlockExhaustedError,
    PhysicalBlockAllocator,
)
from nanoserve.engine.paged.sequence_cache import SequencePagedKVCache
from nanoserve.engine.paged.storage import PagedKVStorage


def create_cache(
    *,
    num_blocks: int = 5,
    block_size: int = 4,
    max_sequence_length: int = 20,
) -> tuple[SequencePagedKVCache, PhysicalBlockAllocator]:
    """Create one sequence cache over a small shared physical pool."""

    allocator = PhysicalBlockAllocator(num_blocks=num_blocks)
    storage = PagedKVStorage(
        num_layers=2,
        num_blocks=num_blocks,
        num_key_value_heads=2,
        block_size=block_size,
        head_dim=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = SequencePagedKVCache(
        allocator=allocator,
        storage=storage,
        max_sequence_length=max_sequence_length,
    )
    return cache, allocator


def append_tokens(
    cache: SequencePagedKVCache,
    values: list[float],
) -> None:
    """Append distinguishable token values to every layer."""

    cache.begin_append(len(values))

    for layer_index in range(cache.storage.num_layers):
        keys = (
            torch.tensor(
                values,
                dtype=torch.float32,
            )
            .view(1, len(values), 1)
            .expand(2, -1, 3)
        )
        keys = keys + layer_index * 100.0
        layer_values = keys + 1000.0
        cache.write_layer(layer_index, keys, layer_values)

    cache.finish_append()


def test_sequence_cache_starts_empty_without_blocks() -> None:
    cache, allocator = create_cache()

    assert cache.current_length == 0
    assert cache.num_blocks == 0
    assert cache.block_ids == ()
    assert allocator.free_count == 5

    keys, values = cache.read_layer(0)
    assert keys.shape == (1, 2, 0, 3)
    assert values.shape == (1, 2, 0, 3)


def test_first_token_allocates_one_block() -> None:
    cache, allocator = create_cache()

    append_tokens(cache, [1.0])

    assert cache.current_length == 1
    assert cache.num_blocks == 1
    assert cache.block_ids == (0,)
    assert allocator.free_count == 4


def test_same_block_appends_do_not_allocate_more_blocks() -> None:
    cache, allocator = create_cache()

    append_tokens(cache, [1.0])
    append_tokens(cache, [2.0])
    append_tokens(cache, [3.0])
    append_tokens(cache, [4.0])

    assert cache.current_length == 4
    assert cache.num_blocks == 1
    assert allocator.free_count == 4


def test_boundary_crossing_allocates_exactly_one_new_block() -> None:
    cache, allocator = create_cache()

    append_tokens(cache, [1.0, 2.0, 3.0, 4.0])
    assert cache.num_blocks == 1

    append_tokens(cache, [5.0])

    assert cache.current_length == 5
    assert cache.num_blocks == 2
    assert cache.block_ids == (0, 1)
    assert allocator.free_count == 3


def test_prefill_across_multiple_blocks_allocates_only_needed_blocks() -> None:
    cache, allocator = create_cache(block_size=4)

    append_tokens(
        cache,
        [float(position) for position in range(10)],
    )

    assert cache.current_length == 10
    assert cache.num_blocks == 3
    assert cache.block_ids == (0, 1, 2)
    assert allocator.free_count == 2


def test_read_reconstructs_original_logical_token_order() -> None:
    cache, _ = create_cache()

    append_tokens(cache, [1.0, 2.0, 3.0, 4.0])
    append_tokens(cache, [5.0, 6.0])

    keys, values = cache.read_layer(0)

    assert keys.shape == (1, 2, 6, 3)
    assert values.shape == (1, 2, 6, 3)

    for position, number in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        torch.testing.assert_close(
            keys[0, :, position, :],
            torch.full((2, 3), number),
        )
        torch.testing.assert_close(
            values[0, :, position, :],
            torch.full((2, 3), number + 1000.0),
        )


def test_layers_store_distinct_values_at_same_logical_positions() -> None:
    cache, _ = create_cache()
    append_tokens(cache, [1.0, 2.0])

    first_keys, _ = cache.read_layer(0)
    second_keys, _ = cache.read_layer(1)

    torch.testing.assert_close(
        second_keys,
        first_keys + 100.0,
    )


def test_pending_positions_are_hidden_until_layer_writes() -> None:
    cache, _ = create_cache()
    cache.begin_append(2)

    committed_keys, _ = cache.read_layer(0)
    assert committed_keys.shape[2] == 0

    with pytest.raises(
        RuntimeError,
        match="pending K/V for this layer have not been written",
    ):
        cache.read_layer(0, include_pending=True)

    keys = torch.ones(2, 2, 3)
    values = torch.full((2, 2, 3), 2.0)
    cache.write_layer(0, keys, values)

    visible_keys, visible_values = cache.read_layer(
        0,
        include_pending=True,
    )
    assert visible_keys.shape == (1, 2, 2, 3)
    torch.testing.assert_close(visible_keys[0], keys)
    torch.testing.assert_close(visible_values[0], values)

    still_hidden_keys, _ = cache.read_layer(1)
    assert still_hidden_keys.shape[2] == 0


def test_finish_requires_all_layers_to_write() -> None:
    cache, _ = create_cache()
    cache.begin_append(1)
    cache.write_layer(
        0,
        torch.ones(2, 1, 3),
        torch.ones(2, 1, 3),
    )

    with pytest.raises(
        RuntimeError,
        match="every layer must write",
    ):
        cache.finish_append()

    assert cache.current_length == 0


def test_failed_exhaustion_leaves_sequence_and_allocator_unchanged() -> None:
    cache, allocator = create_cache(
        num_blocks=1,
        block_size=2,
        max_sequence_length=8,
    )
    append_tokens(cache, [1.0, 2.0])

    with pytest.raises(
        BlockExhaustedError,
        match="not enough free physical KV blocks remain",
    ):
        cache.begin_append(1)

    assert cache.current_length == 2
    assert cache.num_blocks == 1
    assert allocator.free_count == 0


def test_release_restores_free_count_and_invalidates_cache() -> None:
    cache, allocator = create_cache()
    append_tokens(cache, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert allocator.free_count == 3

    cache.release()

    assert allocator.free_count == 5
    assert allocator.allocated_count == 0

    with pytest.raises(RuntimeError, match="sequence cache has been released"):
        cache.read_layer(0)

    with pytest.raises(RuntimeError, match="sequence cache has been released"):
        cache.release()


def test_sequence_rejects_excessive_length_before_allocation() -> None:
    cache, allocator = create_cache(max_sequence_length=4)

    with pytest.raises(
        ValueError,
        match="sequence exceeds max_sequence_length",
    ):
        cache.begin_append(5)

    assert allocator.free_count == 5
    assert cache.current_length == 0
