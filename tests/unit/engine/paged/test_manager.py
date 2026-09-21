import pytest
import torch

from nanoserve.engine.paged.manager import PagedKVCacheManager
from nanoserve.engine.paged.sequence_cache import SequencePagedKVCache


def create_manager(
    *,
    num_blocks: int = 8,
    block_size: int = 2,
) -> PagedKVCacheManager:
    """Create a small shared CPU K/V pool."""

    return PagedKVCacheManager(
        num_layers=2,
        num_blocks=num_blocks,
        num_key_value_heads=2,
        block_size=block_size,
        head_dim=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def append_tokens(
    cache: SequencePagedKVCache,
    token_values: list[float],
) -> None:
    """Write distinct values for each token and decoder layer."""

    cache.begin_append(len(token_values))

    for layer_index in range(cache.storage.num_layers):
        keys = torch.tensor(token_values).view(1, -1, 1).expand(2, -1, 3)
        keys = keys + layer_index * 100.0
        values = keys + 1000.0
        cache.write_layer(layer_index, keys, values)

    cache.finish_append()


def owned_blocks(manager: PagedKVCacheManager) -> list[int]:
    """Collect block IDs owned by all active sequences."""

    return [
        block_id
        for sequence_id in manager.active_sequence_ids
        for block_id in manager.get_sequence(sequence_id).block_ids
    ]


def test_new_manager_has_no_sequences_and_all_blocks_free() -> None:
    manager = create_manager(num_blocks=8)

    assert manager.active_sequence_ids == ()
    assert manager.allocator.free_count == 8
    assert manager.allocator.allocated_count == 0


def test_sequences_share_pool_but_have_distinct_tables() -> None:
    manager = create_manager()
    sequence_a = manager.create_sequence("A", max_sequence_length=8)
    sequence_b = manager.create_sequence("B", max_sequence_length=8)

    assert sequence_a is not sequence_b
    assert sequence_a.allocator is sequence_b.allocator
    assert sequence_a.storage is sequence_b.storage
    assert sequence_a.block_table is not sequence_b.block_table
    assert manager.active_sequence_ids == ("A", "B")


def test_three_sequences_never_own_same_block_simultaneously() -> None:
    manager = create_manager(num_blocks=8)
    sequence_a = manager.create_sequence("A", max_sequence_length=8)
    sequence_b = manager.create_sequence("B", max_sequence_length=8)
    sequence_c = manager.create_sequence("C", max_sequence_length=8)

    append_tokens(sequence_a, [1.0, 2.0, 3.0])
    append_tokens(sequence_b, [4.0, 5.0])
    append_tokens(sequence_c, [6.0, 7.0, 8.0])

    all_blocks = owned_blocks(manager)

    assert len(all_blocks) == len(set(all_blocks))
    assert manager.allocator.allocated_count == len(all_blocks)
    assert manager.allocator.free_count == 8 - len(all_blocks)


def test_releasing_a_preserves_b_and_c_data() -> None:
    manager = create_manager(num_blocks=8)
    sequence_a = manager.create_sequence("A", max_sequence_length=8)
    sequence_b = manager.create_sequence("B", max_sequence_length=8)
    sequence_c = manager.create_sequence("C", max_sequence_length=8)

    append_tokens(sequence_a, [1.0, 2.0, 3.0])
    append_tokens(sequence_b, [4.0, 5.0])
    append_tokens(sequence_c, [6.0, 7.0])

    b_keys_before, b_values_before = sequence_b.read_layer(0)
    c_keys_before, c_values_before = sequence_c.read_layer(1)
    b_keys_before = b_keys_before.clone()
    b_values_before = b_values_before.clone()
    c_keys_before = c_keys_before.clone()
    c_values_before = c_values_before.clone()

    a_blocks = set(sequence_a.block_ids)
    manager.release_sequence("A")

    assert "A" not in manager.active_sequence_ids
    assert a_blocks.isdisjoint(set(owned_blocks(manager)))

    b_keys_after, b_values_after = sequence_b.read_layer(0)
    c_keys_after, c_values_after = sequence_c.read_layer(1)

    torch.testing.assert_close(b_keys_after, b_keys_before)
    torch.testing.assert_close(b_values_after, b_values_before)
    torch.testing.assert_close(c_keys_after, c_keys_before)
    torch.testing.assert_close(c_values_after, c_values_before)


def test_new_sequence_reuses_released_blocks_without_corrupting_others() -> None:
    manager = create_manager(num_blocks=8)
    sequence_a = manager.create_sequence("A", max_sequence_length=8)
    sequence_b = manager.create_sequence("B", max_sequence_length=8)
    sequence_c = manager.create_sequence("C", max_sequence_length=8)

    append_tokens(sequence_a, [1.0, 2.0, 3.0])
    append_tokens(sequence_b, [4.0, 5.0])
    append_tokens(sequence_c, [6.0, 7.0])

    released_blocks = set(sequence_a.block_ids)
    b_keys_before, _ = sequence_b.read_layer(0)
    c_keys_before, _ = sequence_c.read_layer(0)
    b_keys_before = b_keys_before.clone()
    c_keys_before = c_keys_before.clone()

    manager.release_sequence("A")
    sequence_d = manager.create_sequence("D", max_sequence_length=8)
    append_tokens(sequence_d, [9.0, 10.0, 11.0])

    assert set(sequence_d.block_ids) == released_blocks
    assert len(owned_blocks(manager)) == len(set(owned_blocks(manager)))

    d_keys, _ = sequence_d.read_layer(0)
    expected_d = torch.tensor([9.0, 10.0, 11.0]).view(1, 1, 3, 1)
    torch.testing.assert_close(d_keys, expected_d.expand(1, 2, 3, 3))

    b_keys_after, _ = sequence_b.read_layer(0)
    c_keys_after, _ = sequence_c.read_layer(0)
    torch.testing.assert_close(b_keys_after, b_keys_before)
    torch.testing.assert_close(c_keys_after, c_keys_before)


def test_reused_block_does_not_expose_stale_positions() -> None:
    manager = create_manager(num_blocks=2, block_size=4)
    sequence_a = manager.create_sequence("A", max_sequence_length=4)
    append_tokens(sequence_a, [1.0, 2.0, 3.0, 4.0])
    released_block = sequence_a.block_ids[0]

    manager.release_sequence("A")

    sequence_b = manager.create_sequence("B", max_sequence_length=4)
    assert sequence_b.current_length == 0

    empty_keys, empty_values = sequence_b.read_layer(0)
    assert empty_keys.shape[2] == 0
    assert empty_values.shape[2] == 0

    append_tokens(sequence_b, [9.0])

    assert sequence_b.block_ids == (released_block,)
    keys, values = sequence_b.read_layer(0)
    assert keys.shape == (1, 2, 1, 3)
    assert values.shape == (1, 2, 1, 3)
    torch.testing.assert_close(keys, torch.full((1, 2, 1, 3), 9.0))
    torch.testing.assert_close(values, torch.full((1, 2, 1, 3), 1009.0))


def test_released_sequence_object_cannot_access_reused_blocks() -> None:
    manager = create_manager(num_blocks=2)
    sequence_a = manager.create_sequence("A", max_sequence_length=4)
    append_tokens(sequence_a, [1.0])

    manager.release_sequence("A")

    with pytest.raises(RuntimeError, match="sequence cache has been released"):
        sequence_a.read_layer(0)


def test_duplicate_sequence_id_is_rejected_without_changing_state() -> None:
    manager = create_manager()
    original = manager.create_sequence("A", max_sequence_length=8)

    with pytest.raises(ValueError, match="sequence_id is already active"):
        manager.create_sequence("A", max_sequence_length=8)

    assert manager.get_sequence("A") is original
    assert manager.active_sequence_ids == ("A",)


def test_unknown_sequence_id_is_rejected() -> None:
    manager = create_manager()

    with pytest.raises(KeyError, match="unknown sequence_id"):
        manager.get_sequence("missing")

    with pytest.raises(KeyError, match="unknown sequence_id"):
        manager.release_sequence("missing")


def test_empty_sequence_id_is_rejected() -> None:
    manager = create_manager()

    with pytest.raises(ValueError, match="sequence_id must be nonempty"):
        manager.create_sequence("", max_sequence_length=8)


def test_release_restores_allocator_accounting() -> None:
    manager = create_manager(num_blocks=8)
    sequence_a = manager.create_sequence("A", max_sequence_length=8)
    sequence_b = manager.create_sequence("B", max_sequence_length=8)

    append_tokens(sequence_a, [1.0, 2.0, 3.0])
    append_tokens(sequence_b, [4.0, 5.0, 6.0])

    manager.release_sequence("A")
    manager.release_sequence("B")

    assert manager.active_sequence_ids == ()
    assert manager.allocator.free_count == 8
    assert manager.allocator.allocated_count == 0
