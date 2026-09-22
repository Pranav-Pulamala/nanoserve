import pytest

from tokserve.engine.paged.allocator import (
    BlockExhaustedError,
    PhysicalBlockAllocator,
)


def test_fresh_allocator_reports_all_blocks_free() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=4)

    assert allocator.num_blocks == 4
    assert allocator.free_count == 4
    assert allocator.allocated_count == 0
    assert allocator.allocated_blocks == frozenset()


def test_allocate_returns_deterministic_block_ids() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=4)

    first = allocator.allocate()
    second = allocator.allocate()
    third = allocator.allocate()

    assert (first, second, third) == (0, 1, 2)
    assert allocator.free_count == 1
    assert allocator.allocated_count == 3
    assert allocator.allocated_blocks == frozenset({0, 1, 2})


def test_allocate_many_returns_unique_blocks() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=5)

    block_ids = allocator.allocate_many(4)

    assert block_ids == (0, 1, 2, 3)
    assert len(block_ids) == len(set(block_ids))
    assert allocator.free_count == 1
    assert allocator.allocated_count == 4


def test_allocator_raises_clear_exhaustion_error() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=2)
    allocator.allocate_many(2)

    with pytest.raises(
        BlockExhaustedError,
        match="no free physical KV blocks remain",
    ):
        allocator.allocate()

    assert allocator.free_count == 0
    assert allocator.allocated_count == 2


def test_failed_multi_block_allocation_is_atomic() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=3)
    first = allocator.allocate()

    with pytest.raises(
        BlockExhaustedError,
        match="not enough free physical KV blocks remain",
    ):
        allocator.allocate_many(3)

    assert first == 0
    assert allocator.free_count == 2
    assert allocator.allocated_count == 1
    assert allocator.allocated_blocks == frozenset({0})


def test_release_returns_block_to_free_pool() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=3)
    first = allocator.allocate()
    allocator.allocate()

    allocator.release(first)

    assert allocator.free_count == 2
    assert allocator.allocated_count == 1
    assert allocator.allocated_blocks == frozenset({1})


def test_released_block_is_reused_deterministically() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=4)
    allocated = allocator.allocate_many(3)
    allocator.release(allocated[1])

    reused = allocator.allocate()

    assert reused == 1
    assert allocator.allocated_blocks == frozenset({0, 1, 2})


def test_double_release_is_rejected() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=2)
    block_id = allocator.allocate()
    allocator.release(block_id)

    with pytest.raises(
        ValueError,
        match="physical block is not currently allocated",
    ):
        allocator.release(block_id)

    assert allocator.free_count == 2
    assert allocator.allocated_count == 0


@pytest.mark.parametrize("block_id", [-1, 3, 100])
def test_invalid_block_release_is_rejected(block_id: int) -> None:
    allocator = PhysicalBlockAllocator(num_blocks=3)

    with pytest.raises(
        ValueError,
        match="physical block ID is out of range",
    ):
        allocator.release(block_id)

    assert allocator.free_count == 3
    assert allocator.allocated_count == 0


def test_release_many_returns_every_block() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=5)
    blocks = allocator.allocate_many(4)

    allocator.release_many((blocks[1], blocks[3]))

    assert allocator.free_count == 3
    assert allocator.allocated_count == 2
    assert allocator.allocated_blocks == frozenset({0, 2})


def test_failed_release_many_is_atomic() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=4)
    blocks = allocator.allocate_many(2)

    with pytest.raises(
        ValueError,
        match="physical block is not currently allocated",
    ):
        allocator.release_many((blocks[0], 2))

    assert allocator.free_count == 2
    assert allocator.allocated_count == 2
    assert allocator.allocated_blocks == frozenset({0, 1})


def test_release_many_rejects_duplicate_ids() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=3)
    block_id = allocator.allocate()

    with pytest.raises(
        ValueError,
        match="block_ids must not contain duplicates",
    ):
        allocator.release_many((block_id, block_id))

    assert allocator.free_count == 2
    assert allocator.allocated_count == 1


def test_no_blocks_are_simultaneously_allocated_twice() -> None:
    allocator = PhysicalBlockAllocator(num_blocks=8)
    first_group = allocator.allocate_many(4)
    second_group = allocator.allocate_many(4)

    all_blocks = first_group + second_group

    assert len(all_blocks) == len(set(all_blocks))
    assert set(all_blocks) == set(range(8))


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_allocator_requires_positive_block_count(num_blocks: int) -> None:
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        PhysicalBlockAllocator(num_blocks=num_blocks)


@pytest.mark.parametrize("count", [0, -1])
def test_allocate_many_requires_positive_count(count: int) -> None:
    allocator = PhysicalBlockAllocator(num_blocks=3)

    with pytest.raises(ValueError, match="count must be positive"):
        allocator.allocate_many(count)
