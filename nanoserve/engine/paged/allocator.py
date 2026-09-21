"""Deterministic physical block allocation for paged KV caching."""

import heapq
from collections.abc import Iterable


class BlockExhaustedError(RuntimeError):
    """Raised when the physical block pool cannot satisfy an allocation."""


class PhysicalBlockAllocator:
    """Manage ownership of a finite collection of physical block IDs."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        self._num_blocks = num_blocks
        self._free_blocks = list(range(num_blocks))
        self._allocated_blocks: set[int] = set()
        heapq.heapify(self._free_blocks)

    @property
    def num_blocks(self) -> int:
        """Return the total number of physical blocks."""

        return self._num_blocks

    @property
    def free_count(self) -> int:
        """Return the number of currently free blocks."""

        return len(self._free_blocks)

    @property
    def allocated_count(self) -> int:
        """Return the number of currently allocated blocks."""

        return len(self._allocated_blocks)

    @property
    def allocated_blocks(self) -> frozenset[int]:
        """Return an immutable snapshot of allocated block IDs."""

        return frozenset(self._allocated_blocks)

    def allocate(self) -> int:
        """Allocate and return the lowest available physical block ID."""

        if not self._free_blocks:
            raise BlockExhaustedError("no free physical KV blocks remain")

        block_id = heapq.heappop(self._free_blocks)
        self._allocated_blocks.add(block_id)
        return block_id

    def allocate_many(self, count: int) -> tuple[int, ...]:
        """Atomically allocate several physical blocks."""

        if count < 1:
            raise ValueError("count must be positive")

        if count > self.free_count:
            raise BlockExhaustedError("not enough free physical KV blocks remain")

        return tuple(self.allocate() for _ in range(count))

    def release(self, block_id: int) -> None:
        """Release one currently allocated physical block."""

        self._validate_block_id(block_id)

        if block_id not in self._allocated_blocks:
            raise ValueError("physical block is not currently allocated")

        self._allocated_blocks.remove(block_id)
        heapq.heappush(self._free_blocks, block_id)

    def release_many(self, block_ids: Iterable[int]) -> None:
        """Atomically validate and release several physical blocks."""

        released_ids = tuple(block_ids)

        if len(set(released_ids)) != len(released_ids):
            raise ValueError("block_ids must not contain duplicates")

        for block_id in released_ids:
            self._validate_block_id(block_id)

            if block_id not in self._allocated_blocks:
                raise ValueError("physical block is not currently allocated")

        for block_id in released_ids:
            self.release(block_id)

    def _validate_block_id(self, block_id: int) -> None:
        """Validate that an ID belongs to this allocator."""

        if block_id < 0 or block_id >= self._num_blocks:
            raise ValueError("physical block ID is out of range")
