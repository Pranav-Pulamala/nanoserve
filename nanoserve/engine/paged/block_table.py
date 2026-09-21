"""Per-sequence logical-to-physical KV block mapping."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalTokenLocation:
    """Physical block ID and slot for one logical token position."""

    block_id: int
    offset: int


class SequenceBlockTable:
    """Map a sequence's logical token blocks to physical block IDs."""

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self._block_ids: list[int] = []

    @property
    def block_ids(self) -> tuple[int, ...]:
        """Return physical block IDs in logical sequence order."""

        return tuple(self._block_ids)

    @property
    def num_blocks(self) -> int:
        """Return the number of mapped logical blocks."""

        return len(self._block_ids)

    @property
    def capacity(self) -> int:
        """Return the token capacity represented by mapped blocks."""

        return self.num_blocks * self.block_size

    def append_block(self, physical_block_id: int) -> None:
        """Map the next logical block to a physical block ID."""

        if physical_block_id < 0:
            raise ValueError("physical_block_id must be nonnegative")

        if physical_block_id in self._block_ids:
            raise ValueError("physical block is already in this block table")

        self._block_ids.append(physical_block_id)

    def resolve(self, position: int) -> PhysicalTokenLocation:
        """Locate a token position within the mapped physical blocks."""

        if position < 0:
            raise ValueError("position must be nonnegative")

        logical_block = position // self.block_size

        if logical_block >= self.num_blocks:
            raise IndexError("position exceeds block table capacity")

        return PhysicalTokenLocation(
            block_id=self._block_ids[logical_block],
            offset=position % self.block_size,
        )

    def clear(self) -> tuple[int, ...]:
        """Remove all mappings and return their physical block IDs."""

        released_ids = self.block_ids
        self._block_ids.clear()
        return released_ids
