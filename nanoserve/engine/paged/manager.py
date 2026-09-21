"""Shared physical KV pool for multiple independent sequences."""

import torch

from nanoserve.engine.paged.allocator import PhysicalBlockAllocator
from nanoserve.engine.paged.sequence_cache import SequencePagedKVCache
from nanoserve.engine.paged.storage import PagedKVStorage


class PagedKVCacheManager:
    """Manage sequence caches sharing one allocator and K/V storage pool."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        num_key_value_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.allocator = PhysicalBlockAllocator(num_blocks)
        self.storage = PagedKVStorage(
            num_layers=num_layers,
            num_blocks=num_blocks,
            num_key_value_heads=num_key_value_heads,
            block_size=block_size,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self._sequences: dict[str, SequencePagedKVCache] = {}

    @property
    def active_sequence_ids(self) -> tuple[str, ...]:
        """Return active sequence IDs in creation order."""

        return tuple(self._sequences)

    def create_sequence(
        self,
        sequence_id: str,
        *,
        max_sequence_length: int,
    ) -> SequencePagedKVCache:
        """Register a new sequence that uses the shared physical pool."""

        if not sequence_id:
            raise ValueError("sequence_id must be nonempty")

        if sequence_id in self._sequences:
            raise ValueError("sequence_id is already active")

        cache = SequencePagedKVCache(
            allocator=self.allocator,
            storage=self.storage,
            max_sequence_length=max_sequence_length,
        )
        self._sequences[sequence_id] = cache
        return cache

    def get_sequence(self, sequence_id: str) -> SequencePagedKVCache:
        """Return an active sequence cache."""

        try:
            return self._sequences[sequence_id]
        except KeyError as error:
            raise KeyError("unknown sequence_id") from error

    def release_sequence(self, sequence_id: str) -> None:
        """Release a sequence's blocks and remove its registration."""

        cache = self.get_sequence(sequence_id)
        cache.release()
        del self._sequences[sequence_id]
