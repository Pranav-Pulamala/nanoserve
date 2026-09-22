"""One sequence's block ownership and logical paged K/V state."""

import torch

from tokserve.engine.paged.allocator import PhysicalBlockAllocator
from tokserve.engine.paged.block_table import SequenceBlockTable
from tokserve.engine.paged.storage import PagedKVStorage


class SequencePagedKVCache:
    """Manage one sequence's K/V values in a shared physical block pool."""

    def __init__(
        self,
        *,
        allocator: PhysicalBlockAllocator,
        storage: PagedKVStorage,
        max_sequence_length: int,
    ) -> None:
        if allocator.num_blocks != storage.num_blocks:
            raise ValueError("allocator and storage block counts must match")

        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")

        self.allocator = allocator
        self.storage = storage
        self.max_sequence_length = max_sequence_length
        self.block_table = SequenceBlockTable(storage.block_size)
        self._current_length = 0
        self._pending_length = 0
        self._written_layers: set[int] = set()
        self._released = False

    @property
    def current_length(self) -> int:
        """Return the number of fully written, valid token positions."""

        self._require_active()
        return self._current_length

    @property
    def num_blocks(self) -> int:
        """Return the number of physical blocks owned by this sequence."""

        self._require_active()
        return self.block_table.num_blocks

    @property
    def block_ids(self) -> tuple[int, ...]:
        """Return owned physical blocks in logical sequence order."""

        self._require_active()
        return self.block_table.block_ids

    def begin_append(self, sequence_length: int) -> None:
        """Reserve physical locations for new token positions."""

        self._require_active()

        if self._pending_length != 0:
            raise RuntimeError("an append is already in progress")

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        new_length = self._current_length + sequence_length

        if new_length > self.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")

        required_blocks = (
            new_length + self.storage.block_size - 1
        ) // self.storage.block_size
        additional_blocks = required_blocks - self.block_table.num_blocks

        if additional_blocks > 0:
            new_block_ids = self.allocator.allocate_many(additional_blocks)

            for block_id in new_block_ids:
                self.block_table.append_block(block_id)

        self._pending_length = sequence_length
        self._written_layers.clear()

    def write_layer(
        self,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write pending K/V shaped (Hkv, T, Dh) for one layer."""

        self._require_active()

        if self._pending_length == 0:
            raise RuntimeError("begin_append must be called before write_layer")

        if layer_index < 0 or layer_index >= self.storage.num_layers:
            raise IndexError("layer_index is out of range")

        if layer_index in self._written_layers:
            raise ValueError("layer has already written this append")

        expected_shape = (
            self.storage.num_key_value_heads,
            self._pending_length,
            self.storage.head_dim,
        )

        if keys.shape != expected_shape or values.shape != expected_shape:
            raise ValueError("keys and values must have shape (Hkv, T, Dh)")

        if keys.device != self.storage.device or values.device != self.storage.device:
            raise ValueError("keys and values must match the storage device")

        if keys.dtype != self.storage.dtype or values.dtype != self.storage.dtype:
            raise ValueError("keys and values must match the storage dtype")

        for token_offset in range(self._pending_length):
            absolute_position = self._current_length + token_offset
            location = self.block_table.resolve(absolute_position)
            self.storage.write_token(
                layer_index=layer_index,
                block_id=location.block_id,
                offset=location.offset,
                keys=keys[:, token_offset, :],
                values=values[:, token_offset, :],
            )

        self._written_layers.add(layer_index)

    def read_layer(
        self,
        layer_index: int,
        *,
        include_pending: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct one layer's K/V in logical token order.

        Results have shape (1, Hkv, S, Dh).
        """

        self._require_active()

        if layer_index < 0 or layer_index >= self.storage.num_layers:
            raise IndexError("layer_index is out of range")

        length = self._current_length

        if include_pending and self._pending_length != 0:
            if layer_index not in self._written_layers:
                raise RuntimeError("pending K/V for this layer have not been written")

            length += self._pending_length

        key_tokens: list[torch.Tensor] = []
        value_tokens: list[torch.Tensor] = []

        for position in range(length):
            location = self.block_table.resolve(position)
            keys, values = self.storage.read_token(
                layer_index=layer_index,
                block_id=location.block_id,
                offset=location.offset,
            )
            key_tokens.append(keys)
            value_tokens.append(values)

        if not key_tokens:
            empty_shape = (
                1,
                self.storage.num_key_value_heads,
                0,
                self.storage.head_dim,
            )
            return (
                torch.empty(
                    empty_shape,
                    device=self.storage.device,
                    dtype=self.storage.dtype,
                ),
                torch.empty(
                    empty_shape,
                    device=self.storage.device,
                    dtype=self.storage.dtype,
                ),
            )

        return (
            torch.stack(key_tokens, dim=1).unsqueeze(0),
            torch.stack(value_tokens, dim=1).unsqueeze(0),
        )

    def finish_append(self) -> None:
        """Make pending positions valid after every layer has written K/V."""

        self._require_active()

        if self._pending_length == 0:
            raise RuntimeError("no append is in progress")

        if len(self._written_layers) != self.storage.num_layers:
            raise RuntimeError("every layer must write before finishing an append")

        self._current_length += self._pending_length
        self._pending_length = 0
        self._written_layers.clear()

    def release(self) -> None:
        """Return all owned physical blocks and invalidate this sequence cache."""

        self._require_active()
        self.allocator.release_many(self.block_table.clear())
        self._current_length = 0
        self._pending_length = 0
        self._written_layers.clear()
        self._released = True

    def _require_active(self) -> None:
        """Reject use after this sequence has released its blocks."""

        if self._released:
            raise RuntimeError("sequence cache has been released")
