"""Fixed-capacity physical tensor storage for paged K/V blocks."""

import torch


class PagedKVStorage:
    """Store per-layer K/V values in shared physical blocks."""

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
        dimensions = {
            "num_layers": num_layers,
            "num_blocks": num_blocks,
            "num_key_value_heads": num_key_value_heads,
            "block_size": block_size,
            "head_dim": head_dim,
        }

        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_key_value_heads = num_key_value_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        storage_shape = (
            num_layers,
            num_blocks,
            num_key_value_heads,
            block_size,
            head_dim,
        )
        self._key_storage = torch.empty(
            storage_shape,
            device=device,
            dtype=dtype,
        )
        self._value_storage = torch.empty(
            storage_shape,
            device=device,
            dtype=dtype,
        )

    @property
    def storage_shape(self) -> tuple[int, ...]:
        """Return the shape of each physical storage tensor."""

        return tuple(self._key_storage.shape)

    def write_token(
        self,
        *,
        layer_index: int,
        block_id: int,
        offset: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write one token's K/V values shaped (Hkv, Dh)."""

        self._validate_location(layer_index, block_id, offset)

        expected_shape = (self.num_key_value_heads, self.head_dim)

        if keys.shape != expected_shape or values.shape != expected_shape:
            raise ValueError("keys and values must have shape (Hkv, Dh)")

        if keys.device != self.device or values.device != self.device:
            raise ValueError("keys and values must match the storage device")

        if keys.dtype != self.dtype or values.dtype != self.dtype:
            raise ValueError("keys and values must match the storage dtype")

        self._key_storage[layer_index, block_id, :, offset, :].copy_(keys)
        self._value_storage[layer_index, block_id, :, offset, :].copy_(values)

    def read_token(
        self,
        *,
        layer_index: int,
        block_id: int,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return copies of one physical token slot shaped (Hkv, Dh)."""

        self._validate_location(layer_index, block_id, offset)

        keys = self._key_storage[layer_index, block_id, :, offset, :].clone()
        values = self._value_storage[layer_index, block_id, :, offset, :].clone()
        return keys, values

    def _validate_location(
        self,
        layer_index: int,
        block_id: int,
        offset: int,
    ) -> None:
        """Reject indices outside the allocated physical storage."""

        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError("layer_index is out of range")

        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError("block_id is out of range")

        if offset < 0 or offset >= self.block_size:
            raise IndexError("offset is out of range")
