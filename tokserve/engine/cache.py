"""Preallocated contiguous key/value cache data structures."""

import torch


class LayerKVCache:
    """Preallocated key and value storage for one decoder layer."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_key_value_heads: int,
        max_sequence_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if num_key_value_heads < 1:
            raise ValueError("num_key_value_heads must be positive")

        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")

        if head_dim < 1:
            raise ValueError("head_dim must be positive")

        self.batch_size = batch_size
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self._current_length = 0

        storage_shape = (
            batch_size,
            num_key_value_heads,
            max_sequence_length,
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
    def current_length(self) -> int:
        """Return the number of valid cached sequence positions."""

        return self._current_length

    @property
    def remaining_capacity(self) -> int:
        """Return the number of additional positions that can be stored."""

        return self.max_sequence_length - self._current_length

    @property
    def storage_shape(self) -> tuple[int, ...]:
        """Return the shape of the allocated K/V storage."""

        return tuple(self._key_storage.shape)

    @property
    def keys(self) -> torch.Tensor:
        """Return a view containing only valid cached keys."""

        return self._key_storage[:, :, : self._current_length, :]

    @property
    def values(self) -> torch.Tensor:
        """Return a view containing only valid cached values."""

        return self._value_storage[:, :, : self._current_length, :]

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append K/V tensors shaped (B, Hkv, T, Dh)."""

        self._validate_append(keys, values)

        appended_length = keys.shape[2]
        start = self._current_length
        end = start + appended_length

        if end > self.max_sequence_length:
            raise ValueError("KV cache capacity exceeded")

        self._key_storage[:, :, start:end, :].copy_(keys)
        self._value_storage[:, :, start:end, :].copy_(values)
        self._current_length = end

        return self.keys, self.values

    def reset(self) -> None:
        """Logically empty the cache without reallocating storage."""

        self._current_length = 0

    def _validate_append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Validate tensors before writing them into storage."""

        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have shape (B, Hkv, T, Dh)")

        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes")

        expected_prefix = (
            self.batch_size,
            self.num_key_value_heads,
        )

        if keys.shape[:2] != expected_prefix:
            raise ValueError(
                "keys and values must match configured batch and head dimensions"
            )

        if keys.shape[2] < 1:
            raise ValueError("appended sequence length must be positive")

        if keys.shape[3] != self.head_dim:
            raise ValueError("keys and values must match configured head_dim")

        if keys.device != values.device:
            raise ValueError("keys and values must use the same device")

        if keys.device != self.device:
            raise ValueError("keys and values must match the cache device")

        if keys.dtype != values.dtype:
            raise ValueError("keys and values must use the same dtype")

        if keys.dtype != self.dtype:
            raise ValueError("keys and values must match the cache dtype")


class KVCache:
    """Own one preallocated layer cache per decoder layer."""

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_key_value_heads: int,
        max_sequence_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.layers = tuple(
            LayerKVCache(
                batch_size=batch_size,
                num_key_value_heads=num_key_value_heads,
                max_sequence_length=max_sequence_length,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )

    def __len__(self) -> int:
        """Return the number of layer caches."""

        return len(self.layers)

    def __getitem__(self, layer_index: int) -> LayerKVCache:
        """Return one decoder layer's cache."""

        return self.layers[layer_index]

    @property
    def current_length(self) -> int:
        """Return the shared valid length across every layer cache."""

        lengths = {layer.current_length for layer in self.layers}

        if len(lengths) != 1:
            raise RuntimeError("layer caches have inconsistent sequence lengths")

        return next(iter(lengths))

    def reset(self) -> None:
        """Logically empty every decoder-layer cache."""

        for layer in self.layers:
            layer.reset()
