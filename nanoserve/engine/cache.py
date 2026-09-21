"""Contiguous key/value cache data structures."""

import torch


class LayerKVCache:
    """Store contiguous key and value tensors for one decoder layer."""

    def __init__(self) -> None:
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None

    @property
    def current_length(self) -> int:
        """Return the number of cached sequence positions."""

        if self._keys is None:
            return 0

        return self._keys.shape[-2]

    @property
    def keys(self) -> torch.Tensor | None:
        """Return all currently cached keys."""

        return self._keys

    @property
    def values(self) -> torch.Tensor | None:
        """Return all currently cached values."""

        return self._values

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append K/V tensors shaped (B, Hkv, T, Dh)."""

        self._validate_new_tensors(keys, values)

        existing_keys = self._keys
        existing_values = self._values

        if existing_keys is None or existing_values is None:
            self._keys = keys.clone()
            self._values = values.clone()
        else:
            self._validate_existing_tensors(
                existing_keys,
                existing_values,
                keys,
                values,
            )
            self._keys = torch.cat((existing_keys, keys), dim=-2)
            self._values = torch.cat((existing_values, values), dim=-2)

        cached_keys = self._keys
        cached_values = self._values

        if cached_keys is None or cached_values is None:
            raise RuntimeError("cache append failed")

        return cached_keys, cached_values

    def reset(self) -> None:
        """Remove all cached key and value tensors."""

        self._keys = None
        self._values = None

    @staticmethod
    def _validate_new_tensors(
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Validate one incoming K/V pair."""

        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have shape (B, Hkv, T, Dh)")

        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes")

        if keys.shape[0] < 1:
            raise ValueError("batch size must be positive")

        if keys.shape[1] < 1:
            raise ValueError("number of key/value heads must be positive")

        if keys.shape[2] < 1:
            raise ValueError("appended sequence length must be positive")

        if keys.shape[3] < 1:
            raise ValueError("head dimension must be positive")

        if keys.device != values.device:
            raise ValueError("keys and values must use the same device")

        if keys.dtype != values.dtype:
            raise ValueError("keys and values must use the same dtype")

    @staticmethod
    def _validate_existing_tensors(
        existing_keys: torch.Tensor,
        existing_values: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> None:
        """Validate incoming tensors against initialized cache tensors."""

        if existing_keys.shape != existing_values.shape:
            raise RuntimeError("cached keys and values have inconsistent shapes")

        if existing_keys.shape[:2] != new_keys.shape[:2]:
            raise ValueError(
                "new keys and values must match cached batch and head dimensions"
            )

        if existing_keys.shape[-1] != new_keys.shape[-1]:
            raise ValueError("new keys and values must match the cached head dimension")

        if existing_keys.device != new_keys.device:
            raise ValueError("new keys and values must match the cached device")

        if existing_keys.dtype != new_keys.dtype:
            raise ValueError("new keys and values must match the cached dtype")

        if new_keys.shape != new_values.shape:
            raise ValueError("new keys and values must have identical shapes")


class KVCache:
    """Own one independent key/value cache for each decoder layer."""

    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.layers = tuple(LayerKVCache() for _ in range(num_layers))

    def __len__(self) -> int:
        """Return the number of layer caches."""

        return len(self.layers)

    def __getitem__(self, layer_index: int) -> LayerKVCache:
        """Return the cache belonging to one decoder layer."""

        return self.layers[layer_index]

    @property
    def current_length(self) -> int:
        """Return the shared cached length across all layers."""

        lengths = {layer.current_length for layer in self.layers}

        if len(lengths) != 1:
            raise RuntimeError("layer caches have inconsistent sequence lengths")

        return next(iter(lengths))

    def reset(self) -> None:
        """Reset every decoder-layer cache."""

        for layer in self.layers:
            layer.reset()
