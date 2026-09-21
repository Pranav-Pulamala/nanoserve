import pytest
import torch

from nanoserve.engine.cache import KVCache, LayerKVCache


def test_layer_cache_is_initially_empty() -> None:
    cache = LayerKVCache()

    assert cache.current_length == 0
    assert cache.keys is None
    assert cache.values is None


def test_first_append_initializes_layer_cache() -> None:
    cache = LayerKVCache()
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
    values = keys + 100.0

    cached_keys, cached_values = cache.append(keys, values)

    assert cache.current_length == 3
    torch.testing.assert_close(cached_keys, keys)
    torch.testing.assert_close(cached_values, values)


def test_second_append_preserves_exact_tensor_contents() -> None:
    cache = LayerKVCache()
    first_keys = torch.tensor([[[[1.0], [2.0]]]])
    first_values = torch.tensor([[[[11.0], [12.0]]]])
    second_keys = torch.tensor([[[[3.0]]]])
    second_values = torch.tensor([[[[13.0]]]])

    cache.append(first_keys, first_values)
    cached_keys, cached_values = cache.append(
        second_keys,
        second_values,
    )

    expected_keys = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    expected_values = torch.tensor([[[[11.0], [12.0], [13.0]]]])

    assert cache.current_length == 3
    torch.testing.assert_close(cached_keys, expected_keys)
    torch.testing.assert_close(cached_values, expected_values)


def test_append_copies_initial_input_storage() -> None:
    cache = LayerKVCache()
    keys = torch.ones(1, 1, 2, 2)
    values = torch.full((1, 1, 2, 2), 2.0)

    cache.append(keys, values)
    keys.zero_()
    values.zero_()

    assert cache.keys is not None
    assert cache.values is not None
    torch.testing.assert_close(cache.keys, torch.ones(1, 1, 2, 2))
    torch.testing.assert_close(
        cache.values,
        torch.full((1, 1, 2, 2), 2.0),
    )


def test_reset_returns_layer_cache_to_empty_state() -> None:
    cache = LayerKVCache()
    cache.append(
        torch.ones(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2),
    )

    cache.reset()

    assert cache.current_length == 0
    assert cache.keys is None
    assert cache.values is None


@pytest.mark.parametrize(
    ("keys", "values", "message"),
    [
        (
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 3),
            "must have shape",
        ),
        (
            torch.ones(1, 2, 3, 4),
            torch.ones(1, 2, 2, 4),
            "identical shapes",
        ),
        (
            torch.ones(0, 2, 3, 4),
            torch.ones(0, 2, 3, 4),
            "batch size must be positive",
        ),
        (
            torch.ones(1, 0, 3, 4),
            torch.ones(1, 0, 3, 4),
            "number of key/value heads must be positive",
        ),
        (
            torch.ones(1, 2, 0, 4),
            torch.ones(1, 2, 0, 4),
            "appended sequence length must be positive",
        ),
    ],
)
def test_append_rejects_invalid_shapes(
    keys: torch.Tensor,
    values: torch.Tensor,
    message: str,
) -> None:
    cache = LayerKVCache()

    with pytest.raises(ValueError, match=message):
        cache.append(keys, values)


def test_append_rejects_changed_batch_dimension() -> None:
    cache = LayerKVCache()
    cache.append(
        torch.ones(1, 2, 2, 4),
        torch.ones(1, 2, 2, 4),
    )

    with pytest.raises(ValueError, match="batch and head dimensions"):
        cache.append(
            torch.ones(2, 2, 1, 4),
            torch.ones(2, 2, 1, 4),
        )


def test_append_rejects_changed_head_count() -> None:
    cache = LayerKVCache()
    cache.append(
        torch.ones(1, 2, 2, 4),
        torch.ones(1, 2, 2, 4),
    )

    with pytest.raises(ValueError, match="batch and head dimensions"):
        cache.append(
            torch.ones(1, 3, 1, 4),
            torch.ones(1, 3, 1, 4),
        )


def test_append_rejects_changed_head_dimension() -> None:
    cache = LayerKVCache()
    cache.append(
        torch.ones(1, 2, 2, 4),
        torch.ones(1, 2, 2, 4),
    )

    with pytest.raises(ValueError, match="cached head dimension"):
        cache.append(
            torch.ones(1, 2, 1, 8),
            torch.ones(1, 2, 1, 8),
        )


def test_append_rejects_changed_dtype() -> None:
    cache = LayerKVCache()
    cache.append(
        torch.ones(1, 2, 2, 4, dtype=torch.float32),
        torch.ones(1, 2, 2, 4, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="cached dtype"):
        cache.append(
            torch.ones(1, 2, 1, 4, dtype=torch.float64),
            torch.ones(1, 2, 1, 4, dtype=torch.float64),
        )


def test_append_preserves_device_and_dtype() -> None:
    cache = LayerKVCache()
    keys = torch.ones(1, 2, 2, 4, dtype=torch.float64)
    values = torch.ones(1, 2, 2, 4, dtype=torch.float64)

    cached_keys, cached_values = cache.append(keys, values)

    assert cached_keys.device == keys.device
    assert cached_values.device == values.device
    assert cached_keys.dtype == torch.float64
    assert cached_values.dtype == torch.float64


def test_kv_cache_has_independent_layer_entries() -> None:
    cache = KVCache(num_layers=2)

    assert len(cache) == 2
    assert cache[0] is not cache[1]

    cache[0].append(
        torch.ones(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2),
    )

    assert cache[0].current_length == 2
    assert cache[1].current_length == 0


def test_kv_cache_reports_shared_length() -> None:
    cache = KVCache(num_layers=2)

    for layer in cache.layers:
        layer.append(
            torch.ones(1, 1, 3, 2),
            torch.ones(1, 1, 3, 2),
        )

    assert cache.current_length == 3


def test_kv_cache_detects_inconsistent_layer_lengths() -> None:
    cache = KVCache(num_layers=2)
    cache[0].append(
        torch.ones(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2),
    )

    with pytest.raises(RuntimeError, match="inconsistent sequence lengths"):
        _ = cache.current_length


def test_kv_cache_reset_clears_every_layer() -> None:
    cache = KVCache(num_layers=2)

    for layer in cache.layers:
        layer.append(
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )

    cache.reset()

    assert cache.current_length == 0
    assert all(layer.keys is None for layer in cache.layers)
    assert all(layer.values is None for layer in cache.layers)


def test_kv_cache_requires_at_least_one_layer() -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        KVCache(num_layers=0)
