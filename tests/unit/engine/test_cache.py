import pytest
import torch

from nanoserve.engine.cache import KVCache, LayerKVCache


def create_layer_cache(
    *,
    max_sequence_length: int = 6,
    dtype: torch.dtype = torch.float32,
) -> LayerKVCache:
    """Create a small CPU cache for unit tests."""

    return LayerKVCache(
        batch_size=1,
        num_key_value_heads=2,
        max_sequence_length=max_sequence_length,
        head_dim=3,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def test_cache_allocates_full_capacity() -> None:
    cache = create_layer_cache(max_sequence_length=6)

    assert cache.storage_shape == (1, 2, 6, 3)
    assert cache.current_length == 0
    assert cache.remaining_capacity == 6


def test_empty_cache_exposes_only_empty_valid_prefix() -> None:
    cache = create_layer_cache()

    assert cache.keys.shape == (1, 2, 0, 3)
    assert cache.values.shape == (1, 2, 0, 3)


def test_first_write_updates_valid_prefix() -> None:
    cache = create_layer_cache()
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    values = keys + 100.0

    cached_keys, cached_values = cache.append(keys, values)

    assert cache.current_length == 2
    assert cache.remaining_capacity == 4
    assert cached_keys.shape == (1, 2, 2, 3)
    torch.testing.assert_close(cached_keys, keys)
    torch.testing.assert_close(cached_values, values)


def test_multi_token_then_single_token_writes_are_ordered() -> None:
    cache = create_layer_cache()
    first_keys = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    first_values = first_keys + 20.0
    next_keys = torch.full((1, 2, 1, 3), 50.0)
    next_values = torch.full((1, 2, 1, 3), 60.0)

    cache.append(first_keys, first_values)
    cached_keys, cached_values = cache.append(next_keys, next_values)

    expected_keys = torch.cat((first_keys, next_keys), dim=2)
    expected_values = torch.cat((first_values, next_values), dim=2)

    assert cache.current_length == 3
    torch.testing.assert_close(cached_keys, expected_keys)
    torch.testing.assert_close(cached_values, expected_values)


def test_append_copies_values_into_cache_storage() -> None:
    cache = create_layer_cache()
    keys = torch.ones(1, 2, 2, 3)
    values = torch.full((1, 2, 2, 3), 2.0)

    cache.append(keys, values)
    keys.zero_()
    values.zero_()

    torch.testing.assert_close(cache.keys, torch.ones(1, 2, 2, 3))
    torch.testing.assert_close(
        cache.values,
        torch.full((1, 2, 2, 3), 2.0),
    )


def test_reset_logically_empties_without_changing_capacity() -> None:
    cache = create_layer_cache()
    cache.append(
        torch.ones(1, 2, 3, 3),
        torch.ones(1, 2, 3, 3),
    )

    cache.reset()

    assert cache.current_length == 0
    assert cache.remaining_capacity == 6
    assert cache.storage_shape == (1, 2, 6, 3)
    assert cache.keys.shape == (1, 2, 0, 3)
    assert cache.values.shape == (1, 2, 0, 3)


def test_reset_prevents_stale_values_from_being_exposed() -> None:
    cache = create_layer_cache()
    cache.append(
        torch.full((1, 2, 3, 3), 99.0),
        torch.full((1, 2, 3, 3), 88.0),
    )
    cache.reset()

    new_keys = torch.full((1, 2, 1, 3), 1.0)
    new_values = torch.full((1, 2, 1, 3), 2.0)
    cache.append(new_keys, new_values)

    assert cache.keys.shape == (1, 2, 1, 3)
    assert cache.values.shape == (1, 2, 1, 3)
    torch.testing.assert_close(cache.keys, new_keys)
    torch.testing.assert_close(cache.values, new_values)


def test_append_rejects_capacity_overflow() -> None:
    cache = create_layer_cache(max_sequence_length=3)
    cache.append(
        torch.ones(1, 2, 2, 3),
        torch.ones(1, 2, 2, 3),
    )

    with pytest.raises(ValueError, match="KV cache capacity exceeded"):
        cache.append(
            torch.ones(1, 2, 2, 3),
            torch.ones(1, 2, 2, 3),
        )

    assert cache.current_length == 2


@pytest.mark.parametrize(
    ("keys", "values", "message"),
    [
        (
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 3),
            "must have shape",
        ),
        (
            torch.ones(1, 2, 1, 3),
            torch.ones(1, 2, 2, 3),
            "identical shapes",
        ),
        (
            torch.ones(2, 2, 1, 3),
            torch.ones(2, 2, 1, 3),
            "batch and head dimensions",
        ),
        (
            torch.ones(1, 3, 1, 3),
            torch.ones(1, 3, 1, 3),
            "batch and head dimensions",
        ),
        (
            torch.ones(1, 2, 0, 3),
            torch.ones(1, 2, 0, 3),
            "appended sequence length",
        ),
        (
            torch.ones(1, 2, 1, 4),
            torch.ones(1, 2, 1, 4),
            "configured head_dim",
        ),
    ],
)
def test_append_rejects_invalid_shapes(
    keys: torch.Tensor,
    values: torch.Tensor,
    message: str,
) -> None:
    cache = create_layer_cache()

    with pytest.raises(ValueError, match=message):
        cache.append(keys, values)


def test_append_rejects_incorrect_dtype() -> None:
    cache = create_layer_cache(dtype=torch.float32)

    with pytest.raises(ValueError, match="match the cache dtype"):
        cache.append(
            torch.ones(1, 2, 1, 3, dtype=torch.float64),
            torch.ones(1, 2, 1, 3, dtype=torch.float64),
        )


def test_storage_preserves_configured_device_and_dtype() -> None:
    cache = create_layer_cache(dtype=torch.float64)

    assert cache.keys.device.type == "cpu"
    assert cache.values.device.type == "cpu"
    assert cache.keys.dtype == torch.float64
    assert cache.values.dtype == torch.float64


def test_kv_cache_creates_independent_preallocated_layers() -> None:
    cache = KVCache(
        num_layers=3,
        batch_size=1,
        num_key_value_heads=2,
        max_sequence_length=8,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(cache) == 3
    assert cache[0] is not cache[1]
    assert cache[1] is not cache[2]
    assert all(layer.storage_shape == (1, 2, 8, 4) for layer in cache.layers)


def test_kv_cache_reports_shared_length() -> None:
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=1,
        max_sequence_length=4,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for layer in cache.layers:
        layer.append(
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )

    assert cache.current_length == 2


def test_kv_cache_detects_inconsistent_layer_lengths() -> None:
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=1,
        max_sequence_length=4,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache[0].append(
        torch.ones(1, 1, 1, 2),
        torch.ones(1, 1, 1, 2),
    )

    with pytest.raises(RuntimeError, match="inconsistent sequence lengths"):
        _ = cache.current_length


def test_kv_cache_reset_clears_all_valid_prefixes() -> None:
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=1,
        max_sequence_length=4,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for layer in cache.layers:
        layer.append(
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )

    cache.reset()

    assert cache.current_length == 0
    assert all(layer.keys.shape[2] == 0 for layer in cache.layers)
    assert all(layer.values.shape[2] == 0 for layer in cache.layers)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("batch_size", 0, "batch_size must be positive"),
        (
            "num_key_value_heads",
            0,
            "num_key_value_heads must be positive",
        ),
        (
            "max_sequence_length",
            0,
            "max_sequence_length must be positive",
        ),
        ("head_dim", 0, "head_dim must be positive"),
    ],
)
def test_layer_cache_rejects_invalid_dimensions(
    argument: str,
    value: int,
    message: str,
) -> None:
    arguments = {
        "batch_size": 1,
        "num_key_value_heads": 2,
        "max_sequence_length": 4,
        "head_dim": 3,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        LayerKVCache(**arguments)  # type: ignore[arg-type]


def test_kv_cache_requires_at_least_one_layer() -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        KVCache(
            num_layers=0,
            batch_size=1,
            num_key_value_heads=1,
            max_sequence_length=4,
            head_dim=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
