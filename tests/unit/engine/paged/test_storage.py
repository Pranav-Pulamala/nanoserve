import pytest
import torch

from tokserve.engine.paged.storage import PagedKVStorage


def create_storage(
    *,
    dtype: torch.dtype = torch.float32,
) -> PagedKVStorage:
    """Create a small CPU physical K/V pool."""

    return PagedKVStorage(
        num_layers=2,
        num_blocks=3,
        num_key_value_heads=2,
        block_size=4,
        head_dim=3,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def token_values(number: float) -> torch.Tensor:
    """Create distinctive values shaped (Hkv, Dh)."""

    return torch.full((2, 3), number, dtype=torch.float32)


def test_storage_allocates_expected_shape() -> None:
    storage = create_storage()

    assert storage.storage_shape == (2, 3, 2, 4, 3)
    assert storage.num_layers == 2
    assert storage.num_blocks == 3
    assert storage.num_key_value_heads == 2
    assert storage.block_size == 4
    assert storage.head_dim == 3
    assert storage.device.type == "cpu"
    assert storage.dtype == torch.float32


def test_written_token_can_be_read_back() -> None:
    storage = create_storage()
    keys = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    values = keys + 10.0

    storage.write_token(
        layer_index=0,
        block_id=1,
        offset=2,
        keys=keys,
        values=values,
    )
    actual_keys, actual_values = storage.read_token(
        layer_index=0,
        block_id=1,
        offset=2,
    )

    torch.testing.assert_close(actual_keys, keys)
    torch.testing.assert_close(actual_values, values)


def test_layers_blocks_and_offsets_are_isolated() -> None:
    storage = create_storage()

    locations = [
        (0, 0, 0, 1.0),
        (1, 0, 0, 2.0),
        (0, 1, 0, 3.0),
        (0, 0, 1, 4.0),
    ]

    for layer_index, block_id, offset, number in locations:
        storage.write_token(
            layer_index=layer_index,
            block_id=block_id,
            offset=offset,
            keys=token_values(number),
            values=token_values(number + 100.0),
        )

    for layer_index, block_id, offset, number in locations:
        keys, values = storage.read_token(
            layer_index=layer_index,
            block_id=block_id,
            offset=offset,
        )
        torch.testing.assert_close(keys, token_values(number))
        torch.testing.assert_close(values, token_values(number + 100.0))


def test_write_copies_values_from_caller_tensors() -> None:
    storage = create_storage()
    keys = token_values(5.0)
    values = token_values(6.0)

    storage.write_token(
        layer_index=0,
        block_id=0,
        offset=0,
        keys=keys,
        values=values,
    )
    keys.zero_()
    values.zero_()

    cached_keys, cached_values = storage.read_token(
        layer_index=0,
        block_id=0,
        offset=0,
    )

    torch.testing.assert_close(cached_keys, token_values(5.0))
    torch.testing.assert_close(cached_values, token_values(6.0))


def test_read_returns_copies_not_mutable_storage_views() -> None:
    storage = create_storage()
    storage.write_token(
        layer_index=0,
        block_id=0,
        offset=0,
        keys=token_values(7.0),
        values=token_values(8.0),
    )

    keys, values = storage.read_token(
        layer_index=0,
        block_id=0,
        offset=0,
    )
    keys.zero_()
    values.zero_()

    cached_keys, cached_values = storage.read_token(
        layer_index=0,
        block_id=0,
        offset=0,
    )

    torch.testing.assert_close(cached_keys, token_values(7.0))
    torch.testing.assert_close(cached_values, token_values(8.0))


@pytest.mark.parametrize(
    ("layer_index", "block_id", "offset", "message"),
    [
        (-1, 0, 0, "layer_index is out of range"),
        (2, 0, 0, "layer_index is out of range"),
        (0, -1, 0, "block_id is out of range"),
        (0, 3, 0, "block_id is out of range"),
        (0, 0, -1, "offset is out of range"),
        (0, 0, 4, "offset is out of range"),
    ],
)
def test_read_rejects_invalid_location(
    layer_index: int,
    block_id: int,
    offset: int,
    message: str,
) -> None:
    storage = create_storage()

    with pytest.raises(IndexError, match=message):
        storage.read_token(
            layer_index=layer_index,
            block_id=block_id,
            offset=offset,
        )


@pytest.mark.parametrize(
    ("layer_index", "block_id", "offset", "message"),
    [
        (-1, 0, 0, "layer_index is out of range"),
        (0, 3, 0, "block_id is out of range"),
        (0, 0, 4, "offset is out of range"),
    ],
)
def test_write_rejects_invalid_location(
    layer_index: int,
    block_id: int,
    offset: int,
    message: str,
) -> None:
    storage = create_storage()

    with pytest.raises(IndexError, match=message):
        storage.write_token(
            layer_index=layer_index,
            block_id=block_id,
            offset=offset,
            keys=token_values(1.0),
            values=token_values(2.0),
        )


def test_write_rejects_wrong_tensor_shape() -> None:
    storage = create_storage()

    with pytest.raises(ValueError, match="shape \\(Hkv, Dh\\)"):
        storage.write_token(
            layer_index=0,
            block_id=0,
            offset=0,
            keys=torch.ones(1, 3),
            values=torch.ones(2, 3),
        )


def test_write_rejects_wrong_dtype() -> None:
    storage = create_storage(dtype=torch.float32)

    with pytest.raises(ValueError, match="match the storage dtype"):
        storage.write_token(
            layer_index=0,
            block_id=0,
            offset=0,
            keys=torch.ones(2, 3, dtype=torch.float64),
            values=torch.ones(2, 3, dtype=torch.float64),
        )


def test_float64_storage_preserves_dtype_and_device() -> None:
    storage = create_storage(dtype=torch.float64)
    keys = torch.ones(2, 3, dtype=torch.float64)
    values = torch.full((2, 3), 2.0, dtype=torch.float64)

    storage.write_token(
        layer_index=1,
        block_id=2,
        offset=3,
        keys=keys,
        values=values,
    )
    cached_keys, cached_values = storage.read_token(
        layer_index=1,
        block_id=2,
        offset=3,
    )

    assert cached_keys.device.type == "cpu"
    assert cached_values.device.type == "cpu"
    assert cached_keys.dtype == torch.float64
    assert cached_values.dtype == torch.float64
    torch.testing.assert_close(cached_keys, keys)
    torch.testing.assert_close(cached_values, values)


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("num_layers", "num_layers must be positive"),
        ("num_blocks", "num_blocks must be positive"),
        ("num_key_value_heads", "num_key_value_heads must be positive"),
        ("block_size", "block_size must be positive"),
        ("head_dim", "head_dim must be positive"),
    ],
)
def test_storage_rejects_nonpositive_dimensions(
    argument: str,
    message: str,
) -> None:
    dimensions = {
        "num_layers": 2,
        "num_blocks": 3,
        "num_key_value_heads": 2,
        "block_size": 4,
        "head_dim": 3,
    }
    dimensions[argument] = 0

    with pytest.raises(ValueError, match=message):
        PagedKVStorage(
            **dimensions,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
