import pytest

from nanoserve.engine.paged.block_table import (
    PhysicalTokenLocation,
    SequenceBlockTable,
)


def test_new_block_table_is_empty() -> None:
    table = SequenceBlockTable(block_size=4)

    assert table.block_size == 4
    assert table.block_ids == ()
    assert table.num_blocks == 0
    assert table.capacity == 0


def test_appending_blocks_preserves_logical_order() -> None:
    table = SequenceBlockTable(block_size=4)
    table.append_block(7)
    table.append_block(2)
    table.append_block(11)

    assert table.block_ids == (7, 2, 11)
    assert table.num_blocks == 3
    assert table.capacity == 12


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (0, PhysicalTokenLocation(block_id=7, offset=0)),
        (3, PhysicalTokenLocation(block_id=7, offset=3)),
        (4, PhysicalTokenLocation(block_id=2, offset=0)),
        (7, PhysicalTokenLocation(block_id=2, offset=3)),
    ],
)
def test_resolve_block_boundaries(
    position: int,
    expected: PhysicalTokenLocation,
) -> None:
    table = SequenceBlockTable(block_size=4)
    table.append_block(7)
    table.append_block(2)

    assert table.resolve(position) == expected


def test_nonadjacent_physical_blocks_preserve_logical_order() -> None:
    table = SequenceBlockTable(block_size=4)

    for block_id in (8, 1, 13, 4):
        table.append_block(block_id)

    assert table.resolve(0) == PhysicalTokenLocation(8, 0)
    assert table.resolve(4) == PhysicalTokenLocation(1, 0)
    assert table.resolve(8) == PhysicalTokenLocation(13, 0)
    assert table.resolve(12) == PhysicalTokenLocation(4, 0)
    assert table.resolve(15) == PhysicalTokenLocation(4, 3)


def test_resolve_rejects_unmapped_position() -> None:
    table = SequenceBlockTable(block_size=4)
    table.append_block(7)

    with pytest.raises(IndexError, match="exceeds block table capacity"):
        table.resolve(4)


def test_empty_table_cannot_resolve_position() -> None:
    table = SequenceBlockTable(block_size=4)

    with pytest.raises(IndexError, match="exceeds block table capacity"):
        table.resolve(0)


def test_negative_position_is_rejected() -> None:
    table = SequenceBlockTable(block_size=4)

    with pytest.raises(ValueError, match="position must be nonnegative"):
        table.resolve(-1)


def test_negative_physical_block_id_is_rejected() -> None:
    table = SequenceBlockTable(block_size=4)

    with pytest.raises(
        ValueError,
        match="physical_block_id must be nonnegative",
    ):
        table.append_block(-1)

    assert table.block_ids == ()


def test_duplicate_physical_block_id_is_rejected() -> None:
    table = SequenceBlockTable(block_size=4)
    table.append_block(7)

    with pytest.raises(
        ValueError,
        match="already in this block table",
    ):
        table.append_block(7)

    assert table.block_ids == (7,)


def test_clear_returns_ids_and_removes_mappings() -> None:
    table = SequenceBlockTable(block_size=4)
    table.append_block(7)
    table.append_block(2)

    released_ids = table.clear()

    assert released_ids == (7, 2)
    assert table.block_ids == ()
    assert table.num_blocks == 0
    assert table.capacity == 0


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_size_must_be_positive(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        SequenceBlockTable(block_size=block_size)
