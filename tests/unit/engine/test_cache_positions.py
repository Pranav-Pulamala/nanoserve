import pytest
import torch

from nanoserve.engine.attention import GroupedQueryAttention
from nanoserve.engine.rope import apply_rope, positions_for_sequence
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small grouped-query attention configuration."""

    return LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def test_no_cache_positions_start_at_zero() -> None:
    positions = positions_for_sequence(
        4,
        offset=0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        positions,
        torch.tensor([0, 1, 2, 3]),
    )


def test_cached_single_token_position_starts_at_cache_length() -> None:
    positions = positions_for_sequence(
        1,
        offset=4,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(positions, torch.tensor([4]))


def test_multi_token_continuation_uses_consecutive_positions() -> None:
    positions = positions_for_sequence(
        3,
        offset=4,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        positions,
        torch.tensor([4, 5, 6]),
    )


@pytest.mark.parametrize(
    ("sequence_length", "offset", "message"),
    [
        (0, 0, "sequence_length must be positive"),
        (-1, 0, "sequence_length must be positive"),
        (1, -1, "offset must be nonnegative"),
    ],
)
def test_position_construction_rejects_invalid_values(
    sequence_length: int,
    offset: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        positions_for_sequence(
            sequence_length,
            offset=offset,
            device=torch.device("cpu"),
        )


def test_rope_at_cached_position_matches_full_sequence_reference() -> None:
    torch.manual_seed(1)
    complete_query = torch.randn(1, 4, 5, 2)
    complete_key = torch.randn(1, 2, 5, 2)
    complete_positions = positions_for_sequence(
        5,
        offset=0,
        device=torch.device("cpu"),
    )

    rotated_complete_query, rotated_complete_key = apply_rope(
        complete_query,
        complete_key,
        complete_positions,
        theta=10_000.0,
    )

    final_query = complete_query[:, :, -1:, :]
    final_key = complete_key[:, :, -1:, :]
    cached_position = positions_for_sequence(
        1,
        offset=4,
        device=torch.device("cpu"),
    )
    rotated_final_query, rotated_final_key = apply_rope(
        final_query,
        final_key,
        cached_position,
        theta=10_000.0,
    )

    torch.testing.assert_close(
        rotated_final_query,
        rotated_complete_query[:, :, -1:, :],
    )
    torch.testing.assert_close(
        rotated_final_key,
        rotated_complete_key[:, :, -1:, :],
    )


def test_attention_offset_matches_explicit_absolute_positions() -> None:
    torch.manual_seed(2)
    attention = GroupedQueryAttention(tiny_config())
    inputs = torch.randn(1, 2, 8)
    explicit_positions = torch.tensor([4, 5], dtype=torch.int64)

    explicit_output, explicit_weights = attention(
        inputs,
        explicit_positions,
    )
    offset_output, offset_weights = attention(
        inputs,
        position_offset=4,
    )

    torch.testing.assert_close(offset_output, explicit_output)
    torch.testing.assert_close(offset_weights, explicit_weights)


def test_attention_without_positions_starts_at_zero() -> None:
    torch.manual_seed(3)
    attention = GroupedQueryAttention(tiny_config())
    inputs = torch.randn(1, 3, 8)
    explicit_positions = torch.tensor([0, 1, 2], dtype=torch.int64)

    explicit_output, explicit_weights = attention(
        inputs,
        explicit_positions,
    )
    automatic_output, automatic_weights = attention(inputs)

    torch.testing.assert_close(automatic_output, explicit_output)
    torch.testing.assert_close(automatic_weights, explicit_weights)


def test_attention_rejects_positions_and_nonzero_offset_together() -> None:
    attention = GroupedQueryAttention(tiny_config())
    inputs = torch.randn(1, 1, 8)

    with pytest.raises(
        ValueError,
        match="position_offset must be zero when positions are provided",
    ):
        attention(
            inputs,
            torch.tensor([4]),
            position_offset=4,
        )


def test_attention_rejects_negative_position_offset() -> None:
    attention = GroupedQueryAttention(tiny_config())
    inputs = torch.randn(1, 1, 8)

    with pytest.raises(ValueError, match="position_offset must be nonnegative"):
        attention(inputs, position_offset=-1)
