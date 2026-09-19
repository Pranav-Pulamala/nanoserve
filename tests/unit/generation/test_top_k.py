import pytest
import torch

from nanoserve.generation.temperature import sample_from_logits
from nanoserve.generation.top_k import filter_top_k


def test_top_k_retains_highest_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    filtered = filter_top_k(logits, 2)

    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert filtered[0, 1] == 4.0
    assert filtered[0, 3] == 3.0


def test_excluded_tokens_cannot_be_sampled() -> None:
    filtered = filter_top_k(torch.tensor([[1.0, 4.0, 2.0, 3.0]]), 2)
    repeated = filtered.expand(100, -1)

    selected = sample_from_logits(
        repeated,
        generator=torch.Generator().manual_seed(3),
    )

    assert set(selected.flatten().tolist()) <= {1, 3}


def test_top_k_one_matches_unique_greedy_maximum() -> None:
    filtered = filter_top_k(torch.tensor([[1.0, 7.0, 2.0]]), 1)
    selected = sample_from_logits(filtered)

    torch.testing.assert_close(selected, torch.tensor([[1]]))


@pytest.mark.parametrize("top_k", [None, 3, 10])
def test_disabled_or_large_top_k_preserves_logits(top_k: int | None) -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    torch.testing.assert_close(filter_top_k(logits, top_k), logits)


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_rejects_invalid_values(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        filter_top_k(torch.ones(1, 3), top_k)
