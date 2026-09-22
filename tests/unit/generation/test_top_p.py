import pytest
import torch

from tokserve.generation.temperature import sample_from_logits
from tokserve.generation.top_p import filter_top_p


def test_top_p_keeps_token_that_crosses_threshold() -> None:
    probabilities = torch.tensor([[0.50, 0.25, 0.15, 0.10]])
    logits = torch.log(probabilities)

    filtered = filter_top_p(logits, 0.70)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_top_p_always_keeps_highest_probability_token() -> None:
    filtered = filter_top_p(torch.tensor([[8.0, 1.0, 0.0]]), 0.01)

    assert torch.isfinite(filtered[0, 0])


def test_filtered_tokens_cannot_be_sampled() -> None:
    logits = torch.log(torch.tensor([[0.50, 0.25, 0.15, 0.10]]))
    filtered = filter_top_p(logits, 0.70)

    selected = sample_from_logits(
        filtered.expand(100, -1),
        generator=torch.Generator().manual_seed(4),
    )

    assert set(selected.flatten().tolist()) <= {0, 1}


@pytest.mark.parametrize("top_p", [None, 1.0])
def test_disabled_top_p_preserves_logits(top_p: float | None) -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    torch.testing.assert_close(filter_top_p(logits, top_p), logits)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1])
def test_top_p_rejects_invalid_values(top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p must be in the interval"):
        filter_top_p(torch.ones(1, 3), top_p)
