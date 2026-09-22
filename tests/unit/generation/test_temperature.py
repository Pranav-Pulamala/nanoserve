import pytest
import torch

from tokserve.generation.temperature import (
    sample_from_logits,
    scale_temperature,
)


def test_temperature_scaling_divides_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    scaled = scale_temperature(logits, 0.5)

    torch.testing.assert_close(scaled, torch.tensor([[2.0, 4.0, 6.0]]))


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_temperature_must_be_positive(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        scale_temperature(torch.ones(1, 3), temperature)


def test_scaled_probabilities_are_valid() -> None:
    scaled = scale_temperature(torch.tensor([[1.0, 2.0, 3.0]]), 2.0)
    probabilities = torch.softmax(scaled, dim=-1)

    assert torch.all(probabilities >= 0.0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))


def test_fixed_seed_reproduces_sampling() -> None:
    logits = torch.zeros(1, 8)
    first_generator = torch.Generator().manual_seed(12)
    second_generator = torch.Generator().manual_seed(12)

    first = sample_from_logits(logits, generator=first_generator)
    second = sample_from_logits(logits, generator=second_generator)

    torch.testing.assert_close(first, second)


def test_different_seeds_can_produce_different_sequences() -> None:
    logits = torch.zeros(64, 8)
    first = sample_from_logits(
        logits,
        generator=torch.Generator().manual_seed(1),
    )
    second = sample_from_logits(
        logits,
        generator=torch.Generator().manual_seed(2),
    )

    assert not torch.equal(first, second)
