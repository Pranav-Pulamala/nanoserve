import pytest
import torch

from nanoserve.generation.sampling import select_next_token
from nanoserve.generation.types import GenerationConfig


@pytest.mark.parametrize(
    "config",
    [
        GenerationConfig(max_new_tokens=1, do_sample=True),
        GenerationConfig(max_new_tokens=1, do_sample=True, temperature=0.5),
        GenerationConfig(max_new_tokens=1, do_sample=True, top_k=2),
        GenerationConfig(max_new_tokens=1, do_sample=True, top_p=0.8),
        GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            temperature=0.5,
            top_k=2,
        ),
        GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            temperature=2.0,
            top_p=0.8,
        ),
        GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_k=3,
            top_p=0.8,
        ),
        GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            top_k=3,
            top_p=0.9,
        ),
    ],
)
def test_sampling_combinations_return_one_token(
    config: GenerationConfig,
) -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    selected = select_next_token(
        logits,
        config,
        generator=torch.Generator().manual_seed(5),
    )

    assert selected.shape == (1, 1)
    assert selected.device == logits.device


def test_explicit_greedy_mode_ignores_randomness() -> None:
    config = GenerationConfig(max_new_tokens=1, do_sample=False)
    logits = torch.tensor([[1.0, 7.0, 2.0]])

    selected = select_next_token(
        logits,
        config,
        generator=torch.Generator().manual_seed(99),
    )

    torch.testing.assert_close(selected, torch.tensor([[1]]))


def test_seeded_unified_sampling_is_reproducible() -> None:
    config = GenerationConfig(
        max_new_tokens=1,
        do_sample=True,
        temperature=0.8,
        top_k=4,
        top_p=0.9,
    )
    logits = torch.zeros(16, 6)

    first = select_next_token(
        logits,
        config,
        generator=torch.Generator().manual_seed(8),
    )
    second = select_next_token(
        logits,
        config,
        generator=torch.Generator().manual_seed(8),
    )

    torch.testing.assert_close(first, second)
