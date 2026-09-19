import pytest
import torch

from nanoserve.generation.greedy import select_greedy
from nanoserve.generation.types import GenerationConfig


def test_select_greedy_chooses_highest_logit_for_each_batch_row() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0], [5.0, 2.0, 8.0]])

    selected = select_greedy(logits)

    torch.testing.assert_close(selected, torch.tensor([[1], [2]]))


def test_select_greedy_is_deterministic() -> None:
    logits = torch.tensor([[0.1, 0.8, 0.2]])

    first = select_greedy(logits)
    second = select_greedy(logits)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([1.0, 2.0]),
        torch.ones(1, 2, 3),
        torch.empty(0, 3),
        torch.empty(1, 0),
    ],
)
def test_select_greedy_rejects_invalid_shapes(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        select_greedy(logits)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens must be nonnegative"),
        (
            {"max_new_tokens": 1, "temperature": 0.0},
            "temperature must be positive",
        ),
        ({"max_new_tokens": 1, "top_k": 0}, "top_k must be positive"),
        (
            {"max_new_tokens": 1, "top_p": 0.0},
            "top_p must be in the interval",
        ),
        (
            {"max_new_tokens": 1, "top_p": 1.1},
            "top_p must be in the interval",
        ),
        (
            {"max_new_tokens": 1, "eos_token_id": -1},
            "eos_token_id must be nonnegative",
        ),
    ],
)
def test_generation_config_rejects_invalid_values(
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**arguments)  # type: ignore[arg-type]
