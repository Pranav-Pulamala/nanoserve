import torch

from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig


class IncrementModel:
    """Predict the next vocabulary ID after the final input ID."""

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size
        self.grad_enabled_values: list[bool] = []

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_values.append(torch.is_grad_enabled())
        batch_size, sequence_length = token_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -100.0,
            device=token_ids.device,
        )
        next_ids = (token_ids + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 100.0)
        return logits


class UniformModel:
    """Return equal logits for every vocabulary token."""

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            8,
            device=token_ids.device,
        )


def test_greedy_generation_preserves_prompt_and_appends_tokens() -> None:
    model = IncrementModel()

    result = generate(
        model,
        torch.tensor([[1, 2]]),
        GenerationConfig(max_new_tokens=3),
    )

    torch.testing.assert_close(result.token_ids, torch.tensor([[1, 2, 3, 4, 5]]))
    torch.testing.assert_close(
        result.generated_token_ids,
        torch.tensor([[3, 4, 5]]),
    )
    assert result.num_generated_tokens == 3
    assert result.stop_reason == "max_new_tokens"


def test_eos_stops_generation_early() -> None:
    result = generate(
        IncrementModel(),
        torch.tensor([[1, 2]]),
        GenerationConfig(max_new_tokens=10, eos_token_id=4),
    )

    torch.testing.assert_close(result.token_ids, torch.tensor([[1, 2, 3, 4]]))
    assert result.num_generated_tokens == 2
    assert result.stop_reason == "eos"


def test_zero_new_tokens_preserves_prompt() -> None:
    prompt = torch.tensor([[1, 2]])
    result = generate(
        IncrementModel(),
        prompt,
        GenerationConfig(max_new_tokens=0),
    )

    torch.testing.assert_close(result.token_ids, prompt)
    assert result.generated_token_ids.shape == (1, 0)
    assert result.stop_reason == "max_new_tokens"


def test_seeded_generation_is_reproducible() -> None:
    config = GenerationConfig(
        max_new_tokens=8,
        do_sample=True,
        seed=14,
    )
    prompt = torch.tensor([[1]])

    first = generate(UniformModel(), prompt, config)
    second = generate(UniformModel(), prompt, config)

    torch.testing.assert_close(first.token_ids, second.token_ids)


def test_generation_disables_gradient_tracking() -> None:
    model = IncrementModel()

    generate(
        model,
        torch.tensor([[1]]),
        GenerationConfig(max_new_tokens=3),
    )

    assert model.grad_enabled_values == [False, False, False]
