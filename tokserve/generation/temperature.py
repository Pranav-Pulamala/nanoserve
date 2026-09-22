"""Temperature scaling and probabilistic token sampling."""

import torch


def scale_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Scale (B, V) logits by a positive temperature."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    return logits / temperature


def sample_from_logits(
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row from filtered (B, V) logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")

    if logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("logits dimensions must be positive")

    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("each logits row must contain a finite value")

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
