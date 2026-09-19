"""Deterministic greedy token selection."""

import torch


def select_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Select the highest-logit token from each (B, V) row."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")

    if logits.shape[0] < 1:
        raise ValueError("logits must contain at least one batch row")

    if logits.shape[1] < 1:
        raise ValueError("vocabulary size must be positive")

    return torch.argmax(logits, dim=-1, keepdim=True)
