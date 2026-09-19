"""Top-k logit filtering."""

import torch


def filter_top_k(
    logits: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    """Retain only the top-k logits in every (B, V) row."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")

    if top_k is None or top_k >= logits.shape[-1]:
        return logits.clone()

    if top_k < 1:
        raise ValueError("top_k must be positive")

    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, -torch.inf)
