"""Top-p nucleus logit filtering."""

import torch


def filter_top_p(
    logits: torch.Tensor,
    top_p: float | None,
) -> torch.Tensor:
    """Retain the smallest high-probability set reaching top-p mass."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")

    if top_p is None or top_p == 1.0:
        return logits.clone()

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    remove_sorted = cumulative_probabilities > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False

    remove_original = torch.zeros_like(
        remove_sorted,
        dtype=torch.bool,
    )
    remove_original.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)

    return logits.masked_fill(remove_original, -torch.inf)
