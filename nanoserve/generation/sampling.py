"""Unified configurable next-token selection."""

import torch

from nanoserve.generation.greedy import select_greedy
from nanoserve.generation.temperature import (
    sample_from_logits,
    scale_temperature,
)
from nanoserve.generation.top_k import filter_top_k
from nanoserve.generation.top_p import filter_top_p
from nanoserve.generation.types import GenerationConfig


def select_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one next token from every (B, V) logits row."""

    if not config.do_sample:
        return select_greedy(logits)

    filtered = scale_temperature(logits, config.temperature)
    filtered = filter_top_k(filtered, config.top_k)
    filtered = filter_top_p(filtered, config.top_p)

    return sample_from_logits(filtered, generator=generator)
