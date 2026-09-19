"""Naive full-sequence autoregressive token generation."""

from typing import Protocol

import torch

from nanoserve.generation.sampling import select_next_token
from nanoserve.generation.types import (
    GenerationConfig,
    GenerationResult,
    StopReason,
)


class LogitsModel(Protocol):
    """A callable model that maps token IDs to logits."""

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits shaped (B, T, V)."""


@torch.inference_mode()
def generate(
    model: LogitsModel,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> GenerationResult:
    """Generate tokens for one prompt using full-sequence recomputation."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (1, T)")

    if input_ids.shape[0] != 1:
        raise ValueError("generation currently supports batch size 1")

    if input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one token")

    if input_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("input_ids must contain integers")

    generated = input_ids.clone()
    new_tokens: list[torch.Tensor] = []

    generator: torch.Generator | None = None
    if config.do_sample and config.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(config.seed)

    stop_reason: StopReason

    for _ in range(config.max_new_tokens):
        logits = model(generated)

        if logits.ndim != 3:
            raise ValueError("model logits must have shape (B, T, V)")

        if logits.shape[:2] != generated.shape:
            raise ValueError(
                "model logits must match input batch and sequence dimensions"
            )

        next_token_logits = logits[:, -1, :]
        next_token = select_next_token(
            next_token_logits,
            config,
            generator=generator,
        )

        generated = torch.cat((generated, next_token), dim=1)
        new_tokens.append(next_token)

        if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
            stop_reason = "eos"
            break
    else:
        stop_reason = "max_new_tokens"

    if new_tokens:
        generated_token_ids = torch.cat(new_tokens, dim=1)
    else:
        generated_token_ids = input_ids.new_empty((1, 0))

    return GenerationResult(
        token_ids=generated,
        generated_token_ids=generated_token_ids,
        stop_reason=stop_reason,
    )
