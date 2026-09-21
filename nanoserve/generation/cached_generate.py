"""Autoregressive generation using a contiguous KV cache."""

import torch

from nanoserve.engine.inference import decode, prefill
from nanoserve.engine.model import LlamaModel
from nanoserve.generation.sampling import select_next_token
from nanoserve.generation.types import (
    GenerationConfig,
    GenerationResult,
    StopReason,
)


@torch.inference_mode()
def generate_with_cache(
    model: LlamaModel,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> GenerationResult:
    """Generate tokens using prompt prefill and incremental decoding."""

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

    if config.max_new_tokens == 0:
        return GenerationResult(
            token_ids=input_ids.clone(),
            generated_token_ids=input_ids.new_empty((1, 0)),
            stop_reason="max_new_tokens",
        )

    maximum_result_length = input_ids.shape[1] + config.max_new_tokens

    if maximum_result_length > model.config.max_position_embeddings:
        raise ValueError("prompt and generated tokens exceed max_position_embeddings")

    prefill_result = prefill(
        model,
        input_ids,
        max_sequence_length=maximum_result_length,
    )
    logits = prefill_result.logits
    cache = prefill_result.cache

    generated = input_ids.clone()
    new_tokens: list[torch.Tensor] = []

    generator: torch.Generator | None = None
    if config.do_sample and config.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(config.seed)

    stop_reason: StopReason

    for step in range(config.max_new_tokens):
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

        if step + 1 < config.max_new_tokens:
            logits = decode(
                model,
                next_token,
                cache,
            )
    else:
        stop_reason = "max_new_tokens"

    generated_token_ids = torch.cat(new_tokens, dim=1)

    return GenerationResult(
        token_ids=generated,
        generated_token_ids=generated_token_ids,
        stop_reason=stop_reason,
    )
