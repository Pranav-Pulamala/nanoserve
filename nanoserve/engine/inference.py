"""Explicit prompt-prefill and incremental-decode operations."""

from dataclasses import dataclass

import torch

from nanoserve.engine.cache import KVCache
from nanoserve.engine.model import LlamaModel


@dataclass(frozen=True)
class PrefillResult:
    """Logits and populated cache produced by prompt prefill."""

    logits: torch.Tensor
    cache: KVCache


def create_kv_cache(
    model: LlamaModel,
    *,
    batch_size: int,
    max_sequence_length: int | None = None,
) -> KVCache:
    """Create an empty cache compatible with a Llama model."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    capacity = (
        model.config.max_position_embeddings
        if max_sequence_length is None
        else max_sequence_length
    )

    if capacity < 1:
        raise ValueError("max_sequence_length must be positive")

    if capacity > model.config.max_position_embeddings:
        raise ValueError("max_sequence_length cannot exceed max_position_embeddings")

    parameter = model.embed_tokens.weight

    return KVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=batch_size,
        num_key_value_heads=model.config.num_key_value_heads,
        max_sequence_length=capacity,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


@torch.inference_mode()
def prefill(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    *,
    max_sequence_length: int | None = None,
) -> PrefillResult:
    """Process a complete prompt once and populate a new KV cache."""

    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape (B, T)")

    if prompt_ids.shape[0] < 1:
        raise ValueError("prompt batch size must be positive")

    if prompt_ids.shape[1] < 1:
        raise ValueError("prompt must contain at least one token")

    capacity = (
        model.config.max_position_embeddings
        if max_sequence_length is None
        else max_sequence_length
    )

    if prompt_ids.shape[1] > capacity:
        raise ValueError("prompt length exceeds cache capacity")

    cache = create_kv_cache(
        model,
        batch_size=prompt_ids.shape[0],
        max_sequence_length=capacity,
    )
    logits: torch.Tensor = model(prompt_ids, cache=cache)

    if cache.current_length != prompt_ids.shape[1]:
        raise RuntimeError("prefill cache length does not match prompt length")

    return PrefillResult(
        logits=logits,
        cache=cache,
    )


@torch.inference_mode()
def decode(
    model: LlamaModel,
    token_ids: torch.Tensor,
    cache: KVCache,
) -> torch.Tensor:
    """Decode exactly one new token using an existing populated cache."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (B, 1)")

    if token_ids.shape[1] != 1:
        raise ValueError("decode requires exactly one token per sequence")

    if token_ids.shape[0] != cache.batch_size:
        raise ValueError("token_ids batch size must match the cache")

    if cache.current_length < 1:
        raise ValueError("decode requires a populated cache")

    if cache.current_length >= cache.max_sequence_length:
        raise ValueError("KV cache capacity exceeded")

    previous_length = cache.current_length
    logits: torch.Tensor = model(token_ids, cache=cache)

    if cache.current_length != previous_length + 1:
        raise RuntimeError("decode must increase cache length by one")

    if logits.shape[:2] != token_ids.shape:
        raise RuntimeError("decode logits must match token_ids dimensions")

    return logits
