"""Prompt prefill and one-token decode using paged K/V storage."""

from dataclasses import dataclass

import torch

from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.engine.paged.sequence_cache import SequencePagedKVCache


@dataclass(frozen=True)
class PagedPrefillResult:
    """Prompt logits and the registered paged sequence cache."""

    logits: torch.Tensor
    cache: SequencePagedKVCache


def validate_manager(
    model: LlamaModel,
    manager: PagedKVCacheManager,
) -> None:
    """Check physical pool compatibility with the model."""

    storage = manager.storage
    parameter = model.embed_tokens.weight

    if storage.num_layers != model.config.num_hidden_layers:
        raise ValueError("paged pool layer count must match the model")

    if storage.num_key_value_heads != model.config.num_key_value_heads:
        raise ValueError("paged pool KV head count must match the model")

    if storage.head_dim != model.config.head_dim:
        raise ValueError("paged pool head_dim must match the model")

    if storage.device != parameter.device:
        raise ValueError("paged pool device must match the model device")

    if storage.dtype != parameter.dtype:
        raise ValueError("paged pool dtype must match the model dtype")


@torch.inference_mode()
def prefill_paged(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    manager: PagedKVCacheManager,
    *,
    sequence_id: str,
) -> PagedPrefillResult:
    """Run one prompt and populate a newly registered paged sequence."""

    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape (1, T)")

    if prompt_ids.shape[0] != 1:
        raise ValueError("paged prefill currently supports batch size 1")

    if prompt_ids.shape[1] < 1:
        raise ValueError("prompt must contain at least one token")

    validate_manager(model, manager)

    cache = manager.create_sequence(
        sequence_id,
        max_sequence_length=model.config.max_position_embeddings,
    )

    try:
        logits: torch.Tensor = model(prompt_ids, paged_cache=cache)

        if cache.current_length != prompt_ids.shape[1]:
            raise RuntimeError("paged prefill length does not match prompt length")

        return PagedPrefillResult(logits=logits, cache=cache)
    except Exception:
        manager.release_sequence(sequence_id)
        raise


@torch.inference_mode()
def decode_paged(
    model: LlamaModel,
    token_ids: torch.Tensor,
    cache: SequencePagedKVCache,
) -> torch.Tensor:
    """Decode exactly one new token with an existing paged sequence."""

    if token_ids.ndim != 2 or token_ids.shape != (1, 1):
        raise ValueError("paged decode requires token_ids shaped (1, 1)")

    if cache.current_length < 1:
        raise ValueError("paged decode requires a populated cache")

    if cache.current_length >= cache.max_sequence_length:
        raise ValueError("paged cache capacity exceeded")

    previous_length = cache.current_length
    logits: torch.Tensor = model(token_ids, paged_cache=cache)

    if cache.current_length != previous_length + 1:
        raise RuntimeError("paged decode must increase length by one")

    if logits.shape[:2] != token_ids.shape:
        raise RuntimeError("paged decode logits must match token_ids dimensions")

    return logits
