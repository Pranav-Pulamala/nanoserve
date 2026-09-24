"""Validated wrapper for Triton attention over paged K/V storage."""

import torch

from tokserve.engine.paged.storage import PagedKVStorage

BLOCK_M = 16
BLOCK_N = 16


def triton_paged_attention(
    query: torch.Tensor,
    storage: PagedKVStorage,
    block_ids: tuple[int, ...],
    *,
    layer_index: int,
    key_length: int,
    query_position_offset: int,
) -> torch.Tensor:
    """Apply causal attention while reading K/V through a block table."""

    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError("paged query must have shape (1, Hq, Tq, Dh)")

    if not query.is_cuda or storage.device.type != "cuda":
        raise ValueError("paged Triton attention requires CUDA")

    if query.device != storage.device:
        raise ValueError("query and paged storage must use the same device")

    if query.dtype != storage.dtype:
        raise ValueError("query and paged storage must use the same dtype")

    if query.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError("paged attention requires a supported floating dtype")

    if query.shape[1] % storage.num_key_value_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")

    if query.shape[-1] != storage.head_dim:
        raise ValueError("query head_dim must match paged storage")

    if query.shape[-1] > 128:
        raise ValueError("head_dim must not exceed 128")

    if layer_index < 0 or layer_index >= storage.num_layers:
        raise IndexError("layer_index is out of range")

    if key_length < 1:
        raise ValueError("key_length must be positive")

    if query_position_offset < 0:
        raise ValueError("query_position_offset must be nonnegative")

    if query_position_offset + query.shape[2] > key_length:
        raise ValueError("query positions cannot exceed key length")

    required_blocks = (key_length + storage.block_size - 1) // storage.block_size

    if len(block_ids) < required_blocks:
        raise ValueError("block table does not cover the valid key length")

    block_table = torch.tensor(
        block_ids[:required_blocks],
        device=query.device,
        dtype=torch.int32,
    )

    from tokserve.kernels._triton_paged_attention import (
        launch_paged_tiled_attention,
    )

    result: torch.Tensor = launch_paged_tiled_attention(
        query.contiguous(),
        storage.key_storage,
        storage.value_storage,
        block_table,
        layer_index=layer_index,
        key_length=key_length,
        query_position_offset=query_position_offset,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
    )
    return result
