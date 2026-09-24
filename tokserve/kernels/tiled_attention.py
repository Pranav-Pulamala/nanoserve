"""Validated CUDA wrapper for Triton tiled attention."""

import torch

BLOCK_M = 16
BLOCK_N = 16


def triton_tiled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int = 0,
) -> torch.Tensor:
    """Apply causal tiled GQA to contiguous Q/K/V tensors."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, H, T, Dh)")

    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch sizes must match")

    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("sequence lengths must be positive")

    if query.shape[-1] < 1 or query.shape[-1] > 128:
        raise ValueError("head_dim must be in the interval [1, 128]")

    if query_position_offset < 0:
        raise ValueError("query_position_offset must be nonnegative")

    if query_position_offset + query.shape[2] > key.shape[2]:
        raise ValueError("query positions cannot exceed key sequence length")

    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("query, key, and value must be CUDA tensors")

    if query.device != key.device or key.device != value.device:
        raise ValueError("query, key, and value must use the same device")

    if query.dtype != torch.float32:
        raise TypeError("first Triton tiled-attention kernel requires float32")

    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key, and value must use the same dtype")

    from tokserve.kernels._triton_tiled_attention import (
        launch_tiled_attention,
    )

    result: torch.Tensor = launch_tiled_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        query_position_offset=query_position_offset,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
    )
    return result
