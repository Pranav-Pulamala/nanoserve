"""Educational tiled-attention reference implemented with PyTorch."""

from math import sqrt

import torch


def tiled_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int = 0,
    query_block_size: int = 16,
    key_block_size: int = 16,
) -> torch.Tensor:
    """Compute causal GQA attention using online-softmax tiles.

    Shapes:
        query: (B, Hq, Tq, Dh)
        key: (B, Hkv, Tk, Dh)
        value: (B, Hkv, Tk, Dh)
        output: (B, Hq, Tq, Dh)

    The implementation never constructs the complete (Tq, Tk) score matrix.
    """

    _validate_inputs(
        query,
        key,
        value,
        query_position_offset=query_position_offset,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )

    batch_size, num_query_heads, query_length, head_dim = query.shape
    num_key_value_heads = key.shape[1]
    key_length = key.shape[2]
    num_groups = num_query_heads // num_key_value_heads

    kv_head_indices = torch.arange(num_query_heads, device=query.device) // num_groups
    grouped_key = key.index_select(1, kv_head_indices)
    grouped_value = value.index_select(1, kv_head_indices)

    output = torch.empty_like(query)
    scale = 1.0 / sqrt(head_dim)

    for query_start in range(0, query_length, query_block_size):
        query_end = min(query_start + query_block_size, query_length)
        query_tile = query[:, :, query_start:query_end, :].to(torch.float32)
        tile_query_length = query_end - query_start

        running_max = torch.full(
            (
                batch_size,
                num_query_heads,
                tile_query_length,
                1,
            ),
            float("-inf"),
            device=query.device,
            dtype=torch.float32,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (
                batch_size,
                num_query_heads,
                tile_query_length,
                head_dim,
            ),
            device=query.device,
            dtype=torch.float32,
        )

        absolute_query_positions = torch.arange(
            query_position_offset + query_start,
            query_position_offset + query_end,
            device=query.device,
        )

        for key_start in range(0, key_length, key_block_size):
            key_end = min(key_start + key_block_size, key_length)
            key_tile = grouped_key[:, :, key_start:key_end, :].to(torch.float32)
            value_tile = grouped_value[:, :, key_start:key_end, :].to(torch.float32)

            scores = torch.matmul(
                query_tile,
                key_tile.transpose(-1, -2),
            )
            scores = scores * scale

            key_positions = torch.arange(
                key_start,
                key_end,
                device=query.device,
            )
            allowed = key_positions[None, :] <= absolute_query_positions[:, None]
            scores = scores.masked_fill(
                ~allowed[None, None, :, :],
                float("-inf"),
            )

            tile_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, tile_max)

            previous_scale = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max)

            running_sum = previous_scale * running_sum + probabilities.sum(
                dim=-1, keepdim=True
            )
            accumulator = previous_scale * accumulator + torch.matmul(
                probabilities, value_tile
            )
            running_max = new_max

        normalized = accumulator / running_sum
        output[:, :, query_start:query_end, :] = normalized.to(query.dtype)

    return output


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int,
    query_block_size: int,
    key_block_size: int,
) -> None:
    """Validate the tiled-attention reference contract."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, H, T, Dh)")

    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch sizes must match")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    if query.device != key.device or key.device != value.device:
        raise ValueError("query, key, and value must use the same device")

    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("query, key, and value must use the same dtype")

    if query.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError("attention tensors must use a supported floating dtype")

    if query.shape[0] < 1:
        raise ValueError("batch size must be positive")

    if query.shape[1] < 1 or key.shape[1] < 1:
        raise ValueError("attention head counts must be positive")

    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")

    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("sequence lengths must be positive")

    if query.shape[3] < 1:
        raise ValueError("head dimension must be positive")

    if query_position_offset < 0:
        raise ValueError("query_position_offset must be nonnegative")

    if query_position_offset + query.shape[2] > key.shape[2]:
        raise ValueError("query positions cannot exceed key sequence length")

    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")

    if key_block_size < 1:
        raise ValueError("key_block_size must be positive")
