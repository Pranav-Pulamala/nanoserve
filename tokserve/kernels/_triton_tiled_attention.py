"""First Triton tiled-attention kernel using online softmax."""

# mypy: ignore-errors

from math import sqrt

import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    query_length: tl.constexpr,
    key_length: tl.constexpr,
    head_dim: tl.constexpr,
    query_position_offset: tl.constexpr,
    scale: tl.constexpr,
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_sequence: tl.constexpr,
    stride_q_dimension: tl.constexpr,
    stride_k_batch: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_sequence: tl.constexpr,
    stride_k_dimension: tl.constexpr,
    stride_v_batch: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_sequence: tl.constexpr,
    stride_v_dimension: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_o_sequence: tl.constexpr,
    stride_o_dimension: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    query_offsets = query_block * block_m + tl.arange(0, block_m)
    dimension_offsets = tl.arange(0, block_d)
    query_mask = query_offsets < query_length
    dimension_mask = dimension_offsets < head_dim

    query_addresses = (
        query_ptr
        + batch * stride_q_batch
        + head * stride_q_head
        + query_offsets[:, None] * stride_q_sequence
        + dimension_offsets[None, :] * stride_q_dimension
    )
    query = tl.load(
        query_addresses,
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((block_m,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, block_d), tl.float32)

    absolute_query_positions = query_position_offset + query_offsets

    for key_start in range(0, key_length, block_n):
        key_offsets = key_start + tl.arange(0, block_n)
        key_mask = key_offsets < key_length

        key_addresses = (
            key_ptr
            + batch * stride_k_batch
            + head * stride_k_head
            + key_offsets[:, None] * stride_k_sequence
            + dimension_offsets[None, :] * stride_k_dimension
        )
        value_addresses = (
            value_ptr
            + batch * stride_v_batch
            + head * stride_v_head
            + key_offsets[:, None] * stride_v_sequence
            + dimension_offsets[None, :] * stride_v_dimension
        )

        key = tl.load(
            key_addresses,
            mask=key_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            value_addresses,
            mask=key_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(
            query[:, None, :] * key[None, :, :],
            axis=2,
        )
        scores *= scale

        causal_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & (key_offsets[None, :] <= absolute_query_positions[:, None])
        )
        scores = tl.where(causal_mask, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        previous_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(causal_mask, probabilities, 0.0)

        running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * previous_scale[:, None] + tl.sum(
            probabilities[:, :, None] * value[None, :, :],
            axis=1,
        )
        running_max = new_max

    output = accumulator / running_sum[:, None]
    output_addresses = (
        output_ptr
        + batch * stride_o_batch
        + head * stride_o_head
        + query_offsets[:, None] * stride_o_sequence
        + dimension_offsets[None, :] * stride_o_dimension
    )
    tl.store(
        output_addresses,
        output,
        mask=query_mask[:, None] & dimension_mask[None, :],
    )


def launch_tiled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    """Launch causal tiled attention for contiguous (B, H, T, Dh) tensors."""

    batch_size, num_heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    block_d = triton.next_power_of_2(head_dim)

    output = torch.empty_like(query)
    grid = (
        triton.cdiv(query_length, block_m),
        num_heads,
        batch_size,
    )

    _tiled_attention_kernel[grid](
        query,
        key,
        value,
        output,
        query_length,
        key_length,
        head_dim,
        query_position_offset,
        1.0 / sqrt(head_dim),
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        block_m,
        block_n,
        block_d,
        num_warps=4,
    )
    return output
