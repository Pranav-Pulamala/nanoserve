"""Explicit PyTorch grouped-query attention."""

from math import sqrt

import torch
from torch import nn

from nanoserve.engine.cache import LayerKVCache
from nanoserve.engine.rope import apply_rope, positions_for_sequence
from nanoserve.reference.llama.config import LlamaConfig


def reshape_projection(
    projection: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reshape (B, T, H * Dh) into (B, H, T, Dh)."""

    if projection.ndim != 3:
        raise ValueError("projection must have shape (B, T, H * Dh)")

    batch_size, sequence_length, projected_size = projection.shape

    if projected_size != num_heads * head_dim:
        raise ValueError("projection size must equal num_heads * head_dim")

    return projection.reshape(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
    ).transpose(1, 2)


def repeat_key_value(
    inputs: torch.Tensor,
    *,
    num_groups: int,
) -> torch.Tensor:
    """Repeat K/V heads from Hkv heads to Hq heads."""

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (B, Hkv, T, Dh)")

    if num_groups < 1:
        raise ValueError("num_groups must be positive")

    return inputs.repeat_interleave(num_groups, dim=1)


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute causal attention with possibly unequal query and key lengths."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, H, T, Dh)")

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch dimensions must match")

    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key head counts must match")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")

    if query.device != key.device or key.device != value.device:
        raise ValueError("query, key, and value must use the same device")

    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("query, key, and value must use the same dtype")

    if query_position_offset < 0:
        raise ValueError("query_position_offset must be nonnegative")

    query_length = query.shape[-2]
    key_length = key.shape[-2]
    head_dim = query.shape[-1]

    if query_length < 1 or key_length < 1:
        raise ValueError("query and key sequence lengths must be positive")

    if head_dim < 1:
        raise ValueError("head dimension must be positive")

    if query_position_offset + query_length > key_length:
        raise ValueError("query positions cannot exceed the key sequence length")

    scores = torch.matmul(
        query,
        key.transpose(-1, -2),
    ) / sqrt(head_dim)

    query_positions = torch.arange(
        query_position_offset,
        query_position_offset + query_length,
        device=scores.device,
    )
    key_positions = torch.arange(
        key_length,
        device=scores.device,
    )
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    scores = scores.masked_fill(
        ~causal_mask[None, None, :, :],
        float("-inf"),
    )
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)

    return output, weights


class GroupedQueryAttention(nn.Module):
    """Llama grouped-query self-attention."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        key_value_size = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            key_value_size,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            key_value_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        position_offset: int = 0,
        cache: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply grouped-query attention with optional K/V caching."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, D)")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

        if position_offset < 0:
            raise ValueError("position_offset must be nonnegative")

        cache_length = 0

        if cache is not None:
            cache_length = cache.current_length

            if position_offset != 0:
                raise ValueError(
                    "position_offset must be zero when a cache is provided"
                )

        expected_position_offset = (
            cache_length if cache is not None else position_offset
        )

        if positions is None:
            positions = positions_for_sequence(
                inputs.shape[1],
                offset=expected_position_offset,
                device=inputs.device,
            )
        else:
            if position_offset != 0:
                raise ValueError(
                    "position_offset must be zero when positions are provided"
                )

            if cache is not None:
                expected_positions = positions_for_sequence(
                    inputs.shape[1],
                    offset=expected_position_offset,
                    device=inputs.device,
                )

                if not torch.equal(positions, expected_positions):
                    raise ValueError(
                        "positions must match the cache-aware absolute positions"
                    )

        if positions.shape != (inputs.shape[1],):
            raise ValueError("positions must have shape (T,)")

        if positions.device != inputs.device:
            raise ValueError("positions and inputs must use the same device")

        query_projection: torch.Tensor = self.q_proj(inputs)
        key_projection: torch.Tensor = self.k_proj(inputs)
        value_projection: torch.Tensor = self.v_proj(inputs)

        query = reshape_projection(
            query_projection,
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
        )
        key = reshape_projection(
            key_projection,
            num_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )
        value = reshape_projection(
            value_projection,
            num_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )

        query, key = apply_rope(
            query,
            key,
            positions,
            theta=self.rope_theta,
        )

        if cache is not None:
            key, value = cache.append(key, value)

        repeated_key = repeat_key_value(
            key,
            num_groups=self.num_key_value_groups,
        )
        repeated_value = repeat_key_value(
            value,
            num_groups=self.num_key_value_groups,
        )

        attended, attention_weights = causal_attention(
            query,
            repeated_key,
            repeated_value,
            query_position_offset=cache_length,
        )
        merged = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(
                inputs.shape[0],
                inputs.shape[1],
                self.hidden_size,
            )
        )
        output: torch.Tensor = self.o_proj(merged)

        return output, attention_weights
