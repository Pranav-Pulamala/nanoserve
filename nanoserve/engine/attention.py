"""Explicit PyTorch grouped-query attention."""

from math import sqrt

import torch
from torch import nn

from nanoserve.engine.rope import apply_rope
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute explicit causal scaled dot-product attention."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, H, T, Dh)")

    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")

    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError("key and value leading dimensions must match")

    if query.device != key.device or key.device != value.device:
        raise ValueError("query, key, and value must use the same device")

    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]

    if head_dim < 1:
        raise ValueError("head dimension must be positive")

    scores = torch.matmul(
        query,
        key.transpose(-1, -2),
    ) / sqrt(head_dim)

    mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=scores.device,
    ).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
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
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply causal grouped-query attention to (B, T, D)."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, D)")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

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
