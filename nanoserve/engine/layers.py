"""PyTorch implementations of Llama normalization and SwiGLU."""

import torch
from torch import nn
from torch.nn import functional as functional


class RMSNorm(nn.Module):
    """Llama RMSNorm with a learned scale parameter."""

    def __init__(
        self,
        hidden_size: int,
        *,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension without subtracting its mean."""

        if inputs.ndim < 1:
            raise ValueError("inputs must have at least one dimension")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

        input_dtype = inputs.dtype
        float_inputs = inputs.to(torch.float32)
        mean_square = float_inputs.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_inputs * torch.rsqrt(mean_square + self.epsilon)
        return self.weight * normalized.to(input_dtype)


class SwiGLU(nn.Module):
    """Bias-free Llama SwiGLU feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        if intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply SiLU gating and return hidden-size outputs."""

        if inputs.ndim < 1:
            raise ValueError("inputs must have at least one dimension")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

        gate = functional.silu(self.gate_proj(inputs))
        up = self.up_proj(inputs)
        output: torch.Tensor = self.down_proj(gate * up)
        return output
