"""PyTorch Llama model components."""

import torch
from torch import nn

from nanoserve.engine.attention import GroupedQueryAttention
from nanoserve.engine.layers import RMSNorm, SwiGLU
from nanoserve.reference.llama.config import LlamaConfig


class LlamaDecoderBlock(nn.Module):
    """One pre-norm PyTorch Llama decoder block."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.input_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.self_attention = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.mlp = SwiGLU(
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the decoder block to hidden states shaped (B, T, D)."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, D)")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

        normalized_attention_input: torch.Tensor = self.input_norm(inputs)
        attention_result: tuple[torch.Tensor, torch.Tensor] = self.self_attention(
            normalized_attention_input,
            positions,
        )
        attention_output, _ = attention_result
        attention_residual = inputs + attention_output

        normalized_mlp_input: torch.Tensor = self.post_attention_norm(
            attention_residual
        )
        mlp_output: torch.Tensor = self.mlp(normalized_mlp_input)

        return attention_residual + mlp_output
