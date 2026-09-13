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


class LlamaModel(nn.Module):
    """Complete forward-only PyTorch Llama inference model."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            LlamaDecoderBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs shaped (B, T) to logits shaped (B, T, V)."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, T)")

        if token_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("token_ids must contain integers")

        sequence_length = token_ids.shape[1]

        if sequence_length < 1:
            raise ValueError("sequence length must be positive")

        if sequence_length > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        parameter_device = self.embed_tokens.weight.device

        if token_ids.device != parameter_device:
            raise ValueError("token_ids and model parameters must use the same device")

        hidden: torch.Tensor = self.embed_tokens(token_ids)
        positions = torch.arange(
            sequence_length,
            dtype=torch.int64,
            device=token_ids.device,
        )

        for block in self.layers:
            block_output: torch.Tensor = block(hidden, positions)
            hidden = block_output

        normalized: torch.Tensor = self.final_norm(hidden)
        logits: torch.Tensor = self.lm_head(normalized)
        return logits
