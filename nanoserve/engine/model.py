"""PyTorch Llama model components."""

import torch
from torch import nn

from nanoserve.engine.attention import GroupedQueryAttention
from nanoserve.engine.cache import KVCache, LayerKVCache
from nanoserve.engine.layers import RMSNorm, SwiGLU
from nanoserve.engine.rope import positions_for_sequence
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
        *,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """Apply one decoder block with optional layer-specific caching."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, D)")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

        normalized_attention_input: torch.Tensor = self.input_norm(inputs)
        attention_result: tuple[torch.Tensor, torch.Tensor] = self.self_attention(
            normalized_attention_input,
            positions,
            cache=cache,
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
    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Map token IDs to logits with optional per-layer KV caching."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, T)")

        if token_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("token_ids must contain integers")

        batch_size, sequence_length = token_ids.shape

        if batch_size < 1:
            raise ValueError("batch size must be positive")

        if sequence_length < 1:
            raise ValueError("sequence length must be positive")

        parameter_device = self.embed_tokens.weight.device

        if token_ids.device != parameter_device:
            raise ValueError("token_ids and model parameters must use the same device")

        cache_length = 0

        if cache is not None:
            self._validate_cache(cache, batch_size)
            cache_length = cache.current_length

        total_sequence_length = cache_length + sequence_length

        if total_sequence_length > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        hidden: torch.Tensor = self.embed_tokens(token_ids)
        positions = positions_for_sequence(
            sequence_length,
            offset=cache_length,
            device=token_ids.device,
        )

        for layer_index, block in enumerate(self.layers):
            layer_cache = None if cache is None else cache[layer_index]
            block_output: torch.Tensor = block(
                hidden,
                positions,
                cache=layer_cache,
            )
            hidden = block_output

        normalized: torch.Tensor = self.final_norm(hidden)
        logits: torch.Tensor = self.lm_head(normalized)
        return logits

    def _validate_cache(
        self,
        cache: KVCache,
        batch_size: int,
    ) -> None:
        """Validate that a cache is compatible with this model call."""

        if len(cache) != self.config.num_hidden_layers:
            raise ValueError("cache layer count must match the model")

        if cache.batch_size != batch_size:
            raise ValueError("cache batch size must match token_ids")

        if cache.num_key_value_heads != self.config.num_key_value_heads:
            raise ValueError("cache KV head count must match the model")

        if cache.head_dim != self.config.head_dim:
            raise ValueError("cache head_dim must match the model")

        if cache.max_sequence_length > self.config.max_position_embeddings:
            raise ValueError("cache capacity cannot exceed max_position_embeddings")

        parameter = self.embed_tokens.weight

        if cache.device != parameter.device:
            raise ValueError("cache device must match the model device")

        if cache.dtype != parameter.dtype:
            raise ValueError("cache dtype must match the model dtype")
