"""PyTorch Llama model components."""

import torch
from torch import nn

from tokserve.engine.attention import GroupedQueryAttention
from tokserve.engine.cache import KVCache, LayerKVCache
from tokserve.engine.kernel_backend import KernelBackend, apply_rms_norm
from tokserve.engine.layers import RMSNorm, SwiGLU
from tokserve.engine.paged.sequence_cache import SequencePagedKVCache
from tokserve.engine.rope import positions_for_sequence
from tokserve.reference.llama.config import LlamaConfig


class LlamaDecoderBlock(nn.Module):
    """One pre-norm PyTorch Llama decoder block."""

    def __init__(
        self,
        config: LlamaConfig,
        *,
        backend: KernelBackend = "torch",
    ) -> None:
        super().__init__()

        if backend not in ("torch", "triton"):
            raise ValueError("backend must be 'torch' or 'triton'")

        self.backend = backend
        self.hidden_size = config.hidden_size
        self.input_norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )
        self.self_attention = GroupedQueryAttention(config, backend=backend)
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
        paged_cache: SequencePagedKVCache | None = None,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        """Apply one decoder block with optional layer-specific caching."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, D)")

        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("inputs must use the configured hidden_size")

        normalized_attention_input = apply_rms_norm(
            self.input_norm, inputs, self.backend
        )
        attention_result: tuple[torch.Tensor, torch.Tensor] = self.self_attention(
            normalized_attention_input,
            positions,
            cache=cache,
            paged_cache=paged_cache,
            layer_index=layer_index,
        )
        attention_output, _ = attention_result
        attention_residual = inputs + attention_output

        normalized_mlp_input = apply_rms_norm(
            self.post_attention_norm,
            attention_residual,
            self.backend,
        )
        mlp_output: torch.Tensor = self.mlp(normalized_mlp_input)

        return attention_residual + mlp_output


class LlamaModel(nn.Module):
    """Complete forward-only PyTorch Llama inference model."""

    def __init__(
        self,
        config: LlamaConfig,
        *,
        backend: KernelBackend = "torch",
    ) -> None:
        super().__init__()

        if backend not in ("torch", "triton"):
            raise ValueError("backend must be 'torch' or 'triton'")

        self.config = config
        self.backend = backend
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            LlamaDecoderBlock(config, backend=backend)
            for _ in range(config.num_hidden_layers)
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
        paged_cache: SequencePagedKVCache | None = None,
    ) -> torch.Tensor:
        """Map token IDs to logits with optional K/V caching."""

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

        if cache is not None and paged_cache is not None:
            raise ValueError("choose either contiguous or paged cache")

        parameter_device = self.embed_tokens.weight.device

        if token_ids.device != parameter_device:
            raise ValueError("token_ids and model parameters must use the same device")

        if self.backend == "triton":
            if parameter_device.type != "cuda":
                raise ValueError("Triton backend requires CUDA")
            if self.final_norm.weight.dtype != torch.float32:
                raise TypeError("Triton backend requires float32 RMSNorm weights")

        cache_length = 0

        if cache is not None:
            self._validate_cache(cache, batch_size)
            cache_length = cache.current_length
        elif paged_cache is not None:
            self._validate_paged_cache(paged_cache, batch_size)
            cache_length = paged_cache.current_length

        total_sequence_length = cache_length + sequence_length

        if total_sequence_length > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        if (
            paged_cache is not None
            and total_sequence_length > paged_cache.max_sequence_length
        ):
            raise ValueError("sequence exceeds paged cache max_sequence_length")

        hidden: torch.Tensor = self.embed_tokens(token_ids)
        positions = positions_for_sequence(
            sequence_length,
            offset=cache_length,
            device=token_ids.device,
        )

        if paged_cache is not None:
            paged_cache.begin_append(sequence_length)

        for layer_index, block in enumerate(self.layers):
            layer_cache = None if cache is None else cache[layer_index]
            block_output: torch.Tensor = block(
                hidden,
                positions,
                cache=layer_cache,
                paged_cache=paged_cache,
                layer_index=layer_index if paged_cache is not None else None,
            )
            hidden = block_output

        if paged_cache is not None:
            paged_cache.finish_append()

        normalized = apply_rms_norm(self.final_norm, hidden, self.backend)
        logits: torch.Tensor = self.lm_head(normalized)
        return logits

    def _validate_cache(
        self,
        cache: KVCache,
        batch_size: int,
    ) -> None:
        """Validate a contiguous cache for this model call."""

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

    def _validate_paged_cache(
        self,
        paged_cache: SequencePagedKVCache,
        batch_size: int,
    ) -> None:
        """Validate a paged sequence cache for this model call."""

        if batch_size != 1:
            raise ValueError("paged model execution currently supports batch size 1")

        storage = paged_cache.storage

        if storage.num_layers != self.config.num_hidden_layers:
            raise ValueError("paged cache layer count must match the model")

        if storage.num_key_value_heads != self.config.num_key_value_heads:
            raise ValueError("paged cache KV head count must match the model")

        if storage.head_dim != self.config.head_dim:
            raise ValueError("paged cache head_dim must match the model")

        if paged_cache.max_sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                "paged cache maximum cannot exceed max_position_embeddings"
            )

        parameter = self.embed_tokens.weight

        if storage.device != parameter.device:
            raise ValueError("paged cache device must match the model device")

        if storage.dtype != parameter.dtype:
            raise ValueError("paged cache dtype must match the model dtype")
