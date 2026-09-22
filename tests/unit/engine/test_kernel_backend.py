"""CPU behavior and explicit backend selection."""

import pytest
import torch

from tokserve.engine.model import LlamaModel
from tokserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )


def test_default_backend_runs_on_cpu() -> None:
    model = LlamaModel(tiny_config())
    logits = model(torch.tensor([[1, 2, 3]], dtype=torch.int64))

    assert model.backend == "torch"
    assert logits.shape == (1, 3, 32)


def test_triton_backend_requires_cuda() -> None:
    model = LlamaModel(tiny_config(), backend="triton")

    with pytest.raises(ValueError, match="Triton backend requires CUDA"):
        model(torch.tensor([[1, 2, 3]], dtype=torch.int64))
