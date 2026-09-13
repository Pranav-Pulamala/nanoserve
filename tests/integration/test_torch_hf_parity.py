"""Parity tests between Hugging Face and the nanoserve PyTorch engine."""

import torch

from nanoserve.engine.model import LlamaModel
from nanoserve.engine.weights import load_model_weights
from nanoserve.reference.llama.config import LlamaConfig
from nanoserve.reference.llama.hf_bridge import (
    create_hugging_face_model,
    map_hugging_face_weights,
)


def tiny_config() -> LlamaConfig:
    """Return a small deterministic Llama configuration."""

    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )


def create_models() -> tuple[torch.nn.Module, LlamaModel]:
    """Create equivalent Hugging Face and nanoserve models."""

    config = tiny_config()
    hugging_face_model = create_hugging_face_model(config)
    reference_weights = map_hugging_face_weights(hugging_face_model, config)

    nanoserve_model = LlamaModel(config)
    load_model_weights(nanoserve_model, reference_weights)

    hugging_face_model.eval()
    nanoserve_model.eval()

    return hugging_face_model, nanoserve_model


def test_nanoserve_logits_match_hugging_face() -> None:
    """The nanoserve engine should reproduce Hugging Face logits."""

    hugging_face_model, nanoserve_model = create_models()
    token_ids = torch.tensor(
        [
            [1, 4, 7, 2],
            [3, 6, 5, 8],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        expected = hugging_face_model(input_ids=token_ids).logits
        actual = nanoserve_model(token_ids)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_nanoserve_execution_is_deterministic() -> None:
    """Repeated inference should produce identical logits."""

    _, nanoserve_model = create_models()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    first = nanoserve_model(token_ids)
    second = nanoserve_model(token_ids)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_nanoserve_model_preserves_causal_behavior() -> None:
    """A future-token change must not alter earlier-position logits."""

    _, nanoserve_model = create_models()

    original = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)

    original_logits = nanoserve_model(original)
    changed_logits = nanoserve_model(changed)

    torch.testing.assert_close(
        original_logits[:, :3],
        changed_logits[:, :3],
        rtol=1e-4,
        atol=1e-5,
    )
