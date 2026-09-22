"""Parity tests between Hugging Face and the tokserve PyTorch engine."""

import torch

from tokserve.engine.model import LlamaModel
from tokserve.engine.weights import load_model_weights
from tokserve.reference.llama.config import LlamaConfig
from tokserve.reference.llama.hf_bridge import (
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
    """Create equivalent Hugging Face and tokserve models."""

    config = tiny_config()
    hugging_face_model = create_hugging_face_model(config)
    reference_weights = map_hugging_face_weights(hugging_face_model, config)

    tokserve_model = LlamaModel(config)
    load_model_weights(tokserve_model, reference_weights)

    hugging_face_model.eval()
    tokserve_model.eval()

    return hugging_face_model, tokserve_model


def test_tokserve_logits_match_hugging_face() -> None:
    """The tokserve engine should reproduce Hugging Face logits."""

    hugging_face_model, tokserve_model = create_models()
    token_ids = torch.tensor(
        [
            [1, 4, 7, 2],
            [3, 6, 5, 8],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        expected = hugging_face_model(input_ids=token_ids).logits
        actual = tokserve_model(token_ids)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_tokserve_execution_is_deterministic() -> None:
    """Repeated inference should produce identical logits."""

    _, tokserve_model = create_models()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    first = tokserve_model(token_ids)
    second = tokserve_model(token_ids)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_tokserve_model_preserves_causal_behavior() -> None:
    """A future-token change must not alter earlier-position logits."""

    _, tokserve_model = create_models()

    original = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)

    original_logits = tokserve_model(original)
    changed_logits = tokserve_model(changed)

    torch.testing.assert_close(
        original_logits[:, :3],
        changed_logits[:, :3],
        rtol=1e-4,
        atol=1e-5,
    )
