"""Device execution tests for the PyTorch inference engine."""

import pytest
import torch

from nanoserve.engine.model import LlamaModel
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a configuration small enough for device smoke tests."""

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


def run_model_on_device(device: torch.device) -> torch.Tensor:
    """Run the complete nanoserve model on the requested device."""

    model = LlamaModel(tiny_config()).to(device)
    model.eval()

    token_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(token_ids)

    return logits


def test_complete_model_runs_on_cpu() -> None:
    """CPU inference is mandatory on every supported machine."""

    logits = run_model_on_device(torch.device("cpu"))

    assert logits.shape == (2, 4, 32)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()


@pytest.mark.gpu
def test_complete_model_runs_on_cuda() -> None:
    """Run complete-model inference on CUDA when available."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    logits = run_model_on_device(torch.device("cuda"))

    assert logits.shape == (2, 4, 32)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()


@pytest.mark.gpu
def test_complete_model_runs_on_mps() -> None:
    """Run complete-model inference on Apple MPS when available."""

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    logits = run_model_on_device(torch.device("mps"))

    assert logits.shape == (2, 4, 32)
    assert logits.device.type == "mps"
    assert torch.isfinite(logits).all()
