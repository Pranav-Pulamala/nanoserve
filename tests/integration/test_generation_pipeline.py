import torch

from nanoserve.engine.model import LlamaModel
from nanoserve.generation.generate import generate
from nanoserve.generation.types import GenerationConfig
from nanoserve.reference.llama.config import LlamaConfig


def tiny_config() -> LlamaConfig:
    """Return a small configuration for generation smoke tests."""

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


def test_real_nanoserve_model_generates_valid_token_ids() -> None:
    torch.manual_seed(20)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.long)

    result = generate(
        model,
        prompt,
        GenerationConfig(max_new_tokens=4),
    )

    assert result.token_ids.shape == (1, 7)
    torch.testing.assert_close(result.token_ids[:, :3], prompt)
    assert result.num_generated_tokens == 4
    assert result.stop_reason == "max_new_tokens"
    assert torch.all(result.generated_token_ids >= 0)
    assert torch.all(result.generated_token_ids < 32)
    assert result.token_ids.device == prompt.device


def test_real_model_greedy_generation_is_deterministic() -> None:
    torch.manual_seed(21)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt = torch.tensor([[2, 5]], dtype=torch.long)
    config = GenerationConfig(max_new_tokens=3)

    first = generate(model, prompt, config)
    second = generate(model, prompt, config)

    torch.testing.assert_close(first.token_ids, second.token_ids)


def test_real_model_seeded_sampling_is_deterministic() -> None:
    torch.manual_seed(22)
    model = LlamaModel(tiny_config())
    model.eval()
    prompt = torch.tensor([[3, 6]], dtype=torch.long)
    config = GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        seed=23,
    )

    first = generate(model, prompt, config)
    second = generate(model, prompt, config)

    torch.testing.assert_close(first.token_ids, second.token_ids)


def test_real_model_greedy_generation_runs_on_mps_when_available() -> None:
    if not torch.backends.mps.is_available():
        return

    device = torch.device("mps")
    model = LlamaModel(tiny_config()).to(device)
    model.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long, device=device)

    result = generate(
        model,
        prompt,
        GenerationConfig(max_new_tokens=2),
    )

    assert result.token_ids.device.type == "mps"
    assert result.token_ids.shape == (1, 4)
