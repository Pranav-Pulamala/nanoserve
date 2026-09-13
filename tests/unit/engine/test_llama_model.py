import pytest
import torch
from torch.testing import assert_close

from nanoserve.engine.model import LlamaModel
from nanoserve.reference.llama.config import LlamaConfig


def make_config(*, num_hidden_layers: int = 2) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8,
    )


def create_model(*, num_hidden_layers: int = 2) -> LlamaModel:
    torch.manual_seed(17)
    model = LlamaModel(make_config(num_hidden_layers=num_hidden_layers))
    model.eval()
    return model


def test_model_returns_expected_logit_shape() -> None:
    model = create_model()
    token_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    logits = model(token_ids)

    assert logits.shape == (2, 3, 32)
    assert torch.all(torch.isfinite(logits))


def test_model_supports_multiple_decoder_layers() -> None:
    model = create_model(num_hidden_layers=3)

    logits = model(torch.tensor([[1, 2]]))

    assert len(model.layers) == 3
    assert logits.shape == (1, 2, 32)


def test_model_execution_is_deterministic() -> None:
    model = create_model()
    token_ids = torch.tensor([[1, 2, 3]])

    first = model(token_ids)
    second = model(token_ids)

    assert_close(first, second)


def test_model_registers_all_parameters() -> None:
    model = create_model()
    parameter_names = set(dict(model.named_parameters()))

    assert "embed_tokens.weight" in parameter_names
    assert "layers.0.input_norm.weight" in parameter_names
    assert "layers.0.self_attention.q_proj.weight" in parameter_names
    assert "layers.0.mlp.gate_proj.weight" in parameter_names
    assert "final_norm.weight" in parameter_names
    assert "lm_head.weight" in parameter_names


def test_inference_output_does_not_require_gradients() -> None:
    model = create_model()

    logits = model(torch.tensor([[1, 2, 3]]))

    assert logits.requires_grad is False
    assert logits.grad_fn is None


def test_model_moves_parameters_to_cpu() -> None:
    model = create_model().to("cpu")

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())

    token_ids = torch.tensor([[1, 2]], device="cpu")
    logits = model(token_ids)

    assert logits.device.type == "cpu"


def test_model_rejects_excessive_sequence_length() -> None:
    model = create_model()

    with pytest.raises(
        ValueError,
        match="max_position_embeddings",
    ):
        model(torch.ones((1, 9), dtype=torch.int64))


def test_model_rejects_noninteger_token_ids() -> None:
    model = create_model()

    with pytest.raises(TypeError, match="must contain integers"):
        model(torch.ones((1, 2), dtype=torch.float32))


def test_model_rejects_wrong_token_rank() -> None:
    model = create_model()

    with pytest.raises(
        ValueError,
        match=r"token_ids must have shape \(B, T\)",
    ):
        model(torch.tensor([1, 2, 3]))
