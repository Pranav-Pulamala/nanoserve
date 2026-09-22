from dataclasses import replace

import pytest

from tokserve.reference.llama.config import LlamaConfig


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )


def test_config_exposes_derived_attention_dimensions() -> None:
    config = make_config()

    assert config.head_dim == 4
    assert config.num_key_value_groups == 2


@pytest.mark.parametrize(
    "field",
    [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    ],
)
def test_config_rejects_nonpositive_dimensions(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        replace(make_config(), **{field: 0})


def test_config_rejects_invalid_hidden_size_relationship() -> None:
    with pytest.raises(
        ValueError,
        match="hidden_size must be divisible by num_attention_heads",
    ):
        replace(make_config(), hidden_size=15)


def test_config_rejects_invalid_grouped_query_relationship() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        replace(make_config(), num_key_value_heads=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rms_norm_eps", 0.0),
        ("rope_theta", 0.0),
    ],
)
def test_config_rejects_nonpositive_float_parameters(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        replace(make_config(), **{field: value})
