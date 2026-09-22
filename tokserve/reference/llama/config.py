"""Configuration for the NumPy Llama reference implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    """Minimal architectural configuration for a Llama-style model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        """Validate architectural dimensions and numerical parameters."""

        integer_dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }

        for name, value in integer_dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")

        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        """Return the hidden dimension assigned to each attention head."""

        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        """Return the number of query heads sharing each key/value head."""

        return self.num_attention_heads // self.num_key_value_heads
