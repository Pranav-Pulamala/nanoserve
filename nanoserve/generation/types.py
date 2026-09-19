"""Configuration and result types for token generation."""

from dataclasses import dataclass
from typing import Literal

import torch

StopReason = Literal["eos", "max_new_tokens"]


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for autoregressive token generation."""

    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate generation settings."""

        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")

        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")

        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")

        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be nonnegative")


@dataclass(frozen=True)
class GenerationResult:
    """Token IDs and termination information produced by generation."""

    token_ids: torch.Tensor
    generated_token_ids: torch.Tensor
    stop_reason: StopReason

    @property
    def num_generated_tokens(self) -> int:
        """Return the number of newly generated tokens."""

        return self.generated_token_ids.shape[1]
