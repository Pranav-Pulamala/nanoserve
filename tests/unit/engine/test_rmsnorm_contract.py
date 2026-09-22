"""Reference behavior that the future Triton RMSNorm must match."""

from math import prod

import pytest
import torch

from tokserve.engine.layers import RMSNorm


@pytest.mark.parametrize("shape", [(4, 7), (2, 3, 8), (1, 13)])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_matches_explicit_reference(
    shape: tuple[int, ...],
    epsilon: float,
    dtype: torch.dtype,
) -> None:
    hidden_size = shape[-1]
    inputs = (
        torch.arange(prod(shape), dtype=torch.float32).reshape(shape) / 7.0 - 1.0
    ).to(dtype)
    weight = torch.linspace(0.5, 1.5, hidden_size, dtype=torch.float32)

    norm = RMSNorm(hidden_size, epsilon=epsilon)
    with torch.no_grad():
        norm.weight.copy_(weight)

    actual = norm(inputs)

    float_inputs = inputs.to(torch.float32)
    mean_square = float_inputs.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_inputs * torch.rsqrt(mean_square + epsilon)
    expected = weight * normalized.to(dtype)

    assert actual.shape == inputs.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
