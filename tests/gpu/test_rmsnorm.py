"""Check Triton RMSNorm against the existing PyTorch implementation."""

import importlib.util
from math import prod

import pytest
import torch

from tokserve.engine.layers import RMSNorm
from tokserve.kernels.rmsnorm import triton_rms_norm

CUDA_TRITON_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)


def test_triton_rmsnorm_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensors"):
        triton_rms_norm(torch.ones(2, 7), torch.ones(7))


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize(
    ("shape", "epsilon"),
    [
        ((2, 7), 1e-6),
        ((2, 3, 13), 1e-3),
        ((4, 128), 1e-6),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_rmsnorm_matches_pytorch(
    shape: tuple[int, ...],
    epsilon: float,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        pytest.skip("native CUDA bfloat16 is unavailable")

    hidden_size = shape[-1]
    inputs = (
        torch.arange(prod(shape), device="cuda", dtype=torch.float32)
        .reshape(shape)
        .div(7.0)
        .sub(1.0)
        .to(dtype)
    )
    weight = torch.linspace(
        0.5,
        1.5,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    reference = RMSNorm(hidden_size, epsilon=epsilon).to("cuda")
    with torch.no_grad():
        reference.weight.copy_(weight)

    expected = reference(inputs)
    actual = triton_rms_norm(inputs, weight, epsilon=epsilon)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype

    tolerance = {
        torch.float32: 2e-5,
        torch.float16: 2e-3,
        torch.bfloat16: 1e-2,
    }[dtype]
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_triton_rmsnorm_accepts_noncontiguous_inputs() -> None:
    inputs = torch.randn(2, 3, 13, device="cuda").transpose(0, 1)
    weight = torch.linspace(0.5, 1.5, 13, device="cuda")

    reference = RMSNorm(13).to("cuda")
    with torch.no_grad():
        reference.weight.copy_(weight)

    torch.testing.assert_close(
        triton_rms_norm(inputs, weight),
        reference(inputs),
        rtol=2e-5,
        atol=2e-5,
    )
