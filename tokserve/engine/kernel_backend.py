"""Explicit selection between PyTorch and Triton inference kernels."""

from typing import Literal, TypeAlias

import torch

from tokserve.engine.layers import RMSNorm
from tokserve.kernels.rmsnorm import triton_rms_norm

KernelBackend: TypeAlias = Literal["torch", "triton"]


def apply_rms_norm(
    norm: RMSNorm,
    inputs: torch.Tensor,
    backend: KernelBackend,
) -> torch.Tensor:
    """Use the reference RMSNorm or its CUDA Triton implementation."""
    if backend == "torch":
        output: torch.Tensor = norm(inputs)
        return output

    if not inputs.is_cuda:
        raise ValueError("Triton backend requires CUDA")

    if norm.weight.dtype != torch.float32:
        raise TypeError("Triton backend requires float32 RMSNorm weights")

    result: torch.Tensor = triton_rms_norm(
        inputs,
        norm.weight,
        epsilon=norm.epsilon,
    )
    return result
