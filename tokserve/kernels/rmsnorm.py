"""Validated CUDA wrapper for Triton RMSNorm."""

import torch


def triton_rms_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm across the final dimension of CUDA inputs."""
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    hidden_size = inputs.shape[-1]
    if hidden_size < 1:
        raise ValueError("hidden size must be positive")

    if weight.shape != (hidden_size,):
        raise ValueError("weight must have shape (D,)")

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    if not inputs.is_cuda or not weight.is_cuda:
        raise ValueError("inputs and weight must be CUDA tensors")

    if inputs.device != weight.device:
        raise ValueError("inputs and weight must use the same device")

    if inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("inputs must use float32, float16, or bfloat16")

    if weight.dtype != torch.float32:
        raise TypeError("weight must use float32")

    from tokserve.kernels._triton_rmsnorm import launch_rmsnorm

    result: torch.Tensor = launch_rmsnorm(
        inputs.contiguous(),
        weight.contiguous(),
        epsilon,
    )
    return result
