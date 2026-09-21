"""Validated public wrapper for Triton vector addition."""

import torch


def vector_add(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Add matching one-dimensional CUDA vectors with Triton."""

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")

    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")

    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("x and y must be CUDA tensors")

    if x.device != y.device:
        raise ValueError("x and y must use the same CUDA device")

    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")

    if x.dtype not in (torch.float32, torch.float16):
        raise TypeError("vector_add supports float32 and float16")

    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")

    if x.numel() == 0:
        return torch.empty_like(x)

    # Import only when this CUDA operation is called. Importing nanoserve on
    # a Mac must not require the optional Triton package.
    from nanoserve.kernels._triton_vector import launch_vector_add

    output: torch.Tensor = launch_vector_add(x, y)
    return output
