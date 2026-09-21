# mypy: ignore-errors
"""Triton implementation of one-dimensional vector addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    x_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x_values = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0)
    output_values = x_values + y_values

    tl.store(output_ptr + offsets, output_values, mask=mask)


def launch_vector_add(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Launch the Triton kernel for validated one-dimensional CUDA tensors."""

    output = torch.empty_like(x)
    num_elements = x.numel()

    if num_elements == 0:
        return output

    block_size = 1024
    grid = (triton.cdiv(num_elements, block_size),)

    _vector_add_kernel[grid](
        x,
        y,
        output,
        num_elements,
        BLOCK_SIZE=block_size,
    )
    return output
