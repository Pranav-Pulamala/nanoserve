"""Triton implementation of RMSNorm for contiguous CUDA tensors."""

# mypy: ignore-errors

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < hidden_size

    values = tl.load(
        input_ptr + row * hidden_size + columns,
        mask=mask,
        other=0,
    ).to(tl.float32)
    weights = tl.load(weight_ptr + columns, mask=mask, other=0).to(tl.float32)

    mean_square = tl.sum(values * values, axis=0) / hidden_size
    normalized = values * tl.rsqrt(mean_square + epsilon)

    # Match the existing PyTorch RMSNorm: round the normalized value to
    # the input dtype before multiplying by the float32 learned weight.
    normalized = normalized.to(input_ptr.dtype.element_ty).to(tl.float32)
    output = normalized * weights

    tl.store(output_ptr + row * hidden_size + columns, output, mask=mask)


def launch_rmsnorm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Launch one Triton program per flattened input row."""
    hidden_size = inputs.shape[-1]
    row_count = inputs.numel() // hidden_size
    output = torch.empty(inputs.shape, device=inputs.device, dtype=torch.float32)

    if row_count == 0:
        return output

    block_size = triton.next_power_of_2(hidden_size)
    _rmsnorm_kernel[(row_count,)](
        inputs,
        weight,
        output,
        hidden_size,
        epsilon,
        block_size,
        num_warps=4,
    )
    return output
