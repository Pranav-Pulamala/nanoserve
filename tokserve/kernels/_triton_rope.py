"""Triton kernel for applying precomputed rotary cosine and sine values."""

# mypy: ignore-errors

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope_kernel(
    input_ptr,
    cosine_ptr,
    sine_ptr,
    output_ptr,
    num_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_head: tl.constexpr,
    stride_sequence: tl.constexpr,
    stride_dimension: tl.constexpr,
    total_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements

    dimension = offsets % head_dim
    sequence = (offsets // head_dim) % sequence_length
    head = (offsets // (head_dim * sequence_length)) % num_heads
    batch = offsets // (head_dim * sequence_length * num_heads)

    half_dim = head_dim // 2
    partner_dimension = tl.where(
        dimension < half_dim,
        dimension + half_dim,
        dimension - half_dim,
    )

    input_offset = (
        batch * stride_batch
        + head * stride_head
        + sequence * stride_sequence
        + dimension * stride_dimension
    )
    partner_offset = (
        batch * stride_batch
        + head * stride_head
        + sequence * stride_sequence
        + partner_dimension * stride_dimension
    )
    angle_offset = sequence * head_dim + dimension

    value = tl.load(input_ptr + input_offset, mask=mask, other=0)
    partner = tl.load(input_ptr + partner_offset, mask=mask, other=0)
    cosine = tl.load(cosine_ptr + angle_offset, mask=mask, other=0)
    sine = tl.load(sine_ptr + angle_offset, mask=mask, other=0)

    rotated_partner = tl.where(dimension < half_dim, -partner, partner)
    input_dtype = input_ptr.dtype.element_ty

    # Match PyTorch's separate multiplication results before their addition.
    first = (value.to(tl.float32) * cosine.to(tl.float32)).to(input_dtype)
    second = (rotated_partner.to(tl.float32) * sine.to(tl.float32)).to(input_dtype)
    output = first.to(tl.float32) + second.to(tl.float32)

    tl.store(output_ptr + offsets, output, mask=mask)


def launch_rope(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Rotate one (B, H, T, Dh) tensor, respecting its input strides."""
    batch_size, num_heads, sequence_length, head_dim = inputs.shape
    output = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
    total_elements = batch_size * num_heads * sequence_length * head_dim

    if total_elements == 0:
        return output

    block_size = 256
    _apply_rope_kernel[(triton.cdiv(total_elements, block_size),)](
        inputs,
        cosine,
        sine,
        output,
        num_heads,
        sequence_length,
        head_dim,
        *inputs.stride(),
        total_elements,
        block_size,
        num_warps=4,
    )
    return output
