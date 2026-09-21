"""PyTorch rotary positional embeddings."""

import torch


def positions_for_sequence(
    sequence_length: int,
    *,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Return consecutive absolute positions for a sequence segment."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    if offset < 0:
        raise ValueError("offset must be nonnegative")

    return torch.arange(
        offset,
        offset + sequence_length,
        dtype=torch.int64,
        device=device,
    )


def inverse_frequencies(
    head_dim: int,
    *,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Return float32 inverse frequencies on the requested device."""

    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")

    if theta <= 0.0:
        raise ValueError("theta must be positive")

    dimension_indices = torch.arange(
        0,
        head_dim,
        2,
        dtype=torch.float32,
        device=device,
    )
    return theta ** (-dimension_indices / head_dim)


def rotary_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    *,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RoPE cosine and sine tensors shaped (T, Dh)."""

    if positions.ndim != 1:
        raise ValueError("positions must have shape (T,)")

    if positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("positions must contain integers")

    if torch.any(positions < 0):
        raise ValueError("positions must be nonnegative")

    frequencies = inverse_frequencies(
        head_dim,
        theta=theta,
        device=positions.device,
    )
    angles = torch.outer(
        positions.to(torch.float32),
        frequencies,
    )
    duplicated_angles = torch.cat((angles, angles), dim=-1)

    return (
        duplicated_angles.cos().to(dtype),
        duplicated_angles.sin().to(dtype),
    )


def rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the final tensor dimension."""

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    head_dim = inputs.shape[-1]

    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError("final dimension must be a positive even integer")

    first_half = inputs[..., : head_dim // 2]
    second_half = inputs[..., head_dim // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K.

    Shapes:
        query: (B, Hq, T, Dh)
        key: (B, Hkv, T, Dh)
        positions: (T,)
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape (B, H, T, Dh)")

    if query.device != key.device:
        raise ValueError("query and key must use the same device")

    if positions.device != query.device:
        raise ValueError("positions must use the same device as query and key")

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch sizes must match")

    if query.shape[-2:] != key.shape[-2:]:
        raise ValueError("query and key sequence and head dimensions must match")

    sequence_length = query.shape[-2]

    if positions.shape != (sequence_length,):
        raise ValueError("positions must have shape (T,)")

    cosine, sine = rotary_cos_sin(
        positions,
        query.shape[-1],
        theta=theta,
        dtype=query.dtype,
    )
    cosine = cosine[None, None, :, :]
    sine = sine[None, None, :, :]

    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )
