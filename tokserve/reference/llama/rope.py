"""Rotary positional embeddings implemented with NumPy."""

import numpy as np

from tokserve.reference.ops import FloatArray, IntArray


def inverse_frequencies(
    head_dim: int,
    *,
    theta: float = 10_000.0,
) -> FloatArray:
    """Return RoPE inverse frequencies with shape (Dh / 2,)."""

    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")

    if theta <= 0.0:
        raise ValueError("theta must be positive")

    dimension_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    frequencies = theta ** (-dimension_indices / head_dim)
    return np.asarray(frequencies, dtype=np.float64)


def rotary_cos_sin(
    positions: IntArray,
    head_dim: int,
    *,
    theta: float = 10_000.0,
) -> tuple[FloatArray, FloatArray]:
    """Return cosine and sine tensors with shape (T, Dh)."""

    if positions.ndim != 1:
        raise ValueError("positions must have shape (T,)")

    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError("positions must contain integers")

    if np.any(positions < 0):
        raise ValueError("positions must be nonnegative")

    frequencies = inverse_frequencies(head_dim, theta=theta)
    angles = np.outer(
        positions.astype(np.float64),
        frequencies,
    )
    duplicated_angles = np.concatenate((angles, angles), axis=-1)

    return (
        np.asarray(np.cos(duplicated_angles), dtype=np.float64),
        np.asarray(np.sin(duplicated_angles), dtype=np.float64),
    )


def rotate_half(inputs: FloatArray) -> FloatArray:
    """Rotate the two halves of the final dimension."""

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")

    head_dim = inputs.shape[-1]

    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError("final dimension must be a positive even integer")

    first_half = inputs[..., : head_dim // 2]
    second_half = inputs[..., head_dim // 2 :]
    return np.asarray(
        np.concatenate((-second_half, first_half), axis=-1),
        dtype=np.float64,
    )


def apply_rope(
    query: FloatArray,
    key: FloatArray,
    positions: IntArray,
    *,
    theta: float = 10_000.0,
) -> tuple[FloatArray, FloatArray]:
    """Apply RoPE to query and key tensors.

    Shapes:
        query: (B, Hq, T, Dh)
        key: (B, Hkv, T, Dh)
        positions: (T,)
        rotated query: (B, Hq, T, Dh)
        rotated key: (B, Hkv, T, Dh)
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape (B, H, T, Dh)")

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch sizes must match")

    if query.shape[-2:] != key.shape[-2:]:
        raise ValueError("query and key sequence and head dimensions must match")

    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]

    if positions.shape != (sequence_length,):
        raise ValueError("positions must have shape (T,)")

    cosine, sine = rotary_cos_sin(
        positions,
        head_dim,
        theta=theta,
    )
    cosine = cosine[None, None, :, :]
    sine = sine[None, None, :, :]

    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine

    return (
        np.asarray(rotated_query, dtype=np.float64),
        np.asarray(rotated_key, dtype=np.float64),
    )
