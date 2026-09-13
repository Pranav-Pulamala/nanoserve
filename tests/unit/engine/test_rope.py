import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close

from nanoserve.engine.rope import apply_rope, rotate_half
from nanoserve.reference.llama.rope import (
    apply_rope as numpy_apply_rope,
)

RTOL = 1e-5
ATOL = 1e-6


def test_torch_rope_matches_numpy_reference() -> None:
    query = np.arange(32, dtype=np.float64).reshape(1, 2, 4, 4)
    key = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    positions = np.arange(4, dtype=np.int64)

    torch_query, torch_key = apply_rope(
        torch.tensor(query, dtype=torch.float32),
        torch.tensor(key, dtype=torch.float32),
        torch.tensor(positions, dtype=torch.int64),
        theta=10_000.0,
    )
    numpy_query, numpy_key = numpy_apply_rope(
        query,
        key,
        positions,
        theta=10_000.0,
    )

    assert_allclose(
        torch_query.numpy(),
        numpy_query,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        torch_key.numpy(),
        numpy_key,
        rtol=RTOL,
        atol=ATOL,
    )


def test_position_zero_does_not_rotate() -> None:
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = torch.tensor([[[[4.0, 3.0, 2.0, 1.0]]]])

    rotated_query, rotated_key = apply_rope(
        query,
        key,
        torch.tensor([0]),
        theta=10_000.0,
    )

    assert_close(rotated_query, query)
    assert_close(rotated_key, key)


def test_rope_preserves_dtype_and_device() -> None:
    query = torch.ones(1, 2, 3, 4, dtype=torch.float32)
    key = torch.ones(1, 1, 3, 4, dtype=torch.float32)
    positions = torch.arange(3, device=query.device)

    rotated_query, rotated_key = apply_rope(
        query,
        key,
        positions,
        theta=10_000.0,
    )

    assert rotated_query.dtype == query.dtype
    assert rotated_key.dtype == key.dtype
    assert rotated_query.device == query.device
    assert rotated_key.device == key.device


def test_rope_preserves_vector_norms() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 2, 3, 4, generator=generator)
    key = torch.randn(1, 1, 3, 4, generator=generator)
    positions = torch.arange(3)

    rotated_query, rotated_key = apply_rope(
        query,
        key,
        positions,
        theta=10_000.0,
    )

    assert_close(
        torch.linalg.vector_norm(rotated_query, dim=-1),
        torch.linalg.vector_norm(query, dim=-1),
    )
    assert_close(
        torch.linalg.vector_norm(rotated_key, dim=-1),
        torch.linalg.vector_norm(key, dim=-1),
    )


def test_rotate_half_rejects_odd_dimension() -> None:
    with pytest.raises(ValueError, match="positive even integer"):
        rotate_half(torch.ones(2, 3))


def test_rope_rejects_position_device_mismatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    query = torch.ones(1, 2, 3, 4, device="cuda")
    key = torch.ones(1, 1, 3, 4, device="cuda")
    positions = torch.arange(3, device="cpu")

    with pytest.raises(ValueError, match="same device"):
        apply_rope(
            query,
            key,
            positions,
            theta=10_000.0,
        )
