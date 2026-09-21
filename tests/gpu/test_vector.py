import pytest
import torch

from nanoserve.kernels.vector import vector_add


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("length", [1, 1000, 1024, 1025, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_triton_vector_add_matches_torch(
    length: int,
    dtype: torch.dtype,
) -> None:
    x = torch.randn(length, device="cuda", dtype=dtype)
    y = torch.randn(length, device="cuda", dtype=dtype)

    actual = vector_add(x, y)
    expected = torch.add(x, y)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_empty_cuda_vectors_return_empty_output() -> None:
    x = torch.empty(0, device="cuda")
    y = torch.empty(0, device="cuda")

    actual = vector_add(x, y)

    assert actual.shape == (0,)
    assert actual.device.type == "cuda"


def test_cpu_vectors_are_rejected_without_importing_triton() -> None:
    x = torch.ones(4)
    y = torch.ones(4)

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        vector_add(x, y)


def test_mismatched_shapes_are_rejected() -> None:
    x = torch.ones(4)
    y = torch.ones(5)

    with pytest.raises(ValueError, match="matching shapes"):
        vector_add(x, y)


def test_non_vector_inputs_are_rejected() -> None:
    x = torch.ones(2, 2)
    y = torch.ones(2, 2)

    with pytest.raises(ValueError, match="one-dimensional"):
        vector_add(x, y)
