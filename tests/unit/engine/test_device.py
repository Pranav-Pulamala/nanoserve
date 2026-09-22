import pytest
import torch

from tokserve.engine.device import (
    is_mps_available,
    resolve_device,
    resolve_dtype,
)


def test_explicit_cpu_device_always_resolves() -> None:
    device = resolve_device("cpu")

    assert device == torch.device("cpu")


def test_torch_device_input_is_supported() -> None:
    requested = torch.device("cpu")

    assert resolve_device(requested) == requested


def test_automatic_device_resolution_is_predictable() -> None:
    device = resolve_device()

    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif is_mps_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_auto_string_matches_default_resolution() -> None:
    assert resolve_device("auto") == resolve_device()


def test_explicit_cuda_request_is_not_silently_changed() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA.*not available"):
            resolve_device("cuda")


def test_explicit_mps_request_is_not_silently_changed() -> None:
    if is_mps_available():
        assert resolve_device("mps").type == "mps"
    else:
        with pytest.raises(RuntimeError, match="MPS.*not available"):
            resolve_device("mps")


def test_unsupported_device_type_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unsupported device type"):
        resolve_device("meta")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_dtype_names_resolve(
    name: str,
    expected: torch.dtype,
) -> None:
    assert resolve_dtype(name) == expected


def test_default_dtype_is_float32() -> None:
    assert resolve_dtype() == torch.float32


def test_torch_dtype_input_is_preserved() -> None:
    assert resolve_dtype(torch.bfloat16) == torch.bfloat16


@pytest.mark.parametrize(
    "unsupported",
    [
        "float64",
        torch.float64,
        torch.int64,
    ],
)
def test_unsupported_dtype_fails_clearly(
    unsupported: str | torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_dtype(unsupported)
