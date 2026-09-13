"""Explicit PyTorch device and dtype selection."""

import torch

SUPPORTED_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
)

_DTYPE_NAMES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def is_mps_available() -> bool:
    """Return whether the Apple Metal accelerator is available."""

    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def resolve_device(
    requested: str | torch.device | None = None,
) -> torch.device:
    """Resolve an inference device without silently changing explicit requests.

    Automatic selection uses CUDA, then MPS, then CPU.
    """

    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if is_mps_available():
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(requested)

    if device.type == "cpu":
        return device

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    if device.type == "mps":
        if not is_mps_available():
            raise RuntimeError("MPS was requested but is not available")
        return device

    raise ValueError(f"unsupported device type: {device.type}")


def resolve_dtype(
    requested: str | torch.dtype | None = None,
) -> torch.dtype:
    """Resolve one of the supported floating-point inference dtypes."""

    if requested is None:
        return torch.float32

    if isinstance(requested, str):
        try:
            return _DTYPE_NAMES[requested]
        except KeyError as error:
            raise ValueError(f"unsupported dtype: {requested}") from error

    if requested not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {requested}")

    return requested
