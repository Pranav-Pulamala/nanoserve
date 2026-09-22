"""Benchmark PyTorch and Triton vector addition on a CUDA GPU."""

import torch
import triton
import triton.testing

from tokserve.kernels.vector import vector_add


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")
    print("dtype: float32")
    print("Times are median milliseconds; lower is better.")
    print()

    for size in (1_024, 4_096, 65_536, 1_048_576):
        x = torch.randn(size, device=device, dtype=torch.float32)
        y = torch.randn(size, device=device, dtype=torch.float32)

        expected = torch.add(x, y)
        actual = vector_add(x, y)
        torch.testing.assert_close(actual, expected)

        torch_ms = triton.testing.do_bench(
            lambda: torch.add(x, y),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        triton_ms = triton.testing.do_bench(
            lambda: vector_add(x, y),
            warmup=25,
            rep=100,
            return_mode="median",
        )

        print(
            f"elements={size:>9} | "
            f"PyTorch={torch_ms:.6f} ms | "
            f"Triton={triton_ms:.6f} ms"
        )


if __name__ == "__main__":
    main()
