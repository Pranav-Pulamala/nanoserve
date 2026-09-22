"""Benchmark PyTorch and Triton RMSNorm on CUDA."""

import torch
import triton
import triton.testing

from tokserve.engine.layers import RMSNorm
from tokserve.kernels.rmsnorm import triton_rms_norm

EPSILON = 1e-6
SHAPES = (
    (1, 128),
    (32, 1024),
    (256, 1024),
    (256, 4096),
)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")
    print(f"epsilon: {EPSILON}")
    print("Median end-to-end call time in milliseconds; lower is better.")
    print()

    for dtype in (torch.float32, torch.float16):
        for rows, hidden_size in SHAPES:
            inputs = torch.randn(
                rows,
                hidden_size,
                device="cuda",
                dtype=dtype,
            )
            weight = torch.linspace(
                0.5,
                1.5,
                hidden_size,
                device="cuda",
                dtype=torch.float32,
            )

            reference = RMSNorm(hidden_size, epsilon=EPSILON).to("cuda")
            reference.weight.copy_(weight)

            expected = reference(inputs)
            actual = triton_rms_norm(inputs, weight, epsilon=EPSILON)
            tolerance = 2e-5 if dtype == torch.float32 else 2e-3
            torch.testing.assert_close(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
            )

            pytorch_ms = triton.testing.do_bench(
                lambda: reference(inputs),
                warmup=25,
                rep=100,
                return_mode="median",
            )
            triton_ms = triton.testing.do_bench(
                lambda: triton_rms_norm(inputs, weight, epsilon=EPSILON),
                warmup=25,
                rep=100,
                return_mode="median",
            )

            print(
                f"dtype={str(dtype).removeprefix('torch.'):>7} "
                f"shape=({rows:>3}, {hidden_size:>4}) | "
                f"PyTorch={pytorch_ms:.6f} ms | "
                f"Triton={triton_ms:.6f} ms"
            )


if __name__ == "__main__":
    main()
