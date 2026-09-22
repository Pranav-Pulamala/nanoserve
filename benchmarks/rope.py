"""Benchmark PyTorch and Triton RoPE on CUDA."""

import torch
import triton
import triton.testing

from tokserve.engine.rope import apply_rope
from tokserve.kernels.rope import triton_apply_rope

THETA = 10_000.0

# (batch, query heads, key/value heads, sequence length, head dim, start position)
CASES = (
    (1, 8, 2, 1, 64, 128),  # One-token cached decode
    (1, 8, 2, 128, 64, 0),  # Prefill
    (2, 16, 4, 128, 128, 64),
    (4, 16, 4, 256, 64, 0),
)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")

    torch.manual_seed(8)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")
    print(f"RoPE theta: {THETA}")
    print("Median complete-call time in milliseconds; lower is better.")
    print()

    for dtype in (torch.float32, torch.float16):
        tolerance = 2e-5 if dtype == torch.float32 else 2e-3

        for batch, query_heads, kv_heads, length, head_dim, start in CASES:
            query = torch.randn(
                batch,
                query_heads,
                length,
                head_dim,
                device="cuda",
                dtype=dtype,
            )
            key = torch.randn(
                batch,
                kv_heads,
                length,
                head_dim,
                device="cuda",
                dtype=dtype,
            )
            positions = torch.arange(
                start,
                start + length,
                device="cuda",
                dtype=torch.int64,
            )

            expected_query, expected_key = apply_rope(
                query, key, positions, theta=THETA
            )
            actual_query, actual_key = triton_apply_rope(
                query, key, positions, theta=THETA
            )
            torch.testing.assert_close(
                actual_query,
                expected_query,
                rtol=tolerance,
                atol=tolerance,
            )
            torch.testing.assert_close(
                actual_key,
                expected_key,
                rtol=tolerance,
                atol=tolerance,
            )

            pytorch_ms = triton.testing.do_bench(
                lambda: apply_rope(query, key, positions, theta=THETA),
                warmup=25,
                rep=100,
                return_mode="median",
            )
            triton_ms = triton.testing.do_bench(
                lambda: triton_apply_rope(query, key, positions, theta=THETA),
                warmup=25,
                rep=100,
                return_mode="median",
            )

            print(
                f"dtype={str(dtype).removeprefix('torch.'):>7} "
                f"B={batch} Hq={query_heads} Hkv={kv_heads} "
                f"T={length} Dh={head_dim} start={start} | "
                f"PyTorch={pytorch_ms:.6f} ms | "
                f"Triton={triton_ms:.6f} ms"
            )


if __name__ == "__main__":
    main()
