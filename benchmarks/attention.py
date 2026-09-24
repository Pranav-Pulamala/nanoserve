"""Benchmark PyTorch and Triton tiled attention on CUDA."""

from dataclasses import dataclass

import torch
import triton
import triton.testing

from tokserve.engine.attention import (
    causal_attention,
    repeat_key_value,
)
from tokserve.engine.paged.allocator import PhysicalBlockAllocator
from tokserve.engine.paged.sequence_cache import SequencePagedKVCache
from tokserve.engine.paged.storage import PagedKVStorage
from tokserve.kernels.paged_attention import (
    BLOCK_M as PAGED_BLOCK_M,
)
from tokserve.kernels.paged_attention import (
    BLOCK_N as PAGED_BLOCK_N,
)
from tokserve.kernels.paged_attention import triton_paged_attention
from tokserve.kernels.tiled_attention import (
    BLOCK_M as CONTIGUOUS_BLOCK_M,
)
from tokserve.kernels.tiled_attention import (
    BLOCK_N as CONTIGUOUS_BLOCK_N,
)
from tokserve.kernels.tiled_attention import triton_tiled_attention

CONTIGUOUS_WARMUP = 25
CONTIGUOUS_REPETITIONS = 100
PAGED_WARMUP = 10
PAGED_REPETITIONS = 50
PAGE_SIZE = 16


@dataclass(frozen=True)
class AttentionCase:
    """One attention benchmark configuration."""

    name: str
    batch_size: int
    num_query_heads: int
    num_key_value_heads: int
    query_length: int
    key_length: int
    head_dim: int

    @property
    def query_position_offset(self) -> int:
        """Return the absolute starting position of the query."""

        return self.key_length - self.query_length


CASES = (
    AttentionCase("prefill-small", 1, 8, 2, 16, 16, 64),
    AttentionCase("prefill-medium", 1, 8, 2, 64, 64, 64),
    AttentionCase("prefill-large", 1, 8, 2, 256, 256, 64),
    AttentionCase("prefill-mha", 1, 8, 8, 128, 128, 64),
    AttentionCase("decode-short", 1, 8, 2, 1, 16, 64),
    AttentionCase("decode-medium", 1, 8, 2, 1, 128, 64),
    AttentionCase("decode-long", 1, 8, 2, 1, 512, 64),
)


def pytorch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_offset: int,
) -> torch.Tensor:
    """Run TokServe's ordinary PyTorch GQA reference."""

    groups = query.shape[1] // key.shape[1]
    repeated_key = repeat_key_value(key, num_groups=groups)
    repeated_value = repeat_key_value(value, num_groups=groups)
    output, _ = causal_attention(
        query,
        repeated_key,
        repeated_value,
        query_position_offset=query_position_offset,
    )
    return output


def make_tensors(
    case: AttentionCase,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic CUDA Q/K/V tensors."""

    torch.manual_seed(900)
    query = torch.randn(
        case.batch_size,
        case.num_query_heads,
        case.query_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    key = torch.randn(
        case.batch_size,
        case.num_key_value_heads,
        case.key_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    value = torch.randn(
        case.batch_size,
        case.num_key_value_heads,
        case.key_length,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    )
    return query, key, value


def make_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
) -> SequencePagedKVCache:
    """Store one logical K/V sequence in physical pages."""

    key_length = key.shape[2]
    required_blocks = (key_length + PAGE_SIZE - 1) // PAGE_SIZE
    allocator = PhysicalBlockAllocator(required_blocks)
    storage = PagedKVStorage(
        num_layers=1,
        num_blocks=required_blocks,
        num_key_value_heads=key.shape[1],
        block_size=PAGE_SIZE,
        head_dim=key.shape[-1],
        device=key.device,
        dtype=key.dtype,
    )
    cache = SequencePagedKVCache(
        allocator=allocator,
        storage=storage,
        max_sequence_length=key_length,
    )
    cache.begin_append(key_length)
    cache.write_layer(0, key[0], value[0])
    cache.finish_append()
    return cache


def reconstructed_paged_attention(
    query: torch.Tensor,
    cache: SequencePagedKVCache,
    *,
    query_position_offset: int,
) -> torch.Tensor:
    """Run the existing PyTorch path that reconstructs logical paged K/V."""

    key, value = cache.read_layer(0)
    return pytorch_attention(
        query,
        key,
        value,
        query_position_offset=query_position_offset,
    )


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return comparison tolerances for one inference dtype."""

    if dtype == torch.float32:
        return 1e-4, 1e-5

    return 2e-2, 2e-2


def print_case(
    case: AttentionCase,
    dtype: torch.dtype,
    *,
    pytorch_ms: float,
    triton_ms: float,
    pytorch_paged_ms: float,
    triton_paged_ms: float,
) -> None:
    """Print one complete benchmark result row."""

    dtype_name = str(dtype).removeprefix("torch.")
    print(
        f"{case.name:>15} "
        f"dtype={dtype_name:>7} "
        f"B={case.batch_size} "
        f"Hq={case.num_query_heads} "
        f"Hkv={case.num_key_value_heads} "
        f"Tq={case.query_length:>3} "
        f"Tk={case.key_length:>3} "
        f"Dh={case.head_dim:>3} | "
        f"PyTorch={pytorch_ms:.6f} ms | "
        f"Triton={triton_ms:.6f} ms | "
        f"PyTorch-paged-reconstruct={pytorch_paged_ms:.6f} ms | "
        f"Triton-paged={triton_paged_ms:.6f} ms"
    )


@torch.inference_mode()
def main() -> None:
    """Validate and benchmark every configured attention case."""

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")
    print(
        f"Contiguous tiles: BLOCK_M={CONTIGUOUS_BLOCK_M}, BLOCK_N={CONTIGUOUS_BLOCK_N}"
    )
    print(
        "Paged tiles: "
        f"BLOCK_M={PAGED_BLOCK_M}, "
        f"BLOCK_N={PAGED_BLOCK_N}, "
        f"page_size={PAGE_SIZE}"
    )
    print(
        "Contiguous timing: "
        f"warmup={CONTIGUOUS_WARMUP}, "
        f"repetitions={CONTIGUOUS_REPETITIONS}"
    )
    print(f"Paged timing: warmup={PAGED_WARMUP}, repetitions={PAGED_REPETITIONS}")
    print("Times are median milliseconds; lower is better.")
    print(
        "PyTorch paged timing includes logical reconstruction through "
        "SequencePagedKVCache.read_layer()."
    )
    print()

    for dtype in (torch.float32, torch.float16):
        for case in CASES:
            query, key, value = make_tensors(case, dtype)
            cache = make_paged_cache(key, value)
            offset = case.query_position_offset

            expected = pytorch_attention(
                query,
                key,
                value,
                query_position_offset=offset,
            )
            triton_output = triton_tiled_attention(
                query,
                key,
                value,
                query_position_offset=offset,
            )
            paged_output = triton_paged_attention(
                query,
                cache.storage,
                cache.block_ids,
                layer_index=0,
                key_length=case.key_length,
                query_position_offset=offset,
            )

            rtol, atol = tolerance(dtype)
            torch.testing.assert_close(
                triton_output,
                expected,
                rtol=rtol,
                atol=atol,
            )
            torch.testing.assert_close(
                paged_output,
                expected,
                rtol=rtol,
                atol=atol,
            )

            pytorch_ms = triton.testing.do_bench(
                lambda: pytorch_attention(
                    query,
                    key,
                    value,
                    query_position_offset=offset,
                ),
                warmup=CONTIGUOUS_WARMUP,
                rep=CONTIGUOUS_REPETITIONS,
                return_mode="median",
            )
            triton_ms = triton.testing.do_bench(
                lambda: triton_tiled_attention(
                    query,
                    key,
                    value,
                    query_position_offset=offset,
                ),
                warmup=CONTIGUOUS_WARMUP,
                rep=CONTIGUOUS_REPETITIONS,
                return_mode="median",
            )
            pytorch_paged_ms = triton.testing.do_bench(
                lambda: reconstructed_paged_attention(
                    query,
                    cache,
                    query_position_offset=offset,
                ),
                warmup=PAGED_WARMUP,
                rep=PAGED_REPETITIONS,
                return_mode="median",
            )
            triton_paged_ms = triton.testing.do_bench(
                lambda: triton_paged_attention(
                    query,
                    cache.storage,
                    cache.block_ids,
                    layer_index=0,
                    key_length=case.key_length,
                    query_position_offset=offset,
                ),
                warmup=PAGED_WARMUP,
                rep=PAGED_REPETITIONS,
                return_mode="median",
            )

            print_case(
                case,
                dtype,
                pytorch_ms=pytorch_ms,
                triton_ms=triton_ms,
                pytorch_paged_ms=pytorch_paged_ms,
                triton_paged_ms=triton_paged_ms,
            )


if __name__ == "__main__":
    main()
