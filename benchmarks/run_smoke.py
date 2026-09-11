"""Execute a small CPU-only smoke check of the benchmark harness."""

from benchmarks.timing import measure


def operation() -> int:
    """Return a deterministic value for the smoke benchmark."""

    return sum(range(100))


def main() -> None:
    """Run the benchmark harness and report its measured sample count."""

    result = measure(
        operation,
        warmup_iterations=5,
        measured_iterations=20,
    )

    print(f"measured iterations: {result.iterations}")
    print(f"median seconds: {result.median_seconds:.9f}")


if __name__ == "__main__":
    main()