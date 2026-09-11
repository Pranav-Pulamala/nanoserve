import pytest

from benchmarks.timing import measure


def test_measure_separates_warmup_and_measured_iterations() -> None:
    call_count = 0

    def operation() -> None:
        nonlocal call_count
        call_count += 1

    result = measure(
        operation,
        warmup_iterations=2,
        measured_iterations=3,
    )

    assert call_count == 5
    assert result.iterations == 3
    assert len(result.durations_seconds) == 3
    assert all(duration >= 0.0 for duration in result.durations_seconds)
    assert result.median_seconds >= 0.0


@pytest.mark.parametrize(
    ("warmup_iterations", "measured_iterations", "message"),
    [
        (-1, 1, "warmup_iterations must be nonnegative"),
        (0, 0, "measured_iterations must be positive"),
    ],
)
def test_measure_rejects_invalid_iteration_counts(
    warmup_iterations: int,
    measured_iterations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        measure(
            lambda: None,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        )