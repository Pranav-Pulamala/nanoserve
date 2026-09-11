"""Small timing utilities with separate warmup and measurement phases."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter


@dataclass(frozen=True)
class BenchmarkResult:
    """Recorded durations from a benchmark measurement phase."""

    durations_seconds: tuple[float, ...]

    @property
    def iterations(self) -> int:
        """Return the number of measured iterations."""

        return len(self.durations_seconds)

    @property
    def median_seconds(self) -> float:
        """Return the median measured duration in seconds."""

        return median(self.durations_seconds)


def measure(
    function: Callable[[], object],
    *,
    warmup_iterations: int = 5,
    measured_iterations: int = 20,
) -> BenchmarkResult:
    """Time a zero-argument callable after completing warmup iterations."""

    if warmup_iterations < 0:
        message = "warmup_iterations must be nonnegative"
        raise ValueError(message)

    if measured_iterations < 1:
        message = "measured_iterations must be positive"
        raise ValueError(message)

    for _ in range(warmup_iterations):
        function()

    durations: list[float] = []

    for _ in range(measured_iterations):
        start = perf_counter()
        function()
        durations.append(perf_counter() - start)

    return BenchmarkResult(durations_seconds=tuple(durations))
