"""Focused validation of online-softmax accumulation."""

import torch
from torch.testing import assert_close

from tokserve.engine.tiled_attention import online_softmax_update


def initial_state(
    *,
    query_rows: int,
    value_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create empty float32 online-softmax state."""

    running_max = torch.full(
        (1, 1, query_rows, 1),
        float("-inf"),
    )
    running_sum = torch.zeros(1, 1, query_rows, 1)
    accumulator = torch.zeros(1, 1, query_rows, value_size)
    return running_max, running_sum, accumulator


def ordinary_result(
    scores: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute the corresponding ordinary stable-softmax result."""

    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, values)


def finish(
    running_sum: torch.Tensor,
    accumulator: torch.Tensor,
) -> torch.Tensor:
    """Normalize the online output accumulator."""

    return accumulator / running_sum


def test_one_tile_matches_ordinary_softmax() -> None:
    scores = torch.tensor([[[[1.0, -2.0, 3.0]]]])
    values = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]])
    running_max, running_sum, accumulator = initial_state(
        query_rows=1,
        value_size=2,
    )

    running_max, running_sum, accumulator = online_softmax_update(
        running_max,
        running_sum,
        accumulator,
        scores,
        values,
    )

    actual = finish(running_sum, accumulator)
    expected = ordinary_result(scores, values)

    assert torch.isfinite(running_max).all()
    assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_multiple_tiles_match_ordinary_softmax() -> None:
    full_scores = torch.tensor([[[[1.0, -2.0, 3.0, 0.5, 4.0]]]])
    full_values = torch.tensor(
        [
            [
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                ]
            ]
        ]
    )
    running_max, running_sum, accumulator = initial_state(
        query_rows=1,
        value_size=2,
    )

    for start, end in ((0, 2), (2, 4), (4, 5)):
        running_max, running_sum, accumulator = online_softmax_update(
            running_max,
            running_sum,
            accumulator,
            full_scores[..., start:end],
            full_values[..., start:end, :],
        )

    actual = finish(running_sum, accumulator)
    expected = ordinary_result(full_scores, full_values)

    assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_old_accumulator_is_rescaled_when_maximum_increases() -> None:
    """Fail if earlier output contributions remain on the old scale."""

    first_scores = torch.tensor([[[[0.0, 0.0]]]])
    first_values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    second_scores = torch.tensor([[[[12.0]]]])
    second_values = torch.tensor([[[[100.0, 200.0]]]])

    running_max, running_sum, accumulator = initial_state(
        query_rows=1,
        value_size=2,
    )
    running_max, running_sum, accumulator = online_softmax_update(
        running_max,
        running_sum,
        accumulator,
        first_scores,
        first_values,
    )
    running_max, running_sum, accumulator = online_softmax_update(
        running_max,
        running_sum,
        accumulator,
        second_scores,
        second_values,
    )

    full_scores = torch.cat((first_scores, second_scores), dim=-1)
    full_values = torch.cat((first_values, second_values), dim=-2)
    expected = ordinary_result(full_scores, full_values)
    actual = finish(running_sum, accumulator)

    assert running_max.item() == 12.0
    assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_extreme_positive_and_negative_scores_remain_finite() -> None:
    full_scores = torch.tensor([[[[1000.0, -1000.0, 999.0, -999.0]]]])
    full_values = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
    running_max, running_sum, accumulator = initial_state(
        query_rows=1,
        value_size=1,
    )

    for start, end in ((0, 1), (1, 3), (3, 4)):
        running_max, running_sum, accumulator = online_softmax_update(
            running_max,
            running_sum,
            accumulator,
            full_scores[..., start:end],
            full_values[..., start:end, :],
        )

    actual = finish(running_sum, accumulator)
    expected = ordinary_result(full_scores, full_values)

    assert torch.isfinite(actual).all()
    assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_uneven_final_tile_matches_ordinary_softmax() -> None:
    torch.manual_seed(300)
    full_scores = torch.randn(1, 2, 3, 7)
    full_values = torch.randn(1, 2, 7, 5)
    running_max, running_sum, accumulator = initial_state(
        query_rows=3,
        value_size=5,
    )
    running_max = running_max.expand(1, 2, 3, 1).clone()
    running_sum = running_sum.expand(1, 2, 3, 1).clone()
    accumulator = accumulator.expand(1, 2, 3, 5).clone()

    for start, end in ((0, 3), (3, 6), (6, 7)):
        running_max, running_sum, accumulator = online_softmax_update(
            running_max,
            running_sum,
            accumulator,
            full_scores[..., start:end],
            full_values[..., start:end, :],
        )

    actual = finish(running_sum, accumulator)
    expected = ordinary_result(full_scores, full_values)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_causal_masked_scores_contribute_zero_probability() -> None:
    scores = torch.tensor(
        [
            [
                [
                    [1.0, float("-inf"), float("-inf")],
                    [1.0, 2.0, float("-inf")],
                    [1.0, 2.0, 3.0],
                ]
            ]
        ]
    )
    values = torch.tensor([[[[1.0], [10.0], [100.0]]]])
    running_max, running_sum, accumulator = initial_state(
        query_rows=3,
        value_size=1,
    )

    for start, end in ((0, 2), (2, 3)):
        running_max, running_sum, accumulator = online_softmax_update(
            running_max,
            running_sum,
            accumulator,
            scores[..., start:end],
            values[..., start:end, :],
        )

    actual = finish(running_sum, accumulator)
    expected = ordinary_result(scores, values)

    assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual[0, 0, 0, 0] == 1.0
