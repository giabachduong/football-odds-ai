"""Tests for decimal odds and normalized 1X2 probabilities."""

import math

import pytest

from src.odds import implied_probability, normalized_1x2_probabilities


def test_odds_of_two_have_raw_implied_probability_of_one_half() -> None:
    """Odds 2.0 mean one unit staked has a raw implied probability of 0.5."""
    assert implied_probability(2.0) == 0.5


def test_normalized_probabilities_sum_to_one() -> None:
    """The normalized home, draw, and away values form a probability set."""
    result = normalized_1x2_probabilities(2.0, 3.5, 4.0)

    assert math.isclose(result.home + result.draw + result.away, 1.0)


def test_expected_results_for_odds_two_three_point_five_and_four() -> None:
    """Check raw values, overround, and proportional normalization."""
    result = normalized_1x2_probabilities(2.0, 3.5, 4.0)

    assert math.isclose(result.raw_home, 0.5)
    assert math.isclose(result.raw_draw, 2.0 / 7.0)
    assert math.isclose(result.raw_away, 0.25)
    assert math.isclose(result.raw_probability_sum, 29.0 / 28.0)
    assert math.isclose(result.overround, 1.0 / 28.0)
    assert math.isclose(result.home, 14.0 / 29.0)
    assert math.isclose(result.draw, 8.0 / 29.0)
    assert math.isclose(result.away, 7.0 / 29.0)


@pytest.mark.parametrize("odds", [0, -1.0, math.inf, -math.inf, math.nan, "invalid", None])
def test_invalid_single_odds_are_rejected(odds: object) -> None:
    """Zero, negative, non-finite, and non-numeric odds are invalid."""
    with pytest.raises(ValueError):
        implied_probability(odds)  # type: ignore[arg-type]


@pytest.mark.parametrize("position", ["home", "draw", "away"])
def test_missing_1x2_odds_are_rejected(position: str) -> None:
    """Each of the three outcomes must have odds supplied."""
    odds: dict[str, float | None] = {"home": 2.0, "draw": 3.5, "away": 4.0}
    odds[position] = None

    with pytest.raises(ValueError):
        normalized_1x2_probabilities(
            odds["home"],  # type: ignore[arg-type]
            odds["draw"],  # type: ignore[arg-type]
            odds["away"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("home_odds", "draw_odds", "away_odds"),
    [(2.0, 0, 4.0), (2.0, 3.5, -1.0), (math.inf, 3.5, 4.0), (2.0, "invalid", 4.0)],
)
def test_invalid_odds_in_any_1x2_position_are_rejected(
    home_odds: object, draw_odds: object, away_odds: object
) -> None:
    """Invalid values in any outcome position are rejected."""
    with pytest.raises(ValueError):
        normalized_1x2_probabilities(
            home_odds,  # type: ignore[arg-type]
            draw_odds,  # type: ignore[arg-type]
            away_odds,  # type: ignore[arg-type]
        )
