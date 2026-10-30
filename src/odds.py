"""Convert decimal bookmaker odds into 1X2 implied probabilities."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class OneXTwoProbabilities:
    """Raw and normalized probabilities for home, draw, and away outcomes.

    ``overround`` is the sum of the raw implied probabilities minus one.
    ``home``, ``draw``, and ``away`` are normalized to sum to one.
    """

    raw_home: float
    raw_draw: float
    raw_away: float
    raw_probability_sum: float
    overround: float
    home: float
    draw: float
    away: float


def _validate_odds(odds: Real) -> float:
    """Return odds as a float, rejecting non-numeric or invalid values."""
    if isinstance(odds, bool) or not isinstance(odds, Real):
        raise ValueError("Odds must be numeric, finite, and strictly positive.")

    value = float(odds)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Odds must be numeric, finite, and strictly positive.")
    return value


def implied_probability(odds: Real) -> float:
    """Convert positive decimal odds to raw implied probability (1 / odds)."""
    return 1.0 / _validate_odds(odds)


def normalized_1x2_probabilities(
    home_odds: Real,
    draw_odds: Real,
    away_odds: Real,
) -> OneXTwoProbabilities:
    """Calculate raw probabilities, overround, and normalized 1X2 values.

    Normalization divides each raw probability by their sum, making the three
    returned normalized probabilities add up to one.
    """
    raw_home = implied_probability(home_odds)
    raw_draw = implied_probability(draw_odds)
    raw_away = implied_probability(away_odds)

    raw_probability_sum = raw_home + raw_draw + raw_away
    overround = raw_probability_sum - 1.0

    return OneXTwoProbabilities(
        raw_home=raw_home,
        raw_draw=raw_draw,
        raw_away=raw_away,
        raw_probability_sum=raw_probability_sum,
        overround=overround,
        home=raw_home / raw_probability_sum,
        draw=raw_draw / raw_probability_sum,
        away=raw_away / raw_probability_sum,
    )
