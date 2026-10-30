"""Tests for outcome summaries and bookmaker-baseline evaluation."""

import math

import pandas as pd
import pytest

from src.eda import (
    bookmaker_scores,
    data_quality_summary,
    descriptive_statistics,
    outcome_distribution,
    overround_extremes,
)


def test_outcome_distribution_counts_and_percentages() -> None:
    """H, D, and A counts and shares are computed from actual outcomes."""
    matches = pd.DataFrame({"FTR": ["H", "D", "A", "H"]})

    result = outcome_distribution(matches)

    assert result["count"].to_dict() == {"H": 2, "D": 1, "A": 1}
    assert result.loc["H", "percentage"] == 50.0
    assert result.loc["D", "percentage"] == 25.0
    assert result.loc["A", "percentage"] == 25.0


@pytest.mark.parametrize("invalid_label", [None, "X"])
def test_outcome_distribution_rejects_invalid_labels(invalid_label: object) -> None:
    """Missing and unexpected result labels are reported as invalid."""
    with pytest.raises(ValueError, match="invalid outcome"):
        outcome_distribution(pd.DataFrame({"FTR": ["H", invalid_label]}))


def test_descriptive_statistics_use_sample_standard_deviation() -> None:
    """Statistics include mean, median, sample standard deviation, min, max."""
    matches = pd.DataFrame(
        {
            "AvgH": [1.0, 3.0], "AvgD": [2.0, 4.0], "AvgA": [3.0, 5.0],
            "prob_home": [0.2, 0.6], "prob_draw": [0.3, 0.4],
            "prob_away": [0.5, 0.0], "overround": [0.0, 0.2],
        }
    )

    result = descriptive_statistics(matches)

    assert result.loc["AvgH", "Mean"] == 2.0
    assert result.loc["AvgH", "Median"] == 2.0
    assert math.isclose(result.loc["AvgH", "Std Dev"], math.sqrt(2.0))
    assert result.loc["AvgH", "Min"] == 1.0
    assert result.loc["AvgH", "Max"] == 3.0


def test_overround_extremes_identify_matches() -> None:
    """Minimum and maximum overround rows preserve match identifiers."""
    matches = pd.DataFrame(
        {
            "Date": ["Match 1", "Match 2", "Match 3"],
            "HomeTeam": ["A", "B", "C"],
            "AwayTeam": ["D", "E", "F"],
            "overround": [0.04, 0.07, 0.05],
        }
    )

    lowest, highest = overround_extremes(matches)

    assert lowest["Date"] == "Match 1"
    assert highest["Date"] == "Match 2"


def test_scores_map_h_draw_away_and_sum_brier_errors_per_match() -> None:
    """Known predictions verify H/D/A mapping and multiclass Brier definition."""
    matches = pd.DataFrame(
        {
            "FTR": ["H", "D", "A"],
            "prob_home": [0.8, 0.2, 0.1],
            "prob_draw": [0.1, 0.7, 0.2],
            "prob_away": [0.1, 0.1, 0.7],
        }
    )

    log_loss, brier_score = bookmaker_scores(matches)

    assert math.isclose(log_loss, -math.log(0.8 * 0.7 * 0.7) / 3)
    # Each row has squared error sum 0.06, 0.14, and 0.14, respectively.
    assert math.isclose(brier_score, (0.06 + 0.14 + 0.14) / 3)


def test_log_loss_uses_epsilon_for_zero_probability() -> None:
    """A zero probability for the actual class produces a finite score."""
    matches = pd.DataFrame(
        {"FTR": ["H"], "prob_home": [0.0], "prob_draw": [0.5], "prob_away": [0.5]}
    )

    log_loss, _ = bookmaker_scores(matches)

    assert math.isfinite(log_loss)
    assert math.isclose(log_loss, -math.log(1e-15))


def test_quality_summary_finds_missing_invalid_and_unnormalized_values() -> None:
    """Quality checks count missing fields, invalid probability cells, and bad sums."""
    matches = pd.DataFrame(
        {
            "FTR": ["H", "D"],
            "AvgH": [2.0, 3.0], "AvgD": [3.0, 4.0], "AvgA": [4.0, 5.0],
            "prob_home": [0.5, 1.2], "prob_draw": [0.3, 0.0],
            "prob_away": [0.2, 0.0], "overround": [0.05, 0.06],
        }
    )
    matches.loc[1, "AvgH"] = float("nan")

    result = data_quality_summary(matches)

    assert result == {
        "missing_values": 1,
        "invalid_probability_cells": 1,
        "bad_probability_sums": 1,
    }


def test_scores_reject_invalid_outcome_labels_and_probability_rows() -> None:
    """Scoring refuses unknown labels and rows that are not probability vectors."""
    invalid_label = pd.DataFrame(
        {"FTR": ["X"], "prob_home": [0.4], "prob_draw": [0.3], "prob_away": [0.3]}
    )
    invalid_probabilities = pd.DataFrame(
        {"FTR": ["H"], "prob_home": [0.5], "prob_draw": [0.3], "prob_away": [0.3]}
    )

    with pytest.raises(ValueError, match="invalid outcome"):
        bookmaker_scores(invalid_label)
    with pytest.raises(ValueError, match="sum to 1"):
        bookmaker_scores(invalid_probabilities)
