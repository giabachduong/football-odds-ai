"""Explore match outcomes and evaluate the bookmaker probability baseline."""

from pathlib import Path
import math

import pandas as pd


DATA_PATH = Path("data/processed/epl_2526_probabilities.csv")
OUTCOME_COLUMN = "FTR"
OUTCOME_TO_INDEX = {"H": 0, "D": 1, "A": 2}
ODDS_COLUMNS = ("AvgH", "AvgD", "AvgA")
PROBABILITY_COLUMNS = ("prob_home", "prob_draw", "prob_away")
NORMALIZED_COLUMNS = (*ODDS_COLUMNS, *PROBABILITY_COLUMNS, "overround")
PROBABILITY_TOLERANCE = 1e-9
LOG_LOSS_EPSILON = 1e-15


def outcome_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    """Return H/D/A counts and percentages, rejecting invalid labels."""
    if OUTCOME_COLUMN not in matches:
        raise ValueError(f"Dataset is missing required column: {OUTCOME_COLUMN}")

    outcomes = matches[OUTCOME_COLUMN]
    invalid = outcomes.isna() | ~outcomes.isin(OUTCOME_TO_INDEX)
    if invalid.any():
        invalid_count = int(invalid.sum())
        raise ValueError(f"Found {invalid_count} missing or invalid outcome labels.")

    counts = outcomes.value_counts().reindex(OUTCOME_TO_INDEX, fill_value=0)
    total = len(outcomes)
    return pd.DataFrame(
        {
            "count": counts,
            "percentage": counts / total * 100 if total else counts.astype(float),
        }
    )


def descriptive_statistics(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate requested descriptive statistics for odds and probabilities."""
    missing = [column for column in NORMALIZED_COLUMNS if column not in matches]
    if missing:
        raise ValueError(f"Dataset is missing analysis columns: {missing}")

    statistics = matches[list(NORMALIZED_COLUMNS)].agg(
        ["mean", "median", "std", "min", "max"]
    ).T
    return statistics.rename(
        columns={"mean": "Mean", "median": "Median", "std": "Std Dev", "min": "Min", "max": "Max"}
    )


def overround_extremes(matches: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the minimum- and maximum-overround match rows."""
    if "overround" not in matches:
        raise ValueError("Dataset is missing analysis column: overround")
    if matches.empty:
        raise ValueError("Cannot identify overround extremes in an empty dataset.")
    return matches.loc[matches["overround"].idxmin()], matches.loc[matches["overround"].idxmax()]


def data_quality_summary(matches: pd.DataFrame) -> dict[str, int]:
    """Count missing analysis values, invalid probabilities, and bad sums."""
    required = [OUTCOME_COLUMN, *NORMALIZED_COLUMNS]
    missing = [column for column in required if column not in matches]
    if missing:
        raise ValueError(f"Dataset is missing required analysis columns: {missing}")

    probability_values = matches[list(PROBABILITY_COLUMNS)]
    numeric_probabilities = probability_values.apply(pd.to_numeric, errors="coerce")
    missing_values = int(matches[required].isna().sum().sum())

    invalid_cell_count = 0
    for value in numeric_probabilities.to_numpy().ravel():
        if pd.isna(value) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            invalid_cell_count += 1

    bad_sum_count = 0
    for row in numeric_probabilities.itertuples(index=False, name=None):
        if any(pd.isna(value) for value in row):
            bad_sum_count += 1
            continue
        if not math.isclose(
            sum(float(value) for value in row),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_TOLERANCE,
        ):
            bad_sum_count += 1

    return {
        "missing_values": missing_values,
        "invalid_probability_cells": invalid_cell_count,
        "bad_probability_sums": bad_sum_count,
    }


def bookmaker_scores(matches: pd.DataFrame) -> tuple[float, float]:
    """Return multiclass log loss and Brier score using FTR only as labels.

    Log loss floors the probability assigned to the actual outcome at epsilon
    so that a zero probability does not cause an undefined logarithm. The
    multiclass Brier score sums squared errors across H, D, and A per match,
    then averages those per-match sums.
    """
    missing = [column for column in [OUTCOME_COLUMN, *PROBABILITY_COLUMNS] if column not in matches]
    if missing:
        raise ValueError(f"Dataset is missing scoring columns: {missing}")
    if matches.empty:
        raise ValueError("Cannot score an empty dataset.")

    labels = matches[OUTCOME_COLUMN].tolist()
    invalid_labels = [label for label in labels if label not in OUTCOME_TO_INDEX]
    if invalid_labels:
        raise ValueError(f"Found invalid outcome labels: {invalid_labels[:5]}")

    predictions = matches[list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    log_losses: list[float] = []
    brier_scores: list[float] = []

    for label, prediction in zip(labels, predictions.itertuples(index=False, name=None)):
        probabilities = [float(value) for value in prediction]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Prediction probabilities must be finite and between 0 and 1.")
        if not math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
        ):
            raise ValueError("Prediction probabilities must sum to 1 within 1e-9.")

        actual_index = OUTCOME_TO_INDEX[label]
        actual_probability = max(probabilities[actual_index], LOG_LOSS_EPSILON)
        log_losses.append(-math.log(actual_probability))

        squared_errors = [
            (probability - (1.0 if index == actual_index else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        ]
        brier_scores.append(sum(squared_errors))

    return sum(log_losses) / len(log_losses), sum(brier_scores) / len(brier_scores)


def print_report(matches: pd.DataFrame) -> None:
    """Print outcome, distribution, overround, score, and quality results."""
    outcomes = outcome_distribution(matches)
    statistics = descriptive_statistics(matches)
    lowest, highest = overround_extremes(matches)
    log_loss, brier_score = bookmaker_scores(matches)
    quality = data_quality_summary(matches)

    print("=== Football Odds Intelligence AI: Day 3 EDA ===")
    print(f"Matches analyzed: {len(matches)}")
    print("\n=== Outcome distribution ===")
    print(outcomes.to_string(formatters={"percentage": "{:.2f}%".format}))
    valid_outcomes = int(outcomes["count"].sum())
    print(f"Valid H/D/A outcomes: {valid_outcomes} / {len(matches)}")

    print("\n=== Odds, probabilities, and overround ===")
    print(statistics.to_string(float_format=lambda value: f"{value:.6f}"))

    print("\n=== Overround extremes ===")
    print("Lowest:")
    print(_match_description(lowest))
    print("Highest:")
    print(_match_description(highest))

    print("\n=== Bookmaker baseline evaluation (lower is better) ===")
    print(f"Multiclass log loss: {log_loss:.6f}")
    print(f"Multiclass Brier score: {brier_score:.6f}")
    print("Log loss: average negative log probability assigned to the actual result.")
    print("Brier score: average sum of squared H/D/A probability errors per match.")

    print("\n=== Data quality ===")
    print(f"Missing values in analysis columns: {quality['missing_values']}")
    print(f"Invalid probability cells: {quality['invalid_probability_cells']}")
    print(f"Rows with normalized probabilities not summing to 1: {quality['bad_probability_sums']}")


def _match_description(match: pd.Series) -> str:
    """Format identifying fields and overround for one match."""
    date = match.get("Date", "(date unavailable)")
    home = match.get("HomeTeam", "(home team unavailable)")
    away = match.get("AwayTeam", "(away team unavailable)")
    return f"{date}: {home} vs {away} — overround {float(match['overround']):.6f}"


def main() -> None:
    """Load the processed dataset and print the EDA report."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    matches = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    matches.columns = matches.columns.str.strip()
    print_report(matches)


if __name__ == "__main__":
    main()
