"""Add bookmaker-implied 1X2 probabilities to the match dataset."""

from pathlib import Path
import math

import pandas as pd

if __package__ in (None, ""):
    from odds import normalized_1x2_probabilities
else:
    from src.odds import normalized_1x2_probabilities


RAW_DATA_PATH = Path("data/raw/epl_2526.csv")
PROCESSED_DATA_PATH = Path("data/processed/epl_2526_probabilities.csv")
ODDS_COLUMNS = ("AvgH", "AvgD", "AvgA")
PROBABILITY_COLUMNS = (
    "raw_home_probability",
    "raw_draw_probability",
    "raw_away_probability",
    "raw_probability_sum",
    "overround",
    "prob_home",
    "prob_draw",
    "prob_away",
)
NORMALIZATION_TOLERANCE = 1e-9


def process_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return a copy with probabilities and valid/invalid match counts.

    A match is valid when all three bookmaker odds are numeric, finite, and
    strictly positive. Historical result columns, if present, are preserved
    but are never read while calculating probabilities.
    """
    missing_columns = [column for column in ODDS_COLUMNS if column not in matches]
    if missing_columns:
        raise ValueError(f"Dataset is missing required odds columns: {missing_columns}")

    numeric_odds = {
        column: pd.to_numeric(matches[column], errors="coerce")
        for column in ODDS_COLUMNS
    }

    valid_rows: list[bool] = []
    for position in range(len(matches)):
        odds = [numeric_odds[column].iloc[position] for column in ODDS_COLUMNS]
        valid_rows.append(
            all(
                not pd.isna(value)
                and math.isfinite(float(value))
                and float(value) > 0
                for value in odds
            )
        )

    processed = matches.copy()
    calculated = {column: [float("nan")] * len(matches) for column in PROBABILITY_COLUMNS}

    for position, is_valid in enumerate(valid_rows):
        if not is_valid:
            continue

        home_odds, draw_odds, away_odds = (
            float(numeric_odds[column].iloc[position]) for column in ODDS_COLUMNS
        )
        probabilities = normalized_1x2_probabilities(
            home_odds, draw_odds, away_odds
        )
        normalized = (
            probabilities.home,
            probabilities.draw,
            probabilities.away,
        )

        if not all(0.0 <= probability <= 1.0 for probability in normalized):
            raise ArithmeticError("Normalized probabilities must be between 0 and 1.")
        if not math.isclose(
            sum(normalized), 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE
        ):
            raise ArithmeticError(
                "Normalized home, draw, and away probabilities must sum to 1."
            )

        values = (
            probabilities.raw_home,
            probabilities.raw_draw,
            probabilities.raw_away,
            probabilities.raw_probability_sum,
            probabilities.overround,
            probabilities.home,
            probabilities.draw,
            probabilities.away,
        )
        for column, value in zip(PROBABILITY_COLUMNS, values):
            calculated[column][position] = value

    for column, values in calculated.items():
        processed[column] = values

    valid_count = sum(valid_rows)
    invalid_count = len(matches) - valid_count
    return processed, valid_count, invalid_count


def main() -> None:
    """Process the raw CSV and write results only to the processed-data path."""
    if RAW_DATA_PATH.resolve() == PROCESSED_DATA_PATH.resolve():
        raise ValueError("Raw input and processed output paths must be different.")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    matches = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig", low_memory=False)
    matches.columns = matches.columns.str.strip()
    processed, valid_count, invalid_count = process_matches(matches)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Valid matches: {valid_count}")
    print(f"Invalid matches: {invalid_count}")
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
