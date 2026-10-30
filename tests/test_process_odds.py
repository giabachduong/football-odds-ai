"""Tests for processing bookmaker odds into match probabilities."""

import math
from pathlib import Path

import pandas as pd
import pytest

from src import process_odds
from src.process_odds import process_matches


def test_process_matches_calculates_expected_probabilities() -> None:
    """The dataset processor reuses odds calculations for each valid row."""
    matches = pd.DataFrame({"AvgH": [2.0], "AvgD": [3.5], "AvgA": [4.0]})

    result, valid_count, invalid_count = process_matches(matches)

    assert (valid_count, invalid_count) == (1, 0)
    assert math.isclose(result.loc[0, "raw_home_probability"], 0.5)
    assert math.isclose(result.loc[0, "raw_draw_probability"], 2.0 / 7.0)
    assert math.isclose(result.loc[0, "raw_away_probability"], 0.25)
    assert math.isclose(result.loc[0, "raw_probability_sum"], 29.0 / 28.0)
    assert math.isclose(result.loc[0, "overround"], 1.0 / 28.0)
    assert math.isclose(result.loc[0, "prob_home"], 14.0 / 29.0)
    assert math.isclose(result.loc[0, "prob_draw"], 8.0 / 29.0)
    assert math.isclose(result.loc[0, "prob_away"], 7.0 / 29.0)


def test_valid_probabilities_are_bounded_and_sum_to_one() -> None:
    """Every valid row has probabilities in range and normalized to one."""
    matches = pd.DataFrame(
        {"AvgH": [1.08, 2.0], "AvgD": [10.12, 3.5], "AvgA": [22.31, 4.0]}
    )

    result, valid_count, _ = process_matches(matches)
    probability_columns = ["prob_home", "prob_draw", "prob_away"]

    assert valid_count == 2
    for _, row in result.iterrows():
        probabilities = [row[column] for column in probability_columns]
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)
        assert math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9
        )


@pytest.mark.parametrize(
    "bad_odds",
    [None, "not odds", 0, -1, math.inf, -math.inf, math.nan],
)
@pytest.mark.parametrize("odds_column", ["AvgH", "AvgD", "AvgA"])
def test_invalid_odds_make_the_match_invalid(
    odds_column: str, bad_odds: object
) -> None:
    """A missing, malformed, non-finite, or non-positive quote invalidates a row."""
    values: dict[str, object] = {"AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0}
    values[odds_column] = bad_odds

    result, valid_count, invalid_count = process_matches(pd.DataFrame([values]))

    assert (valid_count, invalid_count) == (0, 1)
    assert result.loc[0, "prob_home"] != result.loc[0, "prob_home"]


def test_result_columns_are_preserved_but_do_not_affect_probabilities() -> None:
    """Historical results remain in output and changing them changes no odds result."""
    first = pd.DataFrame(
        {"AvgH": [2.0], "AvgD": [3.5], "AvgA": [4.0], "FTHG": [0], "FTAG": [0], "FTR": ["D"]}
    )
    changed_results = first.assign(FTHG=5, FTAG=1, FTR="H")

    first_result, _, _ = process_matches(first)
    changed_result, _, _ = process_matches(changed_results)

    assert first_result.loc[0, "FTHG"] == 0
    assert first_result.loc[0, "FTAG"] == 0
    assert first_result.loc[0, "FTR"] == "D"
    for column in process_odds.PROBABILITY_COLUMNS:
        assert first_result.loc[0, column] == changed_result.loc[0, column]


def test_missing_required_odds_column_is_reported() -> None:
    """The processor fails clearly when an odds column is absent."""
    matches = pd.DataFrame({"AvgH": [2.0], "AvgD": [3.5]})

    with pytest.raises(ValueError, match="AvgA"):
        process_matches(matches)


def test_main_writes_output_and_preserves_raw_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running the script writes its processed CSV without changing the raw CSV."""
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "probabilities.csv"
    raw_content = "AvgH,AvgD,AvgA,FTR\n2.0,3.5,4.0,D\n"
    raw_path.write_text(raw_content, encoding="utf-8")
    monkeypatch.setattr(process_odds, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(process_odds, "PROCESSED_DATA_PATH", output_path)

    process_odds.main()

    assert raw_path.read_text(encoding="utf-8") == raw_content
    saved = pd.read_csv(output_path)
    assert len(saved) == 1
    assert math.isclose(saved.loc[0, "prob_home"], 14.0 / 29.0)


def test_main_rejects_same_raw_and_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path collision is rejected before the raw CSV can be overwritten."""
    raw_path = tmp_path / "raw.csv"
    raw_content = "AvgH,AvgD,AvgA\n2.0,3.5,4.0\n"
    raw_path.write_text(raw_content, encoding="utf-8")
    monkeypatch.setattr(process_odds, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(process_odds, "PROCESSED_DATA_PATH", raw_path)

    with pytest.raises(ValueError, match="must be different"):
        process_odds.main()

    assert raw_path.read_text(encoding="utf-8") == raw_content
