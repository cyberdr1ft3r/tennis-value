from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tennis_value.cleaning import (
    CANONICAL_COLUMNS,
    clean_matches,
    normalize_round,
    normalize_surface,
    write_cleaning_outputs,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_date": pd.Timestamp("2024-01-02"),
        "tournament": " Brisbane  International ",
        "surface": "Hard",
        "round": "1st Round",
        "best_of": 3,
        "winner": "Alex Example",
        "loser": "Boris Sample",
        "winner_rank": 10,
        "loser_rank": 20,
        "winner_odds": 1.8,
        "loser_odds": 2.05,
        "odds_source": "B365",
        "status_or_comment": "Completed",
        "source_file": "sample.csv",
    }
    row.update(overrides)
    return row


def _clean(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_surface_and_round_normalization() -> None:
    assert normalize_surface("hard") == "Hard"
    assert normalize_surface("Indoor Hard") == "Hard"
    assert normalize_surface("red clay") == "Clay"
    assert normalize_surface("grass") == "Grass"
    assert normalize_surface("carpet") == "Other"
    assert normalize_surface(None) == "Other"

    assert normalize_round("Round of 128") == "R128"
    assert normalize_round("2nd Round") == "R64"
    assert normalize_round("Quarterfinals") == "QF"
    assert normalize_round("Semi-finals") == "SF"
    assert normalize_round("Final") == "F"
    assert normalize_round("Round Robin") == "RR"
    assert normalize_round("Bronze") == "BR"
    assert normalize_round("Something Strange") == "Unknown"


def test_clean_matches_outputs_required_columns_and_sorts_deterministically() -> None:
    result = clean_matches(
        _clean(
            [
                _row(match_date=pd.Timestamp("2024-01-03"), tournament="Zurich"),
                _row(match_date=pd.Timestamp("2024-01-02"), tournament="Basel"),
            ]
        )
    )

    assert list(result.canonical_matches.columns) == CANONICAL_COLUMNS
    assert result.quality_report.rows_accepted == 2
    assert result.canonical_matches.loc[0, "tournament"] == "Basel"
    assert result.canonical_matches["match_date"].is_monotonic_increasing


def test_orientation_swaps_rank_and_odds_when_winner_is_player_2() -> None:
    result = clean_matches(
        _clean(
            [
                _row(
                    winner="Zed Winner",
                    loser="Adam Loser",
                    winner_rank=5,
                    loser_rank=80,
                    winner_odds=1.4,
                    loser_odds=3.1,
                )
            ]
        )
    )
    row = result.canonical_matches.iloc[0]

    assert row["player_1"] == "Adam Loser"
    assert row["player_2"] == "Zed Winner"
    assert row["player_1_won"] is False or row["player_1_won"] == 0
    assert row["player_1_rank"] == 80
    assert row["player_2_rank"] == 5
    assert row["player_1_odds"] == 3.1
    assert row["player_2_odds"] == 1.4


def test_equivalent_source_rows_generate_same_orientation_and_match_id() -> None:
    first = clean_matches(
        _clean([_row(winner="O\u2019Connell, Christopher", loser="Juan  Martin del Potro")])
    )
    second = clean_matches(
        _clean([_row(winner="Juan Martin del Potro", loser="O'Connell, Christopher")])
    )

    first_row = first.canonical_matches.iloc[0]
    second_row = second.canonical_matches.iloc[0]
    assert first_row["player_1_normalized"] == second_row["player_1_normalized"]
    assert first_row["player_2_normalized"] == second_row["player_2_normalized"]
    assert first_row["match_id"] == second_row["match_id"]


def test_walkover_rejected_and_retirement_retained() -> None:
    result = clean_matches(
        _clean(
            [
                _row(status_or_comment="W/O"),
                _row(winner="Carlos Demo", loser="Dani Test", status_or_comment="Retired"),
                _row(winner="Evan Local", loser="Felix Away", status_or_comment="Completed"),
            ]
        )
    )

    assert result.quality_report.walkovers == 1
    assert result.quality_report.retirements == 1
    assert "walkover" in set(result.rejected_rows["rejection_reason"])
    assert sorted(result.canonical_matches["is_retirement"].tolist()) == [False, True]


def test_required_row_rejections_and_missing_optional_values_are_reported() -> None:
    result = clean_matches(
        _clean(
            [
                _row(surface="carpet"),
                _row(winner=""),
                _row(loser=None),
                _row(winner="Same Player", loser="Same  Player"),
                _row(match_date=pd.NaT),
                _row(best_of=4),
                _row(winner_rank=pd.NA, loser_rank=pd.NA, winner_odds=pd.NA, loser_odds=pd.NA),
            ]
        )
    )

    reasons = set(result.rejected_rows["rejection_reason"])
    assert {
        "unsupported_surface",
        "missing_winner",
        "missing_loser",
        "same_player",
        "invalid_match_date",
        "invalid_best_of",
    } <= reasons
    assert result.quality_report.rows_accepted == 1
    assert result.quality_report.missing_rankings == 1
    assert result.quality_report.missing_odds == 1


def test_exact_duplicate_keeps_one_and_rejects_copy() -> None:
    duplicate = _row()
    result = clean_matches(_clean([duplicate, duplicate.copy()]))

    assert result.quality_report.rows_accepted == 1
    assert result.quality_report.exact_duplicates == 1
    assert result.rejected_rows.iloc[0]["rejection_reason"] == "exact_duplicate"


def test_conflicting_duplicate_rejects_all_conflicting_copies() -> None:
    result = clean_matches(
        _clean(
            [
                _row(winner_rank=10),
                _row(winner_rank=11),
            ]
        )
    )

    assert result.quality_report.rows_accepted == 0
    assert result.quality_report.conflicting_duplicates == 2
    assert set(result.rejected_rows["rejection_reason"]) == {"conflicting_duplicate"}


def test_quality_report_is_json_serializable_and_outputs_are_written(tmp_path: Path) -> None:
    result = clean_matches(_clean([_row()]))
    output = tmp_path / "matches.parquet"
    report = tmp_path / "quality.json"
    rejected = tmp_path / "rejected.csv"

    write_cleaning_outputs(result, output, report, rejected)

    assert output.exists()
    assert report.exists()
    assert rejected.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["rows_accepted"] == 1
    assert pd.read_parquet(output).shape[0] == 1


def test_repeated_cleaning_is_deterministic() -> None:
    raw = _clean([_row(), _row(winner="Carlos Demo", loser="Dani Test", match_date="2024-01-03")])

    first = clean_matches(raw)
    second = clean_matches(raw)

    pd.testing.assert_frame_equal(first.canonical_matches, second.canonical_matches)
    pd.testing.assert_frame_equal(first.rejected_rows, second.rejected_rows)
    assert first.quality_report == second.quality_report
