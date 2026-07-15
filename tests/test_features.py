from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from tennis_value.features import (
    MODEL_FEATURE_COLUMNS,
    build_feature_dataset,
    build_features_with_report,
)
from tennis_value.rolling import add_rolling_features


def _match(
    match_id: str,
    match_date: str,
    player_1: str,
    player_2: str,
    player_1_won: bool,
    *,
    surface: str = "Hard",
    best_of: int = 3,
    p1_rank: int | None = 10,
    p2_rank: int | None = 20,
    is_retirement: bool = False,
    tournament: str = "event",
    round_name: str = "R32",
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(match_date),
        "tournament": tournament.title(),
        "tournament_normalized": tournament,
        "surface": surface,
        "round": round_name,
        "best_of": best_of,
        "player_1": player_1.title(),
        "player_2": player_2.title(),
        "player_1_normalized": player_1,
        "player_2_normalized": player_2,
        "player_1_rank": p1_rank,
        "player_2_rank": p2_rank,
        "player_1_odds": 1.8,
        "player_2_odds": 2.0,
        "player_1_won": player_1_won,
        "is_retirement": is_retirement,
        "overall_elo_diff": 12.0,
        "surface_elo_diff": 8.0,
        "history_count_min": 0,
    }


def test_first_match_uses_neutral_defaults_and_null_rest() -> None:
    result = add_rolling_features(pd.DataFrame([_match("m01", "2024-01-01", "alice", "bob", True)]))

    row = result.iloc[0]
    assert row["player_1_prior_match_count"] == 0
    assert row["player_2_prior_match_count"] == 0
    assert row["player_1_recent_10_win_rate"] == pytest.approx(0.5)
    assert row["player_2_recent_10_win_rate"] == pytest.approx(0.5)
    assert row["player_1_surface_recent_10_win_rate"] == pytest.approx(0.5)
    assert row["player_2_surface_recent_10_win_rate"] == pytest.approx(0.5)
    assert pd.isna(row["player_1_days_since_last_match"])
    assert pd.isna(row["player_2_days_since_last_match"])
    assert row["player_1_matches_last_14d"] == 0
    assert row["player_2_matches_last_14d"] == 0


def test_current_result_does_not_affect_own_features_and_next_match_sees_history() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", False),
        ]
    )

    result = add_rolling_features(matches)

    assert result.loc[0, "player_1_recent_10_win_rate"] == pytest.approx(0.5)
    assert result.loc[1, "player_1_prior_match_count"] == 1
    assert result.loc[1, "player_1_recent_10_win_rate"] == pytest.approx(1.0)
    assert result.loc[1, "recent_10_win_rate_diff"] == pytest.approx(0.5)


def test_recent_history_keeps_only_ten_eligible_matches() -> None:
    rows = []
    for index in range(11):
        rows.append(
            _match(
                f"m{index + 1:02d}",
                f"2024-01-{index + 1:02d}",
                "alice",
                f"opponent-{index}",
                index == 0,
            )
        )
    rows.append(_match("m12", "2024-01-12", "alice", "zara", True))

    result = add_rolling_features(pd.DataFrame(rows))

    assert result.loc[11, "player_1_prior_match_count"] == 11
    assert result.loc[11, "player_1_recent_10_win_rate"] == pytest.approx(0.0)


def test_surface_history_is_independent_by_surface() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True, surface="Hard"),
            _match("m02", "2024-01-02", "alice", "cara", False, surface="Clay"),
            _match("m03", "2024-01-03", "alice", "dina", True, surface="Clay"),
        ]
    )

    result = add_rolling_features(matches)

    assert result.loc[1, "player_1_surface_history_count"] == 0
    assert result.loc[1, "player_1_surface_recent_10_win_rate"] == pytest.approx(0.5)
    assert result.loc[2, "player_1_surface_history_count"] == 1
    assert result.loc[2, "player_1_surface_recent_10_win_rate"] == pytest.approx(0.0)


def test_rest_days_ignore_retirements_as_eligible_history() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-05", "alice", "cara", True, is_retirement=True),
            _match("m03", "2024-01-10", "alice", "dina", True),
        ]
    )

    result = add_rolling_features(matches)

    assert pd.isna(result.loc[0, "player_1_days_since_last_match"])
    assert result.loc[1, "player_1_days_since_last_match"] == 4
    assert result.loc[2, "player_1_days_since_last_match"] == 9


def test_schedule_density_uses_prior_fourteen_day_interval() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", True),
            _match("m03", "2024-01-10", "alice", "dina", True, is_retirement=True),
            _match("m04", "2024-01-16", "alice", "erin", True),
            _match("m05", "2024-01-16", "alice", "faye", True, tournament="later"),
        ]
    )

    result = add_rolling_features(matches)

    assert result.loc[3, "player_1_matches_last_14d"] == 1
    assert result.loc[4, "player_1_matches_last_14d"] == 1


def test_ranking_features_missing_values_and_orientation() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True, p1_rank=5, p2_rank=20),
            _match("m02", "2024-01-02", "cara", "dina", True, p1_rank=None, p2_rank=30),
        ]
    )

    result = build_feature_dataset(matches)

    assert result.loc[0, "log_rank_diff"] == pytest.approx(math.log1p(20) - math.log1p(5))
    assert result.loc[0, "log_rank_diff"] > 0
    assert pd.isna(result.loc[1, "log_rank_diff"])
    assert result.loc[1, "rank_missing_player_1"] == 1


def test_surface_and_format_indicators() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True, surface="Hard", best_of=3),
            _match("m02", "2024-01-02", "cara", "dina", True, surface="Clay", best_of=5),
            _match("m03", "2024-01-03", "erin", "faye", True, surface="Grass", best_of=3),
        ]
    )

    result = build_feature_dataset(matches)

    assert result.loc[0, ["surface_clay", "surface_grass", "best_of_5"]].tolist() == [0, 0, 0]
    assert result.loc[1, ["surface_clay", "surface_grass", "best_of_5"]].tolist() == [1, 0, 1]
    assert result.loc[2, ["surface_clay", "surface_grass", "best_of_5"]].tolist() == [0, 1, 0]


def test_output_integrity_and_report_serialization() -> None:
    source = pd.DataFrame(
        [
            _match("m02", "2024-01-02", "cara", "dina", False, p1_rank=None),
            _match("m01", "2024-01-01", "alice", "bob", True),
        ]
    )
    source_before = source.copy(deep=True)

    result = build_features_with_report(source)

    assert set(MODEL_FEATURE_COLUMNS).issubset(result.features.columns)
    assert result.features["match_id"].tolist() == ["m01", "m02"]
    assert result.features["player_1_won"].tolist() == [True, False]
    assert "player_1_odds" in result.features.columns
    assert "player_1_odds" not in MODEL_FEATURE_COLUMNS
    assert result.features["match_id"].is_unique
    pd.testing.assert_frame_equal(source, source_before)
    json.loads(result.report.model_dump_json())
    assert result.report.rows_returned == 2
    assert result.report.rows_with_missing_rank == 1


def test_invalid_inputs_fail_with_useful_messages() -> None:
    matches = pd.DataFrame([_match("m01", "2024-01-01", "alice", "bob", True)])

    with pytest.raises(ValueError, match="duplicate match_id"):
        build_feature_dataset(pd.concat([matches, matches], ignore_index=True))

    with pytest.raises(ValueError, match="missing required feature input columns"):
        build_feature_dataset(matches.drop(columns=["overall_elo_diff"]))

    invalid_surface = matches.copy()
    invalid_surface.loc[0, "surface"] = "Carpet"
    with pytest.raises(ValueError, match="unsupported surface"):
        build_feature_dataset(invalid_surface)

    invalid_best_of = matches.copy()
    invalid_best_of.loc[0, "best_of"] = 7
    with pytest.raises(ValueError, match="best_of"):
        build_feature_dataset(invalid_best_of)


def test_repeated_runs_are_identical() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", False),
        ]
    )

    first = build_feature_dataset(matches)
    second = build_feature_dataset(matches)

    pd.testing.assert_frame_equal(first, second)
