from __future__ import annotations

import pandas as pd
import pytest

from tennis_value.elo import add_elo_features
from tennis_value.features import build_feature_dataset
from tennis_value.rolling import add_rolling_features


def _match(
    match_id: str,
    match_date: str,
    player_1: str,
    player_2: str,
    player_1_won: bool,
    tournament: str = "event",
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(match_date),
        "tournament_normalized": tournament,
        "round": "R32",
        "surface": "Hard",
        "player_1_normalized": player_1,
        "player_2_normalized": player_2,
        "player_1_won": player_1_won,
        "is_retirement": False,
    }


def _feature_match(
    match_id: str,
    match_date: str,
    player_1: str,
    player_2: str,
    player_1_won: bool,
    tournament: str = "event",
    is_retirement: bool = False,
) -> dict[str, object]:
    row = _match(match_id, match_date, player_1, player_2, player_1_won, tournament)
    row.update(
        {
            "tournament": tournament.title(),
            "best_of": 3,
            "player_1": player_1.title(),
            "player_2": player_2.title(),
            "player_1_rank": 10,
            "player_2_rank": 20,
            "player_1_odds": 1.8,
            "player_2_odds": 2.0,
            "is_retirement": is_retirement,
            "overall_elo_diff": 0.0,
            "surface_elo_diff": 0.0,
            "history_count_min": 0,
        }
    )
    return row


def test_same_day_matches_use_start_of_day_state() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", True, tournament="a"),
            _match("m03", "2024-01-02", "alice", "dina", False, tournament="b"),
            _match("m04", "2024-01-03", "alice", "erin", True),
        ]
    )

    result = add_elo_features(matches)
    same_day = result[result["match_date"] == pd.Timestamp("2024-01-02")]

    assert same_day["player_1_elo_before"].tolist() == pytest.approx([1516, 1516])
    assert same_day["player_1_matches_before"].tolist() == [1, 1]
    assert result.loc[3, "player_1_matches_before"] == 3


def test_shuffled_input_produces_identical_elo_output() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", True, tournament="a"),
            _match("m03", "2024-01-02", "alice", "dina", False, tournament="b"),
            _match("m04", "2024-01-03", "bob", "cara", True),
        ]
    )
    shuffled = matches.sample(frac=1, random_state=42).reset_index(drop=True)

    first = add_elo_features(matches)
    second = add_elo_features(shuffled)

    pd.testing.assert_frame_equal(first, second)


def test_repeated_execution_is_deterministic_and_does_not_duplicate_rows() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "bob", "cara", False),
        ]
    )

    first = add_elo_features(matches)
    second = add_elo_features(matches)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(matches)
    assert first["match_id"].is_unique


def test_same_day_matches_use_start_of_day_rolling_state() -> None:
    matches = pd.DataFrame(
        [
            _feature_match("m01", "2024-01-01", "alice", "bob", True),
            _feature_match("m02", "2024-01-02", "alice", "cara", True, tournament="a"),
            _feature_match("m03", "2024-01-02", "alice", "dina", False, tournament="b"),
            _feature_match("m04", "2024-01-03", "alice", "erin", True),
        ]
    )

    result = add_rolling_features(matches)
    same_day = result[result["match_date"] == pd.Timestamp("2024-01-02")]

    assert same_day["player_1_prior_match_count"].tolist() == [1, 1]
    assert same_day["player_1_recent_10_win_rate"].tolist() == pytest.approx([1.0, 1.0])
    assert result.loc[3, "player_1_prior_match_count"] == 3


def test_shuffled_input_produces_identical_feature_output() -> None:
    matches = pd.DataFrame(
        [
            _feature_match("m01", "2024-01-01", "alice", "bob", True),
            _feature_match("m02", "2024-01-02", "alice", "cara", True, tournament="a"),
            _feature_match("m03", "2024-01-02", "alice", "dina", False, tournament="b"),
            _feature_match("m04", "2024-01-03", "bob", "cara", True),
        ]
    )
    shuffled = matches.sample(frac=1, random_state=42).reset_index(drop=True)

    first = build_feature_dataset(matches)
    second = build_feature_dataset(shuffled)

    pd.testing.assert_frame_equal(first, second)
