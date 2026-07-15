from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tennis_value.config import EloConfig
from tennis_value.elo import (
    ELO_COLUMNS,
    add_elo_features,
    add_elo_features_with_report,
    expected_score,
    update_ratings,
    write_elo_outputs,
)


def _match(
    match_id: str,
    match_date: str,
    player_1: str,
    player_2: str,
    player_1_won: bool,
    surface: str = "Hard",
    tournament: str = "event",
    round_name: str = "R32",
    is_retirement: bool = False,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(match_date),
        "tournament_normalized": tournament,
        "round": round_name,
        "surface": surface,
        "player_1_normalized": player_1,
        "player_2_normalized": player_2,
        "player_1_won": player_1_won,
        "is_retirement": is_retirement,
    }


def test_expected_score_formula() -> None:
    assert expected_score(1500, 1500, 400) == pytest.approx(0.5)
    assert expected_score(1600, 1500, 400) > 0.5
    first = expected_score(1600, 1500, 400)
    second = expected_score(1500, 1600, 400)
    assert first + second == pytest.approx(1.0)
    with pytest.raises(ValueError, match="scale"):
        expected_score(1500, 1500, 0)


def test_hand_calculated_update_from_equal_ratings() -> None:
    winner, loser = update_ratings(1500, 1500, 1.0, EloConfig())

    assert winner == pytest.approx(1516)
    assert loser == pytest.approx(1484)


def test_pre_match_snapshot_and_match_counts() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", False),
        ]
    )

    result = add_elo_features(matches)

    first = result.iloc[0]
    second = result.iloc[1]
    assert first["player_1_elo_before"] == pytest.approx(1500)
    assert first["player_2_elo_before"] == pytest.approx(1500)
    assert first["player_1_matches_before"] == 0
    assert second["player_1_elo_before"] == pytest.approx(1516)
    assert second["player_2_elo_before"] == pytest.approx(1500)
    assert second["player_1_matches_before"] == 1
    assert second["player_2_matches_before"] == 0
    assert second["history_count_min"] == 0


def test_surface_elo_updates_only_current_surface() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True, surface="Hard"),
            _match("m02", "2024-01-02", "alice", "cara", True, surface="Hard"),
            _match("m03", "2024-01-03", "alice", "dina", True, surface="Clay"),
        ]
    )

    result = add_elo_features(matches)

    assert result.loc[0, "player_1_surface_elo_before"] == pytest.approx(1500)
    assert result.loc[1, "player_1_surface_elo_before"] == pytest.approx(1516)
    assert result.loc[2, "player_1_surface_elo_before"] == pytest.approx(1500)
    assert result.loc[2, "player_1_elo_before"] > 1516


def test_retirement_keeps_features_but_does_not_update() -> None:
    matches = pd.DataFrame(
        [
            _match("m01", "2024-01-01", "alice", "bob", True),
            _match("m02", "2024-01-02", "alice", "cara", True, is_retirement=True),
            _match("m03", "2024-01-03", "alice", "dina", False),
        ]
    )

    result = add_elo_features(matches)

    assert result.loc[1, "player_1_elo_before"] == pytest.approx(1516)
    assert result.loc[1, "elo_update_applied"] is False or result.loc[1, "elo_update_applied"] == 0
    assert result.loc[2, "player_1_elo_before"] == pytest.approx(1516)
    assert result.loc[2, "player_1_matches_before"] == 1


def test_custom_elo_config_changes_updates() -> None:
    winner, loser = update_ratings(
        1000,
        1000,
        1.0,
        EloConfig(initial_rating=1000, k_factor=20, elo_scale=200),
    )

    assert winner == pytest.approx(1010)
    assert loser == pytest.approx(990)


def test_elo_report_and_output_writer(tmp_path: Path) -> None:
    result = add_elo_features_with_report(
        pd.DataFrame([_match("m01", "2024-01-01", "alice", "bob", True)])
    )
    output = tmp_path / "matches_with_elo.parquet"
    report = tmp_path / "elo_quality.json"

    write_elo_outputs(result, output, report)

    assert output.exists()
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["eligible_updates"] == 1
    assert payload["players_seen"] == 2
    assert pd.read_parquet(output).shape[0] == 1
    for column in ELO_COLUMNS:
        assert column in result.matches.columns


def test_error_handling_for_invalid_inputs() -> None:
    valid = pd.DataFrame([_match("m01", "2024-01-01", "alice", "bob", True)])

    with pytest.raises(ValueError, match="missing required"):
        add_elo_features(valid.drop(columns=["surface"]))

    with pytest.raises(ValueError, match="duplicate"):
        add_elo_features(pd.concat([valid, valid], ignore_index=True))

    with pytest.raises(ValueError, match="player_1_won"):
        add_elo_features(pd.DataFrame([_match("m01", "2024-01-01", "alice", "bob", "maybe")]))

    with pytest.raises(ValueError, match="unsupported surface"):
        add_elo_features(pd.DataFrame([_match("m01", "2024-01-01", "alice", "bob", True, "Other")]))

    with pytest.raises(ValueError, match="distinct"):
        add_elo_features(pd.DataFrame([_match("m01", "2024-01-01", "alice", "alice", True)]))
