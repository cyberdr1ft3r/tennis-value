from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.test_train_v2 import _features

from tennis_value.benchmark_markets import (
    MARKET_SOURCES,
    build_anchor_coverage,
    run_market_benchmark,
)
from tennis_value.cleaning import clean_matches
from tennis_value.ingest import ingest_tennis_data
from tennis_value.odds_audit import build_odds_side_integrity
from tennis_value.train_v2_1 import MODEL_V2_1_FEATURES


def test_all_raw_bookmaker_pairs_are_read_independently(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2024-01-01",
                "Tournament": "T",
                "Surface": "Hard",
                "Winner": "A",
                "Loser": "B",
                "B365W": "1.5",
                "B365L": "2.7",
                "PSW": "1.6",
                "PSL": "",
                "AvgW": "1.55",
                "AvgL": "2.6",
                "MaxW": "1.7",
                "MaxL": "2.8",
            }
        ]
    ).to_csv(raw_dir / "2024.csv", index=False)

    result = ingest_tennis_data(raw_dir).data.iloc[0]

    assert result["winner_b365_odds"] == 1.5
    assert result["loser_b365_odds"] == 2.7
    assert bool(result["b365_pair_available"])
    assert result["winner_ps_odds"] == 1.6
    assert pd.isna(result["loser_ps_odds"])
    assert not bool(result["ps_pair_available"])
    assert bool(result["avg_pair_available"])
    assert bool(result["max_pair_available"])
    assert result["winner_odds"] == 1.5
    assert result["odds_source"] == "B365"


def test_canonical_source_odds_map_to_correct_player_for_both_outcomes() -> None:
    rows = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "tournament": "T",
                "surface": "Hard",
                "winner": "Ada",
                "loser": "Zed",
                "winner_odds": 1.5,
                "loser_odds": 2.7,
                "odds_source": "B365",
                "winner_b365_odds": 1.5,
                "loser_b365_odds": 2.7,
                "b365_pair_available": True,
            },
            {
                "match_date": "2024-01-02",
                "tournament": "T",
                "surface": "Hard",
                "winner": "Zed",
                "loser": "Ada",
                "winner_odds": 1.4,
                "loser_odds": 3.0,
                "odds_source": "B365",
                "winner_b365_odds": 1.4,
                "loser_b365_odds": 3.0,
                "b365_pair_available": True,
            },
        ]
    )

    canonical = clean_matches(rows).canonical_matches

    won = canonical[canonical["player_1_won"]].iloc[0]
    lost = canonical[~canonical["player_1_won"]].iloc[0]
    assert won["player_1_b365_odds"] == won["source_winner_b365_odds"]
    assert won["player_2_b365_odds"] == won["source_loser_b365_odds"]
    assert lost["player_1_b365_odds"] == lost["source_loser_b365_odds"]
    assert lost["player_2_b365_odds"] == lost["source_winner_b365_odds"]


def test_odds_side_mapping_failure_is_direct_not_market_reference_disagreement() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_id": "bad",
                "match_date": "2024-01-01",
                "player_1_won": False,
                "source_winner_b365_odds": 1.5,
                "source_loser_b365_odds": 2.7,
                "player_1_b365_odds": 1.5,
                "player_2_b365_odds": 2.7,
                "b365_pair_available": True,
            }
        ]
    )

    summary, rows = build_odds_side_integrity(frame)

    assert summary["sources"]["b365"]["odds_side_mapping_failures"] == 1
    assert bool(rows.loc[0, "odds_side_mapping_failure"])


def _multi_source_features() -> pd.DataFrame:
    frame = _features()
    for source, p1_offset, p2_offset in (
        ("b365", 0.00, 0.00),
        ("ps", 0.02, -0.02),
        ("avg", 0.01, -0.01),
        ("max", 0.03, -0.03),
    ):
        frame[f"player_1_{source}_odds"] = frame["player_1_odds"] + p1_offset
        frame[f"player_2_{source}_odds"] = frame["player_2_odds"] + p2_offset
        frame[f"{source}_pair_available"] = True
    frame.loc[0, "ps_pair_available"] = False
    return frame


def test_market_benchmark_scopes_max_exclusion_and_common_rows() -> None:
    frame = _multi_source_features()

    coverage = build_anchor_coverage(frame)
    result = run_market_benchmark(frame, bootstrap_samples=20)

    assert set(MARKET_SOURCES) == {"Bet365", "Pinnacle/PS", "Average"}
    assert "Maximum" not in set(result.metrics["source"])
    assert set(coverage["scope"]) == {"source_available", "common_all_sources"}
    common = result.common_rows
    counts = common.groupby(["source", "scope"])["match_id"].nunique()
    assert counts.nunique() == 1
    assert set(result.metrics["architecture"]) == {
        "raw_market",
        "market_recalibration",
        "free_form_workload",
    }
    assert MODEL_V2_1_FEATURES == [
        "market_logit_player_1",
        "recent_10_win_rate_diff",
        "surface_recent_10_win_rate_diff",
        "days_since_last_match_diff",
        "matches_last_14d_diff",
    ]
