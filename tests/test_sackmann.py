from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from tennis_value.features import MODEL_FEATURE_COLUMNS
from tennis_value.sackmann import (
    METRICS,
    build_pre_match_stat_features,
    build_sackmann_enriched_features,
    derive_match_stats,
    link_project_matches,
    name_keys,
    normalize_sackmann_frame,
)
from tennis_value.train import MODEL_FEATURES as MODEL_V1_FEATURES
from tennis_value.train_v2 import MARKET_FEATURES as MODEL_V2_FEATURES
from tennis_value.train_v2_1 import MODEL_V2_1_FEATURES


def _sackmann_source(
    date: str,
    winner: str,
    loser: str,
    winner_id: int,
    loser_id: int,
    *,
    tourney: str = "Sample Open",
    match_num: int = 1,
    surface: str = "Hard",
    score: str = "6-4 6-4",
    minutes: int | None = 90,
) -> dict[str, object]:
    return {
        "tourney_id": f"{date[:4]}-001",
        "tourney_name": tourney,
        "surface": surface,
        "draw_size": 32,
        "tourney_level": "A",
        "tourney_date": date.replace("-", ""),
        "match_num": match_num,
        "winner_id": winner_id,
        "winner_name": winner,
        "loser_id": loser_id,
        "loser_name": loser,
        "score": score,
        "best_of": 3,
        "round": "R32",
        "minutes": minutes,
        "w_ace": 5,
        "w_df": 2,
        "w_svpt": 50,
        "w_1stIn": 30,
        "w_1stWon": 24,
        "w_2ndWon": 10,
        "w_SvGms": 10,
        "w_bpSaved": 3,
        "w_bpFaced": 4,
        "l_ace": 3,
        "l_df": 4,
        "l_svpt": 60,
        "l_1stIn": 36,
        "l_1stWon": 20,
        "l_2ndWon": 8,
        "l_SvGms": 10,
        "l_bpSaved": 4,
        "l_bpFaced": 8,
    }


def _project_match(
    match_id: str,
    date: str,
    player_1: str,
    player_2: str,
    player_1_won: bool,
    *,
    tournament: str = "Sample Open",
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(date),
        "tournament": tournament,
        "tournament_normalized": "sample open",
        "surface": "Hard",
        "round": "R32",
        "best_of": 3,
        "player_1": player_1,
        "player_2": player_2,
        "player_1_normalized": player_1.casefold(),
        "player_2_normalized": player_2.casefold(),
        "player_1_rank": 10,
        "player_2_rank": 20,
        "player_1_odds": 1.8,
        "player_2_odds": 2.0,
        "player_1_won": player_1_won,
        "is_retirement": False,
    }


def _base_feature(match_id: str, date: str, player_1_won: bool) -> dict[str, object]:
    row = {
        "match_id": match_id,
        "match_date": pd.Timestamp(date),
        "tournament": "Sample Open",
        "surface": "Hard",
        "player_1": "A",
        "player_2": "B",
        "player_1_won": player_1_won,
        "player_1_odds": 1.8,
        "player_2_odds": 2.0,
        "is_retirement": False,
    }
    row.update({feature: 0.0 for feature in MODEL_FEATURE_COLUMNS})
    return row


def test_normalization_handles_missing_optional_columns_and_does_not_mutate_source() -> None:
    source = pd.DataFrame(
        [_sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2)]
    )
    source = source.drop(columns=["w_ace"])
    before = source.copy(deep=True)

    result = normalize_sackmann_frame(source, source_file="atp_matches_2020.csv")

    pd.testing.assert_frame_equal(source, before)
    assert result.loc[0, "surface"] == "Hard"
    assert pd.isna(result.loc[0, "winner_ace"])


def test_name_keys_cover_abbreviations_accents_and_compounds() -> None:
    assert name_keys("A Zverev") & name_keys("Alexander Zverev")
    assert name_keys("F Auger Aliassime") & name_keys("Felix Auger-Aliassime")
    assert name_keys("M Cilic") & name_keys("Marin Cilic")


def test_strict_and_unique_matching_and_player_orientation() -> None:
    sackmann = normalize_sackmann_frame(
        pd.DataFrame([_sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2)]),
        source_file="atp_matches_2020.csv",
    )
    project = pd.DataFrame([_project_match("m01", "2020-01-01", "A Zverev", "R Nadal", True)])

    links, summary, failures, manual = link_project_matches(project, sackmann)

    assert failures.empty
    assert manual.empty
    assert summary["strict_matches"] == 1
    assert links.loc[0, "player_1_sackmann_id"] == 1
    assert links.loc[0, "player_2_sackmann_id"] == 2


def test_player_1_loser_orientation_assigns_loser_stats_to_player_1() -> None:
    sackmann = normalize_sackmann_frame(
        pd.DataFrame([_sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2)]),
        source_file="atp_matches_2020.csv",
    )
    project = pd.DataFrame([_project_match("m01", "2020-01-01", "R Nadal", "A Zverev", False)])

    links, _, _, _ = link_project_matches(project, sackmann)

    assert links.loc[0, "player_1_sackmann_id"] == 2
    assert links.loc[0, "player_2_sackmann_id"] == 1


def test_ambiguous_candidates_remain_unmatched() -> None:
    source = pd.DataFrame(
        [
            _sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2, match_num=1),
            _sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2, match_num=2),
        ]
    )
    sackmann = normalize_sackmann_frame(source, source_file="atp_matches_2020.csv")
    project = pd.DataFrame([_project_match("m01", "2020-01-01", "A Zverev", "R Nadal", True)])

    links, summary, failures, manual = link_project_matches(project, sackmann)

    assert links.empty
    assert summary["ambiguous_matches"] == 1
    assert failures.loc[0, "failure_reason"] == "ambiguous"
    assert not manual.empty


def test_stat_formulas_and_zero_denominators() -> None:
    sackmann = normalize_sackmann_frame(
        pd.DataFrame([_sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2)]),
        source_file="atp_matches_2020.csv",
    )

    stats = derive_match_stats(sackmann.iloc[0])

    assert stats["winner"]["serve_points_won_pct"] == pytest.approx((34, 50))
    assert stats["winner"]["return_points_won_pct"] == pytest.approx((32, 60))
    assert stats["winner"]["first_serve_in_pct"] == pytest.approx((30, 50))
    assert stats["winner"]["first_serve_points_won_pct"] == pytest.approx((24, 30))
    assert stats["winner"]["second_serve_points_won_pct"] == pytest.approx((10, 20))
    assert stats["winner"]["ace_rate"] == pytest.approx((5, 50))
    assert stats["winner"]["double_fault_rate"] == pytest.approx((2, 50))
    assert stats["winner"]["break_points_saved_pct"] == pytest.approx((3, 4))
    assert stats["winner"]["break_points_converted_pct"] == pytest.approx((4, 8))

    broken = sackmann.copy()
    broken.loc[0, "winner_svpt"] = 0
    invalid: Counter[str] = Counter()
    assert derive_match_stats(broken.iloc[0], invalid) == {}
    assert invalid["winner_serve_points_positive"] == 1


def test_leakage_safe_warmup_same_day_and_workload_features() -> None:
    sackmann = normalize_sackmann_frame(
        pd.DataFrame(
            [
                _sackmann_source(
                    "2019-12-30",
                    "Alexander Zverev",
                    "Rafael Nadal",
                    1,
                    2,
                    match_num=1,
                ),
                _sackmann_source(
                    "2020-01-01",
                    "Alexander Zverev",
                    "Rafael Nadal",
                    1,
                    2,
                    match_num=2,
                ),
                _sackmann_source(
                    "2020-01-01",
                    "Alexander Zverev",
                    "Novak Djokovic",
                    1,
                    3,
                    match_num=3,
                ),
            ]
        ),
        source_file="atp_matches_2020.csv",
    )
    project = pd.DataFrame(
        [
            _project_match("m01", "2020-01-01", "A Zverev", "R Nadal", True),
            _project_match("m02", "2020-01-01", "A Zverev", "N Djokovic", True),
        ]
    )
    links, _, _, _ = link_project_matches(project, sackmann)

    features, quality, leakage = build_pre_match_stat_features(project, sackmann, links)
    first = features[features["match_id"] == "m01"].iloc[0]
    second = features[features["match_id"] == "m02"].iloc[0]

    assert first["player_1_point_stats_match_count"] == 1
    assert second["player_1_point_stats_match_count"] == 1
    assert first["player_1_minutes_last_3d"] == 90
    assert quality["retirements_skipped_for_point_updates"] == 0
    assert leakage["same_day_leakage_checks_passed"] is True
    assert leakage["warmup_2020_rows_with_prior_history"] == 2


def test_retirements_skip_point_updates_but_valid_minutes_update_workload() -> None:
    sackmann = normalize_sackmann_frame(
        pd.DataFrame(
            [
                _sackmann_source(
                    "2019-12-30",
                    "Alexander Zverev",
                    "Rafael Nadal",
                    1,
                    2,
                    score="6-4 RET",
                    minutes=55,
                    match_num=1,
                ),
                _sackmann_source(
                    "2020-01-01",
                    "Alexander Zverev",
                    "Rafael Nadal",
                    1,
                    2,
                    match_num=2,
                ),
            ]
        ),
        source_file="atp_matches_2020.csv",
    )
    project = pd.DataFrame([_project_match("m01", "2020-01-01", "A Zverev", "R Nadal", True)])
    links, _, _, _ = link_project_matches(project, sackmann)

    features, quality, _ = build_pre_match_stat_features(project, sackmann, links)

    assert features.loc[0, "player_1_point_stats_match_count"] == 0
    assert features.loc[0, "player_1_minutes_last_3d"] == 0
    assert quality["retirements_skipped_for_point_updates"] == 1


def test_enriched_dataset_preserves_base_columns_and_model_allowlists() -> None:
    sackmann = normalize_sackmann_frame(
        pd.DataFrame([_sackmann_source("2020-01-01", "Alexander Zverev", "Rafael Nadal", 1, 2)]),
        source_file="atp_matches_2020.csv",
    )
    project = pd.DataFrame([_project_match("m01", "2020-01-01", "A Zverev", "R Nadal", True)])
    base = pd.DataFrame([_base_feature("m01", "2020-01-01", True)])
    before_columns = list(base.columns)

    result = build_sackmann_enriched_features(
        project_matches=project,
        base_features=base,
        sackmann_matches=sackmann,
    )

    assert len(result.enriched_features) == len(base)
    assert list(result.enriched_features.columns[: len(before_columns)]) == before_columns
    assert result.feature_quality["row_count_preservation"] is True
    assert set(MODEL_V1_FEATURES) == set(MODEL_FEATURE_COLUMNS)
    assert "player_1_ewm_serve_points_won_pct" not in MODEL_V2_FEATURES
    assert "player_1_ewm_serve_points_won_pct" not in MODEL_V2_1_FEATURES
    assert all("roi" not in metric for metric in METRICS)
