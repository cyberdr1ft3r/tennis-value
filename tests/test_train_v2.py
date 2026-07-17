from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tennis_value.train import MODEL_FEATURES as MODEL_V1_FEATURES
from tennis_value.train_v2 import (
    FORBIDDEN_FEATURES,
    LOGIT_EPSILON,
    MARKET_FEATURES,
    ModelV2OutputPaths,
    add_market_features,
    market_logit,
    train_model_v2,
    write_model_v2_artifacts,
)


def _row(
    match_id: str,
    year: int,
    target: bool,
    *,
    player_1_odds: float = 2.0,
    player_2_odds: float = 2.0,
    log_rank_diff: float = 0.0,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(year=year, month=1, day=1),
        "surface": "Hard",
        "player_1": f"{match_id} A",
        "player_2": f"{match_id} B",
        "player_1_won": target,
        "player_1_odds": player_1_odds,
        "player_2_odds": player_2_odds,
        "is_retirement": False,
        "overall_elo_diff": 10.0 if target else -10.0,
        "surface_elo_diff": 5.0 if target else -5.0,
        "log_rank_diff": log_rank_diff,
        "recent_10_win_rate_diff": 0.1 if target else -0.1,
        "surface_recent_10_win_rate_diff": 0.1 if target else -0.1,
        "days_since_last_match_diff": 1.0,
        "matches_last_14d_diff": 0,
        "history_count_min": 2,
        "best_of_5": 0,
        "surface_clay": 0,
        "surface_grass": 0,
    }


def _features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2020, 2026):
        rows.append(_row(f"{year}a", year, True, player_1_odds=1.8, player_2_odds=2.1))
        rows.append(_row(f"{year}b", year, False, player_1_odds=2.2, player_2_odds=1.7))
    return pd.DataFrame(rows)


def test_market_features_calculate_no_vig_probability_and_logit() -> None:
    frame = add_market_features(
        pd.DataFrame(
            [_row("m01", 2024, True, player_1_odds=2.0, player_2_odds=4.0)]
        )
    )

    assert frame.loc[0, "overround"] == pytest.approx(0.75)
    assert frame.loc[0, "market_probability_player_1"] == pytest.approx((1 / 2.0) / 0.75)
    expected_logit = np.log(
        frame.loc[0, "market_probability_player_1"]
        / (1 - frame.loc[0, "market_probability_player_1"])
    )
    assert frame.loc[0, "market_logit_player_1"] == pytest.approx(expected_logit)
    assert market_logit(pd.Series([0.0, 1.0]), LOGIT_EPSILON).between(-14, 14).all()


def test_invalid_odds_are_excluded_not_repaired() -> None:
    frame = pd.DataFrame(
        [
            _row("valid", 2024, True),
            _row("bad", 2024, False, player_1_odds=1.0),
        ]
    )

    result = add_market_features(frame)

    assert result["match_id"].tolist() == ["valid"]


def test_model_v2_features_exclude_leakage_and_v1_remains_unchanged() -> None:
    assert "market_logit_player_1" in MARKET_FEATURES
    assert FORBIDDEN_FEATURES.isdisjoint(MARKET_FEATURES)
    assert "player_1_odds" not in MARKET_FEATURES
    assert "player_2_odds" not in MARKET_FEATURES
    assert "edge" not in MARKET_FEATURES
    assert "expected_value" not in MARKET_FEATURES
    assert "result" not in MARKET_FEATURES
    assert "settlement_reason" not in MARKET_FEATURES
    assert "bankroll_after" not in MARKET_FEATURES
    assert "market_logit_player_1" not in MODEL_V1_FEATURES


def test_train_model_v2_is_deterministic_and_does_not_mutate_input() -> None:
    frame = _features()
    original = frame.copy(deep=True)

    first = train_model_v2(frame)
    second = train_model_v2(frame)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert set(first.predictions["evaluation_year"]) == {2023, 2024, 2025}
    assert first.predictions["model_version"].unique().tolist() == ["model_v2"]
    assert np.allclose(
        first.predictions["predicted_player_1_probability"]
        + first.predictions["predicted_player_2_probability"],
        1.0,
    )
    assert "market_logit_player_1" in first.folds[-1].coefficient_by_feature


def test_train_only_preprocessing_excludes_evaluation_extremes() -> None:
    frame = _features()
    frame.loc[frame["match_date"].dt.year == 2023, "log_rank_diff"] = 1_000_000.0

    result = train_model_v2(frame)
    fold_one = result.folds[0]
    numeric_pipeline = fold_one.model.named_steps["preprocessor"].named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    log_rank_index = list(numeric_pipeline.feature_names_in_).index("log_rank_diff")

    assert abs(imputer.statistics_[log_rank_index]) < 100


def test_artifacts_are_valid() -> None:
    output_dir = Path(".tmp-model-v2-artifacts")
    paths = ModelV2OutputPaths(
        model_output=output_dir / "model.joblib",
        metadata_output=output_dir / "metadata.json",
        predictions_output=output_dir / "predictions.parquet",
        metrics_output=output_dir / "metrics.json",
        corrections_output=output_dir / "corrections.parquet",
        calibration_output=output_dir / "calibration.parquet",
        correction_distribution_plot=output_dir / "corrections.png",
    )
    for path in paths.__dict__.values():
        path.unlink(missing_ok=True)

    result = train_model_v2(_features())
    write_model_v2_artifacts(result, paths)

    assert paths.model_output.exists()
    metadata = json.loads(paths.metadata_output.read_text(encoding="utf-8"))
    metrics = json.loads(paths.metrics_output.read_text(encoding="utf-8"))
    assert metadata["model_version"] == "model_v2"
    assert not pd.read_parquet(paths.predictions_output).empty
    assert not pd.read_parquet(paths.corrections_output).empty
    assert not pd.read_parquet(paths.calibration_output).empty
    assert metrics["model_version"] == "model_v2"
    assert paths.correction_distribution_plot.exists()
    assert paths.correction_distribution_plot.stat().st_size > 0
