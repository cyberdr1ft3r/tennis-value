from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tests.test_train_v2 import _features

from tennis_value.train import MODEL_FEATURES as MODEL_V1_FEATURES
from tennis_value.train_v2 import (
    FORBIDDEN_FEATURES,
    MARKET_FEATURES,
    build_walk_forward_folds,
    prepare_model_v2_dataset,
)
from tennis_value.train_v2_1 import (
    CORRECTION_CAP,
    FORM_WORKLOAD_FEATURES,
    MODEL_V2_1_FEATURES,
    FixedOffsetLogisticCorrection,
    ModelV21OutputPaths,
    block_bootstrap_comparison,
    build_correction_direction,
    correction_direction_slope,
    signed_correction_bucket,
    train_model_v2_1,
    write_model_v2_1_artifacts,
)


def test_model_v2_1_feature_allowlist_and_existing_models_unchanged() -> None:
    assert MODEL_V2_1_FEATURES == [
        "market_logit_player_1",
        "recent_10_win_rate_diff",
        "surface_recent_10_win_rate_diff",
        "days_since_last_match_diff",
        "matches_last_14d_diff",
    ]
    assert FORBIDDEN_FEATURES.isdisjoint(MODEL_V2_1_FEATURES)
    assert {"edge", "expected_value", "stake", "bankroll_after", "roi"}.isdisjoint(
        MODEL_V2_1_FEATURES
    )
    assert "market_logit_player_1" in MARKET_FEATURES
    assert "market_logit_player_1" not in MODEL_V1_FEATURES


def test_v2_1_uses_strict_walk_forward_boundaries_and_train_only_preprocessing() -> None:
    frame = _features()
    result = train_model_v2_1(frame, bootstrap_samples=20)
    prepared = result.predictions

    assert set(prepared["evaluation_year"]) == {2023, 2024, 2025}
    assert set(prepared["architecture"]) == {
        "market_recalibration",
        "free_form_workload",
        "fixed_offset_form_workload",
        "fixed_offset_form_workload_capped",
    }
    for fold in build_walk_forward_folds(prepare_model_v2_dataset(frame)):
        assert fold.train["match_date"].max() < fold.evaluation["match_date"].min()


def test_fixed_offset_model_keeps_market_coefficient_fixed_to_one() -> None:
    frame = pd.DataFrame(
        {
            "recent_10_win_rate_diff": [0.1, -0.1, 0.2, -0.2],
            "surface_recent_10_win_rate_diff": [0.1, -0.1, 0.2, -0.2],
            "days_since_last_match_diff": [1.0, 2.0, 1.0, 2.0],
            "matches_last_14d_diff": [0, 1, 0, 1],
        }
    )
    target = pd.Series([1, 0, 1, 0])
    offset = pd.Series([0.0, 0.0, 0.2, -0.2])
    model = FixedOffsetLogisticCorrection(feature_names=FORM_WORKLOAD_FEATURES, max_iter=50)

    model.fit(frame, target, offset)

    assert model.coefficient_by_feature()["market_logit_player_1"] == 1.0
    probabilities, correction_logit, capped = model.predict_proba(frame, offset)
    assert probabilities.between(0, 1).all()
    assert correction_logit.notna().all()
    assert not capped.any()


def test_correction_cap_is_applied_on_logit_not_probability() -> None:
    frame = _features()
    result = train_model_v2_1(frame, bootstrap_samples=20)
    capped = result.predictions[
        result.predictions["architecture"].eq("fixed_offset_form_workload_capped")
    ]

    assert capped["correction_logit"].abs().le(CORRECTION_CAP + 1e-12).all()
    assert capped["model_probability"].between(0, 1).all()


def test_block_bootstrap_keeps_whole_blocks_and_positive_improvement_semantics() -> None:
    rows = pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "match_date": ["2024-01-01", "2024-01-02", "2024-01-08", "2024-01-09"],
            "evaluation_year": [2024, 2024, 2024, 2024],
            "iso_week": [1, 1, 2, 2],
            "actual_player_1_won": [1, 0, 1, 0],
            "model_probability": [0.7, 0.3, 0.7, 0.3],
            "market_probability_player_1": [0.5, 0.5, 0.5, 0.5],
        }
    )

    report = block_bootstrap_comparison(
        rows,
        candidate_probability_column="model_probability",
        comparator_probability_column="market_probability_player_1",
        samples=50,
        seed=4,
    )

    assert report["block_count"] == 2
    assert report["mean_log_loss_improvement"] > 0
    assert 0 <= report["probability_model_beats_comparator"] <= 1


def test_signed_correction_buckets_and_direction_slope() -> None:
    assert signed_correction_bucket(-0.06) == "below_-0.05"
    assert signed_correction_bucket(-0.02) == "-0.02_to_-0.01"
    assert signed_correction_bucket(0.0) == "-0.01_to_0.01"
    assert signed_correction_bucket(0.05) == "above_0.05"
    rows = pd.DataFrame(
        {
            "architecture": ["free_form_workload"] * 4,
            "signed_correction_bucket": ["-0.01_to_0.01"] * 4,
            "correction": [-0.2, -0.1, 0.1, 0.2],
            "market_residual": [-0.3, -0.1, 0.1, 0.3],
            "actual_player_1_won": [0, 0, 1, 1],
            "market_probability_player_1": [0.5, 0.5, 0.5, 0.5],
            "model_probability": [0.3, 0.4, 0.6, 0.7],
        }
    )

    assert correction_direction_slope(rows) > 0
    direction = build_correction_direction(rows)
    assert {"rows", "log_loss_improvement", "correction_direction_slope"}.issubset(direction)


def test_v2_1_does_not_mutate_input_and_artifacts_have_stable_schema(tmp_path: Path) -> None:
    frame = _features()
    original = frame.copy(deep=True)
    paths = ModelV21OutputPaths(
        model_output=tmp_path / "model.joblib",
        metadata_output=tmp_path / "metadata.json",
        predictions_output=tmp_path / "predictions.parquet",
        architecture_metrics=tmp_path / "metrics.parquet",
        block_bootstrap=tmp_path / "bootstrap.json",
        correction_direction=tmp_path / "direction.parquet",
        odds_sensitivity=tmp_path / "odds.parquet",
        summary=tmp_path / "summary.json",
        architecture_comparison_plot=tmp_path / "architecture.png",
        correction_calibration_plot=tmp_path / "correction.png",
    )

    result = train_model_v2_1(frame, bootstrap_samples=20)
    write_model_v2_1_artifacts(result, paths)

    pd.testing.assert_frame_equal(frame, original)
    assert json.loads(paths.metadata_output.read_text(encoding="utf-8"))["model_version"]
    assert not pd.read_parquet(paths.predictions_output).empty
    assert not pd.read_parquet(paths.architecture_metrics).empty
    assert json.loads(paths.block_bootstrap.read_text(encoding="utf-8"))["comparisons"]
    assert not pd.read_parquet(paths.correction_direction).empty
    assert not pd.read_parquet(paths.odds_sensitivity).empty
    assert paths.architecture_comparison_plot.stat().st_size > 0
    assert paths.correction_calibration_plot.stat().st_size > 0


def test_predictions_have_no_business_leakage_columns() -> None:
    result = train_model_v2_1(_features(), bootstrap_samples=20)

    forbidden = {"edge", "expected_value", "result", "settlement", "stake", "bankroll", "roi"}
    for column in result.predictions.columns:
        assert not any(token in column for token in forbidden)
    assert np.isfinite(result.predictions["model_probability"]).all()
