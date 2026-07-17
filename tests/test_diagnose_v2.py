from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.test_train_v2 import _features

from tennis_value.diagnose_v2 import (
    DiagnosticOutputPaths,
    build_bootstrap_significance,
    build_correction_buckets,
    build_correction_diagnostics,
    build_odds_quality_metrics,
    calculate_row_differences,
    correction_bucket,
    paired_bootstrap,
    run_diagnostics,
    write_diagnostic_artifacts,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m2023a",
                "match_date": "2023-01-01",
                "evaluation_year": 2023,
                "surface": "Hard",
                "player_1": "A",
                "player_2": "B",
                "actual_player_1_won": 1,
                "player_1_odds": 2.0,
                "player_2_odds": 2.0,
                "market_probability_player_1": 0.50,
                "predicted_player_1_probability": 0.60,
                "predicted_player_2_probability": 0.40,
                "probability_correction": 0.10,
                "model_version": "model_v2",
            },
            {
                "match_id": "m2023b",
                "match_date": "2023-01-02",
                "evaluation_year": 2023,
                "surface": "Hard",
                "player_1": "C",
                "player_2": "D",
                "actual_player_1_won": 0,
                "player_1_odds": 1.8,
                "player_2_odds": 2.1,
                "market_probability_player_1": 0.52,
                "predicted_player_1_probability": 0.40,
                "predicted_player_2_probability": 0.60,
                "probability_correction": -0.12,
                "model_version": "model_v2",
            },
            {
                "match_id": "m2024a",
                "match_date": "2024-01-01",
                "evaluation_year": 2024,
                "surface": "Clay",
                "player_1": "E",
                "player_2": "F",
                "actual_player_1_won": 1,
                "player_1_odds": 1.5,
                "player_2_odds": 2.7,
                "market_probability_player_1": 0.64,
                "predicted_player_1_probability": 0.63,
                "predicted_player_2_probability": 0.37,
                "probability_correction": -0.01,
                "model_version": "model_v2",
            },
            {
                "match_id": "m2025a",
                "match_date": "2025-01-01",
                "evaluation_year": 2025,
                "surface": "Grass",
                "player_1": "G",
                "player_2": "H",
                "actual_player_1_won": 0,
                "player_1_odds": 2.2,
                "player_2_odds": 1.7,
                "market_probability_player_1": 0.435,
                "predicted_player_1_probability": 0.45,
                "predicted_player_2_probability": 0.55,
                "probability_correction": 0.015,
                "model_version": "model_v2",
            },
        ]
    )


def test_row_level_log_loss_and_brier_differences() -> None:
    result = calculate_row_differences(_predictions())
    first = result.iloc[0]

    assert first["model_log_loss_row"] == pytest.approx(-__import__("math").log(0.60))
    assert first["market_log_loss_row"] == pytest.approx(-__import__("math").log(0.50))
    assert first["log_loss_difference"] > 0
    assert first["brier_difference"] == pytest.approx((0.50 - 1) ** 2 - (0.60 - 1) ** 2)


def test_bootstrap_is_deterministic_paired_and_ci_is_percentile() -> None:
    differences = pd.Series([0.1, 0.2, -0.1, 0.0])

    first = paired_bootstrap(differences, samples=200, seed=7)
    second = paired_bootstrap(differences, samples=200, seed=7)

    assert first["mean_improvement"] == second["mean_improvement"]
    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]
    assert (first["bootstrap_distribution"] == second["bootstrap_distribution"]).all()
    assert first["mean_improvement"] == pytest.approx(differences.mean())
    assert first["ci_lower"] <= first["median_bootstrap_improvement"] <= first["ci_upper"]
    assert 0 <= first["probability_model_beats_market"] <= 1


def test_bootstrap_significance_reports_years_and_combined() -> None:
    report, distribution = build_bootstrap_significance(_predictions(), samples=50, seed=3)

    assert set(report["segments"]) == {"2023", "2024", "2025", "combined_2023_2025"}
    assert {"segment", "sample_index", "metric", "improvement"}.issubset(distribution.columns)
    assert report["segments"]["combined_2023_2025"]["log_loss"]["sample_count"] == 4


def test_probability_clipping_handles_extreme_probabilities() -> None:
    frame = _predictions()
    frame.loc[0, "predicted_player_1_probability"] = 1.0
    frame.loc[0, "predicted_player_2_probability"] = 0.0

    result = calculate_row_differences(frame)

    assert result["model_log_loss_row"].notna().all()


def test_correction_buckets_and_diagnostics() -> None:
    diagnostics = build_correction_diagnostics(_predictions())
    buckets = build_correction_buckets(_predictions())

    assert correction_bucket(0.0) == "0.00-0.01"
    assert correction_bucket(0.01) == "0.01-0.02"
    assert correction_bucket(0.02) == "0.02-0.05"
    assert correction_bucket(0.05) == "0.05-0.10"
    assert correction_bucket(0.10) == "0.10+"
    assert "combined_2023_2025" in diagnostics["segment"].tolist()
    assert {"model_log_loss", "market_log_loss", "log_loss_improvement"}.issubset(
        buckets.columns
    )


def test_overround_filters_report_included_and_excluded_counts() -> None:
    metrics = build_odds_quality_metrics(_predictions())

    assert set(metrics["filter"]) == {
        "standard_valid_odds",
        "normal_overround",
        "strict_overround",
    }
    assert (metrics["included_rows"] > 0).any()
    assert "excluded_rows_total" in metrics.columns


def test_input_frames_are_not_mutated_and_artifacts_have_stable_schemas() -> None:
    predictions = _predictions()
    features = _features()
    original_predictions = predictions.copy(deep=True)
    original_features = features.copy(deep=True)
    output_dir = Path(".tmp-v2-diagnostics")
    paths = DiagnosticOutputPaths(
        bootstrap_significance=output_dir / "bootstrap.json",
        ablation_metrics=output_dir / "ablation.parquet",
        ablation_summary=output_dir / "ablation.json",
        coefficients=output_dir / "coefficients.parquet",
        correction_diagnostics=output_dir / "correction_diagnostics.parquet",
        correction_buckets=output_dir / "correction_buckets.parquet",
        odds_quality_metrics=output_dir / "odds_quality.parquet",
        diagnostic_summary=output_dir / "summary.json",
        bootstrap_distribution_plot=output_dir / "bootstrap.png",
        ablation_log_loss_plot=output_dir / "ablation.png",
        correction_performance_plot=output_dir / "correction.png",
    )
    for path in paths.__dict__.values():
        path.unlink(missing_ok=True)

    result = run_diagnostics(predictions=predictions, features=features, bootstrap_samples=20)
    write_diagnostic_artifacts(result, paths)

    pd.testing.assert_frame_equal(predictions, original_predictions)
    pd.testing.assert_frame_equal(features, original_features)
    assert json.loads(paths.bootstrap_significance.read_text(encoding="utf-8"))["segments"]
    assert not pd.read_parquet(paths.ablation_metrics).empty
    assert not pd.read_parquet(paths.coefficients).empty
    assert paths.bootstrap_distribution_plot.stat().st_size > 0
