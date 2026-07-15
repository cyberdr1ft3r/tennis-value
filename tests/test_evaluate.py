from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tennis_value.evaluate import (
    EvaluationOutputPaths,
    build_calibration_table,
    calculate_bookmaker_baseline,
    calculate_calibration_summary,
    calculate_probability_metrics,
    evaluate_predictions,
    join_elo_baseline,
    write_evaluation_artifacts,
)


def _prediction(
    match_id: str,
    partition: str,
    target: int,
    probability: float,
    *,
    surface: str = "Hard",
    odds_1: float | None = 1.8,
    odds_2: float | None = 2.0,
    elo_probability: float | None = 0.5,
) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": match_id,
        "match_date": pd.Timestamp("2025-01-01"),
        "partition": partition,
        "surface": surface,
        "player_1": f"{match_id} A",
        "player_2": f"{match_id} B",
        "actual_player_1_won": target,
        "predicted_player_1_probability": probability,
        "predicted_player_2_probability": 1.0 - probability,
        "player_1_odds": odds_1,
        "player_2_odds": odds_2,
        "model_version": "model_v1",
    }
    if elo_probability is not None:
        row["elo_expected_player_1"] = elo_probability
    return row


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _prediction("tr01", "train", 1, 0.9, surface="Hard", elo_probability=0.6),
            _prediction("tr02", "train", 0, 0.1, surface="Clay", elo_probability=0.4),
            _prediction("va01", "validation", 1, 0.8, surface="Grass", elo_probability=0.6),
            _prediction("va02", "validation", 0, 0.2, surface="Hard", elo_probability=0.4),
            _prediction("te01", "test", 1, 0.7, surface="Clay", elo_probability=0.55),
            _prediction("te02", "test", 0, 0.3, surface="Grass", elo_probability=0.45),
        ]
    )


def test_probability_metrics_correctness_and_clipping() -> None:
    perfect = calculate_probability_metrics(pd.Series([1, 0]), pd.Series([1.0, 0.0]))
    inverted = calculate_probability_metrics(pd.Series([1, 0]), pd.Series([0.0, 1.0]))

    assert perfect.log_loss < inverted.log_loss
    assert perfect.brier_score == pytest.approx(0.0)
    assert perfect.accuracy == pytest.approx(1.0)
    assert perfect.roc_auc == pytest.approx(1.0)
    assert inverted.log_loss < 40


def test_accuracy_threshold_and_single_class_roc_auc() -> None:
    metrics = calculate_probability_metrics(
        pd.Series([1, 0, 1, 0]),
        pd.Series([0.5, 0.49, 0.2, 0.8]),
    )
    single_class = calculate_probability_metrics(pd.Series([1, 1]), pd.Series([0.2, 0.8]))

    assert metrics.accuracy == pytest.approx(0.5)
    assert single_class.roc_auc is None
    assert single_class.warnings


def test_metrics_do_not_modify_original_probabilities() -> None:
    probabilities = pd.Series([1.0, 0.0])
    before = probabilities.copy(deep=True)

    calculate_probability_metrics(pd.Series([1, 0]), probabilities)

    pd.testing.assert_series_equal(probabilities, before)


def test_calibration_table_bins_edges_and_summary() -> None:
    frame = pd.DataFrame(
        [
            _prediction("a", "test", 0, 0.0),
            _prediction("b", "test", 1, 1.0),
            _prediction("c", "test", 1, 0.55),
        ]
    )

    table = build_calibration_table(
        frame,
        probability_column="predicted_player_1_probability",
        target_column="actual_player_1_won",
    )
    summary = calculate_calibration_summary(table)["test"]

    assert table["bucket_label"].tolist() == ["0.00-0.10", "0.50-0.60", "0.90-1.00"]
    assert table["sample_count"].tolist() == [1, 1, 1]
    assert table.loc[0, "observed_win_rate"] == pytest.approx(0.0)
    assert table.loc[1, "observed_win_rate"] == pytest.approx(1.0)
    assert summary.expected_calibration_error >= 0
    assert summary.maximum_calibration_error >= summary.expected_calibration_error


def test_elo_comparison_uses_same_rows_and_positive_improvement() -> None:
    result = evaluate_predictions(_predictions(), model_version="model_v1")
    test = result.comparison_report["elo_baseline"]["partitions"]["test"]

    assert test["rows_compared"] == 2
    assert test["log_loss_improvement"] > 0
    assert test["brier_improvement"] > 0


def test_missing_elo_values_are_handled_explicitly() -> None:
    predictions = _predictions()
    predictions.loc[0, "elo_expected_player_1"] = pd.NA

    result = evaluate_predictions(predictions, model_version="model_v1")

    assert "missing Elo probability" in " ".join(result.comparison_report["warnings"])


def test_join_elo_baseline_requires_unique_match_ids() -> None:
    predictions = _predictions().drop(columns=["elo_expected_player_1"])
    features = pd.DataFrame(
        [
            {"match_id": "tr01", "elo_expected_player_1": 0.5},
            {"match_id": "tr01", "elo_expected_player_1": 0.5},
        ]
    )

    with pytest.raises(ValueError, match="unique"):
        join_elo_baseline(predictions, features)


def test_bookmaker_no_vig_baseline_and_invalid_odds() -> None:
    predictions = pd.DataFrame(
        [
            _prediction("a", "test", 1, 0.8, odds_1=2.0, odds_2=2.0),
            _prediction("b", "test", 0, 0.2, odds_1=1.5, odds_2=3.0),
            _prediction("c", "test", 1, 0.7, odds_1=1.0, odds_2=2.0),
            _prediction("d", "test", 1, 0.7, odds_1=None, odds_2=2.0),
        ]
    )

    result = calculate_bookmaker_baseline(predictions)
    test = result["partitions"]["test"]

    assert result["rows_with_valid_odds"] == 2
    assert test["rows_with_valid_odds"] == 2
    assert test["average_overround"] == pytest.approx(((0.5 + 0.5) + (1 / 1.5 + 1 / 3.0)) / 2)
    assert "model_log_loss_on_odds_subset" in test


def test_partition_validation_and_duplicate_ids() -> None:
    unknown = _predictions()
    unknown.loc[0, "partition"] = "future"
    with pytest.raises(ValueError, match="unknown"):
        evaluate_predictions(unknown, model_version="model_v1")

    duplicated = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate match IDs"):
        evaluate_predictions(duplicated, model_version="model_v1")

    overlap = _predictions()
    overlap.loc[1, "match_id"] = overlap.loc[0, "match_id"]
    overlap.loc[1, "partition"] = "validation"
    with pytest.raises(ValueError, match="multiple partitions"):
        evaluate_predictions(overlap, model_version="model_v1")


def test_surface_metrics_grouping_and_single_class_roc_auc() -> None:
    result = evaluate_predictions(_predictions(), model_version="model_v1")
    surfaces = result.surface_metrics

    assert set(surfaces["surface"]) == {"Hard", "Clay", "Grass"}
    assert surfaces["sample_count"].min() == 1
    assert surfaces["roc_auc"].isna().any()


def test_invalid_inputs_fail_helpfully() -> None:
    missing = _predictions().drop(columns=["predicted_player_1_probability"])
    with pytest.raises(ValueError, match="missing required prediction columns"):
        evaluate_predictions(missing, model_version="model_v1")

    invalid_probability = _predictions()
    invalid_probability.loc[0, "predicted_player_1_probability"] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_predictions(invalid_probability, model_version="model_v1")

    invalid_sum = _predictions()
    invalid_sum.loc[0, "predicted_player_2_probability"] = 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate_predictions(invalid_sum, model_version="model_v1")


def test_artifacts_are_written_and_readable() -> None:
    output_dir = Path(".tmp-task8-artifacts")
    output_dir.mkdir(exist_ok=True)
    paths = EvaluationOutputPaths(
        metrics_output=output_dir / "metrics.json",
        comparison_output=output_dir / "comparison.json",
        calibration_output=output_dir / "calibration.parquet",
        surface_output=output_dir / "surface.parquet",
        calibration_plot=output_dir / "calibration.png",
        distribution_plot=output_dir / "distribution.png",
    )
    predictions = _predictions()
    result = evaluate_predictions(predictions, model_version="model_v1")

    write_evaluation_artifacts(result, paths, predictions=predictions)

    json.loads(paths.metrics_output.read_text(encoding="utf-8"))
    json.loads(paths.comparison_output.read_text(encoding="utf-8"))
    assert not pd.read_parquet(paths.calibration_output).empty
    assert not pd.read_parquet(paths.surface_output).empty
    assert paths.calibration_plot.stat().st_size > 0
    assert paths.distribution_plot.stat().st_size > 0


def test_input_integrity_and_repeated_evaluation_tables_are_deterministic() -> None:
    predictions = _predictions()
    before = predictions.copy(deep=True)

    first = evaluate_predictions(predictions, model_version="model_v1")
    second = evaluate_predictions(predictions, model_version="model_v1")

    pd.testing.assert_frame_equal(predictions, before)
    pd.testing.assert_frame_equal(first.calibration_table, second.calibration_table)
    pd.testing.assert_frame_equal(first.surface_metrics, second.surface_metrics)
