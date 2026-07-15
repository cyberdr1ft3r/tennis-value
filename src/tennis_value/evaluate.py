"""Probability-model evaluation and calibration artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

SUPPORTED_PARTITIONS = ("train", "validation", "test")
SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
PROBABILITY_EPSILON = 1e-15
CALIBRATION_BINS = tuple((index / 10.0, (index + 1) / 10.0) for index in range(10))
REQUIRED_PREDICTION_COLUMNS = (
    "match_id",
    "match_date",
    "partition",
    "surface",
    "player_1",
    "player_2",
    "actual_player_1_won",
    "predicted_player_1_probability",
    "predicted_player_2_probability",
    "player_1_odds",
    "player_2_odds",
    "model_version",
)


class ProbabilityMetrics(BaseModel):
    """Core binary probability metrics."""

    model_config = ConfigDict(frozen=True)

    sample_count: int
    positive_rate: float
    log_loss: float
    brier_score: float
    accuracy: float
    roc_auc: float | None = None
    mean_predicted_probability: float
    minimum_predicted_probability: float
    maximum_predicted_probability: float
    warnings: list[str] = Field(default_factory=list)


class CalibrationSummary(BaseModel):
    """Calibration summary for one partition."""

    model_config = ConfigDict(frozen=True)

    expected_calibration_error: float
    maximum_calibration_error: float


class EvaluationResult(BaseModel):
    """Evaluation metrics, tables, and JSON reports."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metrics_report: dict[str, Any]
    comparison_report: dict[str, Any]
    calibration_table: pd.DataFrame
    surface_metrics: pd.DataFrame


@dataclass(frozen=True)
class EvaluationOutputPaths:
    """Output paths for evaluation artifacts."""

    metrics_output: Path
    comparison_output: Path
    calibration_output: Path
    surface_output: Path
    calibration_plot: Path
    distribution_plot: Path


def calculate_probability_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> ProbabilityMetrics:
    """Calculate binary probability metrics with safe log-loss clipping."""
    target = _coerce_target(y_true)
    predicted = _coerce_probability_series(probabilities, "probabilities")
    if len(target) != len(predicted):
        msg = "target and probabilities must have the same length"
        raise ValueError(msg)
    if target.empty:
        msg = "at least one row is required for metrics"
        raise ValueError(msg)

    clipped = predicted.clip(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    predicted_class = (predicted >= 0.5).astype("int64")
    warnings: list[str] = []
    roc_auc: float | None = None
    if target.nunique() == 2:
        roc_auc = float(roc_auc_score(target, predicted))
    else:
        warnings.append("roc_auc unavailable: target contains one class")

    return ProbabilityMetrics(
        sample_count=len(target),
        positive_rate=float(target.mean()),
        log_loss=float(log_loss(target, clipped, labels=[0, 1])),
        brier_score=float(brier_score_loss(target, predicted)),
        accuracy=float(accuracy_score(target, predicted_class)),
        roc_auc=roc_auc,
        mean_predicted_probability=float(predicted.mean()),
        minimum_predicted_probability=float(predicted.min()),
        maximum_predicted_probability=float(predicted.max()),
        warnings=warnings,
    )


def build_calibration_table(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    partition_column: str = "partition",
) -> pd.DataFrame:
    """Build fixed-bin calibration rows for each partition."""
    required = [probability_column, target_column, partition_column]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        msg = f"missing required calibration columns: {missing}"
        raise ValueError(msg)
    frame = predictions.copy(deep=True)
    frame[target_column] = _coerce_target(frame[target_column])
    frame[probability_column] = _coerce_probability_series(
        frame[probability_column],
        probability_column,
    )
    rows: list[dict[str, Any]] = []
    for partition in sorted(frame[partition_column].astype(str).unique()):
        partition_frame = frame[frame[partition_column].astype(str) == partition]
        for lower, upper in CALIBRATION_BINS:
            bucket_mask = _bucket_mask(partition_frame[probability_column], lower, upper)
            bucket = partition_frame[bucket_mask]
            if bucket.empty:
                continue
            mean_probability = float(bucket[probability_column].mean())
            observed_rate = float(bucket[target_column].mean())
            rows.append(
                {
                    "partition": partition,
                    "bucket_lower": lower,
                    "bucket_upper": upper,
                    "bucket_label": f"{lower:.2f}-{upper:.2f}",
                    "sample_count": len(bucket),
                    "mean_predicted_probability": mean_probability,
                    "observed_win_rate": observed_rate,
                    "calibration_error": observed_rate - mean_probability,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "partition",
            "bucket_lower",
            "bucket_upper",
            "bucket_label",
            "sample_count",
            "mean_predicted_probability",
            "observed_win_rate",
            "calibration_error",
        ],
    )


def calculate_calibration_summary(calibration_table: pd.DataFrame) -> dict[str, CalibrationSummary]:
    """Calculate ECE and maximum absolute calibration error by partition."""
    summaries: dict[str, CalibrationSummary] = {}
    if calibration_table.empty:
        return summaries
    for partition, group in calibration_table.groupby("partition", sort=True):
        total = int(group["sample_count"].sum())
        absolute_errors = group["calibration_error"].abs()
        expected_error = float((absolute_errors * group["sample_count"]).sum() / total)
        summaries[str(partition)] = CalibrationSummary(
            expected_calibration_error=expected_error,
            maximum_calibration_error=float(absolute_errors.max()),
        )
    return summaries


def calculate_bookmaker_baseline(predictions: pd.DataFrame) -> dict[str, Any]:
    """Calculate bookmaker no-vig baseline metrics by partition."""
    frame = predictions.copy(deep=True)
    valid = _valid_odds_mask(frame)
    frame = frame.loc[valid].copy()
    if frame.empty:
        return {
            "rows_with_valid_odds": 0,
            "warnings": ["bookmaker baseline unavailable: no valid paired odds"],
        }
    raw_p1 = 1.0 / pd.to_numeric(frame["player_1_odds"], errors="coerce")
    raw_p2 = 1.0 / pd.to_numeric(frame["player_2_odds"], errors="coerce")
    frame["overround"] = raw_p1 + raw_p2
    frame["bookmaker_probability_player_1"] = raw_p1 / frame["overround"]

    partitions: dict[str, Any] = {}
    for partition, group in frame.groupby("partition", sort=True):
        bookmaker_metrics = calculate_probability_metrics(
            group["actual_player_1_won"],
            group["bookmaker_probability_player_1"],
        )
        model_metrics = calculate_probability_metrics(
            group["actual_player_1_won"],
            group["predicted_player_1_probability"],
        )
        partitions[str(partition)] = {
            "rows_with_valid_odds": len(group),
            "average_overround": float(group["overround"].mean()),
            "minimum_overround": float(group["overround"].min()),
            "maximum_overround": float(group["overround"].max()),
            "bookmaker_log_loss": bookmaker_metrics.log_loss,
            "bookmaker_brier_score": bookmaker_metrics.brier_score,
            "bookmaker_accuracy": bookmaker_metrics.accuracy,
            "bookmaker_roc_auc": bookmaker_metrics.roc_auc,
            "model_log_loss_on_odds_subset": model_metrics.log_loss,
            "model_brier_on_odds_subset": model_metrics.brier_score,
            "model_accuracy_on_odds_subset": model_metrics.accuracy,
            "log_loss_improvement": bookmaker_metrics.log_loss - model_metrics.log_loss,
            "brier_improvement": bookmaker_metrics.brier_score - model_metrics.brier_score,
        }
    return {
        "rows_with_valid_odds": len(frame),
        "partitions": partitions,
    }


def join_elo_baseline(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join Elo baseline probability by unique match_id."""
    if "match_id" not in features or "elo_expected_player_1" not in features:
        msg = "features must contain match_id and elo_expected_player_1 for Elo comparison"
        raise ValueError(msg)
    if features["match_id"].duplicated().any():
        msg = "features match_id values must be unique for Elo comparison"
        raise ValueError(msg)
    baseline = features[["match_id", "elo_expected_player_1"]].copy()
    return predictions.merge(baseline, on="match_id", how="left", validate="one_to_one")


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    model_version: str,
) -> EvaluationResult:
    """Evaluate partition, calibration, surface, Elo, and bookmaker metrics."""
    frame = predictions.copy(deep=True)
    _validate_predictions(frame)

    warnings: list[str] = []
    partition_metrics: dict[str, Any] = {}
    for partition in SUPPORTED_PARTITIONS:
        partition_frame = frame[frame["partition"] == partition]
        if partition_frame.empty:
            continue
        metrics = calculate_probability_metrics(
            partition_frame["actual_player_1_won"],
            partition_frame["predicted_player_1_probability"],
        )
        warnings.extend([f"{partition}: {warning}" for warning in metrics.warnings])
        partition_metrics[partition] = metrics.model_dump()

    calibration_table = build_calibration_table(
        frame,
        probability_column="predicted_player_1_probability",
        target_column="actual_player_1_won",
    )
    calibration_summaries = calculate_calibration_summary(calibration_table)
    surface_metrics = calculate_surface_metrics(frame)
    comparison_report = _build_comparison_report(frame)

    primary_test_metrics = partition_metrics.get("test", {})
    if "test" in calibration_summaries:
        primary_test_metrics = {
            **primary_test_metrics,
            **calibration_summaries["test"].model_dump(),
        }
    metrics_report = {
        "model_version": model_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partitions": {
            partition: {
                **metrics,
                **(
                    calibration_summaries[partition].model_dump()
                    if partition in calibration_summaries
                    else {}
                ),
            }
            for partition, metrics in partition_metrics.items()
        },
        "primary_test_metrics": primary_test_metrics,
        "calibration_bins": [
            {"lower": lower, "upper": upper, "label": f"{lower:.2f}-{upper:.2f}"}
            for lower, upper in CALIBRATION_BINS
        ],
        "warnings": warnings + comparison_report.get("warnings", []),
    }
    return EvaluationResult(
        metrics_report=_to_jsonable(metrics_report),
        comparison_report=_to_jsonable(comparison_report),
        calibration_table=calibration_table,
        surface_metrics=surface_metrics,
    )


def calculate_surface_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics by partition and surface."""
    invalid_surfaces = sorted(set(predictions["surface"].astype(str)) - SUPPORTED_SURFACES)
    if invalid_surfaces:
        msg = f"unsupported surfaces for surface metrics: {invalid_surfaces}"
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for (partition, surface), group in predictions.groupby(["partition", "surface"], sort=True):
        metrics = calculate_probability_metrics(
            group["actual_player_1_won"],
            group["predicted_player_1_probability"],
        )
        rows.append(
            {
                "partition": partition,
                "surface": surface,
                "sample_count": metrics.sample_count,
                "positive_rate": metrics.positive_rate,
                "log_loss": metrics.log_loss,
                "brier_score": metrics.brier_score,
                "accuracy": metrics.accuracy,
                "roc_auc": metrics.roc_auc,
            }
        )
    return pd.DataFrame(rows)


def write_evaluation_artifacts(
    result: EvaluationResult,
    output_paths: EvaluationOutputPaths,
    *,
    predictions: pd.DataFrame,
) -> None:
    """Write JSON, Parquet, and PNG evaluation artifacts."""
    for path in (
        output_paths.metrics_output,
        output_paths.comparison_output,
        output_paths.calibration_output,
        output_paths.surface_output,
        output_paths.calibration_plot,
        output_paths.distribution_plot,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_paths.metrics_output.write_text(
        _json_dumps(result.metrics_report),
        encoding="utf-8",
    )
    output_paths.comparison_output.write_text(
        _json_dumps(result.comparison_report),
        encoding="utf-8",
    )
    result.calibration_table.to_parquet(output_paths.calibration_output, index=False)
    result.surface_metrics.to_parquet(output_paths.surface_output, index=False)
    write_calibration_plot(result.calibration_table, output_paths.calibration_plot)
    write_probability_distribution_plot(predictions, output_paths.distribution_plot)


def write_calibration_plot(calibration_table: pd.DataFrame, output_path: Path) -> None:
    """Write a simple headless calibration PNG."""
    image, draw = _blank_plot("Calibration by Partition", "Mean predicted probability", "Observed")
    _draw_diagonal(draw)
    colors = {"train": "blue", "validation": "green", "test": "red"}
    for partition, group in calibration_table.groupby("partition", sort=True):
        color = colors.get(str(partition), "black")
        for _, row in group.iterrows():
            x, y = _plot_point(
                float(row["mean_predicted_probability"]),
                float(row["observed_win_rate"]),
            )
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text((x + 5, y - 5), str(int(row["sample_count"])), fill=color)
    image.save(output_path)


def write_probability_distribution_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write a simple headless probability-distribution PNG."""
    image, draw = _blank_plot("Predicted Probability Distribution", "Probability", "Count")
    colors = {"train": "blue", "validation": "green", "test": "red"}
    max_count = 1
    histograms: dict[str, np.ndarray[Any, np.dtype[np.int64]]] = {}
    for partition, group in predictions.groupby("partition", sort=True):
        counts, _ = np.histogram(group["predicted_player_1_probability"], bins=10, range=(0, 1))
        histograms[str(partition)] = counts
        max_count = max(max_count, int(counts.max()))
    bar_width = 10
    for offset, (partition, counts) in enumerate(histograms.items()):
        color = colors.get(partition, "black")
        for index, count in enumerate(counts):
            x = 60 + index * 50 + offset * bar_width
            y = 340 - int((int(count) / max_count) * 260)
            draw.rectangle((x, y, x + bar_width - 2, 340), fill=color)
    image.save(output_path)


def _build_comparison_report(predictions: pd.DataFrame) -> dict[str, Any]:
    warnings: list[str] = []
    elo_report = _calculate_elo_comparison(predictions, warnings)
    bookmaker_report = calculate_bookmaker_baseline(predictions)
    return {
        "model_version": _single_model_version(predictions),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elo_baseline": elo_report,
        "bookmaker_no_vig_baseline": bookmaker_report,
        "warnings": warnings + bookmaker_report.get("warnings", []),
    }


def _calculate_elo_comparison(predictions: pd.DataFrame, warnings: list[str]) -> dict[str, Any]:
    if "elo_expected_player_1" not in predictions.columns:
        warnings.append("elo baseline unavailable: elo_expected_player_1 missing")
        return {"available": False, "partitions": {}}
    frame = predictions[predictions["elo_expected_player_1"].notna()].copy()
    missing = len(predictions) - len(frame)
    if missing:
        warnings.append(f"elo baseline skipped {missing} row(s) with missing Elo probability")
    if frame.empty:
        return {"available": False, "partitions": {}}
    frame["elo_expected_player_1"] = _coerce_probability_series(
        frame["elo_expected_player_1"],
        "elo_expected_player_1",
    )
    partitions: dict[str, Any] = {}
    for partition, group in frame.groupby("partition", sort=True):
        model_metrics = calculate_probability_metrics(
            group["actual_player_1_won"],
            group["predicted_player_1_probability"],
        )
        elo_metrics = calculate_probability_metrics(
            group["actual_player_1_won"],
            group["elo_expected_player_1"],
        )
        partitions[str(partition)] = {
            "rows_compared": len(group),
            "model_log_loss": model_metrics.log_loss,
            "elo_log_loss": elo_metrics.log_loss,
            "log_loss_improvement": elo_metrics.log_loss - model_metrics.log_loss,
            "model_brier_score": model_metrics.brier_score,
            "elo_brier_score": elo_metrics.brier_score,
            "brier_improvement": elo_metrics.brier_score - model_metrics.brier_score,
            "model_accuracy": model_metrics.accuracy,
            "elo_accuracy": elo_metrics.accuracy,
            "model_roc_auc": model_metrics.roc_auc,
            "elo_roc_auc": elo_metrics.roc_auc,
        }
    return {"available": True, "partitions": partitions}


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = [
        column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing:
        msg = f"missing required prediction columns: {missing}"
        raise ValueError(msg)
    unknown_partitions = sorted(
        set(predictions["partition"].astype(str)) - set(SUPPORTED_PARTITIONS)
    )
    if unknown_partitions:
        msg = f"unknown prediction partitions: {unknown_partitions}"
        raise ValueError(msg)
    duplicated_within = predictions.duplicated(["partition", "match_id"])
    if duplicated_within.any():
        msg = "duplicate match IDs within a partition are not allowed"
        raise ValueError(msg)
    partition_counts = predictions.groupby("match_id")["partition"].nunique()
    if (partition_counts > 1).any():
        msg = "same match ID appears in multiple partitions"
        raise ValueError(msg)
    _coerce_target(predictions["actual_player_1_won"])
    p1 = _coerce_probability_series(
        predictions["predicted_player_1_probability"],
        "predicted_player_1_probability",
    )
    p2 = _coerce_probability_series(
        predictions["predicted_player_2_probability"],
        "predicted_player_2_probability",
    )
    if not np.allclose(p1 + p2, 1.0, atol=1e-9):
        msg = "player probabilities must sum to 1"
        raise ValueError(msg)


def _valid_odds_mask(frame: pd.DataFrame) -> pd.Series:
    odds_1 = pd.to_numeric(frame["player_1_odds"], errors="coerce")
    odds_2 = pd.to_numeric(frame["player_2_odds"], errors="coerce")
    raw_1 = 1.0 / odds_1
    raw_2 = 1.0 / odds_2
    overround = raw_1 + raw_2
    mask = (
        odds_1.notna()
        & odds_2.notna()
        & np.isfinite(odds_1)
        & np.isfinite(odds_2)
        & (odds_1 > 1.0)
        & (odds_2 > 1.0)
        & np.isfinite(overround)
        & (overround > 0)
    )
    return pd.Series(mask, index=frame.index, dtype="bool")


def _bucket_mask(series: pd.Series, lower: float, upper: float) -> pd.Series:
    if upper == 1.0:
        return (series >= lower) & (series <= upper)
    return (series >= lower) & (series < upper)


def _coerce_target(values: pd.Series) -> pd.Series:
    mapped = values.map(_target_to_int)
    if mapped.isna().any():
        msg = "target values must be valid binary values"
        raise ValueError(msg)
    return mapped.astype("int64")


def _target_to_int(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return int(value)
    if value in {0, 1}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return 1
        if normalized in {"false", "0", "no"}:
            return 0
    return None


def _coerce_probability_series(values: pd.Series, column_name: str) -> pd.Series:
    probabilities = pd.to_numeric(values, errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        msg = f"{column_name} contains non-finite probabilities"
        raise ValueError(msg)
    if not probabilities.between(0, 1).all():
        msg = f"{column_name} probabilities must be in [0, 1]"
        raise ValueError(msg)
    return probabilities.astype("float64")


def _single_model_version(predictions: pd.DataFrame) -> str:
    versions = sorted(set(predictions["model_version"].astype(str)))
    return versions[0] if len(versions) == 1 else ",".join(versions)


def _blank_plot(title: str, x_label: str, y_label: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 580, 340), outline="black")
    draw.text((60, 12), title, fill="black")
    draw.text((250, 375), x_label, fill="black")
    draw.text((8, 180), y_label, fill="black")
    for tick in range(11):
        x = 60 + tick * 52
        y = 340 - tick * 30
        draw.line((x, 340, x, 345), fill="black")
        draw.line((55, y, 60, y), fill="black")
    return image, draw


def _draw_diagonal(draw: ImageDraw.ImageDraw) -> None:
    draw.line((_plot_point(0, 0), _plot_point(1, 1)), fill="gray", width=1)


def _plot_point(x_value: float, y_value: float) -> tuple[int, int]:
    x = 60 + int(x_value * 520)
    y = 340 - int(y_value * 300)
    return x, y


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(_to_jsonable(value), indent=2)


__all__ = [
    "CALIBRATION_BINS",
    "PROBABILITY_EPSILON",
    "EvaluationOutputPaths",
    "EvaluationResult",
    "ProbabilityMetrics",
    "build_calibration_table",
    "calculate_bookmaker_baseline",
    "calculate_calibration_summary",
    "calculate_probability_metrics",
    "calculate_surface_metrics",
    "evaluate_predictions",
    "join_elo_baseline",
    "write_evaluation_artifacts",
]
