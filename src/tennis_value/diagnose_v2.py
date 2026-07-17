"""Diagnostics for the market-anchored Model v2 experiment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_value.evaluate import PROBABILITY_EPSILON, calculate_probability_metrics
from tennis_value.train_v2 import (
    BINARY_FEATURES,
    FORBIDDEN_FEATURES,
    MARKET_FEATURES,
    build_walk_forward_folds,
    prepare_model_v2_dataset,
)

BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
MODEL_COLUMN = "predicted_player_1_probability"
MARKET_COLUMN = "market_probability_player_1"
TARGET_COLUMN = "actual_player_1_won"
YEAR_COLUMN = "evaluation_year"
AblationKind = Literal[
    "market_baseline",
    "logistic_regression",
]
REQUIRED_PREDICTION_COLUMNS = (
    "match_id",
    "match_date",
    "evaluation_year",
    "actual_player_1_won",
    "player_1_odds",
    "player_2_odds",
    "market_probability_player_1",
    "predicted_player_1_probability",
)
CORRECTION_BUCKETS = (
    (0.00, 0.01, "0.00-0.01"),
    (0.01, 0.02, "0.01-0.02"),
    (0.02, 0.05, "0.02-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, math.inf, "0.10+"),
)
ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "A_market_baseline": {
        "kind": "market_baseline",
        "features": [],
    },
    "B_market_recalibration": {
        "kind": "logistic_regression",
        "features": ["market_logit_player_1"],
    },
    "C_market_plus_overall_elo": {
        "kind": "logistic_regression",
        "features": ["market_logit_player_1", "overall_elo_diff"],
    },
    "D_market_plus_elo_package": {
        "kind": "logistic_regression",
        "features": [
            "market_logit_player_1",
            "overall_elo_diff",
            "surface_elo_diff",
            "history_count_min",
        ],
    },
    "E_market_plus_ranking": {
        "kind": "logistic_regression",
        "features": ["market_logit_player_1", "log_rank_diff"],
    },
    "F_market_plus_form_workload": {
        "kind": "logistic_regression",
        "features": [
            "market_logit_player_1",
            "recent_10_win_rate_diff",
            "surface_recent_10_win_rate_diff",
            "days_since_last_match_diff",
            "matches_last_14d_diff",
        ],
    },
    "G_full_model_v2": {
        "kind": "logistic_regression",
        "features": MARKET_FEATURES,
    },
}


class DiagnosticResult(BaseModel):
    """All Model v2 diagnostic outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    bootstrap_significance: dict[str, Any]
    ablation_metrics: pd.DataFrame
    ablation_summary: dict[str, Any]
    coefficients: pd.DataFrame
    correction_diagnostics: pd.DataFrame
    correction_buckets: pd.DataFrame
    odds_quality_metrics: pd.DataFrame
    diagnostic_summary: dict[str, Any]
    bootstrap_distribution: pd.DataFrame


@dataclass(frozen=True)
class DiagnosticOutputPaths:
    """Output paths for Model v2 diagnostic artifacts."""

    bootstrap_significance: Path
    ablation_metrics: Path
    ablation_summary: Path
    coefficients: Path
    correction_diagnostics: Path
    correction_buckets: Path
    odds_quality_metrics: Path
    diagnostic_summary: Path
    bootstrap_distribution_plot: Path
    ablation_log_loss_plot: Path
    correction_performance_plot: Path


def calculate_row_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired row-level log-loss and Brier improvements."""
    _validate_predictions(predictions)
    frame = predictions.copy(deep=True)
    y = _target(frame[TARGET_COLUMN])
    model_probability = _probability(frame[MODEL_COLUMN], MODEL_COLUMN)
    market_probability = _probability(frame[MARKET_COLUMN], MARKET_COLUMN)
    model_clipped = model_probability.clip(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    market_clipped = market_probability.clip(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    model_log_loss = -(y * np.log(model_clipped) + (1 - y) * np.log(1 - model_clipped))
    market_log_loss = -(y * np.log(market_clipped) + (1 - y) * np.log(1 - market_clipped))
    model_brier = (model_probability - y) ** 2
    market_brier = (market_probability - y) ** 2
    output = frame[["match_id", "evaluation_year"]].copy()
    output["model_log_loss_row"] = model_log_loss.astype("float64")
    output["market_log_loss_row"] = market_log_loss.astype("float64")
    output["log_loss_difference"] = (
        output["market_log_loss_row"] - output["model_log_loss_row"]
    )
    output["model_brier_row"] = model_brier.astype("float64")
    output["market_brier_row"] = market_brier.astype("float64")
    output["brier_difference"] = output["market_brier_row"] - output["model_brier_row"]
    return output


def paired_bootstrap(
    differences: pd.Series,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap mean paired row differences by resampling row indices with replacement."""
    values = pd.to_numeric(differences, errors="raise").astype("float64").to_numpy()
    if len(values) == 0:
        msg = "at least one paired difference is required"
        raise ValueError(msg)
    if samples <= 0:
        msg = "bootstrap samples must be greater than zero"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrapped = values[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrapped, [2.5, 97.5])
    return {
        "sample_count": int(len(values)),
        "mean_improvement": float(values.mean()),
        "median_bootstrap_improvement": float(np.median(bootstrapped)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "probability_model_beats_market": float((bootstrapped > 0).mean()),
        "interval_excludes_zero": bool(lower > 0 or upper < 0),
        "bootstrap_distribution": bootstrapped,
    }


def build_bootstrap_significance(
    predictions: pd.DataFrame,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build paired bootstrap summaries by year and combined 2023-2025."""
    differences = calculate_row_differences(predictions)
    reports: dict[str, Any] = {}
    distribution_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(year), group) for year, group in differences.groupby("evaluation_year", sort=True)
    ]
    groups.append(("combined_2023_2025", differences))
    for label, group in groups:
        log_report = paired_bootstrap(group["log_loss_difference"], samples=samples, seed=seed)
        brier_report = paired_bootstrap(group["brier_difference"], samples=samples, seed=seed + 1)
        for index, value in enumerate(log_report.pop("bootstrap_distribution")):
            distribution_rows.append(
                {
                    "segment": label,
                    "sample_index": index,
                    "metric": "log_loss",
                    "improvement": float(value),
                }
            )
        for index, value in enumerate(brier_report.pop("bootstrap_distribution")):
            distribution_rows.append(
                {
                    "segment": label,
                    "sample_index": index,
                    "metric": "brier",
                    "improvement": float(value),
                }
            )
        reports[label] = {
            "sample_count": len(group),
            "log_loss": log_report,
            "brier": brier_report,
        }
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": samples,
        "seed": seed,
        "segments": reports,
    }
    return payload, pd.DataFrame(distribution_rows)


def run_ablation_study(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed walk-forward ablation variants using isolated fold preprocessing."""
    prepared = prepare_model_v2_dataset(features)
    folds = build_walk_forward_folds(prepared)
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for variant_name, spec in ABLATION_VARIANTS.items():
        kind = cast(AblationKind, spec["kind"])
        feature_names = cast(list[str], spec["features"])
        _validate_ablation_features(feature_names)
        for fold in folds:
            if kind == "market_baseline":
                probabilities = fold.evaluation[MARKET_COLUMN].astype("float64")
                coefficient_rows.append(
                    _coefficient_row(
                        variant_name,
                        fold.evaluation_year,
                        "market_probability_direct",
                        1.0,
                        "direct_baseline",
                    )
                )
            else:
                model = _build_variant_pipeline(feature_names)
                model.fit(fold.train[feature_names], fold.train["target"])
                probabilities = pd.Series(
                    model.predict_proba(fold.evaluation[feature_names])[:, 1],
                    index=fold.evaluation.index,
                    dtype="float64",
                )
                coefficient_rows.extend(
                    _coefficient_rows_from_model(variant_name, fold.evaluation_year, model)
                )
            metric_rows.append(
                _variant_metrics(
                    variant_name=variant_name,
                    evaluation_year=fold.evaluation_year,
                    features=feature_names,
                    y_true=fold.evaluation["target"],
                    probabilities=probabilities,
                    market_probabilities=fold.evaluation[MARKET_COLUMN],
                )
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def build_ablation_summary(ablation_metrics: pd.DataFrame) -> dict[str, Any]:
    """Summarize best variants and full-vs-recalibration comparisons."""
    best_by_year: dict[str, Any] = {}
    full_vs_recalibration: dict[str, Any] = {}
    for year, group in ablation_metrics.groupby("evaluation_year", sort=True):
        best = group.sort_values("model_log_loss", kind="mergesort").iloc[0]
        best_by_year[str(year)] = {
            "variant": best["variant"],
            "model_log_loss": float(best["model_log_loss"]),
            "log_loss_improvement_vs_market": float(best["log_loss_improvement_vs_market"]),
        }
        full = group[group["variant"] == "G_full_model_v2"].iloc[0]
        recalibration = group[group["variant"] == "B_market_recalibration"].iloc[0]
        full_vs_recalibration[str(year)] = {
            "full_model_log_loss": float(full["model_log_loss"]),
            "recalibration_log_loss": float(recalibration["model_log_loss"]),
            "full_beats_recalibration": bool(
                full["model_log_loss"] < recalibration["model_log_loss"]
            ),
            "log_loss_difference": float(
                recalibration["model_log_loss"] - full["model_log_loss"]
            ),
        }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "best_variant_by_year": best_by_year,
        "full_model_v2_vs_market_recalibration": full_vs_recalibration,
        "variants": {
            name: {"kind": spec["kind"], "features": spec["features"]}
            for name, spec in ABLATION_VARIANTS.items()
        },
    }


def build_correction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build correction distribution diagnostics by evaluation year plus combined."""
    frame = _with_correction(predictions)
    rows = [
        _correction_summary(str(year), group)
        for year, group in frame.groupby("evaluation_year", sort=True)
    ]
    rows.append(_correction_summary("combined_2023_2025", frame))
    return pd.DataFrame(rows)


def build_correction_buckets(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether larger correction magnitudes are useful or harmful."""
    frame = _with_correction(predictions)
    frame["correction_bucket"] = frame["probability_correction"].abs().map(correction_bucket)
    rows: list[dict[str, Any]] = []
    for (year, bucket), group in frame.groupby(["evaluation_year", "correction_bucket"], sort=True):
        rows.append(_bucket_metrics(str(year), str(bucket), group))
    for bucket, group in frame.groupby("correction_bucket", sort=True):
        rows.append(_bucket_metrics("combined_2023_2025", str(bucket), group))
    return pd.DataFrame(rows)


def build_odds_quality_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report metric robustness under progressively stricter odds-quality filters."""
    frame = predictions.copy(deep=True)
    overround = _overround(frame)
    frame["overround"] = overround
    filters = {
        "standard_valid_odds": _valid_odds_mask(frame),
        "normal_overround": _valid_odds_mask(frame) & overround.between(1.00, 1.10),
        "strict_overround": _valid_odds_mask(frame) & overround.between(1.02, 1.08),
    }
    rows: list[dict[str, Any]] = []
    for filter_name, mask in filters.items():
        included = frame.loc[mask].copy()
        excluded_count = len(frame) - len(included)
        for label, group in _year_and_combined_groups(included):
            if group.empty:
                continue
            model_metrics = _simple_metrics(group[TARGET_COLUMN], group[MODEL_COLUMN])
            market_metrics = _simple_metrics(group[TARGET_COLUMN], group[MARKET_COLUMN])
            rows.append(
                {
                    "filter": filter_name,
                    "segment": label,
                    "included_rows": len(group),
                    "excluded_rows_total": excluded_count,
                    "model_log_loss": model_metrics["log_loss"],
                    "market_log_loss": market_metrics["log_loss"],
                    "log_loss_improvement": market_metrics["log_loss"] - model_metrics["log_loss"],
                    "model_brier": model_metrics["brier_score"],
                    "market_brier": market_metrics["brier_score"],
                    "brier_improvement": market_metrics["brier_score"]
                    - model_metrics["brier_score"],
                    "model_accuracy": model_metrics["accuracy"],
                    "market_accuracy": market_metrics["accuracy"],
                }
            )
    return pd.DataFrame(rows)


def run_diagnostics(
    *,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> DiagnosticResult:
    """Run all Model v2 diagnostic analyses."""
    prediction_frame = predictions.copy(deep=True)
    feature_frame = features.copy(deep=True)
    bootstrap, distribution = build_bootstrap_significance(
        prediction_frame,
        samples=bootstrap_samples,
    )
    ablation_metrics, coefficients = run_ablation_study(feature_frame)
    ablation_summary = build_ablation_summary(ablation_metrics)
    correction_diagnostics = build_correction_diagnostics(prediction_frame)
    correction_buckets = build_correction_buckets(prediction_frame)
    odds_quality = build_odds_quality_metrics(prediction_frame)
    diagnostic_summary = _diagnostic_summary(
        bootstrap=bootstrap,
        ablation_summary=ablation_summary,
        correction_buckets=correction_buckets,
        odds_quality=odds_quality,
    )
    return DiagnosticResult(
        bootstrap_significance=bootstrap,
        ablation_metrics=ablation_metrics,
        ablation_summary=ablation_summary,
        coefficients=coefficients,
        correction_diagnostics=correction_diagnostics,
        correction_buckets=correction_buckets,
        odds_quality_metrics=odds_quality,
        diagnostic_summary=diagnostic_summary,
        bootstrap_distribution=distribution,
    )


def write_diagnostic_artifacts(result: DiagnosticResult, paths: DiagnosticOutputPaths) -> None:
    """Write Model v2 diagnostic JSON, Parquet, and PNG artifacts."""
    for path in paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.bootstrap_significance.write_text(
        json.dumps(_jsonable(result.bootstrap_significance), indent=2),
        encoding="utf-8",
    )
    result.ablation_metrics.to_parquet(paths.ablation_metrics, index=False)
    paths.ablation_summary.write_text(
        json.dumps(_jsonable(result.ablation_summary), indent=2),
        encoding="utf-8",
    )
    result.coefficients.to_parquet(paths.coefficients, index=False)
    result.correction_diagnostics.to_parquet(paths.correction_diagnostics, index=False)
    result.correction_buckets.to_parquet(paths.correction_buckets, index=False)
    result.odds_quality_metrics.to_parquet(paths.odds_quality_metrics, index=False)
    paths.diagnostic_summary.write_text(
        json.dumps(_jsonable(result.diagnostic_summary), indent=2),
        encoding="utf-8",
    )
    _write_distribution_plot(
        result.bootstrap_distribution,
        paths.bootstrap_distribution_plot,
        title="Model v2 Bootstrap Log-Loss Improvements",
        value_column="improvement",
    )
    _write_ablation_plot(result.ablation_metrics, paths.ablation_log_loss_plot)
    _write_correction_performance_plot(
        result.correction_buckets,
        paths.correction_performance_plot,
    )


def correction_bucket(abs_correction: float) -> str:
    """Return the configured correction magnitude bucket."""
    value = abs(float(abs_correction))
    for lower, upper, label in CORRECTION_BUCKETS:
        if lower <= value < upper:
            return label
    return "0.10+"


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = [
        column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing:
        msg = f"missing required Model v2 diagnostic columns: {missing}"
        raise ValueError(msg)
    _target(predictions[TARGET_COLUMN])
    _probability(predictions[MODEL_COLUMN], MODEL_COLUMN)
    _probability(predictions[MARKET_COLUMN], MARKET_COLUMN)


def _target(values: pd.Series) -> pd.Series:
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


def _probability(values: pd.Series, name: str) -> pd.Series:
    probabilities = pd.to_numeric(values, errors="coerce").astype("float64")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        msg = f"{name} contains non-finite probabilities"
        raise ValueError(msg)
    if not probabilities.between(0, 1).all():
        msg = f"{name} must be in [0, 1]"
        raise ValueError(msg)
    return probabilities


def _simple_metrics(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    metrics = calculate_probability_metrics(y_true, probabilities)
    return {
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "accuracy": metrics.accuracy,
    }


def _variant_metrics(
    *,
    variant_name: str,
    evaluation_year: int,
    features: list[str],
    y_true: pd.Series,
    probabilities: pd.Series,
    market_probabilities: pd.Series,
) -> dict[str, Any]:
    model_metrics = calculate_probability_metrics(y_true, probabilities)
    market_metrics = calculate_probability_metrics(y_true, market_probabilities)
    return {
        "variant": variant_name,
        "evaluation_year": evaluation_year,
        "features": "|".join(features) if features else "market_probability_player_1",
        "sample_count": model_metrics.sample_count,
        "model_log_loss": model_metrics.log_loss,
        "market_log_loss": market_metrics.log_loss,
        "log_loss_improvement_vs_market": market_metrics.log_loss - model_metrics.log_loss,
        "model_brier": model_metrics.brier_score,
        "market_brier": market_metrics.brier_score,
        "brier_improvement_vs_market": market_metrics.brier_score - model_metrics.brier_score,
        "model_accuracy": model_metrics.accuracy,
        "market_accuracy": market_metrics.accuracy,
        "roc_auc": model_metrics.roc_auc,
    }


def _validate_ablation_features(feature_names: list[str]) -> None:
    forbidden = set(feature_names) & FORBIDDEN_FEATURES
    if forbidden:
        msg = f"ablation features contain forbidden leakage columns: {sorted(forbidden)}"
        raise ValueError(msg)
    if feature_names and "market_logit_player_1" not in feature_names:
        msg = "trained ablation variants must include market_logit_player_1"
        raise ValueError(msg)
    allowed = set(MARKET_FEATURES)
    unknown = sorted(set(feature_names) - allowed)
    if unknown:
        msg = f"unknown ablation features: {unknown}"
        raise ValueError(msg)


def _build_variant_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_features = [feature for feature in feature_names if feature not in BINARY_FEATURES]
    binary_features = [feature for feature in feature_names if feature in BINARY_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "binary",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _coefficient_rows_from_model(
    variant_name: str,
    evaluation_year: int,
    model: Pipeline,
) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())
    rows = [
        _coefficient_row(
            variant_name,
            evaluation_year,
            "intercept",
            float(classifier.intercept_[0]),
            "intercept",
        )
    ]
    for feature, coefficient in zip(feature_names, classifier.coef_[0], strict=True):
        rows.append(
            _coefficient_row(
                variant_name,
                evaluation_year,
                str(feature),
                float(coefficient),
                _market_coefficient_category(str(feature), float(coefficient)),
            )
        )
    return rows


def _coefficient_row(
    variant_name: str,
    evaluation_year: int,
    feature_name: str,
    coefficient: float,
    category: str,
) -> dict[str, Any]:
    return {
        "variant": variant_name,
        "evaluation_year": evaluation_year,
        "feature_name": feature_name,
        "coefficient": coefficient,
        "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
        "absolute_coefficient": abs(coefficient),
        "market_logit_coefficient_category": category,
        "note": "continuous features are standardized before coefficient estimation",
    }


def _market_coefficient_category(feature_name: str, coefficient: float) -> str:
    if feature_name != "market_logit_player_1":
        return "not_market_logit"
    if coefficient < 0.9:
        return "below_0.9"
    if coefficient <= 1.1:
        return "between_0.9_and_1.1"
    return "above_1.1"


def _with_correction(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy(deep=True)
    frame["probability_correction"] = frame[MODEL_COLUMN] - frame[MARKET_COLUMN]
    return frame


def _correction_summary(segment: str, group: pd.DataFrame) -> dict[str, Any]:
    corrections = group["probability_correction"].astype(float)
    absolute = corrections.abs()
    quantiles = corrections.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "segment": segment,
        "rows": len(group),
        "mean_correction": float(corrections.mean()),
        "mean_absolute_correction": float(absolute.mean()),
        "median_absolute_correction": float(absolute.median()),
        "standard_deviation": float(corrections.std(ddof=0)),
        "minimum": float(corrections.min()),
        "maximum": float(corrections.max()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "pct_abs_below_0.01": float((absolute < 0.01).mean()),
        "pct_abs_below_0.02": float((absolute < 0.02).mean()),
        "pct_abs_below_0.05": float((absolute < 0.05).mean()),
        "pct_abs_above_0.10": float((absolute > 0.10).mean()),
    }


def _bucket_metrics(segment: str, bucket: str, group: pd.DataFrame) -> dict[str, Any]:
    model_metrics = _simple_metrics(group[TARGET_COLUMN], group[MODEL_COLUMN])
    market_metrics = _simple_metrics(group[TARGET_COLUMN], group[MARKET_COLUMN])
    return {
        "segment": segment,
        "correction_bucket": bucket,
        "rows": len(group),
        "average_absolute_correction": float(group["probability_correction"].abs().mean()),
        "model_log_loss": model_metrics["log_loss"],
        "market_log_loss": market_metrics["log_loss"],
        "log_loss_improvement": market_metrics["log_loss"] - model_metrics["log_loss"],
        "model_brier": model_metrics["brier_score"],
        "market_brier": market_metrics["brier_score"],
        "brier_improvement": market_metrics["brier_score"] - model_metrics["brier_score"],
        "model_accuracy": model_metrics["accuracy"],
        "market_accuracy": market_metrics["accuracy"],
    }


def _overround(frame: pd.DataFrame) -> pd.Series:
    odds_1 = pd.to_numeric(frame["player_1_odds"], errors="coerce")
    odds_2 = pd.to_numeric(frame["player_2_odds"], errors="coerce")
    return 1.0 / odds_1 + 1.0 / odds_2


def _valid_odds_mask(frame: pd.DataFrame) -> pd.Series:
    odds_1 = pd.to_numeric(frame["player_1_odds"], errors="coerce")
    odds_2 = pd.to_numeric(frame["player_2_odds"], errors="coerce")
    overround = _overround(frame)
    return pd.Series(
        odds_1.notna()
        & odds_2.notna()
        & np.isfinite(odds_1)
        & np.isfinite(odds_2)
        & (odds_1 > 1.0)
        & (odds_2 > 1.0)
        & np.isfinite(overround),
        index=frame.index,
        dtype="bool",
    )


def _year_and_combined_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(str(year), group) for year, group in frame.groupby("evaluation_year", sort=True)]
    groups.append(("combined_2023_2025", frame))
    return groups


def _diagnostic_summary(
    *,
    bootstrap: dict[str, Any],
    ablation_summary: dict[str, Any],
    correction_buckets: pd.DataFrame,
    odds_quality: pd.DataFrame,
) -> dict[str, Any]:
    combined = bootstrap["segments"]["combined_2023_2025"]["log_loss"]
    bucket_frame = correction_buckets[
        correction_buckets["segment"].astype(str) == "combined_2023_2025"
    ]
    best_bucket = bucket_frame.sort_values("log_loss_improvement", ascending=False).iloc[0]
    worst_bucket = bucket_frame.sort_values("log_loss_improvement", ascending=True).iloc[0]
    strict = odds_quality[
        (odds_quality["filter"] == "strict_overround")
        & (odds_quality["segment"] == "combined_2023_2025")
    ]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "combined_log_loss_ci": [combined["ci_lower"], combined["ci_upper"]],
        "combined_ci_excludes_zero": combined["interval_excludes_zero"],
        "combined_probability_model_beats_market": combined[
            "probability_model_beats_market"
        ],
        "best_ablation_variant_by_year": ablation_summary["best_variant_by_year"],
        "full_model_v2_vs_market_recalibration": ablation_summary[
            "full_model_v2_vs_market_recalibration"
        ],
        "best_correction_bucket": best_bucket.to_dict(),
        "worst_correction_bucket": worst_bucket.to_dict(),
        "strict_overround_combined": strict.iloc[0].to_dict() if not strict.empty else None,
        "interpretation_warning": (
            "Do not declare Model v2 successful unless confidence intervals exclude zero "
            "and tennis-feature variants improve beyond market-only recalibration."
        ),
    }


def _write_distribution_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
    value_column: str,
) -> None:
    values = frame[
        (frame["segment"] == "combined_2023_2025") & (frame["metric"] == "log_loss")
    ][value_column].astype(float)
    _write_histogram(values.to_numpy(), path, title=title, x_label="Improvement")


def _write_ablation_plot(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame["evaluation_year"] == frame["evaluation_year"].max()].copy()
    _write_bar_plot(
        subset["variant"].astype(str).tolist(),
        subset["model_log_loss"].astype(float).tolist(),
        path,
        title="Model v2 Ablation Log Loss",
        y_label="Log loss",
    )


def _write_correction_performance_plot(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[frame["segment"] == "combined_2023_2025"].copy()
    _write_bar_plot(
        subset["correction_bucket"].astype(str).tolist(),
        subset["log_loss_improvement"].astype(float).tolist(),
        path,
        title="Correction Bucket Log-Loss Improvement",
        y_label="Improvement",
    )


def _write_histogram(values: np.ndarray[Any, Any], path: Path, *, title: str, x_label: str) -> None:
    image = Image.new("RGB", (720, 440), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 660, 340), outline="black")
    draw.text((60, 14), title, fill="black")
    draw.text((300, 380), x_label, fill="black")
    if len(values) == 0:
        values = np.array([0.0])
    counts, _ = np.histogram(values, bins=30)
    max_count = max(int(counts.max()), 1)
    bar_width = 600 / len(counts)
    for index, count in enumerate(counts):
        x0 = 60 + index * bar_width
        x1 = x0 + bar_width - 2
        y0 = 340 - (int(count) / max_count) * 280
        draw.rectangle((x0, y0, x1, 340), fill="steelblue")
    image.save(path)


def _write_bar_plot(
    labels: list[str],
    values: list[float],
    path: Path,
    *,
    title: str,
    y_label: str,
) -> None:
    image = Image.new("RGB", (900, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 50, 850, 350), outline="black")
    draw.text((70, 18), title, fill="black")
    draw.text((10, 180), y_label, fill="black")
    if not values:
        values = [0.0]
        labels = ["empty"]
    minimum = min(values + [0.0])
    maximum = max(values + [0.0])
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    zero_y = 50 + (maximum - 0.0) / (maximum - minimum) * 300
    draw.line((70, zero_y, 850, zero_y), fill="gray")
    width = 780 / len(values)
    for index, value in enumerate(values):
        x0 = 70 + index * width + 5
        x1 = x0 + width - 10
        y = 50 + (maximum - value) / (maximum - minimum) * 300
        draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill="teal")
        draw.text((x0, 360), labels[index][:12], fill="black")
    image.save(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "ABLATION_VARIANTS",
    "BOOTSTRAP_SEED",
    "CORRECTION_BUCKETS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DiagnosticOutputPaths",
    "DiagnosticResult",
    "build_ablation_summary",
    "build_bootstrap_significance",
    "build_correction_buckets",
    "build_correction_diagnostics",
    "build_odds_quality_metrics",
    "calculate_row_differences",
    "correction_bucket",
    "paired_bootstrap",
    "run_ablation_study",
    "run_diagnostics",
    "write_diagnostic_artifacts",
]
