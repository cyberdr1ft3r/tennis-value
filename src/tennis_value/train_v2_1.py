"""Focused market correction experiment using form and workload features."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_value.evaluate import PROBABILITY_EPSILON, calculate_probability_metrics
from tennis_value.train_v2 import (
    FORBIDDEN_FEATURES,
    WALK_FORWARD_FOLDS,
    WalkForwardSplit,
    build_walk_forward_folds,
    market_logit,
    prepare_model_v2_dataset,
)

MODEL_VERSION = "model_v2_1_form_workload"
BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
CORRECTION_CAP = 0.15
MARKET_LOGIT = "market_logit_player_1"
FORM_WORKLOAD_FEATURES = [
    "recent_10_win_rate_diff",
    "surface_recent_10_win_rate_diff",
    "days_since_last_match_diff",
    "matches_last_14d_diff",
]
MODEL_V2_1_FEATURES = [MARKET_LOGIT, *FORM_WORKLOAD_FEATURES]
ARCHITECTURES = (
    "market_recalibration",
    "free_form_workload",
    "fixed_offset_form_workload",
    "fixed_offset_form_workload_capped",
)
COMPARATORS = ("raw_market", "market_recalibration")
SIGNED_CORRECTION_BUCKETS = (
    (-math.inf, -0.05, "below_-0.05"),
    (-0.05, -0.02, "-0.05_to_-0.02"),
    (-0.02, -0.01, "-0.02_to_-0.01"),
    (-0.01, 0.01, "-0.01_to_0.01"),
    (0.01, 0.02, "0.01_to_0.02"),
    (0.02, 0.05, "0.02_to_0.05"),
    (0.05, math.inf, "above_0.05"),
)

ArchitectureName = Literal[
    "market_recalibration",
    "free_form_workload",
    "fixed_offset_form_workload",
    "fixed_offset_form_workload_capped",
]


class FixedOffsetLogisticCorrection(BaseModel):
    """Small deterministic logistic correction model with a fixed market-logit offset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_names: list[str]
    c_value: float = 1.0
    learning_rate: float = 0.05
    max_iter: int = 600
    tolerance: float = 1e-8
    intercept_: float = 0.0
    coefficients_: np.ndarray[Any, Any] | None = None
    imputer_: SimpleImputer | None = None
    scaler_: StandardScaler | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series, offset_logit: pd.Series) -> None:
        """Fit correction coefficients while keeping the market-logit coefficient fixed to one."""
        x = self._fit_transform_features(features)
        y = pd.to_numeric(target, errors="raise").astype("float64").to_numpy()
        offset = pd.to_numeric(offset_logit, errors="raise").astype("float64").to_numpy()
        if len(y) != len(offset) or len(y) != x.shape[0]:
            msg = "features, target, and offset must have matching row counts"
            raise ValueError(msg)
        weights = np.zeros(x.shape[1], dtype="float64")
        intercept = 0.0
        alpha = 1.0 / self.c_value if self.c_value > 0 else 1.0
        previous_loss = math.inf
        for _ in range(self.max_iter):
            logits = offset + intercept + x @ weights
            probabilities = _sigmoid(logits)
            error = probabilities - y
            intercept -= self.learning_rate * float(error.mean())
            weights -= self.learning_rate * (
                (x.T @ error) / len(y) + alpha * weights / len(y)
            )
            loss = _offset_log_loss(y, offset + intercept + x @ weights)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss
        self.intercept_ = float(intercept)
        self.coefficients_ = weights

    def predict_correction_logit(self, features: pd.DataFrame) -> pd.Series:
        """Predict the learned additive correction logit."""
        if self.coefficients_ is None:
            msg = "fixed-offset model must be fitted before prediction"
            raise ValueError(msg)
        x = self._transform_features(features)
        values = self.intercept_ + x @ self.coefficients_
        return pd.Series(values, index=features.index, dtype="float64")

    def predict_proba(
        self,
        features: pd.DataFrame,
        offset_logit: pd.Series,
        *,
        correction_cap: float | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return probability, raw correction logit, and capped correction mask."""
        correction = self.predict_correction_logit(features)
        capped_mask = pd.Series(False, index=features.index, dtype="bool")
        if correction_cap is not None:
            capped_mask = correction.abs() > correction_cap
            correction = correction.clip(-correction_cap, correction_cap)
        offset = pd.to_numeric(offset_logit, errors="raise").astype("float64")
        probabilities = pd.Series(_sigmoid((offset + correction).to_numpy()), index=features.index)
        return probabilities.astype("float64"), correction.astype("float64"), capped_mask

    def coefficient_by_feature(self) -> dict[str, float]:
        """Return fitted coefficients, including the fixed market coefficient."""
        if self.coefficients_ is None:
            return {MARKET_LOGIT: 1.0}
        feature_names = list(self.feature_names)
        if len(feature_names) < len(self.coefficients_):
            feature_names.extend(
                f"generated_feature_{index}"
                for index in range(len(feature_names), len(self.coefficients_))
            )
        return {
            MARKET_LOGIT: 1.0,
            "intercept": self.intercept_,
            **{
                feature: float(value)
                for feature, value in zip(feature_names, self.coefficients_, strict=False)
            },
        }

    def _fit_transform_features(self, features: pd.DataFrame) -> np.ndarray[Any, Any]:
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler_ = StandardScaler()
        imputed = self.imputer_.fit_transform(features[self.feature_names])
        return cast(np.ndarray[Any, Any], self.scaler_.fit_transform(imputed))

    def _transform_features(self, features: pd.DataFrame) -> np.ndarray[Any, Any]:
        if self.imputer_ is None or self.scaler_ is None:
            msg = "fixed-offset model must be fitted before transforming features"
            raise ValueError(msg)
        imputed = self.imputer_.transform(features[self.feature_names])
        return cast(np.ndarray[Any, Any], self.scaler_.transform(imputed))


class ModelV21Result(BaseModel):
    """Model v2.1 experiment artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: Any
    predictions: pd.DataFrame
    architecture_metrics: pd.DataFrame
    block_bootstrap: dict[str, Any]
    correction_direction: pd.DataFrame
    odds_sensitivity: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelV21OutputPaths:
    """Output paths for Model v2.1 artifacts."""

    model_output: Path
    metadata_output: Path
    predictions_output: Path
    architecture_metrics: Path
    block_bootstrap: Path
    correction_direction: Path
    odds_sensitivity: Path
    summary: Path
    architecture_comparison_plot: Path
    correction_calibration_plot: Path


def train_model_v2_1(
    features: pd.DataFrame,
    *,
    input_dataset_path: Path | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_state: int = 42,
) -> ModelV21Result:
    """Train and evaluate the focused Model v2.1 market-correction experiment."""
    _validate_feature_allowlist()
    prepared = prepare_model_v2_dataset(features)
    folds = build_walk_forward_folds(prepared)
    prediction_frames: list[pd.DataFrame] = []
    final_models: dict[str, Any] = {}
    coefficient_rows: list[dict[str, Any]] = []

    for fold in folds:
        _validate_fold(fold)
        models = _fit_fold_models(fold, random_state=random_state)
        final_models = models
        fold_predictions, fold_coefficients = _predict_fold(fold, models)
        prediction_frames.append(fold_predictions)
        coefficient_rows.extend(fold_coefficients)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    architecture_metrics = build_architecture_metrics(predictions)
    bootstrap = build_block_bootstrap(predictions, samples=bootstrap_samples)
    correction_direction = build_correction_direction(predictions)
    odds_sensitivity = build_odds_sensitivity(predictions)
    summary = build_v2_1_summary(
        architecture_metrics=architecture_metrics,
        block_bootstrap=bootstrap,
        correction_direction=correction_direction,
        predictions=predictions,
    )
    metadata = _metadata(
        prepared=prepared,
        input_dataset_path=input_dataset_path,
        coefficient_rows=coefficient_rows,
        bootstrap_samples=bootstrap_samples,
    )
    return ModelV21Result(
        model=final_models,
        predictions=predictions,
        architecture_metrics=architecture_metrics,
        block_bootstrap=bootstrap,
        correction_direction=correction_direction,
        odds_sensitivity=odds_sensitivity,
        summary=summary,
        metadata=metadata,
    )


def build_architecture_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for each architecture and evaluation year."""
    rows: list[dict[str, Any]] = []
    for (architecture, year), group in predictions.groupby(
        ["architecture", "evaluation_year"],
        sort=True,
    ):
        rows.append(_metrics_row(str(architecture), str(year), group))
    for architecture, group in predictions.groupby("architecture", sort=True):
        rows.append(_metrics_row(str(architecture), "combined_2023_2025", group))
    return pd.DataFrame(rows)


def build_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build paired block-bootstrap comparisons by ISO-week blocks."""
    if samples <= 0:
        msg = "bootstrap samples must be greater than zero"
        raise ValueError(msg)
    reports: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        candidate = predictions[predictions["architecture"] == architecture]
        if candidate.empty:
            continue
        reports[architecture] = {}
        for comparator in COMPARATORS:
            reports[architecture][comparator] = {}
            for label, group in _year_and_combined(candidate):
                reports[architecture][comparator][label] = block_bootstrap_comparison(
                    group,
                    candidate_probability_column="model_probability",
                    comparator_probability_column=_comparator_column(comparator),
                    samples=samples,
                    seed=seed,
                )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "bootstrap_samples": samples,
        "block_definition": "evaluation_year + ISO week",
        "comparisons": reports,
    }


def block_bootstrap_comparison(
    rows: pd.DataFrame,
    *,
    candidate_probability_column: str,
    comparator_probability_column: str,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare candidate against comparator using paired block resampling."""
    frame = rows.copy(deep=True)
    if frame.empty:
        msg = "at least one row is required for block bootstrap"
        raise ValueError(msg)
    frame["block_id"] = _block_ids(frame)
    frame["log_loss_difference"] = _row_log_loss(
        frame["actual_player_1_won"],
        frame[comparator_probability_column],
    ) - _row_log_loss(frame["actual_player_1_won"], frame[candidate_probability_column])
    frame["brier_difference"] = (
        (frame[comparator_probability_column].astype(float) - frame["actual_player_1_won"]) ** 2
        - (frame[candidate_probability_column].astype(float) - frame["actual_player_1_won"]) ** 2
    )
    block_sums = frame.groupby("block_id", sort=True)[
        ["log_loss_difference", "brier_difference"]
    ].sum()
    block_counts = frame.groupby("block_id", sort=True).size().astype("int64")
    rng = np.random.default_rng(seed)
    block_count = len(block_counts)
    sampled_indices = rng.integers(0, block_count, size=(samples, block_count))
    log_sums = block_sums["log_loss_difference"].to_numpy(dtype="float64")
    brier_sums = block_sums["brier_difference"].to_numpy(dtype="float64")
    counts = block_counts.to_numpy(dtype="float64")
    sampled_counts = counts[sampled_indices].sum(axis=1)
    log_array = log_sums[sampled_indices].sum(axis=1) / sampled_counts
    brier_array = brier_sums[sampled_indices].sum(axis=1) / sampled_counts
    log_lower, log_upper = np.percentile(log_array, [2.5, 97.5])
    brier_lower, brier_upper = np.percentile(brier_array, [2.5, 97.5])
    return {
        "sample_count": int(len(frame)),
        "block_count": int(block_count),
        "mean_log_loss_improvement": float(frame["log_loss_difference"].mean()),
        "median_bootstrap_log_loss_improvement": float(np.median(log_array)),
        "log_loss_ci_lower": float(log_lower),
        "log_loss_ci_upper": float(log_upper),
        "probability_model_beats_comparator": float((log_array > 0).mean()),
        "mean_brier_improvement": float(frame["brier_difference"].mean()),
        "brier_ci_lower": float(brier_lower),
        "brier_ci_upper": float(brier_upper),
        "log_loss_interval_excludes_zero": bool(log_lower > 0 or log_upper < 0),
        "brier_interval_excludes_zero": bool(brier_lower > 0 or brier_upper < 0),
    }


def build_correction_direction(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report signed correction buckets and correction-direction slope."""
    rows: list[dict[str, Any]] = []
    candidate_rows = predictions[predictions["architecture"].isin(ARCHITECTURES)].copy()
    for (architecture, bucket), group in candidate_rows.groupby(
        ["architecture", "signed_correction_bucket"],
        sort=True,
    ):
        rows.append(_correction_bucket_row(str(architecture), str(bucket), group))
    for architecture, group in candidate_rows.groupby("architecture", sort=True):
        slope = correction_direction_slope(group)
        rows.append(
            {
                **_correction_bucket_row(str(architecture), "all", group),
                "correction_direction_slope": slope,
            }
        )
    return pd.DataFrame(rows)


def correction_direction_slope(rows: pd.DataFrame) -> float:
    """Return linear slope of market residual on model correction."""
    correction = rows["correction"].astype(float).to_numpy()
    residual = rows["market_residual"].astype(float).to_numpy()
    if len(correction) < 2 or math.isclose(float(np.var(correction)), 0.0):
        return 0.0
    return float(np.cov(correction, residual, ddof=0)[0, 1] / np.var(correction))


def signed_correction_bucket(correction: float) -> str:
    """Return signed correction bucket label."""
    value = float(correction)
    for lower, upper, label in SIGNED_CORRECTION_BUCKETS:
        if lower <= value < upper:
            return label
    return "above_0.05"


def build_odds_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each architecture under odds-quality filters."""
    rows: list[dict[str, Any]] = []
    for architecture, group in predictions.groupby("architecture", sort=True):
        overround = group["overround"].astype(float)
        filters = {
            "all_valid_paired_odds": pd.Series(True, index=group.index),
            "normal_overround": overround.between(1.00, 1.10),
            "strict_overround": overround.between(1.02, 1.08),
        }
        if "odds_source" in group:
            filters["same_source_paired_odds"] = group["odds_source"].notna()
            filters["pinnacle_only"] = group["odds_source"].astype(str).isin({"PS", "Pinnacle"})
            filters["bet365_only"] = group["odds_source"].astype(str).eq("B365")
        for filter_name, mask in filters.items():
            included = group.loc[mask]
            if included.empty:
                rows.append(
                    {
                        "architecture": architecture,
                        "filter": filter_name,
                        "included_rows": 0,
                        "excluded_rows": len(group),
                        "model_log_loss": None,
                        "raw_market_log_loss": None,
                        "log_loss_improvement_vs_market": None,
                    }
                )
                continue
            model_metrics = calculate_probability_metrics(
                included["actual_player_1_won"],
                included["model_probability"],
            )
            market_metrics = calculate_probability_metrics(
                included["actual_player_1_won"],
                included["market_probability_player_1"],
            )
            rows.append(
                {
                    "architecture": architecture,
                    "filter": filter_name,
                    "included_rows": len(included),
                    "excluded_rows": len(group) - len(included),
                    "model_log_loss": model_metrics.log_loss,
                    "raw_market_log_loss": market_metrics.log_loss,
                    "log_loss_improvement_vs_market": market_metrics.log_loss
                    - model_metrics.log_loss,
                }
            )
    return pd.DataFrame(rows)


def build_v2_1_summary(
    *,
    architecture_metrics: pd.DataFrame,
    block_bootstrap: dict[str, Any],
    correction_direction: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Build concise JSON summary for the Model v2.1 experiment."""
    pooled = architecture_metrics[
        architecture_metrics["segment"].astype(str).eq("combined_2023_2025")
    ]
    best = pooled.sort_values("model_log_loss", kind="mergesort").iloc[0]
    capped = predictions[predictions["architecture"] == "fixed_offset_form_workload_capped"]
    capped_rows = int(capped["correction_capped"].sum()) if not capped.empty else 0
    slope_rows = correction_direction[
        correction_direction["signed_correction_bucket"].astype(str).eq("all")
    ]
    pooled_bootstrap = block_bootstrap["comparisons"].get(str(best["architecture"]), {}).get(
        "raw_market",
        {},
    ).get("combined_2023_2025")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": MODEL_VERSION,
        "feature_names": MODEL_V2_1_FEATURES,
        "selected_after_inspecting_2023_2025": True,
        "best_pooled_architecture": best.to_dict(),
        "pooled_bootstrap_vs_raw_market_for_best": pooled_bootstrap,
        "correction_direction_slope_by_architecture": {
            str(row["architecture"]): float(row["correction_direction_slope"])
            for _, row in slope_rows.iterrows()
        },
        "capped_rows": capped_rows,
        "capped_row_rate": capped_rows / len(capped) if len(capped) else 0.0,
        "interpretation_warning": (
            "Do not call Model v2.1 successful unless it beats raw market and market-only "
            "recalibration with supportive pooled block-bootstrap intervals."
        ),
    }


def write_model_v2_1_artifacts(result: ModelV21Result, paths: ModelV21OutputPaths) -> None:
    """Write Model v2.1 model, JSON, Parquet, and PNG artifacts."""
    for path in paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, paths.model_output)
    paths.metadata_output.write_text(json.dumps(_jsonable(result.metadata), indent=2), "utf-8")
    result.predictions.to_parquet(paths.predictions_output, index=False)
    result.architecture_metrics.to_parquet(paths.architecture_metrics, index=False)
    paths.block_bootstrap.write_text(
        json.dumps(_jsonable(result.block_bootstrap), indent=2),
        "utf-8",
    )
    result.correction_direction.to_parquet(paths.correction_direction, index=False)
    result.odds_sensitivity.to_parquet(paths.odds_sensitivity, index=False)
    paths.summary.write_text(json.dumps(_jsonable(result.summary), indent=2), "utf-8")
    _write_architecture_plot(result.architecture_metrics, paths.architecture_comparison_plot)
    _write_correction_plot(result.correction_direction, paths.correction_calibration_plot)


def _fit_fold_models(fold: WalkForwardSplit, *, random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "market_recalibration": _build_pipeline([MARKET_LOGIT], random_state=random_state),
        "free_form_workload": _build_pipeline(MODEL_V2_1_FEATURES, random_state=random_state),
        "fixed_offset_form_workload": FixedOffsetLogisticCorrection(
            feature_names=FORM_WORKLOAD_FEATURES
        ),
    }
    models["market_recalibration"].fit(fold.train[[MARKET_LOGIT]], fold.train["target"])
    models["free_form_workload"].fit(fold.train[MODEL_V2_1_FEATURES], fold.train["target"])
    fixed = cast(FixedOffsetLogisticCorrection, models["fixed_offset_form_workload"])
    fixed.fit(
        fold.train[FORM_WORKLOAD_FEATURES],
        fold.train["target"],
        fold.train[MARKET_LOGIT],
    )
    models["fixed_offset_form_workload_capped"] = fixed
    return models


def _predict_fold(
    fold: WalkForwardSplit,
    models: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        if architecture == "fixed_offset_form_workload":
            model = cast(FixedOffsetLogisticCorrection, models[architecture])
            probabilities, correction_logit, capped = model.predict_proba(
                fold.evaluation[FORM_WORKLOAD_FEATURES],
                fold.evaluation[MARKET_LOGIT],
            )
            coefficient_rows.extend(_fixed_coefficient_rows(architecture, fold, model))
        elif architecture == "fixed_offset_form_workload_capped":
            model = cast(FixedOffsetLogisticCorrection, models[architecture])
            probabilities, correction_logit, capped = model.predict_proba(
                fold.evaluation[FORM_WORKLOAD_FEATURES],
                fold.evaluation[MARKET_LOGIT],
                correction_cap=CORRECTION_CAP,
            )
            coefficient_rows.extend(_fixed_coefficient_rows(architecture, fold, model))
        else:
            feature_names = (
                [MARKET_LOGIT]
                if architecture == "market_recalibration"
                else MODEL_V2_1_FEATURES
            )
            pipeline = cast(Pipeline, models[architecture])
            probabilities = pd.Series(
                pipeline.predict_proba(fold.evaluation[feature_names])[:, 1],
                index=fold.evaluation.index,
                dtype="float64",
            )
            correction_logit = market_logit(probabilities) - fold.evaluation[MARKET_LOGIT]
            capped = pd.Series(False, index=fold.evaluation.index, dtype="bool")
            coefficient_rows.extend(_pipeline_coefficient_rows(architecture, fold, pipeline))
        frames.append(
            _prediction_frame(fold, architecture, probabilities, correction_logit, capped)
        )
    output = pd.concat(frames, ignore_index=True)
    recalibration = output[output["architecture"] == "market_recalibration"][
        ["match_id", "model_probability"]
    ].rename(columns={"model_probability": "market_recalibration_probability_value"})
    output = output.drop(columns=["market_recalibration_probability"]).merge(
        recalibration,
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    output = output.rename(
        columns={"market_recalibration_probability_value": "market_recalibration_probability"}
    )
    return output, coefficient_rows


def _prediction_frame(
    fold: WalkForwardSplit,
    architecture: str,
    probabilities: pd.Series,
    correction_logit: pd.Series,
    capped: pd.Series,
) -> pd.DataFrame:
    frame = fold.evaluation[
        [
            "match_id",
            "match_date",
            "surface",
            "player_1",
            "player_2",
            "target",
            "player_1_odds",
            "player_2_odds",
            "market_probability_player_1",
            "overround",
        ]
    ].copy()
    if "odds_source" in fold.evaluation:
        frame["odds_source"] = fold.evaluation["odds_source"].astype(str)
    else:
        frame["odds_source"] = pd.NA
    frame["evaluation_year"] = fold.evaluation_year
    frame["iso_week"] = pd.to_datetime(frame["match_date"]).dt.isocalendar().week.astype("int64")
    frame["architecture"] = architecture
    frame["actual_player_1_won"] = frame["target"].astype("int64")
    frame["model_probability"] = probabilities.astype("float64")
    frame["raw_market_probability"] = frame["market_probability_player_1"].astype("float64")
    frame["market_recalibration_probability"] = np.nan
    frame["correction"] = frame["model_probability"] - frame["raw_market_probability"]
    frame["market_residual"] = frame["actual_player_1_won"] - frame["raw_market_probability"]
    frame["correction_logit"] = correction_logit.astype("float64")
    frame["correction_capped"] = capped.astype("bool")
    frame["signed_correction_bucket"] = frame["correction"].map(signed_correction_bucket)
    frame["model_version"] = MODEL_VERSION
    frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.strftime("%Y-%m-%d")
    return frame.drop(columns=["target"]).reset_index(drop=True)


def _metrics_row(architecture: str, segment: str, group: pd.DataFrame) -> dict[str, Any]:
    model = calculate_probability_metrics(group["actual_player_1_won"], group["model_probability"])
    market = calculate_probability_metrics(
        group["actual_player_1_won"],
        group["market_probability_player_1"],
    )
    return {
        "architecture": architecture,
        "segment": segment,
        "sample_count": model.sample_count,
        "model_log_loss": model.log_loss,
        "raw_market_log_loss": market.log_loss,
        "log_loss_improvement_vs_market": market.log_loss - model.log_loss,
        "model_brier": model.brier_score,
        "raw_market_brier": market.brier_score,
        "brier_improvement_vs_market": market.brier_score - model.brier_score,
        "model_accuracy": model.accuracy,
        "raw_market_accuracy": market.accuracy,
    }


def _build_pipeline(feature_names: list[str], *, random_state: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _year_and_combined(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(str(year), group) for year, group in frame.groupby("evaluation_year", sort=True)]
    groups.append(("combined_2023_2025", frame))
    return groups


def _comparator_column(comparator: str) -> str:
    if comparator == "raw_market":
        return "market_probability_player_1"
    if comparator == "market_recalibration":
        return "market_recalibration_probability"
    msg = f"unknown comparator: {comparator}"
    raise ValueError(msg)


def _block_ids(frame: pd.DataFrame) -> pd.Series:
    if "iso_week" in frame:
        week = frame["iso_week"].astype("int64")
    else:
        week = pd.to_datetime(frame["match_date"]).dt.isocalendar().week.astype("int64")
    return frame["evaluation_year"].astype(str) + "-W" + week.astype(str)


def _row_log_loss(target: pd.Series, probabilities: pd.Series) -> pd.Series:
    y = pd.to_numeric(target, errors="raise").astype("float64")
    p = pd.to_numeric(probabilities, errors="raise").astype("float64").clip(
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _correction_bucket_row(architecture: str, bucket: str, group: pd.DataFrame) -> dict[str, Any]:
    market = calculate_probability_metrics(
        group["actual_player_1_won"],
        group["market_probability_player_1"],
    )
    model = calculate_probability_metrics(group["actual_player_1_won"], group["model_probability"])
    return {
        "architecture": architecture,
        "signed_correction_bucket": bucket,
        "rows": len(group),
        "mean_market_probability": float(group["market_probability_player_1"].mean()),
        "mean_model_probability": float(group["model_probability"].mean()),
        "observed_win_rate": float(group["actual_player_1_won"].mean()),
        "mean_correction": float(group["correction"].mean()),
        "mean_market_residual": float(group["market_residual"].mean()),
        "market_log_loss": market.log_loss,
        "model_log_loss": model.log_loss,
        "log_loss_improvement": market.log_loss - model.log_loss,
        "market_brier": market.brier_score,
        "model_brier": model.brier_score,
        "brier_improvement": market.brier_score - model.brier_score,
        "correction_direction_slope": correction_direction_slope(group),
    }


def _pipeline_coefficient_rows(
    architecture: str,
    fold: WalkForwardSplit,
    model: Pipeline,
) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    rows = [_coefficient_row(architecture, fold, "intercept", float(classifier.intercept_[0]))]
    feature_names = [str(feature) for feature in preprocessor.get_feature_names_out()]
    coefficients = classifier.coef_[0]
    if len(feature_names) < len(coefficients):
        feature_names.extend(
            f"generated_feature_{index}"
            for index in range(len(feature_names), len(coefficients))
        )
    for feature, coefficient in zip(feature_names, coefficients, strict=False):
        rows.append(_coefficient_row(architecture, fold, str(feature), float(coefficient)))
    return rows


def _fixed_coefficient_rows(
    architecture: str,
    fold: WalkForwardSplit,
    model: FixedOffsetLogisticCorrection,
) -> list[dict[str, Any]]:
    return [
        _coefficient_row(architecture, fold, feature, coefficient)
        for feature, coefficient in model.coefficient_by_feature().items()
    ]


def _coefficient_row(
    architecture: str,
    fold: WalkForwardSplit,
    feature_name: str,
    coefficient: float,
) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "evaluation_year": fold.evaluation_year,
        "feature_name": feature_name,
        "coefficient": coefficient,
        "market_logit_fixed_to_one": architecture.startswith("fixed_offset")
        and feature_name == MARKET_LOGIT,
    }


def _validate_feature_allowlist() -> None:
    forbidden = set(MODEL_V2_1_FEATURES) & FORBIDDEN_FEATURES
    if forbidden:
        msg = f"Model v2.1 features contain leakage columns: {sorted(forbidden)}"
        raise ValueError(msg)
    banned_tokens = {
        "outcome",
        "edge",
        "expected_value",
        "result",
        "settlement",
        "stake",
        "bankroll",
        "roi",
    }
    bad = [
        feature
        for feature in MODEL_V2_1_FEATURES
        if any(token in feature for token in banned_tokens)
    ]
    if bad:
        msg = f"Model v2.1 features contain forbidden business-result columns: {bad}"
        raise ValueError(msg)


def _validate_fold(fold: WalkForwardSplit) -> None:
    if fold.train.empty or fold.evaluation.empty:
        msg = f"fold {fold.fold} has empty train or evaluation rows"
        raise ValueError(msg)
    if fold.train["target"].nunique() < 2:
        msg = f"fold {fold.fold} training rows must contain both target classes"
        raise ValueError(msg)


def _sigmoid(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    clipped = np.clip(values, -40.0, 40.0)
    return cast(np.ndarray[Any, Any], 1.0 / (1.0 + np.exp(-clipped)))


def _offset_log_loss(target: np.ndarray[Any, Any], logits: np.ndarray[Any, Any]) -> float:
    probabilities = np.clip(_sigmoid(logits), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    losses = target * np.log(probabilities) + (1.0 - target) * np.log(1.0 - probabilities)
    return float(-losses.mean())


def _metadata(
    *,
    prepared: pd.DataFrame,
    input_dataset_path: Path | None,
    coefficient_rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    git_commit, warnings = _git_commit()
    return {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_names": MODEL_V2_1_FEATURES,
        "form_workload_features": FORM_WORKLOAD_FEATURES,
        "architectures": list(ARCHITECTURES),
        "walk_forward_folds": list(WALK_FORWARD_FOLDS),
        "eligible_rows": len(prepared),
        "bootstrap_samples": bootstrap_samples,
        "correction_cap_logit": CORRECTION_CAP,
        "input_dataset_path": str(input_dataset_path) if input_dataset_path else None,
        "input_dataset_sha256": (
            _dataset_sha256(input_dataset_path)
            if input_dataset_path is not None and input_dataset_path.exists()
            else None
        ),
        "coefficient_rows": coefficient_rows,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "git_commit_when_available": git_commit,
        "warnings": warnings,
    }


def _write_architecture_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    subset = metrics[metrics["segment"].astype(str).eq("combined_2023_2025")]
    _write_bar_plot(
        subset["architecture"].astype(str).tolist(),
        subset["model_log_loss"].astype(float).tolist(),
        output_path,
        title="Model v2.1 Architecture Log Loss",
        y_label="Log loss",
    )


def _write_correction_plot(correction_direction: pd.DataFrame, output_path: Path) -> None:
    subset = correction_direction[
        correction_direction["signed_correction_bucket"].astype(str).ne("all")
    ]
    grouped = subset.groupby("signed_correction_bucket", sort=True)["log_loss_improvement"].mean()
    _write_bar_plot(
        grouped.index.astype(str).tolist(),
        grouped.astype(float).tolist(),
        output_path,
        title="Model v2.1 Correction Direction",
        y_label="Improvement",
    )


def _write_bar_plot(
    labels: list[str],
    values: list[float],
    path: Path,
    *,
    title: str,
    y_label: str,
) -> None:
    image = Image.new("RGB", (920, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 50, 860, 350), outline="black")
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
    draw.line((70, zero_y, 860, zero_y), fill="gray")
    width = 790 / len(values)
    for index, value in enumerate(values):
        x0 = 70 + index * width + 5
        x1 = x0 + width - 10
        y = 50 + (maximum - value) / (maximum - minimum) * 300
        draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill="teal")
        draw.text((x0, 360), labels[index][:14], fill="black")
    image.save(path)


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> tuple[str | None, list[str]]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, [f"git commit unavailable: {exc}"]
    return result.stdout.strip(), []


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
    "ARCHITECTURES",
    "BOOTSTRAP_SEED",
    "CORRECTION_CAP",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "FORM_WORKLOAD_FEATURES",
    "MODEL_V2_1_FEATURES",
    "MODEL_VERSION",
    "FixedOffsetLogisticCorrection",
    "ModelV21OutputPaths",
    "ModelV21Result",
    "block_bootstrap_comparison",
    "build_architecture_metrics",
    "build_block_bootstrap",
    "build_correction_direction",
    "build_odds_sensitivity",
    "correction_direction_slope",
    "signed_correction_bucket",
    "train_model_v2_1",
    "write_model_v2_1_artifacts",
]
