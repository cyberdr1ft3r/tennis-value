"""Market-anchored walk-forward Model v2 training."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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

from tennis_value.evaluate import (
    build_calibration_table,
    calculate_calibration_summary,
    calculate_probability_metrics,
)
from tennis_value.features import MODEL_FEATURE_COLUMNS

MODEL_VERSION = "model_v2"
LOGIT_EPSILON = 1e-6
ELO_SCALE = 400.0
MARKET_FEATURES = ["market_logit_player_1", *list(MODEL_FEATURE_COLUMNS)]
NUMERIC_FEATURES = [
    "market_logit_player_1",
    "overall_elo_diff",
    "surface_elo_diff",
    "log_rank_diff",
    "recent_10_win_rate_diff",
    "surface_recent_10_win_rate_diff",
    "days_since_last_match_diff",
    "matches_last_14d_diff",
    "history_count_min",
]
BINARY_FEATURES = ["best_of_5", "surface_clay", "surface_grass"]
FORBIDDEN_FEATURES = {
    "match_id",
    "match_date",
    "player_1",
    "player_2",
    "player_1_won",
    "actual_player_1_won",
    "player_1_odds",
    "player_2_odds",
    "is_retirement",
    "edge",
    "expected_value",
    "result",
    "settlement_reason",
    "bankroll_before",
    "bankroll_after",
    "profit_loss",
    "stake",
}
REQUIRED_COLUMNS = [
    "match_id",
    "match_date",
    "surface",
    "player_1",
    "player_2",
    "player_1_won",
    "player_1_odds",
    "player_2_odds",
    "is_retirement",
    *list(MODEL_FEATURE_COLUMNS),
]
PREDICTION_COLUMNS = [
    "match_id",
    "match_date",
    "evaluation_year",
    "surface",
    "player_1",
    "player_2",
    "actual_player_1_won",
    "player_1_odds",
    "player_2_odds",
    "market_probability_player_1",
    "predicted_player_1_probability",
    "predicted_player_2_probability",
    "probability_correction",
    "model_version",
]
WALK_FORWARD_FOLDS = (
    {"fold": 1, "train_start_year": 2020, "train_end_year": 2022, "evaluation_year": 2023},
    {"fold": 2, "train_start_year": 2020, "train_end_year": 2023, "evaluation_year": 2024},
    {"fold": 3, "train_start_year": 2020, "train_end_year": 2024, "evaluation_year": 2025},
)


class FoldResult(BaseModel):
    """One walk-forward fold result."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    fold: int
    train_start_year: int
    train_end_year: int
    evaluation_year: int
    model: Pipeline
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    coefficient_by_feature: dict[str, float]


class ModelV2Result(BaseModel):
    """Model v2 training artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: Pipeline
    predictions: pd.DataFrame
    corrections: pd.DataFrame
    calibration: pd.DataFrame
    metrics_report: dict[str, Any]
    metadata: dict[str, Any]
    folds: tuple[FoldResult, ...]


@dataclass(frozen=True)
class WalkForwardSplit:
    """Prepared train/evaluation rows for one walk-forward fold."""

    fold: int
    train_start_year: int
    train_end_year: int
    evaluation_year: int
    train: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class ModelV2OutputPaths:
    """Output paths for Model v2 artifacts."""

    model_output: Path
    metadata_output: Path
    predictions_output: Path
    metrics_output: Path
    corrections_output: Path
    calibration_output: Path
    correction_distribution_plot: Path


def add_market_features(features: pd.DataFrame, *, epsilon: float = LOGIT_EPSILON) -> pd.DataFrame:
    """Add no-vig market probability, market logit, and overround for valid paired odds."""
    _validate_required_columns(features)
    frame = features.copy(deep=True)
    odds_1 = pd.to_numeric(frame["player_1_odds"], errors="coerce")
    odds_2 = pd.to_numeric(frame["player_2_odds"], errors="coerce")
    raw_1 = 1.0 / odds_1
    raw_2 = 1.0 / odds_2
    overround = raw_1 + raw_2
    valid = (
        odds_1.notna()
        & odds_2.notna()
        & np.isfinite(odds_1)
        & np.isfinite(odds_2)
        & (odds_1 > 1.0)
        & (odds_2 > 1.0)
        & np.isfinite(overround)
        & (overround > 0)
    )
    frame = frame.loc[valid].copy()
    frame["player_1_odds"] = odds_1.loc[valid].astype("float64")
    frame["player_2_odds"] = odds_2.loc[valid].astype("float64")
    raw_1 = raw_1.loc[valid]
    raw_2 = raw_2.loc[valid]
    frame["overround"] = (raw_1 + raw_2).astype("float64")
    frame["market_probability_player_1"] = (raw_1 / frame["overround"]).astype("float64")
    frame["market_logit_player_1"] = market_logit(frame["market_probability_player_1"], epsilon)
    return cast(pd.DataFrame, frame)


def market_logit(
    probabilities: pd.Series | np.ndarray[Any, Any],
    epsilon: float = LOGIT_EPSILON,
) -> pd.Series:
    """Calculate a safe logit; probabilities are clipped only for the logit transform."""
    series = pd.Series(probabilities, dtype="float64")
    if series.isna().any() or not np.isfinite(series).all():
        msg = "market probabilities must be finite"
        raise ValueError(msg)
    if not series.between(0, 1).all():
        msg = "market probabilities must be in [0, 1]"
        raise ValueError(msg)
    if not math.isfinite(epsilon) or epsilon <= 0 or epsilon >= 0.5:
        msg = "epsilon must be finite and between 0 and 0.5"
        raise ValueError(msg)
    clipped = series.clip(epsilon, 1.0 - epsilon)
    return pd.Series(np.log(clipped / (1.0 - clipped)), index=series.index, dtype="float64")


def prepare_model_v2_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare eligible market-anchored rows without mutating input."""
    frame = add_market_features(features)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    invalid_dates = frame["match_date"].isna()
    invalid_targets = ~frame["player_1_won"].map(_target_is_valid)
    retirements = frame["is_retirement"].map(_coerce_bool).fillna(False).astype(bool)
    unsupported_surfaces = ~frame["surface"].astype(str).isin({"Hard", "Clay", "Grass"})
    duplicate_ids = frame["match_id"].duplicated(keep=False)
    exclusions = (
        invalid_dates
        | invalid_targets
        | retirements
        | unsupported_surfaces
        | duplicate_ids
    )
    prepared = frame.loc[~exclusions].copy()
    prepared["target"] = prepared["player_1_won"].map(_target_to_int).astype("int64")
    prepared["year"] = prepared["match_date"].dt.year.astype("int64")
    _validate_feature_allowlist(MARKET_FEATURES)
    prepared = prepared.sort_values(["match_date", "match_id"], kind="mergesort").reset_index(
        drop=True
    )
    return prepared


def build_walk_forward_folds(prepared: pd.DataFrame) -> list[WalkForwardSplit]:
    """Return strict chronological walk-forward train/evaluation frames."""
    folds: list[WalkForwardSplit] = []
    for spec in WALK_FORWARD_FOLDS:
        train = prepared[
            (prepared["year"] >= spec["train_start_year"])
            & (prepared["year"] <= spec["train_end_year"])
        ].copy()
        evaluation = prepared[prepared["year"] == spec["evaluation_year"]].copy()
        if not train.empty and not evaluation.empty:
            if train["match_date"].max() >= evaluation["match_date"].min():
                msg = "walk-forward fold has overlapping train/evaluation dates"
                raise ValueError(msg)
            if set(train["match_id"]) & set(evaluation["match_id"]):
                msg = "walk-forward fold includes an evaluation row in training"
                raise ValueError(msg)
        folds.append(
            WalkForwardSplit(
                fold=int(spec["fold"]),
                train_start_year=int(spec["train_start_year"]),
                train_end_year=int(spec["train_end_year"]),
                evaluation_year=int(spec["evaluation_year"]),
                train=train.reset_index(drop=True),
                evaluation=evaluation.reset_index(drop=True),
            )
        )
    return folds


def build_market_anchored_pipeline(
    *,
    c_value: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    """Build the fixed regularized logistic-regression pipeline for Model v2."""
    _validate_feature_allowlist(MARKET_FEATURES)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_model_v2(
    features: pd.DataFrame,
    *,
    input_dataset_path: Path | None = None,
    model_v1_predictions: pd.DataFrame | None = None,
    c_value: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> ModelV2Result:
    """Train Model v2 walk-forward folds and return all artifacts."""
    prepared = prepare_model_v2_dataset(features)
    fold_specs = build_walk_forward_folds(prepared)
    fold_results: list[FoldResult] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in fold_specs:
        train = spec.train
        evaluation = spec.evaluation
        if train.empty or evaluation.empty:
            msg = f"fold {spec.fold} has empty train or evaluation rows"
            raise ValueError(msg)
        if train["target"].nunique() < 2:
            msg = f"fold {spec.fold} training rows must contain both target classes"
            raise ValueError(msg)
        model = build_market_anchored_pipeline(
            c_value=c_value,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(train[MARKET_FEATURES], train["target"])
        predictions = generate_fold_predictions(
            model,
            evaluation,
            evaluation_year=spec.evaluation_year,
        )
        metrics = _fold_metrics(
            evaluation=evaluation,
            predictions=predictions,
            model_v1_predictions=model_v1_predictions,
        )
        coefficients = _coefficient_by_feature(model)
        fold_results.append(
            FoldResult(
                fold=spec.fold,
                train_start_year=spec.train_start_year,
                train_end_year=spec.train_end_year,
                evaluation_year=spec.evaluation_year,
                model=model,
                predictions=predictions,
                metrics=metrics,
                coefficient_by_feature=coefficients,
            )
        )
        prediction_frames.append(predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    corrections = _build_corrections(predictions)
    calibration = build_calibration_table(
        predictions.assign(evaluation_year=predictions["evaluation_year"].astype(str)),
        probability_column="predicted_player_1_probability",
        target_column="actual_player_1_won",
        partition_column="evaluation_year",
    )
    calibration_summary = {
        year: summary.model_dump()
        for year, summary in calculate_calibration_summary(calibration).items()
    }
    final_model = fold_results[-1].model
    metrics_report = _build_metrics_report(
        fold_results=fold_results,
        corrections=corrections,
        calibration_summary=calibration_summary,
        c_value=c_value,
        max_iter=max_iter,
        random_state=random_state,
    )
    metadata = _build_metadata(
        prepared=prepared,
        fold_results=fold_results,
        input_dataset_path=input_dataset_path,
        c_value=c_value,
        max_iter=max_iter,
        random_state=random_state,
    )
    return ModelV2Result(
        model=final_model,
        predictions=predictions,
        corrections=corrections,
        calibration=calibration,
        metrics_report=metrics_report,
        metadata=metadata,
        folds=tuple(fold_results),
    )


def generate_fold_predictions(
    model: Pipeline,
    evaluation: pd.DataFrame,
    *,
    evaluation_year: int,
) -> pd.DataFrame:
    """Generate Model v2 predictions for one evaluation year."""
    probabilities = model.predict_proba(evaluation[MARKET_FEATURES])[:, 1]
    prediction_rows = evaluation[
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
        ]
    ].copy()
    prediction_rows["evaluation_year"] = evaluation_year
    prediction_rows["actual_player_1_won"] = prediction_rows["target"].astype("int64")
    prediction_rows["predicted_player_1_probability"] = probabilities.astype("float64")
    prediction_rows["predicted_player_2_probability"] = (
        1.0 - prediction_rows["predicted_player_1_probability"]
    )
    prediction_rows["probability_correction"] = (
        prediction_rows["predicted_player_1_probability"]
        - prediction_rows["market_probability_player_1"]
    )
    prediction_rows["model_version"] = MODEL_VERSION
    prediction_rows["match_date"] = pd.to_datetime(prediction_rows["match_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    prediction_rows = prediction_rows[PREDICTION_COLUMNS]
    if not np.allclose(
        prediction_rows["predicted_player_1_probability"]
        + prediction_rows["predicted_player_2_probability"],
        1.0,
        atol=1e-9,
    ):
        msg = "prediction probabilities must sum to 1"
        raise ValueError(msg)
    return prediction_rows.reset_index(drop=True)


def write_model_v2_artifacts(result: ModelV2Result, output_paths: ModelV2OutputPaths) -> None:
    """Write Model v2 model, JSON, Parquet, and PNG artifacts."""
    for path in output_paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output_paths.model_output)
    output_paths.metadata_output.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    result.predictions.to_parquet(output_paths.predictions_output, index=False)
    output_paths.metrics_output.write_text(
        json.dumps(_jsonable(result.metrics_report), indent=2),
        encoding="utf-8",
    )
    result.corrections.to_parquet(output_paths.corrections_output, index=False)
    result.calibration.to_parquet(output_paths.calibration_output, index=False)
    write_correction_distribution_plot(
        result.corrections,
        output_paths.correction_distribution_plot,
    )


def write_correction_distribution_plot(corrections: pd.DataFrame, output_path: Path) -> None:
    """Write a simple headless correction-distribution PNG."""
    image = Image.new("RGB", (720, 440), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 660, 340), outline="black")
    draw.text((60, 14), "Model v2 Probability Correction Distribution", fill="black")
    draw.text((270, 380), "Correction", fill="black")
    values = corrections["probability_correction"].astype(float).to_numpy()
    if len(values) == 0:
        values = np.array([0.0])
    lower = min(float(values.min()), -0.2)
    upper = max(float(values.max()), 0.2)
    counts, edges = np.histogram(values, bins=20, range=(lower, upper))
    max_count = max(int(counts.max()), 1)
    bar_width = 600 / len(counts)
    for index, count in enumerate(counts):
        x0 = 60 + index * bar_width
        x1 = x0 + bar_width - 2
        y0 = 340 - (int(count) / max_count) * 280
        draw.rectangle((x0, y0, x1, 340), fill="steelblue")
    zero_x = 60 + ((0.0 - edges[0]) / (edges[-1] - edges[0])) * 600
    draw.line((zero_x, 40, zero_x, 340), fill="gray")
    image.save(output_path)


def _fold_metrics(
    *,
    evaluation: pd.DataFrame,
    predictions: pd.DataFrame,
    model_v1_predictions: pd.DataFrame | None,
) -> dict[str, Any]:
    target = predictions["actual_player_1_won"]
    model_metrics = calculate_probability_metrics(
        target,
        predictions["predicted_player_1_probability"],
    )
    market_metrics = calculate_probability_metrics(
        target,
        predictions["market_probability_player_1"],
    )
    elo_probability = _elo_probability_from_diff(evaluation["overall_elo_diff"])
    elo_metrics = calculate_probability_metrics(target, elo_probability)
    diagnostics = _correction_diagnostics(predictions["probability_correction"])
    report: dict[str, Any] = {
        "sample_count": model_metrics.sample_count,
        "model_v2": model_metrics.model_dump(),
        "market": market_metrics.model_dump(),
        "elo": elo_metrics.model_dump(),
        "log_loss_improvement_vs_market": market_metrics.log_loss - model_metrics.log_loss,
        "brier_improvement_vs_market": market_metrics.brier_score - model_metrics.brier_score,
        "correction_diagnostics": diagnostics,
    }
    if model_v1_predictions is not None:
        v1 = _comparable_v1_metrics(predictions, model_v1_predictions)
        if v1 is not None:
            report["model_v1"] = v1
            report["log_loss_improvement_vs_model_v1"] = (
                v1["log_loss"] - model_metrics.log_loss
            )
    return cast(dict[str, Any], _jsonable(report))


def _comparable_v1_metrics(
    predictions: pd.DataFrame,
    model_v1_predictions: pd.DataFrame,
) -> dict[str, Any] | None:
    required = {"match_id", "predicted_player_1_probability", "actual_player_1_won"}
    if not required.issubset(model_v1_predictions.columns):
        return None
    v1 = model_v1_predictions[list(required)].copy()
    joined = predictions[["match_id"]].merge(v1, on="match_id", how="left", validate="one_to_one")
    if joined["predicted_player_1_probability"].isna().any():
        return None
    metrics = calculate_probability_metrics(
        joined["actual_player_1_won"],
        joined["predicted_player_1_probability"],
    )
    return metrics.model_dump()


def _build_metrics_report(
    *,
    fold_results: list[FoldResult],
    corrections: pd.DataFrame,
    calibration_summary: dict[str, dict[str, float]],
    c_value: float,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _jsonable(
        {
            "model_version": MODEL_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "implementation": "regularized_logistic_regression_with_market_logit_feature",
            "logit_epsilon": LOGIT_EPSILON,
            "classifier_parameters": {
                "C": c_value,
                "max_iter": max_iter,
                "random_state": random_state,
                "solver": "lbfgs",
            },
            "feature_names": MARKET_FEATURES,
            "folds": [
                {
                    "fold": fold.fold,
                    "train_start_year": fold.train_start_year,
                    "train_end_year": fold.train_end_year,
                    "evaluation_year": fold.evaluation_year,
                    "market_logit_coefficient": fold.coefficient_by_feature.get(
                        "market_logit_player_1"
                    ),
                    "metrics": fold.metrics,
                    "coefficient_by_feature": fold.coefficient_by_feature,
                }
                for fold in fold_results
            ],
            "overall_correction_diagnostics": _correction_diagnostics(
                corrections["probability_correction"]
            ),
            "calibration_summary": calibration_summary,
        }
        ),
    )


def _build_metadata(
    *,
    prepared: pd.DataFrame,
    fold_results: list[FoldResult],
    input_dataset_path: Path | None,
    c_value: float,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    git_commit, warnings = _git_commit()
    return cast(
        dict[str, Any],
        _jsonable(
        {
            "model_version": MODEL_VERSION,
            "model_class": "sklearn.pipeline.Pipeline",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "implementation": "regularized_logistic_regression_with_market_logit_feature",
            "feature_names": MARKET_FEATURES,
            "target_name": "player_1_won",
            "logit_epsilon": LOGIT_EPSILON,
            "walk_forward_folds": [
                {
                    "fold": fold.fold,
                    "train_start_year": fold.train_start_year,
                    "train_end_year": fold.train_end_year,
                    "evaluation_year": fold.evaluation_year,
                    "evaluation_rows": len(fold.predictions),
                    "market_logit_coefficient": fold.coefficient_by_feature.get(
                        "market_logit_player_1"
                    ),
                }
                for fold in fold_results
            ],
            "final_model_training_years": "2020-2024",
            "final_model_evaluation_year": 2025,
            "eligible_rows": len(prepared),
            "classifier_parameters": {
                "C": c_value,
                "max_iter": max_iter,
                "random_state": random_state,
                "solver": "lbfgs",
            },
            "input_dataset_path": str(input_dataset_path) if input_dataset_path else None,
            "input_dataset_sha256": (
                dataset_sha256(input_dataset_path)
                if input_dataset_path is not None and input_dataset_path.exists()
                else None
            ),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "git_commit_when_available": git_commit,
            "warnings": warnings,
        }
        ),
    )


def _build_corrections(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_id",
        "match_date",
        "evaluation_year",
        "surface",
        "market_probability_player_1",
        "predicted_player_1_probability",
        "probability_correction",
        "model_version",
    ]
    return predictions[columns].copy()


def _correction_diagnostics(corrections: pd.Series) -> dict[str, float]:
    absolute = corrections.astype(float).abs()
    return {
        "mean_absolute_correction": float(absolute.mean()),
        "median_absolute_correction": float(absolute.median()),
        "maximum_absolute_correction": float(absolute.max()),
        "percentage_below_0_02": float((absolute < 0.02).mean()),
        "percentage_below_0_05": float((absolute < 0.05).mean()),
    }


def _coefficient_by_feature(model: Pipeline) -> dict[str, float]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())
    coefficients = classifier.coef_[0]
    return {
        str(name): float(value)
        for name, value in zip(feature_names, coefficients, strict=True)
    }


def _elo_probability_from_diff(overall_elo_diff: pd.Series) -> pd.Series:
    diff = pd.to_numeric(overall_elo_diff, errors="coerce")
    if diff.isna().any() or not np.isfinite(diff).all():
        msg = "overall_elo_diff must be finite for Elo baseline"
        raise ValueError(msg)
    return pd.Series(1.0 / (1.0 + 10.0 ** (-diff / ELO_SCALE)), index=overall_elo_diff.index)


def _validate_required_columns(features: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        msg = f"missing required model v2 input columns: {missing}"
        raise ValueError(msg)


def _validate_feature_allowlist(feature_names: list[str]) -> None:
    if feature_names != MARKET_FEATURES:
        msg = f"feature_names must exactly match MARKET_FEATURES: {MARKET_FEATURES}"
        raise ValueError(msg)
    forbidden = FORBIDDEN_FEATURES & set(feature_names)
    if forbidden:
        msg = f"Model v2 features contain forbidden leakage columns: {sorted(forbidden)}"
        raise ValueError(msg)


def _target_is_valid(value: Any) -> bool:
    return _target_to_int(value) is not None


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


def _coerce_bool(value: Any) -> bool | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def dataset_sha256(path: Path) -> str:
    """Return a SHA-256 hash for a dataset file."""
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


__all__ = [
    "BINARY_FEATURES",
    "FORBIDDEN_FEATURES",
    "LOGIT_EPSILON",
    "MARKET_FEATURES",
    "MODEL_VERSION",
    "ModelV2OutputPaths",
    "ModelV2Result",
    "WALK_FORWARD_FOLDS",
    "add_market_features",
    "build_market_anchored_pipeline",
    "build_walk_forward_folds",
    "generate_fold_predictions",
    "market_logit",
    "prepare_model_v2_dataset",
    "train_model_v2",
    "write_model_v2_artifacts",
]
