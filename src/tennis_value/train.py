"""Chronological baseline probability-model training."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_value.config import DateSplitConfig, TrainingConfig
from tennis_value.features import MODEL_FEATURE_COLUMNS

MODEL_FEATURES = list(MODEL_FEATURE_COLUMNS)
NUMERIC_FEATURES = [
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
    *MODEL_FEATURES,
]
PREDICTION_COLUMNS = [
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
]


class DatasetPartitions(BaseModel):
    """Chronological train/validation/test partitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class TrainingSummary(BaseModel):
    """JSON-serializable training summary."""

    model_config = ConfigDict(frozen=True)

    rows_received: int
    rows_eligible: int
    rows_excluded: int
    excluded_retirements: int
    excluded_invalid_dates: int
    excluded_invalid_targets: int
    excluded_duplicate_ids: int
    excluded_unsupported_surfaces: int
    excluded_outside_date_ranges: int
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_names: list[str]
    missingness_by_partition: dict[str, dict[str, int]] = Field(default_factory=dict)
    class_balance_by_partition: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    model_version: str
    model_output_path: str
    prediction_output_path: str


class ModelMetadata(BaseModel):
    """JSON-serializable model metadata."""

    model_config = ConfigDict(frozen=True)

    model_version: str
    model_class: str
    created_at_utc: str
    feature_names: list[str]
    target_name: str
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    training_rows: int
    validation_rows: int
    test_rows: int
    training_positive_rate: float
    validation_positive_rate: float
    test_positive_rate: float
    classifier_parameters: dict[str, Any]
    selected_c: float
    selection_metric: str
    input_dataset_path: str | None
    input_dataset_sha256: str | None
    python_version: str
    scikit_learn_version: str
    pandas_version: str
    git_commit_when_available: str | None
    warnings: list[str] = Field(default_factory=list)


class TrainingResult(BaseModel):
    """Trained model, predictions, metadata, and summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: Pipeline
    predictions: pd.DataFrame
    metadata: ModelMetadata
    summary: TrainingSummary
    partitions: DatasetPartitions


def split_feature_dataset(
    features: pd.DataFrame,
    config: DateSplitConfig,
) -> DatasetPartitions:
    """Split eligible feature rows into strict chronological partitions."""
    prepared, _ = _prepare_eligible_features(features)
    return _partition_prepared_features(prepared, config)


def build_model_pipeline(
    feature_names: list[str],
    *,
    c_value: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    """Build the baseline scikit-learn probability pipeline."""
    _validate_feature_allowlist(feature_names)
    numeric_features = [feature for feature in NUMERIC_FEATURES if feature in feature_names]
    binary_features = [feature for feature in BINARY_FEATURES if feature in feature_names]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("binary", binary_pipeline, binary_features),
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


def train_probability_model(
    features: pd.DataFrame,
    config: TrainingConfig | None = None,
    *,
    input_dataset_path: Path | None = None,
    model_output_path: Path | None = None,
    prediction_output_path: Path | None = None,
) -> TrainingResult:
    """Fit the baseline model and generate chronological partition predictions."""
    active_config = config or TrainingConfig()
    prepared, exclusions = _prepare_eligible_features(features)
    partitions = _partition_prepared_features(prepared, active_config.date_splits)
    _validate_partitions(partitions)

    model = build_model_pipeline(
        MODEL_FEATURES,
        c_value=active_config.classifier_c,
        max_iter=active_config.max_iter,
        random_state=active_config.random_state,
    )
    model.fit(partitions.train[MODEL_FEATURES], partitions.train["target"])

    predictions = pd.concat(
        [
            generate_predictions(
                model,
                partitions.train,
                partition_name="train",
                model_version=active_config.model_version,
            ),
            generate_predictions(
                model,
                partitions.validation,
                partition_name="validation",
                model_version=active_config.model_version,
            ),
            generate_predictions(
                model,
                partitions.test,
                partition_name="test",
                model_version=active_config.model_version,
            ),
        ],
        ignore_index=True,
    )

    summary = _build_training_summary(
        rows_received=len(features),
        exclusions=exclusions,
        partitions=partitions,
        model_version=active_config.model_version,
        model_output_path=model_output_path,
        prediction_output_path=prediction_output_path,
    )
    metadata = _build_metadata(
        config=active_config,
        partitions=partitions,
        input_dataset_path=input_dataset_path,
    )
    return TrainingResult(
        model=model,
        predictions=predictions,
        metadata=metadata,
        summary=summary,
        partitions=partitions,
    )


def generate_predictions(
    model: Pipeline,
    partition: pd.DataFrame,
    *,
    partition_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Generate P(player_1 wins) predictions for one partition."""
    probabilities = model.predict_proba(partition[MODEL_FEATURES])[:, 1]
    if not pd.Series(probabilities).between(0, 1).all():
        msg = "model produced probabilities outside [0, 1]"
        raise ValueError(msg)
    prediction_rows = partition[
        [
            "match_id",
            "match_date",
            "surface",
            "player_1",
            "player_2",
            "player_1_odds",
            "player_2_odds",
            "target",
        ]
    ].copy()
    prediction_rows["partition"] = partition_name
    prediction_rows["actual_player_1_won"] = prediction_rows["target"].astype("int64")
    prediction_rows["predicted_player_1_probability"] = probabilities.astype("float64")
    prediction_rows["predicted_player_2_probability"] = (
        1.0 - prediction_rows["predicted_player_1_probability"]
    )
    prediction_rows["model_version"] = model_version
    prediction_rows = prediction_rows[PREDICTION_COLUMNS]
    if not _probabilities_are_valid(prediction_rows):
        msg = "prediction probabilities must be finite and within [0, 1]"
        raise ValueError(msg)
    return prediction_rows


def write_training_outputs(
    result: TrainingResult,
    model_output_path: Path,
    metadata_output_path: Path,
    predictions_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Write trained model, metadata, predictions, and summary."""
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_output_path)
    metadata_output_path.write_text(result.metadata.model_dump_json(indent=2), encoding="utf-8")
    result.predictions.to_parquet(predictions_output_path, index=False)
    summary_output_path.write_text(result.summary.model_dump_json(indent=2), encoding="utf-8")


def dataset_sha256(path: Path) -> str:
    """Return a SHA-256 hash for a dataset file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_eligible_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        msg = f"missing required training input columns: {missing}"
        raise ValueError(msg)

    prepared = features.copy(deep=True)
    rows_received = len(prepared)
    prepared["match_date"] = pd.to_datetime(prepared["match_date"], errors="coerce")
    invalid_dates = prepared["match_date"].isna()

    duplicate_ids = prepared["match_id"].duplicated(keep=False)
    invalid_targets = ~prepared["player_1_won"].map(_target_is_valid)
    unsupported_surfaces = ~prepared["surface"].astype(str).isin({"Hard", "Clay", "Grass"})
    retirements = prepared["is_retirement"].map(_coerce_bool).fillna(False).astype(bool)

    exclusion_mask = (
        invalid_dates | duplicate_ids | invalid_targets | unsupported_surfaces | retirements
    )
    eligible = prepared.loc[~exclusion_mask].copy()
    eligible["target"] = eligible["player_1_won"].map(_target_to_int).astype("int64")
    for column in BINARY_FEATURES:
        values = set(pd.to_numeric(eligible[column], errors="coerce").dropna().astype(int))
        if values - {0, 1}:
            msg = f"{column} must contain only binary values"
            raise ValueError(msg)

    eligible = eligible.sort_values(["match_date", "match_id"], kind="mergesort").reset_index(
        drop=True
    )
    exclusions = {
        "rows_received": rows_received,
        "excluded_retirements": int(retirements.sum()),
        "excluded_invalid_dates": int(invalid_dates.sum()),
        "excluded_invalid_targets": int(invalid_targets.sum()),
        "excluded_duplicate_ids": int(duplicate_ids.sum()),
        "excluded_unsupported_surfaces": int(unsupported_surfaces.sum()),
    }
    return eligible, exclusions


def _partition_prepared_features(
    prepared: pd.DataFrame,
    config: DateSplitConfig,
) -> DatasetPartitions:
    train_mask = _between(prepared["match_date"], config.train_start, config.train_end)
    validation_mask = _between(
        prepared["match_date"],
        config.validation_start,
        config.validation_end,
    )
    test_mask = _between(prepared["match_date"], config.test_start, config.test_end)
    if ((train_mask.astype(int) + validation_mask.astype(int) + test_mask.astype(int)) > 1).any():
        msg = "configured date ranges produce overlapping partitions"
        raise ValueError(msg)
    return DatasetPartitions(
        train=prepared.loc[train_mask].copy().reset_index(drop=True),
        validation=prepared.loc[validation_mask].copy().reset_index(drop=True),
        test=prepared.loc[test_mask].copy().reset_index(drop=True),
    )


def _between(series: pd.Series, start: Any, end: Any) -> pd.Series:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    return (series >= start_timestamp) & (series <= end_timestamp)


def _validate_partitions(partitions: DatasetPartitions) -> None:
    if partitions.train.empty:
        msg = "training partition is empty"
        raise ValueError(msg)
    if partitions.validation.empty:
        msg = "validation partition is empty"
        raise ValueError(msg)
    if partitions.test.empty:
        msg = "test partition is empty"
        raise ValueError(msg)
    if partitions.train["target"].nunique() < 2:
        msg = "training partition must contain both target classes"
        raise ValueError(msg)
    _assert_no_overlapping_match_ids(partitions)


def _assert_no_overlapping_match_ids(partitions: DatasetPartitions) -> None:
    train_ids = set(partitions.train["match_id"])
    validation_ids = set(partitions.validation["match_id"])
    test_ids = set(partitions.test["match_id"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        msg = "partition match IDs must not overlap"
        raise ValueError(msg)


def _build_training_summary(
    *,
    rows_received: int,
    exclusions: dict[str, int],
    partitions: DatasetPartitions,
    model_version: str,
    model_output_path: Path | None,
    prediction_output_path: Path | None,
) -> TrainingSummary:
    partitioned_rows = len(partitions.train) + len(partitions.validation) + len(partitions.test)
    rows_eligible = rows_received - sum(
        exclusions[key]
        for key in (
            "excluded_retirements",
            "excluded_invalid_dates",
            "excluded_invalid_targets",
            "excluded_duplicate_ids",
            "excluded_unsupported_surfaces",
        )
    )
    excluded_outside = max(rows_eligible - partitioned_rows, 0)
    return TrainingSummary(
        rows_received=rows_received,
        rows_eligible=partitioned_rows,
        rows_excluded=rows_received - partitioned_rows,
        excluded_retirements=exclusions["excluded_retirements"],
        excluded_invalid_dates=exclusions["excluded_invalid_dates"],
        excluded_invalid_targets=exclusions["excluded_invalid_targets"],
        excluded_duplicate_ids=exclusions["excluded_duplicate_ids"],
        excluded_unsupported_surfaces=exclusions["excluded_unsupported_surfaces"],
        excluded_outside_date_ranges=excluded_outside,
        train_rows=len(partitions.train),
        validation_rows=len(partitions.validation),
        test_rows=len(partitions.test),
        feature_names=MODEL_FEATURES,
        missingness_by_partition={
            "train": _missingness(partitions.train),
            "validation": _missingness(partitions.validation),
            "test": _missingness(partitions.test),
        },
        class_balance_by_partition={
            "train": _class_balance(partitions.train),
            "validation": _class_balance(partitions.validation),
            "test": _class_balance(partitions.test),
        },
        model_version=model_version,
        model_output_path=str(model_output_path) if model_output_path else "",
        prediction_output_path=str(prediction_output_path) if prediction_output_path else "",
    )


def _build_metadata(
    *,
    config: TrainingConfig,
    partitions: DatasetPartitions,
    input_dataset_path: Path | None,
) -> ModelMetadata:
    git_commit, warnings = _git_commit()
    classifier_parameters = {
        "C": config.classifier_c,
        "max_iter": config.max_iter,
        "random_state": config.random_state,
        "solver": "lbfgs",
    }
    return ModelMetadata(
        model_version=config.model_version,
        model_class="sklearn.pipeline.Pipeline",
        created_at_utc=datetime.now(UTC).isoformat(),
        feature_names=MODEL_FEATURES,
        target_name="player_1_won",
        training_start=config.date_splits.train_start.isoformat(),
        training_end=config.date_splits.train_end.isoformat(),
        validation_start=config.date_splits.validation_start.isoformat(),
        validation_end=config.date_splits.validation_end.isoformat(),
        test_start=config.date_splits.test_start.isoformat(),
        test_end=config.date_splits.test_end.isoformat(),
        training_rows=len(partitions.train),
        validation_rows=len(partitions.validation),
        test_rows=len(partitions.test),
        training_positive_rate=float(partitions.train["target"].mean()),
        validation_positive_rate=float(partitions.validation["target"].mean()),
        test_positive_rate=float(partitions.test["target"].mean()),
        classifier_parameters=classifier_parameters,
        selected_c=config.classifier_c,
        selection_metric="fixed_C_no_validation_selection",
        input_dataset_path=str(input_dataset_path) if input_dataset_path else None,
        input_dataset_sha256=(
            dataset_sha256(input_dataset_path)
            if input_dataset_path is not None and input_dataset_path.exists()
            else None
        ),
        python_version=platform.python_version(),
        scikit_learn_version=sklearn.__version__,
        pandas_version=pd.__version__,
        git_commit_when_available=git_commit,
        warnings=warnings,
    )


def _missingness(partition: pd.DataFrame) -> dict[str, int]:
    return {feature: int(partition[feature].isna().sum()) for feature in MODEL_FEATURES}


def _class_balance(partition: pd.DataFrame) -> dict[str, float | int]:
    positives = int(partition["target"].sum())
    rows = len(partition)
    return {
        "rows": rows,
        "positives": positives,
        "negatives": rows - positives,
        "positive_rate": float(partition["target"].mean()) if rows else 0.0,
    }


def _validate_feature_allowlist(feature_names: list[str]) -> None:
    if feature_names != MODEL_FEATURES:
        msg = f"feature_names must exactly match MODEL_FEATURES: {MODEL_FEATURES}"
        raise ValueError(msg)
    forbidden = {
        "match_id",
        "match_date",
        "player_1",
        "player_2",
        "player_1_odds",
        "player_2_odds",
        "player_1_won",
        "is_retirement",
    }
    if forbidden & set(feature_names):
        msg = "model features contain forbidden leakage columns"
        raise ValueError(msg)


def _probabilities_are_valid(predictions: pd.DataFrame) -> bool:
    p1 = pd.to_numeric(predictions["predicted_player_1_probability"], errors="coerce")
    p2 = pd.to_numeric(predictions["predicted_player_2_probability"], errors="coerce")
    return bool(
        p1.notna().all()
        and p2.notna().all()
        and p1.between(0, 1).all()
        and p2.between(0, 1).all()
    )


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


__all__ = [
    "BINARY_FEATURES",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "DatasetPartitions",
    "ModelMetadata",
    "TrainingResult",
    "TrainingSummary",
    "build_model_pipeline",
    "dataset_sha256",
    "generate_predictions",
    "split_feature_dataset",
    "train_probability_model",
    "write_training_outputs",
]
