"""Build the final model-ready feature dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tennis_value.config import FeatureConfig
from tennis_value.rolling import SORT_COLUMNS, add_rolling_features

SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
RETAINED_COLUMNS = [
    "match_id",
    "match_date",
    "tournament",
    "surface",
    "player_1",
    "player_2",
    "player_1_won",
    "player_1_odds",
    "player_2_odds",
    "is_retirement",
]
SOURCE_ODDS_CONTEXT_COLUMNS = [
    column
    for source in ("b365", "ps", "avg", "max")
    for column in (
        f"source_winner_{source}_odds",
        f"source_loser_{source}_odds",
        f"player_1_{source}_odds",
        f"player_2_{source}_odds",
        f"{source}_pair_available",
    )
]
MODEL_FEATURE_COLUMNS = [
    "overall_elo_diff",
    "surface_elo_diff",
    "log_rank_diff",
    "recent_10_win_rate_diff",
    "surface_recent_10_win_rate_diff",
    "days_since_last_match_diff",
    "matches_last_14d_diff",
    "history_count_min",
    "best_of_5",
    "surface_clay",
    "surface_grass",
]
OPTIONAL_CONTEXT_COLUMNS = [
    "rank_diff",
    "rank_missing_player_1",
    "rank_missing_player_2",
    "player_1_prior_match_count",
    "player_2_prior_match_count",
    "player_1_recent_10_win_rate",
    "player_2_recent_10_win_rate",
    "player_1_surface_recent_10_win_rate",
    "player_2_surface_recent_10_win_rate",
    "player_1_days_since_last_match",
    "player_2_days_since_last_match",
    "player_1_matches_last_14d",
    "player_2_matches_last_14d",
    "player_1_surface_history_count",
    "player_2_surface_history_count",
    "surface_history_count_min",
]
FEATURE_OUTPUT_COLUMNS = (
    RETAINED_COLUMNS
    + SOURCE_ODDS_CONTEXT_COLUMNS
    + MODEL_FEATURE_COLUMNS
    + OPTIONAL_CONTEXT_COLUMNS
)
REQUIRED_COLUMNS = (
    "match_id",
    "match_date",
    "tournament_normalized",
    "round",
    "surface",
    "best_of",
    "player_1",
    "player_2",
    "player_1_normalized",
    "player_2_normalized",
    "player_1_rank",
    "player_2_rank",
    "player_1_odds",
    "player_2_odds",
    "player_1_won",
    "is_retirement",
    "overall_elo_diff",
    "surface_elo_diff",
    "history_count_min",
)


class FeatureQualityReport(BaseModel):
    """JSON-serializable feature quality report."""

    model_config = ConfigDict(frozen=True)

    rows_received: int
    rows_returned: int
    eligible_history_updates: int
    skipped_retirements: int
    players_seen: int
    missingness_by_feature: dict[str, int] = Field(default_factory=dict)
    minimum_by_numeric_feature: dict[str, float | None] = Field(default_factory=dict)
    maximum_by_numeric_feature: dict[str, float | None] = Field(default_factory=dict)
    rows_with_missing_rank: int
    rows_with_missing_rest: int
    rows_with_zero_history: int
    surface_counts: dict[str, int] = Field(default_factory=dict)
    best_of_counts: dict[str, int] = Field(default_factory=dict)
    minimum_match_date: str | None = None
    maximum_match_date: str | None = None


class FeatureResult(BaseModel):
    """Feature rows and quality report."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    features: pd.DataFrame
    report: FeatureQualityReport


def build_feature_dataset(
    matches_with_elo: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build deterministic, model-ready feature rows from Elo-enriched matches."""
    active_config = config or FeatureConfig()
    _validate_input(matches_with_elo)

    enriched = add_rolling_features(matches_with_elo, active_config.rolling)
    enriched["log_rank_diff"] = _log_rank_diff(enriched)
    enriched["rank_diff"] = _rank_diff(enriched)
    enriched["rank_missing_player_1"] = enriched["player_1_rank"].isna().astype("int64")
    enriched["rank_missing_player_2"] = enriched["player_2_rank"].isna().astype("int64")
    enriched["best_of_5"] = (pd.to_numeric(enriched["best_of"], errors="raise") == 5).astype(
        "int64"
    )
    enriched["surface_clay"] = (enriched["surface"].astype(str) == "Clay").astype("int64")
    enriched["surface_grass"] = (enriched["surface"].astype(str) == "Grass").astype("int64")
    for column in SOURCE_ODDS_CONTEXT_COLUMNS:
        if column not in enriched:
            enriched[column] = False if column.endswith("_pair_available") else pd.NA

    feature_rows = enriched.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
    feature_rows = feature_rows[FEATURE_OUTPUT_COLUMNS].copy()
    return _coerce_feature_dtypes(feature_rows)


def build_features_with_report(
    matches_with_elo: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> FeatureResult:
    """Build feature rows and their quality report."""
    features = build_feature_dataset(matches_with_elo, config)
    return FeatureResult(features=features, report=build_feature_report(features))


def build_feature_report(features: pd.DataFrame) -> FeatureQualityReport:
    """Summarize feature dataset quality."""
    numeric_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS + OPTIONAL_CONTEXT_COLUMNS
        if column in features.columns
    ]
    min_values: dict[str, float | None] = {}
    max_values: dict[str, float | None] = {}
    for column in numeric_columns:
        series = pd.to_numeric(features[column], errors="coerce")
        min_values[column] = _nullable_float(series.min())
        max_values[column] = _nullable_float(series.max())

    min_date = features["match_date"].min() if not features.empty else None
    max_date = features["match_date"].max() if not features.empty else None
    missing_rank = features["rank_missing_player_1"].astype(bool) | features[
        "rank_missing_player_2"
    ].astype(bool)
    missing_rest = features["player_1_days_since_last_match"].isna() | features[
        "player_2_days_since_last_match"
    ].isna()

    return FeatureQualityReport(
        rows_received=len(features),
        rows_returned=len(features),
        eligible_history_updates=int((~features["is_retirement"].astype(bool)).sum()),
        skipped_retirements=int(features["is_retirement"].astype(bool).sum()),
        players_seen=len(set(features["player_1"]) | set(features["player_2"])),
        missingness_by_feature={
            column: int(features[column].isna().sum()) for column in MODEL_FEATURE_COLUMNS
        },
        minimum_by_numeric_feature=min_values,
        maximum_by_numeric_feature=max_values,
        rows_with_missing_rank=int(missing_rank.sum()),
        rows_with_missing_rest=int(missing_rest.sum()),
        rows_with_zero_history=int((features["history_count_min"] == 0).sum()),
        surface_counts=_value_counts(features, "surface"),
        best_of_counts=_best_of_counts(features),
        minimum_match_date=_date_to_string(min_date),
        maximum_match_date=_date_to_string(max_date),
    )


def write_feature_outputs(result: FeatureResult, output_path: Path, report_path: Path) -> None:
    """Write model-ready features and quality report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.features.to_parquet(output_path, index=False)
    report_path.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")


def feature_report_to_json(report: FeatureQualityReport) -> str:
    """Serialize a feature report as formatted JSON."""
    return json.dumps(report.model_dump(mode="json"), indent=2)


def _validate_input(matches: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing:
        msg = f"missing required feature input columns: {missing}"
        raise ValueError(msg)
    if matches["match_id"].duplicated().any():
        duplicates = sorted(matches.loc[matches["match_id"].duplicated(), "match_id"].astype(str))
        msg = f"duplicate match_id values are not allowed: {duplicates}"
        raise ValueError(msg)
    if pd.to_datetime(matches["match_date"], errors="coerce").isna().any():
        msg = "match_date contains invalid values"
        raise ValueError(msg)
    invalid_surfaces = sorted(set(matches["surface"].astype(str)) - SUPPORTED_SURFACES)
    if invalid_surfaces:
        msg = f"unsupported surface values for features: {invalid_surfaces}"
        raise ValueError(msg)
    invalid_best_of = sorted(
        set(pd.to_numeric(matches["best_of"], errors="coerce").dropna().astype(int)) - {3, 5}
    )
    if invalid_best_of:
        msg = f"best_of must be 3 or 5, got: {invalid_best_of}"
        raise ValueError(msg)
    if pd.to_numeric(matches["best_of"], errors="coerce").isna().any():
        msg = "best_of contains invalid values"
        raise ValueError(msg)
    for column in ("player_1_won", "is_retirement"):
        try:
            matches[column].map(lambda value, field_name=column: _coerce_bool(value, field_name))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    same_player = matches["player_1_normalized"].astype(str) == matches[
        "player_2_normalized"
    ].astype(str)
    if same_player.any():
        msg = "player_1_normalized and player_2_normalized must be distinct"
        raise ValueError(msg)


def _coerce_bool(value: Any, field_name: str) -> bool:
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
    msg = f"{field_name} must be Boolean or safely coercible"
    raise ValueError(msg)


def _log_rank_diff(frame: pd.DataFrame) -> pd.Series:
    player_1_rank = pd.to_numeric(frame["player_1_rank"], errors="coerce")
    player_2_rank = pd.to_numeric(frame["player_2_rank"], errors="coerce")
    return pd.Series(np.log1p(player_2_rank) - np.log1p(player_1_rank), index=frame.index)


def _rank_diff(frame: pd.DataFrame) -> pd.Series:
    player_1_rank = pd.to_numeric(frame["player_1_rank"], errors="coerce")
    player_2_rank = pd.to_numeric(frame["player_2_rank"], errors="coerce")
    return player_2_rank - player_1_rank


def _coerce_feature_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    coerced["match_date"] = pd.to_datetime(coerced["match_date"], errors="raise")
    for column in ("player_1_won", "is_retirement"):
        coerced[column] = coerced[column].astype("bool")
    float_columns = [
        "overall_elo_diff",
        "surface_elo_diff",
        "log_rank_diff",
        "recent_10_win_rate_diff",
        "surface_recent_10_win_rate_diff",
        "days_since_last_match_diff",
        "rank_diff",
        "player_1_recent_10_win_rate",
        "player_2_recent_10_win_rate",
        "player_1_surface_recent_10_win_rate",
        "player_2_surface_recent_10_win_rate",
        "player_1_days_since_last_match",
        "player_2_days_since_last_match",
    ]
    source_float_columns = [
        column for column in SOURCE_ODDS_CONTEXT_COLUMNS if not column.endswith("_pair_available")
    ]
    source_bool_columns = [
        column for column in SOURCE_ODDS_CONTEXT_COLUMNS if column.endswith("_pair_available")
    ]
    int_columns = [
        "matches_last_14d_diff",
        "history_count_min",
        "best_of_5",
        "surface_clay",
        "surface_grass",
        "rank_missing_player_1",
        "rank_missing_player_2",
        "player_1_prior_match_count",
        "player_2_prior_match_count",
        "player_1_matches_last_14d",
        "player_2_matches_last_14d",
        "player_1_surface_history_count",
        "player_2_surface_history_count",
        "surface_history_count_min",
    ]
    for column in float_columns + source_float_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype("Float64")
    for column in source_bool_columns:
        coerced[column] = coerced[column].fillna(False).astype("bool")
    for column in int_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype("int64")
    return coerced


def _nullable_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _best_of_counts(features: pd.DataFrame) -> dict[str, int]:
    if "best_of_5" not in features:
        return {}
    counts = features["best_of_5"].map({0: "3", 1: "5"}).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _date_to_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


__all__ = [
    "FEATURE_OUTPUT_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "SOURCE_ODDS_CONTEXT_COLUMNS",
    "FeatureQualityReport",
    "FeatureResult",
    "build_feature_dataset",
    "build_feature_report",
    "build_features_with_report",
    "feature_report_to_json",
    "write_feature_outputs",
]
