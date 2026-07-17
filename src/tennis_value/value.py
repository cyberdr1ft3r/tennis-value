"""Theoretical value assessment for model predictions and market odds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tennis_value.config import ValueThresholds
from tennis_value.odds import NoVigMarket, remove_two_way_margin, validate_decimal_odds

SelectionSide = Literal["player_1", "player_2", "none"]
SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
PROBABILITY_SUM_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12
REQUIRED_COLUMNS = (
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
REASON_ORDER = (
    "missing_match_id",
    "same_player",
    "unsupported_surface",
    "unsupported_match",
    "retirement",
    "walkover",
    "missing_odds",
    "invalid_odds",
    "invalid_probability",
    "probability_below_threshold",
    "edge_below_threshold",
    "expected_value_below_threshold",
    "odds_below_minimum",
    "odds_above_maximum",
    "multiple_eligible_selections",
)


class SelectionAssessment(BaseModel):
    """Per-player value assessment."""

    model_config = ConfigDict(frozen=True)

    match_id: str | None
    selection_side: SelectionSide
    selection_player: str | None
    model_probability: float | None = None
    decimal_odds: float | None = None
    raw_implied_probability: float | None = None
    market_probability: float | None = None
    overround: float | None = None
    edge: float | None = None
    expected_value: float | None = None
    eligible: bool
    reason_codes: tuple[str, ...] = ()


class MatchValueAssessment(BaseModel):
    """One-match value decision with at most one recommendation."""

    model_config = ConfigDict(frozen=True)

    match_id: str | None
    match_date: str | None
    partition: str | None
    surface: str | None
    player_1: str | None
    player_2: str | None
    actual_player_1_won: bool | None = None
    model_version: str | None
    odds_source: str | None = None
    is_retirement: bool = False
    player_1_assessment: SelectionAssessment
    player_2_assessment: SelectionAssessment
    recommended_side: SelectionSide = "none"
    recommended_player: str | None = None
    recommended_probability: float | None = None
    recommended_odds: float | None = None
    recommended_market_probability: float | None = None
    recommended_edge: float | None = None
    recommended_expected_value: float | None = None
    has_recommendation: bool = False
    decision_reason_codes: tuple[str, ...] = ()


class ValueSummary(BaseModel):
    """JSON-serializable value assessment summary."""

    model_config = ConfigDict(frozen=True)

    model_version: str | None
    created_at_utc: str
    rows_received: int
    rows_assessed: int
    rows_with_valid_odds: int
    rows_with_invalid_odds: int
    rows_with_recommendations: int
    recommendation_rate: float
    recommendations_player_1: int
    recommendations_player_2: int
    average_recommended_probability: float | None = None
    average_recommended_odds: float | None = None
    average_recommended_market_probability: float | None = None
    average_recommended_edge: float | None = None
    average_recommended_expected_value: float | None = None
    recommendations_by_surface: dict[str, int] = Field(default_factory=dict)
    recommendations_by_partition: dict[str, int] = Field(default_factory=dict)
    skip_reason_counts: dict[str, int] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ValueResult(BaseModel):
    """Flattened assessment rows and summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    assessments: pd.DataFrame
    summary: ValueSummary


@dataclass(frozen=True)
class ValueOutputPaths:
    """Output paths for Task 9 artifacts."""

    output: Path
    summary: Path


def validate_model_probability(value: Any) -> float:
    """Return a valid model probability or raise ValueError."""
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        msg = "model probability must be numeric"
        raise ValueError(msg) from exc
    if not np.isfinite(probability):
        msg = "model probability must be finite"
        raise ValueError(msg)
    if probability < 0.0 or probability > 1.0:
        msg = "model probability must be between 0 and 1"
        raise ValueError(msg)
    return probability


def validate_probability_pair(
    player_1_probability: Any,
    player_2_probability: Any,
) -> tuple[float, float]:
    """Validate a two-player probability pair."""
    probability_1 = validate_model_probability(player_1_probability)
    probability_2 = validate_model_probability(player_2_probability)
    if not np.isclose(probability_1 + probability_2, 1.0, atol=PROBABILITY_SUM_TOLERANCE):
        msg = "model probabilities must sum to 1"
        raise ValueError(msg)
    return probability_1, probability_2


def calculate_edge(model_probability: float, market_probability: float) -> float:
    """Return model probability minus no-vig market probability."""
    return validate_model_probability(model_probability) - validate_model_probability(
        market_probability
    )


def calculate_expected_value(model_probability: float, decimal_odds: float) -> float:
    """Return theoretical expected value per one unit staked."""
    return validate_model_probability(model_probability) * validate_decimal_odds(decimal_odds) - 1.0


def assess_selection(
    *,
    match_id: str | None,
    selection_side: SelectionSide,
    selection_player: str | None,
    model_probability: float | None,
    market_probability: float | None,
    raw_implied_probability: float | None,
    decimal_odds: float | None,
    overround: float | None,
    thresholds: ValueThresholds,
    base_reason_codes: tuple[str, ...] = (),
) -> SelectionAssessment:
    """Assess one player side against fixed value thresholds."""
    reason_codes = list(base_reason_codes)
    edge: float | None = None
    expected_value: float | None = None
    if model_probability is None or market_probability is None or decimal_odds is None:
        reason_codes.append("unsupported_match")
    else:
        edge = calculate_edge(model_probability, market_probability)
        expected_value = calculate_expected_value(model_probability, decimal_odds)
        if model_probability < thresholds.min_model_probability:
            reason_codes.append("probability_below_threshold")
        if edge < thresholds.min_edge:
            reason_codes.append("edge_below_threshold")
        if expected_value < thresholds.min_expected_value:
            reason_codes.append("expected_value_below_threshold")
        if decimal_odds < thresholds.min_odds:
            reason_codes.append("odds_below_minimum")
        if decimal_odds > thresholds.max_odds:
            reason_codes.append("odds_above_maximum")

    normalized_reasons = _normalize_reasons(reason_codes)
    return SelectionAssessment(
        match_id=match_id,
        selection_side=selection_side,
        selection_player=selection_player,
        model_probability=model_probability,
        decimal_odds=decimal_odds,
        raw_implied_probability=raw_implied_probability,
        market_probability=market_probability,
        overround=overround,
        edge=edge,
        expected_value=expected_value,
        eligible=not normalized_reasons,
        reason_codes=normalized_reasons,
    )


def assess_match_value(
    prediction: pd.Series | dict[str, Any],
    thresholds: ValueThresholds | None = None,
) -> MatchValueAssessment:
    """Assess both players and return at most one theoretical selection."""
    active_thresholds = thresholds or ValueThresholds()
    row = dict(prediction)
    match_id = _optional_text(row.get("match_id"))
    player_1 = _optional_text(row.get("player_1"))
    player_2 = _optional_text(row.get("player_2"))
    surface = _optional_text(row.get("surface"))
    base_reasons = _base_reason_codes(row, match_id, player_1, player_2, surface)

    market: NoVigMarket | None = None
    odds_reasons: list[str] = []
    try:
        market = remove_two_way_margin(row.get("player_1_odds"), row.get("player_2_odds"))
    except ValueError:
        odds_reasons.append(_odds_reason(row.get("player_1_odds"), row.get("player_2_odds")))

    probability_1: float | None = None
    probability_2: float | None = None
    probability_reasons: list[str] = []
    try:
        probability_1, probability_2 = validate_probability_pair(
            row.get("predicted_player_1_probability"),
            row.get("predicted_player_2_probability"),
        )
    except ValueError:
        probability_reasons.append("invalid_probability")

    common_reasons = _normalize_reasons([*base_reasons, *odds_reasons, *probability_reasons])
    player_1_assessment = assess_selection(
        match_id=match_id,
        selection_side="player_1",
        selection_player=player_1,
        model_probability=probability_1,
        market_probability=market.player_1_market_probability if market else None,
        raw_implied_probability=market.raw_player_1_probability if market else None,
        decimal_odds=market.player_1_odds if market else None,
        overround=market.overround if market else None,
        thresholds=active_thresholds,
        base_reason_codes=common_reasons,
    )
    player_2_assessment = assess_selection(
        match_id=match_id,
        selection_side="player_2",
        selection_player=player_2,
        model_probability=probability_2,
        market_probability=market.player_2_market_probability if market else None,
        raw_implied_probability=market.raw_player_2_probability if market else None,
        decimal_odds=market.player_2_odds if market else None,
        overround=market.overround if market else None,
        thresholds=active_thresholds,
        base_reason_codes=common_reasons,
    )
    selected, decision_reasons = _select_recommendation(player_1_assessment, player_2_assessment)
    return MatchValueAssessment(
        match_id=match_id,
        match_date=_date_to_string(row.get("match_date")),
        partition=_optional_text(row.get("partition")),
        surface=surface,
        player_1=player_1,
        player_2=player_2,
        actual_player_1_won=(
            None if _is_missing(row.get("actual_player_1_won")) else _coerce_optional_bool(
                row.get("actual_player_1_won")
            )
        ),
        model_version=_optional_text(row.get("model_version")),
        odds_source=_optional_text(row.get("odds_source")),
        is_retirement=_coerce_optional_bool(row.get("is_retirement")),
        player_1_assessment=player_1_assessment,
        player_2_assessment=player_2_assessment,
        recommended_side=selected.selection_side if selected else "none",
        recommended_player=selected.selection_player if selected else None,
        recommended_probability=selected.model_probability if selected else None,
        recommended_odds=selected.decimal_odds if selected else None,
        recommended_market_probability=selected.market_probability if selected else None,
        recommended_edge=selected.edge if selected else None,
        recommended_expected_value=selected.expected_value if selected else None,
        has_recommendation=selected is not None,
        decision_reason_codes=decision_reasons,
    )


def assess_prediction_dataframe(
    predictions: pd.DataFrame,
    thresholds: ValueThresholds | None = None,
) -> pd.DataFrame:
    """Assess value for every prediction row without mutating input."""
    active_thresholds = thresholds or ValueThresholds()
    _validate_prediction_frame(predictions)
    rows = [
        _flatten_assessment(assess_match_value(row, active_thresholds))
        for _, row in predictions.copy(deep=True).iterrows()
    ]
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["match_date", "partition", "match_id"], kind="mergesort")
        result = result.reset_index(drop=True)
    return result


def build_value_summary(
    assessments: pd.DataFrame,
    *,
    thresholds: ValueThresholds,
    rows_received: int | None = None,
) -> ValueSummary:
    """Build a JSON-serializable value assessment summary."""
    rows_assessed = len(assessments)
    recommendation_rows = assessments[assessments["has_recommendation"].astype(bool)]
    rows_received_value = rows_assessed if rows_received is None else rows_received
    skip_counts: Counter[str] = Counter()
    for column in ("player_1_reason_codes", "player_2_reason_codes", "decision_reason_codes"):
        for value in assessments.get(column, pd.Series(dtype="string")):
            skip_counts.update(_split_reason_codes(value))

    return ValueSummary(
        model_version=_single_value(assessments, "model_version"),
        created_at_utc=datetime.now(UTC).isoformat(),
        rows_received=rows_received_value,
        rows_assessed=rows_assessed,
        rows_with_valid_odds=int(assessments["odds_valid"].sum()) if rows_assessed else 0,
        rows_with_invalid_odds=int((~assessments["odds_valid"]).sum()) if rows_assessed else 0,
        rows_with_recommendations=len(recommendation_rows),
        recommendation_rate=len(recommendation_rows) / rows_assessed if rows_assessed else 0.0,
        recommendations_player_1=int((recommendation_rows["recommended_side"] == "player_1").sum()),
        recommendations_player_2=int((recommendation_rows["recommended_side"] == "player_2").sum()),
        average_recommended_probability=_nullable_mean(
            recommendation_rows,
            "recommended_probability",
        ),
        average_recommended_odds=_nullable_mean(recommendation_rows, "recommended_odds"),
        average_recommended_market_probability=_nullable_mean(
            recommendation_rows,
            "recommended_market_probability",
        ),
        average_recommended_edge=_nullable_mean(recommendation_rows, "recommended_edge"),
        average_recommended_expected_value=_nullable_mean(
            recommendation_rows,
            "recommended_expected_value",
        ),
        recommendations_by_surface=_counts(recommendation_rows, "surface"),
        recommendations_by_partition=_counts(recommendation_rows, "partition"),
        skip_reason_counts=dict(sorted(skip_counts.items())),
        thresholds=_threshold_dict(thresholds),
        warnings=[],
    )


def assess_predictions_with_summary(
    predictions: pd.DataFrame,
    thresholds: ValueThresholds | None = None,
) -> ValueResult:
    """Assess predictions and return rows plus summary."""
    active_thresholds = thresholds or ValueThresholds()
    assessments = assess_prediction_dataframe(predictions, active_thresholds)
    summary = build_value_summary(
        assessments,
        thresholds=active_thresholds,
        rows_received=len(predictions),
    )
    return ValueResult(assessments=assessments, summary=summary)


def write_value_outputs(result: ValueResult, output_path: Path, summary_path: Path) -> None:
    """Write value assessments and summary artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.assessments.to_parquet(output_path, index=False)
    summary_path.write_text(result.summary.model_dump_json(indent=2), encoding="utf-8")


def _validate_prediction_frame(predictions: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        msg = f"missing required value input columns: {missing}"
        raise ValueError(msg)
    non_missing_ids = predictions["match_id"].dropna()
    if non_missing_ids.duplicated().any():
        msg = "duplicate match_id values are not allowed"
        raise ValueError(msg)
    try:
        for _, row in predictions.iterrows():
            validate_probability_pair(
                row["predicted_player_1_probability"],
                row["predicted_player_2_probability"],
            )
    except ValueError as exc:
        msg = f"invalid probability pair: {exc}"
        raise ValueError(msg) from exc


def _base_reason_codes(
    row: dict[str, Any],
    match_id: str | None,
    player_1: str | None,
    player_2: str | None,
    surface: str | None,
) -> list[str]:
    reasons: list[str] = []
    if match_id is None:
        reasons.append("missing_match_id")
    if player_1 is not None and player_2 is not None and player_1.casefold() == player_2.casefold():
        reasons.append("same_player")
    if surface not in SUPPORTED_SURFACES:
        reasons.append("unsupported_surface")
    if _coerce_optional_bool(row.get("is_retirement")):
        reasons.append("retirement")
    if _coerce_optional_bool(row.get("is_walkover")):
        reasons.append("walkover")
    if _optional_text(row.get("odds_source")) == "inconsistent":
        reasons.append("unsupported_match")
    return reasons


def _select_recommendation(
    player_1_assessment: SelectionAssessment,
    player_2_assessment: SelectionAssessment,
) -> tuple[SelectionAssessment | None, tuple[str, ...]]:
    eligible = [
        assessment
        for assessment in (player_1_assessment, player_2_assessment)
        if assessment.eligible
    ]
    if not eligible:
        reasons = _normalize_reasons(
            [*player_1_assessment.reason_codes, *player_2_assessment.reason_codes]
        )
        return None, reasons
    decision_reasons: list[str] = []
    if len(eligible) > 1:
        decision_reasons.append("multiple_eligible_selections")
    selected = sorted(eligible, key=_selection_sort_key)[0]
    return selected, _normalize_reasons(decision_reasons)


def _selection_sort_key(assessment: SelectionAssessment) -> tuple[float, float, str, str]:
    expected_value = assessment.expected_value if assessment.expected_value is not None else -np.inf
    edge = assessment.edge if assessment.edge is not None else -np.inf
    player = assessment.selection_player or ""
    return (-expected_value, -edge, player.casefold(), assessment.selection_side)


def _flatten_assessment(assessment: MatchValueAssessment) -> dict[str, Any]:
    player_1 = assessment.player_1_assessment
    player_2 = assessment.player_2_assessment
    return {
        "match_id": assessment.match_id,
        "match_date": assessment.match_date,
        "partition": assessment.partition,
        "surface": assessment.surface,
        "player_1": assessment.player_1,
        "player_2": assessment.player_2,
        "actual_player_1_won": assessment.actual_player_1_won,
        "model_version": assessment.model_version,
        "odds_source": assessment.odds_source,
        "overround": player_1.overround,
        "odds_valid": player_1.overround is not None,
        "player_1_model_probability": player_1.model_probability,
        "player_1_decimal_odds": player_1.decimal_odds,
        "player_1_raw_implied_probability": player_1.raw_implied_probability,
        "player_1_market_probability": player_1.market_probability,
        "player_1_edge": player_1.edge,
        "player_1_expected_value": player_1.expected_value,
        "player_1_eligible": player_1.eligible,
        "player_1_reason_codes": "|".join(player_1.reason_codes),
        "player_2_model_probability": player_2.model_probability,
        "player_2_decimal_odds": player_2.decimal_odds,
        "player_2_raw_implied_probability": player_2.raw_implied_probability,
        "player_2_market_probability": player_2.market_probability,
        "player_2_edge": player_2.edge,
        "player_2_expected_value": player_2.expected_value,
        "player_2_eligible": player_2.eligible,
        "player_2_reason_codes": "|".join(player_2.reason_codes),
        "recommended_side": assessment.recommended_side,
        "recommended_player": assessment.recommended_player,
        "recommended_odds": assessment.recommended_odds,
        "recommended_probability": assessment.recommended_probability,
        "recommended_market_probability": assessment.recommended_market_probability,
        "recommended_edge": assessment.recommended_edge,
        "recommended_expected_value": assessment.recommended_expected_value,
        "has_recommendation": assessment.has_recommendation,
        "is_retirement": assessment.is_retirement,
        "decision_reason_codes": "|".join(assessment.decision_reason_codes),
    }


def _odds_reason(player_1_odds: Any, player_2_odds: Any) -> str:
    if _is_missing(player_1_odds) or _is_missing(player_2_odds):
        return "missing_odds"
    return "invalid_odds"


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_bool(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "walkover", "retirement"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return False


def _date_to_string(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalize_reasons(reasons: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    unique = set(reasons)
    return tuple(reason for reason in REASON_ORDER if reason in unique)


def _split_reason_codes(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    return [reason for reason in str(value).split("|") if reason]


def _nullable_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _single_value(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    values = sorted({str(value) for value in frame[column].dropna().unique()})
    return values[0] if len(values) == 1 else ",".join(values)


def _threshold_dict(thresholds: ValueThresholds) -> dict[str, float]:
    return {
        "minimum_model_probability": thresholds.min_model_probability,
        "minimum_edge": thresholds.min_edge,
        "minimum_expected_value": thresholds.min_expected_value,
        "minimum_odds": thresholds.min_odds,
        "maximum_odds": thresholds.max_odds,
    }


__all__ = [
    "PROBABILITY_SUM_TOLERANCE",
    "REASON_ORDER",
    "SelectionAssessment",
    "MatchValueAssessment",
    "ValueOutputPaths",
    "ValueResult",
    "ValueSummary",
    "assess_match_value",
    "assess_prediction_dataframe",
    "assess_predictions_with_summary",
    "assess_selection",
    "build_value_summary",
    "calculate_edge",
    "calculate_expected_value",
    "validate_model_probability",
    "validate_probability_pair",
    "write_value_outputs",
]
