"""Chronological overall and surface Elo features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tennis_value.config import EloConfig

SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
SORT_COLUMNS = ["match_date", "tournament_normalized", "round", "match_id"]
REQUIRED_COLUMNS = (
    "match_id",
    "match_date",
    "surface",
    "tournament_normalized",
    "round",
    "player_1_normalized",
    "player_2_normalized",
    "player_1_won",
    "is_retirement",
)
ELO_COLUMNS = [
    "player_1_elo_before",
    "player_2_elo_before",
    "overall_elo_diff",
    "player_1_surface_elo_before",
    "player_2_surface_elo_before",
    "surface_elo_diff",
    "player_1_matches_before",
    "player_2_matches_before",
    "history_count_min",
    "elo_expected_player_1",
    "elo_update_applied",
]


class EloQualityReport(BaseModel):
    """JSON-serializable report for Elo feature generation."""

    model_config = ConfigDict(frozen=True)

    rows_received: int
    rows_returned: int
    eligible_updates: int
    skipped_retirements: int
    skipped_invalid_results: int
    players_seen: int
    matches_by_surface: dict[str, int] = Field(default_factory=dict)
    minimum_rating: float
    maximum_rating: float


class EloResult(BaseModel):
    """Elo-enriched match rows and quality report."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    matches: pd.DataFrame
    report: EloQualityReport


@dataclass
class EloState:
    """Mutable Elo state local to one feature-generation run."""

    config: EloConfig
    overall_ratings: dict[str, float] = field(default_factory=dict)
    surface_ratings: dict[str, dict[str, float]] = field(
        default_factory=lambda: {surface: {} for surface in SUPPORTED_SURFACES}
    )
    match_counts: dict[str, int] = field(default_factory=dict)

    def overall_rating(self, player: str) -> float:
        return self.overall_ratings.get(player, self.config.initial_rating)

    def surface_rating(self, player: str, surface: str) -> float:
        return self.surface_ratings[surface].get(player, self.config.initial_rating)

    def match_count(self, player: str) -> int:
        return self.match_counts.get(player, 0)

    def apply_update(self, player_1: str, player_2: str, surface: str, player_1_won: bool) -> None:
        actual_player_1 = 1.0 if player_1_won else 0.0
        new_p1, new_p2 = update_ratings(
            self.overall_rating(player_1),
            self.overall_rating(player_2),
            actual_player_1,
            self.config,
        )
        self.overall_ratings[player_1] = new_p1
        self.overall_ratings[player_2] = new_p2

        new_surface_p1, new_surface_p2 = update_ratings(
            self.surface_rating(player_1, surface),
            self.surface_rating(player_2, surface),
            actual_player_1,
            self.config,
        )
        self.surface_ratings[surface][player_1] = new_surface_p1
        self.surface_ratings[surface][player_2] = new_surface_p2
        self.match_counts[player_1] = self.match_count(player_1) + 1
        self.match_counts[player_2] = self.match_count(player_2) + 1


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Return Elo expected score for player A."""
    _validate_finite_rating(rating_a, "rating_a")
    _validate_finite_rating(rating_b, "rating_b")
    if not math.isfinite(scale) or scale <= 0:
        msg = "scale must be finite and greater than zero"
        raise ValueError(msg)
    return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale)))


def update_ratings(
    rating_a: float,
    rating_b: float,
    actual_a: float,
    config: EloConfig,
) -> tuple[float, float]:
    """Return updated ratings for a completed two-player match."""
    if actual_a not in {0.0, 1.0}:
        msg = "actual_a must be 0.0 or 1.0"
        raise ValueError(msg)
    expected_a = expected_score(rating_a, rating_b, config.elo_scale)
    expected_b = 1.0 - expected_a
    actual_b = 1.0 - actual_a
    k_factor = float(config.k_factor)
    return (
        rating_a + k_factor * (actual_a - expected_a),
        rating_b + k_factor * (actual_b - expected_b),
    )


def add_elo_features(matches: pd.DataFrame, config: EloConfig | None = None) -> pd.DataFrame:
    """Add deterministic pre-match overall and surface Elo features."""
    active_config = config or EloConfig()
    _validate_config(active_config)
    _validate_input(matches)

    sorted_matches = matches.copy()
    sorted_matches["match_date"] = pd.to_datetime(sorted_matches["match_date"], errors="raise")
    sorted_matches = sorted_matches.sort_values(
        SORT_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)

    state = EloState(active_config)
    output_groups: list[pd.DataFrame] = []

    for _, day_group in sorted_matches.groupby("match_date", sort=False, dropna=False):
        snapshots = day_group.copy()
        updates: list[tuple[str, str, str, bool]] = []

        for index, row in day_group.iterrows():
            player_1 = str(row["player_1_normalized"])
            player_2 = str(row["player_2_normalized"])
            surface = str(row["surface"])
            player_1_won = _coerce_bool(row["player_1_won"], "player_1_won")
            is_retirement = _coerce_bool(row["is_retirement"], "is_retirement")

            p1_overall = state.overall_rating(player_1)
            p2_overall = state.overall_rating(player_2)
            p1_surface = state.surface_rating(player_1, surface)
            p2_surface = state.surface_rating(player_2, surface)
            p1_count = state.match_count(player_1)
            p2_count = state.match_count(player_2)
            expected_p1 = expected_score(p1_overall, p2_overall, active_config.elo_scale)
            should_update = not is_retirement

            snapshots.loc[index, "player_1_elo_before"] = p1_overall
            snapshots.loc[index, "player_2_elo_before"] = p2_overall
            snapshots.loc[index, "overall_elo_diff"] = p1_overall - p2_overall
            snapshots.loc[index, "player_1_surface_elo_before"] = p1_surface
            snapshots.loc[index, "player_2_surface_elo_before"] = p2_surface
            snapshots.loc[index, "surface_elo_diff"] = p1_surface - p2_surface
            snapshots.loc[index, "player_1_matches_before"] = p1_count
            snapshots.loc[index, "player_2_matches_before"] = p2_count
            snapshots.loc[index, "history_count_min"] = min(p1_count, p2_count)
            snapshots.loc[index, "elo_expected_player_1"] = expected_p1
            snapshots.loc[index, "elo_update_applied"] = should_update

            if should_update:
                updates.append((player_1, player_2, surface, player_1_won))

        output_groups.append(snapshots)
        for player_1, player_2, surface, player_1_won in updates:
            state.apply_update(player_1, player_2, surface, player_1_won)

    enriched = pd.concat(output_groups, ignore_index=True) if output_groups else sorted_matches
    return _coerce_elo_dtypes(enriched)


def build_elo_report(matches_with_elo: pd.DataFrame) -> EloQualityReport:
    """Build a compact report for Elo-enriched matches."""
    rating_columns = [
        "player_1_elo_before",
        "player_2_elo_before",
        "player_1_surface_elo_before",
        "player_2_surface_elo_before",
    ]
    ratings = pd.concat([matches_with_elo[column] for column in rating_columns], ignore_index=True)
    players = set(matches_with_elo["player_1_normalized"]) | set(
        matches_with_elo["player_2_normalized"]
    )
    return EloQualityReport(
        rows_received=len(matches_with_elo),
        rows_returned=len(matches_with_elo),
        eligible_updates=int(matches_with_elo["elo_update_applied"].sum()),
        skipped_retirements=int(
            (matches_with_elo["is_retirement"] & ~matches_with_elo["elo_update_applied"]).sum()
        ),
        skipped_invalid_results=0,
        players_seen=len(players),
        matches_by_surface={
            str(key): int(value)
            for key, value in matches_with_elo["surface"].value_counts().sort_index().items()
        },
        minimum_rating=float(ratings.min()),
        maximum_rating=float(ratings.max()),
    )


def add_elo_features_with_report(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
) -> EloResult:
    """Add Elo features and return a quality report."""
    enriched = add_elo_features(matches, config)
    return EloResult(matches=enriched, report=build_elo_report(enriched))


def write_elo_outputs(result: EloResult, output_path: Path, report_path: Path) -> None:
    """Write Elo-enriched matches and report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.matches.to_parquet(output_path, index=False)
    report_path.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")


def elo_report_to_json(report: EloQualityReport) -> str:
    """Serialize an Elo report as formatted JSON."""
    return json.dumps(report.model_dump(mode="json"), indent=2)


def _validate_input(matches: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing:
        msg = f"missing required Elo input columns: {missing}"
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
        msg = f"unsupported surface values for Elo: {invalid_surfaces}"
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


def _validate_config(config: EloConfig) -> None:
    _validate_finite_rating(config.initial_rating, "initial_rating")
    if not math.isfinite(config.k_factor) or config.k_factor <= 0:
        msg = "k_factor must be finite and greater than zero"
        raise ValueError(msg)
    if not math.isfinite(config.elo_scale) or config.elo_scale <= 0:
        msg = "elo_scale must be finite and greater than zero"
        raise ValueError(msg)


def _validate_finite_rating(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        msg = f"{field_name} must be finite"
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


def _coerce_elo_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    float_columns = [
        "player_1_elo_before",
        "player_2_elo_before",
        "overall_elo_diff",
        "player_1_surface_elo_before",
        "player_2_surface_elo_before",
        "surface_elo_diff",
        "elo_expected_player_1",
    ]
    int_columns = ["player_1_matches_before", "player_2_matches_before", "history_count_min"]
    for column in float_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype("float64")
    for column in int_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype("int64")
    coerced["elo_update_applied"] = coerced["elo_update_applied"].astype("bool")
    return coerced


__all__ = [
    "ELO_COLUMNS",
    "EloQualityReport",
    "EloResult",
    "add_elo_features",
    "add_elo_features_with_report",
    "build_elo_report",
    "elo_report_to_json",
    "expected_score",
    "update_ratings",
    "write_elo_outputs",
]
