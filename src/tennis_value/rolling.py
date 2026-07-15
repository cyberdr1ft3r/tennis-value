"""Chronological rolling pre-match player features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tennis_value.config import RollingFeatureConfig

SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
SORT_COLUMNS = ["match_date", "tournament_normalized", "round", "match_id"]
REQUIRED_COLUMNS = (
    "match_id",
    "match_date",
    "tournament_normalized",
    "round",
    "surface",
    "player_1_normalized",
    "player_2_normalized",
    "player_1_won",
    "is_retirement",
)
ROLLING_COLUMNS = [
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
    "recent_10_win_rate_diff",
    "surface_recent_10_win_rate_diff",
    "days_since_last_match_diff",
    "matches_last_14d_diff",
]


@dataclass
class PlayerHistory:
    """Eligible-match history for one player."""

    recent_results: deque[float] = field(default_factory=deque)
    surface_results: dict[str, deque[float]] = field(
        default_factory=lambda: {surface: deque() for surface in SUPPORTED_SURFACES}
    )
    match_dates: deque[pd.Timestamp] = field(default_factory=deque)
    prior_match_count: int = 0
    last_match_date: pd.Timestamp | None = None


@dataclass(frozen=True)
class PlayerSnapshot:
    """Pre-match rolling values for one player."""

    prior_match_count: int
    recent_win_rate: float
    surface_recent_win_rate: float
    days_since_last_match: int | None
    matches_last_14d: int
    surface_history_count: int


@dataclass
class RollingState:
    """Mutable rolling state local to one feature-generation run."""

    config: RollingFeatureConfig
    histories: dict[str, PlayerHistory] = field(default_factory=dict)

    def history(self, player: str) -> PlayerHistory:
        if player not in self.histories:
            self.histories[player] = PlayerHistory()
        return self.histories[player]

    def snapshot(self, player: str, surface: str, match_date: pd.Timestamp) -> PlayerSnapshot:
        history = self.history(player)
        surface_results = history.surface_results[surface]
        days_since_last_match = (
            None
            if history.last_match_date is None
            else int((match_date - history.last_match_date).days)
        )
        return PlayerSnapshot(
            prior_match_count=history.prior_match_count,
            recent_win_rate=_win_rate(history.recent_results, self.config.neutral_win_rate),
            surface_recent_win_rate=_win_rate(surface_results, self.config.neutral_win_rate),
            days_since_last_match=days_since_last_match,
            matches_last_14d=_matches_in_schedule_window(
                history.match_dates,
                match_date,
                self.config.schedule_window_days,
            ),
            surface_history_count=len(surface_results),
        )

    def apply_update(
        self,
        player_1: str,
        player_2: str,
        surface: str,
        match_date: pd.Timestamp,
        player_1_won: bool,
    ) -> None:
        p1_result = 1.0 if player_1_won else 0.0
        p2_result = 1.0 - p1_result
        self._apply_player_update(player_1, surface, match_date, p1_result)
        self._apply_player_update(player_2, surface, match_date, p2_result)

    def _apply_player_update(
        self,
        player: str,
        surface: str,
        match_date: pd.Timestamp,
        result: float,
    ) -> None:
        history = self.history(player)
        _append_limited(history.recent_results, result, self.config.rolling_window)
        _append_limited(history.surface_results[surface], result, self.config.rolling_window)
        history.match_dates.append(match_date)
        history.prior_match_count += 1
        history.last_match_date = match_date


def add_rolling_features(
    matches: pd.DataFrame,
    config: RollingFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add chronological pre-match rolling player features.

    Same-date rows are snapshotted from the state available before that date because the
    source data does not include reliable intra-day match times.
    """
    active_config = config or RollingFeatureConfig()
    _validate_input(matches)

    sorted_matches = matches.copy(deep=True)
    sorted_matches["match_date"] = pd.to_datetime(sorted_matches["match_date"], errors="raise")
    sorted_matches = sorted_matches.sort_values(
        SORT_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)

    state = RollingState(active_config)
    output_groups: list[pd.DataFrame] = []

    for _, day_group in sorted_matches.groupby("match_date", sort=False, dropna=False):
        snapshots = day_group.copy()
        updates: list[tuple[str, str, str, pd.Timestamp, bool]] = []
        current_date = pd.Timestamp(day_group["match_date"].iloc[0]).normalize()

        for index, row in day_group.iterrows():
            player_1 = str(row["player_1_normalized"])
            player_2 = str(row["player_2_normalized"])
            surface = str(row["surface"])
            player_1_won = _coerce_bool(row["player_1_won"], "player_1_won")
            is_retirement = _coerce_bool(row["is_retirement"], "is_retirement")

            player_1_snapshot = state.snapshot(player_1, surface, current_date)
            player_2_snapshot = state.snapshot(player_2, surface, current_date)
            _assign_snapshot(snapshots, index, "player_1", player_1_snapshot)
            _assign_snapshot(snapshots, index, "player_2", player_2_snapshot)
            snapshots.loc[index, "surface_history_count_min"] = min(
                player_1_snapshot.surface_history_count,
                player_2_snapshot.surface_history_count,
            )
            snapshots.loc[index, "recent_10_win_rate_diff"] = (
                player_1_snapshot.recent_win_rate - player_2_snapshot.recent_win_rate
            )
            snapshots.loc[index, "surface_recent_10_win_rate_diff"] = (
                player_1_snapshot.surface_recent_win_rate
                - player_2_snapshot.surface_recent_win_rate
            )
            snapshots.loc[index, "days_since_last_match_diff"] = _nullable_difference(
                player_1_snapshot.days_since_last_match,
                player_2_snapshot.days_since_last_match,
            )
            snapshots.loc[index, "matches_last_14d_diff"] = (
                player_1_snapshot.matches_last_14d - player_2_snapshot.matches_last_14d
            )

            if not is_retirement:
                updates.append((player_1, player_2, surface, current_date, player_1_won))

        output_groups.append(snapshots)
        for player_1, player_2, surface, update_date, player_1_won in updates:
            state.apply_update(player_1, player_2, surface, update_date, player_1_won)

    enriched = pd.concat(output_groups, ignore_index=True) if output_groups else sorted_matches
    return _coerce_rolling_dtypes(enriched)


def _assign_snapshot(
    frame: pd.DataFrame,
    index: Any,
    player_prefix: str,
    snapshot: PlayerSnapshot,
) -> None:
    frame.loc[index, f"{player_prefix}_prior_match_count"] = snapshot.prior_match_count
    frame.loc[index, f"{player_prefix}_recent_10_win_rate"] = snapshot.recent_win_rate
    frame.loc[index, f"{player_prefix}_surface_recent_10_win_rate"] = (
        snapshot.surface_recent_win_rate
    )
    frame.loc[index, f"{player_prefix}_days_since_last_match"] = snapshot.days_since_last_match
    frame.loc[index, f"{player_prefix}_matches_last_14d"] = snapshot.matches_last_14d
    frame.loc[index, f"{player_prefix}_surface_history_count"] = snapshot.surface_history_count


def _append_limited(values: deque[float], value: float, max_length: int) -> None:
    values.append(value)
    while len(values) > max_length:
        values.popleft()


def _win_rate(results: deque[float], neutral_win_rate: float) -> float:
    if not results:
        return neutral_win_rate
    return float(sum(results) / len(results))


def _matches_in_schedule_window(
    match_dates: deque[pd.Timestamp],
    current_date: pd.Timestamp,
    schedule_window_days: int,
) -> int:
    start_date = current_date - pd.Timedelta(days=schedule_window_days)
    return sum(start_date <= match_date < current_date for match_date in match_dates)


def _nullable_difference(first: int | None, second: int | None) -> float | None:
    if first is None or second is None:
        return None
    return float(first - second)


def _validate_input(matches: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing:
        msg = f"missing required rolling input columns: {missing}"
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
        msg = f"unsupported surface values for rolling features: {invalid_surfaces}"
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


def _coerce_rolling_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    float_columns = [
        "player_1_recent_10_win_rate",
        "player_2_recent_10_win_rate",
        "player_1_surface_recent_10_win_rate",
        "player_2_surface_recent_10_win_rate",
        "recent_10_win_rate_diff",
        "surface_recent_10_win_rate_diff",
        "days_since_last_match_diff",
    ]
    nullable_float_columns = [
        "player_1_days_since_last_match",
        "player_2_days_since_last_match",
    ]
    int_columns = [
        "player_1_prior_match_count",
        "player_2_prior_match_count",
        "player_1_matches_last_14d",
        "player_2_matches_last_14d",
        "player_1_surface_history_count",
        "player_2_surface_history_count",
        "surface_history_count_min",
        "matches_last_14d_diff",
    ]
    for column in float_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype("Float64")
    for column in nullable_float_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype("Float64")
    for column in int_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype("int64")
    return coerced


__all__ = [
    "ROLLING_COLUMNS",
    "RollingState",
    "add_rolling_features",
]
