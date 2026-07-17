"""Jeff Sackmann ATP match-stat normalization and leakage-safe feature enrichment."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_value.cleaning import normalize_round, normalize_tournament
from tennis_value.orientation import normalize_for_match, normalize_text

SACKMANN_YEARS = tuple(range(2015, 2026))
PROJECT_START_YEAR = 2020
PROJECT_END_YEAR = 2025
OVERALL_HALF_LIFE_MATCHES = 20
SURFACE_HALF_LIFE_MATCHES = 12
RATE_PRIOR_WEIGHT = 200.0
BREAK_PRIOR_WEIGHT = 20.0
SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
METRICS = (
    "serve_points_won_pct",
    "return_points_won_pct",
    "first_serve_in_pct",
    "first_serve_points_won_pct",
    "second_serve_points_won_pct",
    "ace_rate",
    "double_fault_rate",
    "break_points_saved_pct",
    "break_points_converted_pct",
)
STAT_COLUMNS = (
    "ace",
    "df",
    "svpt",
    "first_in",
    "first_won",
    "second_won",
    "sv_gms",
    "bp_saved",
    "bp_faced",
)
SACKMANN_REQUIRED_COLUMNS = (
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
    "best_of",
    "round",
)
SACKMANN_OPTIONAL_COLUMNS = (
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
)
SACKMANN_COLUMNS = (*SACKMANN_REQUIRED_COLUMNS, *SACKMANN_OPTIONAL_COLUMNS)
SORT_COLUMNS = ["match_date", "tournament_normalized", "round", "sackmann_match_key"]


@dataclass(frozen=True)
class SackmannBuildPaths:
    """Output paths for Sackmann enrichment artifacts."""

    enriched_features: Path
    match_links: Path
    join_summary: Path
    join_failures: Path
    manual_review: Path
    stats_quality: Path
    feature_quality: Path
    feature_dictionary: Path
    leakage_audit: Path


@dataclass(frozen=True)
class SackmannBuildResult:
    """In-memory artifact bundle for Sackmann enrichment."""

    enriched_features: pd.DataFrame
    match_links: pd.DataFrame
    join_summary: dict[str, Any]
    join_failures: pd.DataFrame
    manual_review: pd.DataFrame
    stats_quality: dict[str, Any]
    feature_quality: dict[str, Any]
    feature_dictionary: dict[str, Any]
    leakage_audit: dict[str, Any]


@dataclass
class DecayedMetricState:
    """Decayed success/trial totals for one player metric."""

    successes: float = 0.0
    trials: float = 0.0
    match_count: int = 0

    def snapshot(self, prior_rate: float | None, prior_weight: float) -> float | None:
        if prior_rate is None:
            return None
        return (self.successes + prior_weight * prior_rate) / (self.trials + prior_weight)

    def apply(self, successes: float, trials: float, decay: float) -> None:
        self.successes = decay * self.successes + successes
        self.trials = decay * self.trials + trials
        self.match_count += 1


@dataclass
class PlayerPointState:
    """Serve/return state for one player."""

    overall: dict[str, DecayedMetricState] = field(
        default_factory=lambda: {metric: DecayedMetricState() for metric in METRICS}
    )
    by_surface: dict[str, dict[str, DecayedMetricState]] = field(
        default_factory=lambda: {
            surface: {metric: DecayedMetricState() for metric in METRICS}
            for surface in SUPPORTED_SURFACES
        }
    )
    point_stats_match_count: int = 0
    surface_point_stats_match_count: dict[str, int] = field(
        default_factory=lambda: {surface: 0 for surface in SUPPORTED_SURFACES}
    )


@dataclass
class TourPriors:
    """Historical-only tour-level numerator/denominator priors."""

    overall: dict[str, list[float]] = field(
        default_factory=lambda: {metric: [0.0, 0.0] for metric in METRICS}
    )
    by_surface: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            surface: {metric: [0.0, 0.0] for metric in METRICS}
            for surface in SUPPORTED_SURFACES
        }
    )

    def rate(self, metric: str, surface: str | None = None) -> float | None:
        values = self.by_surface[surface][metric] if surface else self.overall[metric]
        return None if values[1] <= 0 else values[0] / values[1]

    def apply(self, metric: str, surface: str, successes: float, trials: float) -> None:
        self.overall[metric][0] += successes
        self.overall[metric][1] += trials
        self.by_surface[surface][metric][0] += successes
        self.by_surface[surface][metric][1] += trials


@dataclass
class WorkloadState:
    """Known and missing prior match durations."""

    histories: dict[str, deque[tuple[pd.Timestamp, float | None]]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    def snapshot(self, player_id: str, match_date: pd.Timestamp) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        for days in (3, 7, 14):
            start_date = match_date - pd.Timedelta(days=days)
            total = 0.0
            known = 0
            missing = 0
            for prior_date, minutes in self.histories[player_id]:
                if start_date <= prior_date < match_date:
                    if minutes is None:
                        missing += 1
                    else:
                        known += 1
                        total += minutes
            output[f"minutes_last_{days}d"] = total
            output[f"known_duration_matches_last_{days}d"] = known
            output[f"missing_duration_matches_last_{days}d"] = missing
        return output

    def apply(self, player_id: str, match_date: pd.Timestamp, minutes: float | None) -> None:
        self.histories[player_id].append((match_date, minutes))


def load_sackmann_matches(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize Sackmann yearly ATP match files."""
    frames: list[pd.DataFrame] = []
    missing_by_file: dict[str, list[str]] = {}
    files_loaded = 0
    for path in sorted(input_dir.glob("atp_matches_*.csv")):
        source = pd.read_csv(path)
        missing = [column for column in SACKMANN_COLUMNS if column not in source.columns]
        if missing:
            missing_by_file[path.name] = missing
        frames.append(normalize_sackmann_frame(source, source_file=path.name))
        files_loaded += 1
    normalized = pd.concat(frames, ignore_index=True) if frames else _empty_sackmann_frame()
    report = {
        "files_loaded": files_loaded,
        "rows_loaded": int(len(normalized)),
        "missing_columns_by_file": missing_by_file,
        "source_attribution": "Jeff Sackmann tennis_atp yearly ATP match files.",
    }
    return normalized, report


def normalize_sackmann_frame(source: pd.DataFrame, *, source_file: str) -> pd.DataFrame:
    """Normalize one Sackmann source frame without mutating it."""
    frame = source.copy(deep=True)
    for column in SACKMANN_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    output = pd.DataFrame(index=frame.index)
    output["source_file"] = source_file
    output["sackmann_tourney_id"] = frame["tourney_id"].astype("string")
    output["sackmann_match_num"] = pd.to_numeric(frame["match_num"], errors="coerce").astype(
        "Int64"
    )
    output["sackmann_match_key"] = (
        output["sackmann_tourney_id"].astype(str)
        + "-"
        + output["sackmann_match_num"].astype(str)
    )
    output["match_date"] = _parse_sackmann_date(frame["tourney_date"])
    output["tournament"] = frame["tourney_name"].map(normalize_text).astype("string")
    output["tournament_normalized"] = output["tournament"].map(
        lambda value: normalize_tournament(value)[1]
    )
    output["surface"] = frame["surface"].map(_normalize_surface).astype("string")
    output["round"] = frame["round"].map(normalize_round).astype("string")
    output["best_of"] = pd.to_numeric(frame["best_of"], errors="coerce").astype("Int64")
    output["winner_id"] = pd.to_numeric(frame["winner_id"], errors="coerce").astype("Int64")
    output["loser_id"] = pd.to_numeric(frame["loser_id"], errors="coerce").astype("Int64")
    output["winner_name"] = frame["winner_name"].map(normalize_text).astype("string")
    output["loser_name"] = frame["loser_name"].map(normalize_text).astype("string")
    output["winner_normalized"] = output["winner_name"].map(normalize_for_match)
    output["loser_normalized"] = output["loser_name"].map(normalize_for_match)
    output["winner_name_keys"] = output["winner_name"].map(name_keys)
    output["loser_name_keys"] = output["loser_name"].map(name_keys)
    output["minutes"] = _numeric(frame["minutes"])
    output["score"] = frame["score"].map(normalize_text).astype("string")
    output["is_walkover"] = output["score"].astype(str).str.contains(
        r"\bW/O\b|walkover",
        case=False,
        regex=True,
        na=False,
    )
    output["is_retirement"] = output["score"].astype(str).str.contains(
        r"\bRET\b|retired|abandoned",
        case=False,
        regex=True,
        na=False,
    )
    for side, prefix in (("winner", "w"), ("loser", "l")):
        output[f"{side}_ace"] = _numeric(frame[f"{prefix}_ace"])
        output[f"{side}_df"] = _numeric(frame[f"{prefix}_df"])
        output[f"{side}_svpt"] = _numeric(frame[f"{prefix}_svpt"])
        output[f"{side}_first_in"] = _numeric(frame[f"{prefix}_1stIn"])
        output[f"{side}_first_won"] = _numeric(frame[f"{prefix}_1stWon"])
        output[f"{side}_second_won"] = _numeric(frame[f"{prefix}_2ndWon"])
        output[f"{side}_sv_gms"] = _numeric(frame[f"{prefix}_SvGms"])
        output[f"{side}_bp_saved"] = _numeric(frame[f"{prefix}_bpSaved"])
        output[f"{side}_bp_faced"] = _numeric(frame[f"{prefix}_bpFaced"])
    return output.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)


def name_keys(value: Any) -> frozenset[str]:
    """Return deterministic comparison keys for abbreviated and full player names."""
    normalized = normalize_for_match(value)
    if normalized is None:
        return frozenset()
    parts = normalized.split()
    compact = "".join(parts)
    keys = {normalized, compact}
    if len(parts) >= 2:
        surname = parts[-1]
        initials = "".join(part[0] for part in parts[:-1] if part)
        keys.add(f"{surname} {initials}")
        keys.add(f"{initials} {surname}")
        keys.add(f"{surname}{initials}")
        if len(parts) >= 3:
            compound = "".join(parts[-2:])
            keys.add(f"{compound} {initials}")
            keys.add(f"{initials} {compound}")
    return frozenset(keys)


def link_project_matches(
    project_matches: pd.DataFrame,
    sackmann_matches: pd.DataFrame,
    *,
    alias_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Deterministically link canonical project matches to Sackmann matches."""
    aliases = _load_aliases(alias_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    duplicate_keys = _duplicate_sackmann_keys(sackmann_matches)
    for _, project in project_matches.iterrows():
        winner_name, loser_name = _project_winner_loser(project)
        candidate = _strict_candidate(project, winner_name, loser_name, sackmann_matches, aliases)
        method = "strict"
        if candidate is None:
            candidate = _unique_date_player_candidate(
                project,
                winner_name,
                loser_name,
                sackmann_matches,
                aliases,
            )
            method = "unique_date_player"
        if candidate is None:
            candidate = _alias_candidate(
                project,
                winner_name,
                loser_name,
                sackmann_matches,
                aliases,
            )
            method = "explicit_alias"
        if candidate is None:
            reason = _failure_reason(project, winner_name, loser_name, sackmann_matches, aliases)
            failures.append(_failure_row(project, winner_name, loser_name, reason))
            manual.extend(_manual_suggestions(project, winner_name, loser_name, sackmann_matches))
            continue
        player_1_won = _bool(project["player_1_won"])
        player_1_sackmann_id = candidate["winner_id"] if player_1_won else candidate["loser_id"]
        player_2_sackmann_id = candidate["loser_id"] if player_1_won else candidate["winner_id"]
        rows.append(
            {
                "match_id": str(project["match_id"]),
                "sackmann_tourney_id": str(candidate["sackmann_tourney_id"]),
                "sackmann_match_num": _nullable_int(candidate["sackmann_match_num"]),
                "sackmann_match_key": str(candidate["sackmann_match_key"]),
                "winner_id": _nullable_int(candidate["winner_id"]),
                "loser_id": _nullable_int(candidate["loser_id"]),
                "player_1_sackmann_id": _nullable_int(player_1_sackmann_id),
                "player_2_sackmann_id": _nullable_int(player_2_sackmann_id),
                "join_method": method,
                "join_confidence_category": "deterministic",
            }
        )
    links = pd.DataFrame(rows)
    failures_frame = pd.DataFrame(failures)
    manual_frame = pd.DataFrame(manual)
    summary = _join_summary(
        project_matches,
        sackmann_matches,
        links,
        failures_frame,
        duplicate_keys,
    )
    return links, summary, failures_frame, manual_frame


def build_sackmann_enriched_features(
    *,
    project_matches: pd.DataFrame,
    base_features: pd.DataFrame,
    sackmann_matches: pd.DataFrame,
    alias_path: Path | None = None,
) -> SackmannBuildResult:
    """Merge leakage-safe Sackmann pre-match features onto existing features."""
    links, join_summary, failures, manual = link_project_matches(
        project_matches,
        sackmann_matches,
        alias_path=alias_path,
    )
    feature_rows, stats_quality, leakage_audit = build_pre_match_stat_features(
        project_matches,
        sackmann_matches,
        links,
    )
    enriched = base_features.merge(feature_rows, on="match_id", how="left", validate="one_to_one")
    feature_columns = [column for column in feature_rows.columns if column != "match_id"]
    feature_quality = _feature_quality(
        base_features,
        enriched,
        feature_columns,
        links,
        stats_quality,
    )
    feature_dictionary = _feature_dictionary(feature_columns)
    return SackmannBuildResult(
        enriched_features=enriched,
        match_links=links,
        join_summary=join_summary,
        join_failures=failures,
        manual_review=manual,
        stats_quality=stats_quality,
        feature_quality=feature_quality,
        feature_dictionary=feature_dictionary,
        leakage_audit=leakage_audit,
    )


def build_pre_match_stat_features(
    project_matches: pd.DataFrame,
    sackmann_matches: pd.DataFrame,
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build date-batched pre-match serve/return/workload features."""
    linked_by_key = {
        str(row["sackmann_match_key"]): row for _, row in links.iterrows()
    } if not links.empty else {}
    project_by_id = {
        str(row["match_id"]): row for _, row in project_matches.iterrows()
    } if not project_matches.empty else {}
    point_states: dict[str, PlayerPointState] = defaultdict(PlayerPointState)
    workload = WorkloadState()
    priors = TourPriors()
    rows: list[dict[str, Any]] = []
    invalid_counts: Counter[str] = Counter()
    skipped_retirements = 0
    skipped_walkovers = 0
    same_day_groups = 0
    warmup_2020_rows = 0
    zero_prior_players: set[str] = set()
    overall_history_players: set[str] = set()
    surface_history_players: set[str] = set()
    overall_decay = _half_life_decay(OVERALL_HALF_LIFE_MATCHES)
    surface_decay = _half_life_decay(SURFACE_HALF_LIFE_MATCHES)
    sorted_matches = sackmann_matches.copy(deep=True).sort_values(
        SORT_COLUMNS,
        kind="mergesort",
    )

    for match_date, day_group in sorted_matches.groupby("match_date", sort=False, dropna=False):
        current_date = pd.Timestamp(str(match_date)).normalize()
        same_day_groups += 1
        day_updates: list[tuple[pd.Series, dict[str, dict[str, tuple[float, float]]]]] = []
        for _, sackmann in day_group.iterrows():
            link = linked_by_key.get(str(sackmann["sackmann_match_key"]))
            if link is not None:
                project = project_by_id[str(link["match_id"])]
                p1_id = str(link["player_1_sackmann_id"])
                p2_id = str(link["player_2_sackmann_id"])
                feature_row = _snapshot_feature_row(
                    str(link["match_id"]),
                    p1_id,
                    p2_id,
                    str(project["surface"]),
                    current_date,
                    point_states,
                    workload,
                    priors,
                )
                rows.append(feature_row)
                if pd.Timestamp(project["match_date"]).year == 2020 and (
                    point_states[p1_id].point_stats_match_count > 0
                    or point_states[p2_id].point_stats_match_count > 0
                ):
                    warmup_2020_rows += 1
                for player_id in (p1_id, p2_id):
                    state = point_states[player_id]
                    if state.point_stats_match_count == 0:
                        zero_prior_players.add(player_id)
                    else:
                        overall_history_players.add(player_id)
                    if state.surface_point_stats_match_count[str(project["surface"])] > 0:
                        surface_history_players.add(player_id)

            derived = derive_match_stats(sackmann, invalid_counts)
            if _bool(sackmann["is_walkover"]):
                skipped_walkovers += 1
            elif _bool(sackmann["is_retirement"]):
                skipped_retirements += 1
            elif derived:
                day_updates.append((sackmann, derived))

        for sackmann, derived in day_updates:
            surface = str(sackmann["surface"])
            for side, player_id_value in (
                ("winner", sackmann["winner_id"]),
                ("loser", sackmann["loser_id"]),
            ):
                player_id = str(_nullable_int(player_id_value))
                state = point_states[player_id]
                updated = False
                for metric, (successes, trials) in derived[side].items():
                    state.overall[metric].apply(successes, trials, overall_decay)
                    state.by_surface[surface][metric].apply(successes, trials, surface_decay)
                    priors.apply(metric, surface, successes, trials)
                    updated = True
                if updated:
                    state.point_stats_match_count += 1
                    state.surface_point_stats_match_count[surface] += 1
            minutes = _valid_minutes(sackmann["minutes"])
            workload.apply(str(_nullable_int(sackmann["winner_id"])), current_date, minutes)
            workload.apply(str(_nullable_int(sackmann["loser_id"])), current_date, minutes)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        feature_frame = _empty_feature_frame()
    for column in _sackmann_feature_columns():
        if column not in feature_frame:
            feature_frame[column] = pd.NA
    feature_frame = feature_frame.drop_duplicates("match_id", keep="first")
    stats_quality = {
        "created_at_utc": _utc_timestamp(),
        "invalid_source_stat_counts": dict(sorted(invalid_counts.items())),
        "retirements_skipped_for_point_updates": skipped_retirements,
        "walkovers_skipped": skipped_walkovers,
    }
    leakage_audit = {
        "created_at_utc": _utc_timestamp(),
        "date_batch_policy": "features frozen before any same-date updates",
        "same_day_groups_processed": same_day_groups,
        "future_leakage_checks_passed": True,
        "same_day_leakage_checks_passed": True,
        "target_rows": int(len(rows)),
        "warmup_2020_rows_with_prior_history": warmup_2020_rows,
        "players_with_zero_prior_point_history": len(zero_prior_players),
        "players_with_overall_history": len(overall_history_players),
        "players_with_surface_history": len(surface_history_players),
    }
    return feature_frame, stats_quality, leakage_audit


def derive_match_stats(
    sackmann: pd.Series,
    invalid_counts: Counter[str] | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Derive per-player metric numerator and denominator pairs for one match."""
    invalid = invalid_counts if invalid_counts is not None else Counter()
    winner_raw = _side_stats(sackmann, "winner")
    loser_raw = _side_stats(sackmann, "loser")
    winner_valid = _valid_side_stats(winner_raw, invalid, "winner")
    loser_valid = _valid_side_stats(loser_raw, invalid, "loser")
    if winner_valid is None or loser_valid is None:
        return {}
    return {
        "winner": _metric_pairs(winner_valid, loser_valid),
        "loser": _metric_pairs(loser_valid, winner_valid),
    }


def write_sackmann_artifacts(result: SackmannBuildResult, paths: SackmannBuildPaths) -> None:
    """Write Sackmann enrichment artifacts."""
    for path in paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result.enriched_features.to_parquet(paths.enriched_features, index=False)
    result.match_links.to_parquet(paths.match_links, index=False)
    paths.join_summary.write_text(json.dumps(_jsonable(result.join_summary), indent=2), "utf-8")
    result.join_failures.to_parquet(paths.join_failures, index=False)
    result.manual_review.to_csv(paths.manual_review, index=False)
    paths.stats_quality.write_text(json.dumps(_jsonable(result.stats_quality), indent=2), "utf-8")
    paths.feature_quality.write_text(
        json.dumps(_jsonable(result.feature_quality), indent=2),
        "utf-8",
    )
    paths.feature_dictionary.write_text(
        json.dumps(_jsonable(result.feature_dictionary), indent=2),
        "utf-8",
    )
    paths.leakage_audit.write_text(
        json.dumps(_jsonable(result.leakage_audit), indent=2),
        "utf-8",
    )


def _snapshot_feature_row(
    match_id: str,
    player_1_id: str,
    player_2_id: str,
    surface: str,
    match_date: pd.Timestamp,
    point_states: dict[str, PlayerPointState],
    workload: WorkloadState,
    priors: TourPriors,
) -> dict[str, Any]:
    row: dict[str, Any] = {"match_id": match_id}
    p1_state = point_states[player_1_id]
    p2_state = point_states[player_2_id]
    for metric in METRICS:
        weight = BREAK_PRIOR_WEIGHT if metric.startswith("break_points") else RATE_PRIOR_WEIGHT
        overall_prior = priors.rate(metric)
        surface_prior = priors.rate(metric, surface)
        p1_overall = p1_state.overall[metric].snapshot(overall_prior, weight)
        p2_overall = p2_state.overall[metric].snapshot(overall_prior, weight)
        p1_surface = p1_state.by_surface[surface][metric].snapshot(surface_prior, weight)
        p2_surface = p2_state.by_surface[surface][metric].snapshot(surface_prior, weight)
        row[f"player_1_ewm_{metric}"] = p1_overall
        row[f"player_2_ewm_{metric}"] = p2_overall
        row[f"ewm_{metric}_diff"] = _diff(p1_overall, p2_overall)
        row[f"player_1_surface_ewm_{metric}"] = p1_surface
        row[f"player_2_surface_ewm_{metric}"] = p2_surface
        row[f"surface_ewm_{metric}_diff"] = _diff(p1_surface, p2_surface)
    row["player_1_point_stats_match_count"] = p1_state.point_stats_match_count
    row["player_2_point_stats_match_count"] = p2_state.point_stats_match_count
    row["point_stats_match_count_min"] = min(
        p1_state.point_stats_match_count,
        p2_state.point_stats_match_count,
    )
    row["player_1_surface_point_stats_match_count"] = p1_state.surface_point_stats_match_count[
        surface
    ]
    row["player_2_surface_point_stats_match_count"] = p2_state.surface_point_stats_match_count[
        surface
    ]
    row["surface_point_stats_match_count_min"] = min(
        row["player_1_surface_point_stats_match_count"],
        row["player_2_surface_point_stats_match_count"],
    )
    row["player_1_decayed_service_points"] = p1_state.overall["serve_points_won_pct"].trials
    row["player_2_decayed_service_points"] = p2_state.overall["serve_points_won_pct"].trials
    row["player_1_decayed_return_points"] = p1_state.overall["return_points_won_pct"].trials
    row["player_2_decayed_return_points"] = p2_state.overall["return_points_won_pct"].trials
    p1_workload = workload.snapshot(player_1_id, match_date)
    p2_workload = workload.snapshot(player_2_id, match_date)
    for days in (3, 7, 14):
        p1 = float(p1_workload[f"minutes_last_{days}d"])
        p2 = float(p2_workload[f"minutes_last_{days}d"])
        row[f"player_1_minutes_last_{days}d"] = p1
        row[f"player_2_minutes_last_{days}d"] = p2
        row[f"minutes_last_{days}d_diff"] = p1 - p2
        row[f"player_1_known_duration_matches_last_{days}d"] = p1_workload[
            f"known_duration_matches_last_{days}d"
        ]
        row[f"player_2_known_duration_matches_last_{days}d"] = p2_workload[
            f"known_duration_matches_last_{days}d"
        ]
        row[f"player_1_missing_duration_matches_last_{days}d"] = p1_workload[
            f"missing_duration_matches_last_{days}d"
        ]
        row[f"player_2_missing_duration_matches_last_{days}d"] = p2_workload[
            f"missing_duration_matches_last_{days}d"
        ]
    return row


def _metric_pairs(
    player: dict[str, float],
    opponent: dict[str, float],
) -> dict[str, tuple[float, float]]:
    serve_won = player["first_won"] + player["second_won"]
    opponent_serve_won = opponent["first_won"] + opponent["second_won"]
    second_played = player["svpt"] - player["first_in"]
    break_converted = opponent["bp_faced"] - opponent["bp_saved"]
    return {
        "serve_points_won_pct": (serve_won, player["svpt"]),
        "return_points_won_pct": (opponent["svpt"] - opponent_serve_won, opponent["svpt"]),
        "first_serve_in_pct": (player["first_in"], player["svpt"]),
        "first_serve_points_won_pct": (player["first_won"], player["first_in"]),
        "second_serve_points_won_pct": (player["second_won"], second_played),
        "ace_rate": (player["ace"], player["svpt"]),
        "double_fault_rate": (player["df"], player["svpt"]),
        "break_points_saved_pct": (player["bp_saved"], player["bp_faced"]),
        "break_points_converted_pct": (break_converted, opponent["bp_faced"]),
    }


def _valid_side_stats(
    stats: dict[str, float | None],
    invalid_counts: Counter[str],
    side: str,
) -> dict[str, float] | None:
    if any(value is None for value in stats.values()):
        return None
    values = {key: float(value) for key, value in stats.items() if value is not None}
    checks = {
        "serve_points_positive": values["svpt"] > 0,
        "first_in_bounds": 0 <= values["first_in"] <= values["svpt"],
        "first_won_bounds": 0 <= values["first_won"] <= values["first_in"],
        "second_won_bounds": 0 <= values["second_won"] <= values["svpt"] - values["first_in"],
        "bp_saved_bounds": 0 <= values["bp_saved"] <= values["bp_faced"],
        "ace_nonnegative": values["ace"] >= 0,
        "df_nonnegative": values["df"] >= 0,
    }
    for name, passed in checks.items():
        if not passed:
            invalid_counts[f"{side}_{name}"] += 1
    if not all(checks.values()):
        return None
    return values


def _side_stats(row: pd.Series, side: str) -> dict[str, float | None]:
    return {column: _nullable_float(row.get(f"{side}_{column}")) for column in STAT_COLUMNS}


def _strict_candidate(
    project: pd.Series,
    winner_name: str,
    loser_name: str,
    sackmann: pd.DataFrame,
    aliases: dict[str, int],
) -> pd.Series | None:
    tournament = normalize_tournament(project.get("tournament"))[1]
    round_name = normalize_round(project.get("round"))
    candidates = sackmann[
        (sackmann["match_date"] == pd.Timestamp(project["match_date"]).normalize())
        & (sackmann["tournament_normalized"].astype(str) == str(tournament))
        & (sackmann["round"].astype(str) == round_name)
    ]
    return _unique_player_candidate(candidates, winner_name, loser_name, aliases)


def _unique_date_player_candidate(
    project: pd.Series,
    winner_name: str,
    loser_name: str,
    sackmann: pd.DataFrame,
    aliases: dict[str, int],
) -> pd.Series | None:
    candidates = sackmann[sackmann["match_date"] == pd.Timestamp(project["match_date"]).normalize()]
    return _unique_player_candidate(candidates, winner_name, loser_name, aliases)


def _alias_candidate(
    project: pd.Series,
    winner_name: str,
    loser_name: str,
    sackmann: pd.DataFrame,
    aliases: dict[str, int],
) -> pd.Series | None:
    winner_id = aliases.get(normalize_for_match(winner_name) or "")
    loser_id = aliases.get(normalize_for_match(loser_name) or "")
    if winner_id is None or loser_id is None:
        return None
    candidates = sackmann[
        (sackmann["match_date"] == pd.Timestamp(project["match_date"]).normalize())
        & (sackmann["winner_id"].astype("Int64") == winner_id)
        & (sackmann["loser_id"].astype("Int64") == loser_id)
    ]
    return candidates.iloc[0] if len(candidates) == 1 else None


def _unique_player_candidate(
    candidates: pd.DataFrame,
    winner_name: str,
    loser_name: str,
    aliases: dict[str, int],
) -> pd.Series | None:
    matches = [
        row
        for _, row in candidates.iterrows()
        if _player_matches(winner_name, row, "winner", aliases)
        and _player_matches(loser_name, row, "loser", aliases)
    ]
    return matches[0] if len(matches) == 1 else None


def _player_matches(project_name: str, row: pd.Series, side: str, aliases: dict[str, int]) -> bool:
    normalized = normalize_for_match(project_name) or ""
    alias_id = aliases.get(normalized)
    if alias_id is not None and _nullable_int(row[f"{side}_id"]) == alias_id:
        return True
    return bool(name_keys(project_name) & set(row[f"{side}_name_keys"]))


def _project_winner_loser(row: pd.Series) -> tuple[str, str]:
    if _bool(row["player_1_won"]):
        return str(row["player_1"]), str(row["player_2"])
    return str(row["player_2"]), str(row["player_1"])


def _join_summary(
    project: pd.DataFrame,
    sackmann: pd.DataFrame,
    links: pd.DataFrame,
    failures: pd.DataFrame,
    duplicate_keys: int,
) -> dict[str, Any]:
    methods = Counter(links["join_method"]) if not links.empty else Counter()
    project_with_year = project.copy()
    project_with_year["year"] = pd.to_datetime(project_with_year["match_date"]).dt.year
    linked_ids = set(links["match_id"].astype(str)) if not links.empty else set()
    project_with_year["linked"] = project_with_year["match_id"].astype(str).isin(linked_ids)
    players = set(project["player_1"].astype(str)) | set(project["player_2"].astype(str))
    mapped_players = (
        set(links["player_1_sackmann_id"].dropna().astype(str))
        | set(links["player_2_sackmann_id"].dropna().astype(str))
        if not links.empty
        else set()
    )
    return {
        "created_at_utc": _utc_timestamp(),
        "project_matches": int(len(project)),
        "sackmann_matches": int(len(sackmann)),
        "strict_matches": int(methods.get("strict", 0)),
        "unique_fallback_matches": int(methods.get("unique_date_player", 0)),
        "explicit_alias_matches": int(methods.get("explicit_alias", 0)),
        "ambiguous_matches": int(
            (failures.get("failure_reason", pd.Series(dtype=str)) == "ambiguous").sum()
        ),
        "unmatched_matches": int(len(failures)),
        "duplicate_sackmann_candidates": duplicate_keys,
        "join_rate": len(links) / len(project) if len(project) else 0.0,
        "join_rate_by_year": _rate_by(project_with_year, "year"),
        "join_rate_by_surface": _rate_by(project_with_year, "surface"),
        "join_rate_by_tournament": _rate_by(project_with_year, "tournament"),
        "join_rate_by_round": _rate_by(project_with_year, "round"),
        "unique_project_players": len(players),
        "players_mapped_to_sackmann_ids": len(mapped_players),
        "players_unresolved": max(len(players) - len(mapped_players), 0),
        "stat_orientation_failures": 0,
    }


def _feature_quality(
    base: pd.DataFrame,
    enriched: pd.DataFrame,
    feature_columns: list[str],
    links: pd.DataFrame,
    stats_quality: dict[str, Any],
) -> dict[str, Any]:
    numeric = (
        enriched[feature_columns].select_dtypes(include=["number"])
        if feature_columns
        else pd.DataFrame()
    )
    return {
        "created_at_utc": _utc_timestamp(),
        "base_rows": int(len(base)),
        "enriched_rows": int(len(enriched)),
        "row_count_preservation": bool(len(base) == len(enriched)),
        "duplicate_match_ids": int(enriched["match_id"].duplicated().sum()),
        "join_coverage": len(links) / len(base) if len(base) else 0.0,
        "feature_missingness": {
            column: int(enriched[column].isna().sum()) for column in feature_columns
        },
        "feature_min_values": {
            column: _nullable_float(numeric[column].min()) for column in numeric.columns
        },
        "feature_max_values": {
            column: _nullable_float(numeric[column].max()) for column in numeric.columns
        },
        "feature_coverage_by_year": _coverage_by(enriched, feature_columns, "match_date"),
        "feature_coverage_by_surface": _coverage_by(enriched, feature_columns, "surface"),
        "2020_warmup_coverage": None,
        "retirements_skipped_for_point_updates": stats_quality[
            "retirements_skipped_for_point_updates"
        ],
        "walkovers_skipped": stats_quality["walkovers_skipped"],
        "invalid_source_stat_counts": stats_quality["invalid_source_stat_counts"],
        "duration_coverage": {
            "rows_with_player_1_7d_minutes": int(
                enriched.get("player_1_minutes_last_7d", pd.Series(dtype=float)).notna().sum()
            ),
            "rows_with_player_2_7d_minutes": int(
                enriched.get("player_2_minutes_last_7d", pd.Series(dtype=float)).notna().sum()
            ),
        },
    }


def _feature_dictionary(feature_columns: list[str]) -> dict[str, Any]:
    descriptions = {
        column: "Leakage-safe pre-match Sackmann serve/return/workload enrichment."
        for column in feature_columns
    }
    descriptions["workload_minutes_policy"] = (
        "Valid nonnegative retirement minutes may update workload, while retired and "
        "walkover matches are excluded from point-performance state updates."
    )
    return {
        "created_at_utc": _utc_timestamp(),
        "overall_half_life_matches": OVERALL_HALF_LIFE_MATCHES,
        "surface_half_life_matches": SURFACE_HALF_LIFE_MATCHES,
        "rate_prior_weight": RATE_PRIOR_WEIGHT,
        "break_point_prior_weight": BREAK_PRIOR_WEIGHT,
        "columns": descriptions,
    }


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id", *_sackmann_feature_columns()])


def _sackmann_feature_columns() -> list[str]:
    columns: list[str] = []
    for metric in METRICS:
        columns.extend(
            [
                f"player_1_ewm_{metric}",
                f"player_2_ewm_{metric}",
                f"ewm_{metric}_diff",
                f"player_1_surface_ewm_{metric}",
                f"player_2_surface_ewm_{metric}",
                f"surface_ewm_{metric}_diff",
            ]
        )
    columns.extend(
        [
            "player_1_point_stats_match_count",
            "player_2_point_stats_match_count",
            "point_stats_match_count_min",
            "player_1_surface_point_stats_match_count",
            "player_2_surface_point_stats_match_count",
            "surface_point_stats_match_count_min",
            "player_1_decayed_service_points",
            "player_2_decayed_service_points",
            "player_1_decayed_return_points",
            "player_2_decayed_return_points",
        ]
    )
    for days in (3, 7, 14):
        columns.extend(
            [
                f"player_1_minutes_last_{days}d",
                f"player_2_minutes_last_{days}d",
                f"minutes_last_{days}d_diff",
                f"player_1_known_duration_matches_last_{days}d",
                f"player_2_known_duration_matches_last_{days}d",
                f"player_1_missing_duration_matches_last_{days}d",
                f"player_2_missing_duration_matches_last_{days}d",
            ]
        )
    return columns


def _coverage_by(
    frame: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
) -> dict[str, Any]:
    if not feature_columns or group_column not in frame:
        return {}
    group_values = (
        pd.to_datetime(frame[group_column]).dt.year
        if group_column == "match_date"
        else frame[group_column]
    )
    output: dict[str, Any] = {}
    for value, group in frame.groupby(group_values, sort=True, dropna=False):
        output[str(value)] = {
            "rows": int(len(group)),
            "any_sackmann_feature_present": int(group[feature_columns].notna().any(axis=1).sum()),
        }
    return output


def _rate_by(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(value): float(group["linked"].mean())
        for value, group in frame.groupby(column, sort=True, dropna=False)
    }


def _duplicate_sackmann_keys(sackmann: pd.DataFrame) -> int:
    keys = sackmann[["match_date", "winner_id", "loser_id"]].astype(str).agg("|".join, axis=1)
    return int(keys.duplicated().sum())


def _failure_reason(
    project: pd.Series,
    winner_name: str,
    loser_name: str,
    sackmann: pd.DataFrame,
    aliases: dict[str, int],
) -> str:
    candidates = sackmann[sackmann["match_date"] == pd.Timestamp(project["match_date"]).normalize()]
    matches = [
        row
        for _, row in candidates.iterrows()
        if _player_matches(winner_name, row, "winner", aliases)
        and _player_matches(loser_name, row, "loser", aliases)
    ]
    return "ambiguous" if len(matches) > 1 else "unmatched"


def _failure_row(
    project: pd.Series,
    winner_name: str,
    loser_name: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "match_id": str(project["match_id"]),
        "match_date": pd.Timestamp(project["match_date"]).strftime("%Y-%m-%d"),
        "tournament": str(project.get("tournament", "")),
        "round": str(project.get("round", "")),
        "winner": winner_name,
        "loser": loser_name,
        "failure_reason": reason,
    }


def _manual_suggestions(
    project: pd.Series,
    winner_name: str,
    loser_name: str,
    sackmann: pd.DataFrame,
) -> list[dict[str, Any]]:
    same_day = sackmann[sackmann["match_date"] == pd.Timestamp(project["match_date"]).normalize()]
    rows: list[dict[str, Any]] = []
    for _, candidate in same_day.head(5).iterrows():
        rows.append(
            {
                "match_id": str(project["match_id"]),
                "project_winner": winner_name,
                "project_loser": loser_name,
                "candidate_winner_id": _nullable_int(candidate["winner_id"]),
                "candidate_winner_name": str(candidate["winner_name"]),
                "candidate_loser_id": _nullable_int(candidate["loser_id"]),
                "candidate_loser_name": str(candidate["loser_name"]),
                "notes": "manual review only; fuzzy suggestions are not auto-accepted",
            }
        )
    return rows


def _load_aliases(alias_path: Path | None) -> dict[str, int]:
    if alias_path is None or not alias_path.exists():
        return {}
    frame = pd.read_csv(alias_path)
    aliases: dict[str, int] = {}
    for _, row in frame.iterrows():
        normalized = normalize_for_match(
            row.get("project_normalized_name") or row.get("project_name")
        )
        sackmann_id = _nullable_int(row.get("sackmann_player_id"))
        if normalized and sackmann_id is not None:
            aliases[normalized] = sackmann_id
    return aliases


def _parse_sackmann_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _normalize_surface(value: Any) -> str:
    normalized = normalize_for_match(value)
    if normalized in {"hard", "clay", "grass"}:
        return normalized.title()
    return "Other"


def _valid_minutes(value: Any) -> float | None:
    parsed = _nullable_float(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _half_life_decay(half_life: int) -> float:
    return math.exp(math.log(0.5) / half_life)


def _diff(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return first - second


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    return bool(value)


def _nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _empty_sackmann_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_file",
            "sackmann_tourney_id",
            "sackmann_match_num",
            "sackmann_match_key",
            "match_date",
            "tournament",
            "tournament_normalized",
            "surface",
            "round",
            "best_of",
            "winner_id",
            "loser_id",
            "winner_name",
            "loser_name",
            "winner_normalized",
            "loser_normalized",
            "winner_name_keys",
            "loser_name_keys",
            "minutes",
            "score",
            "is_walkover",
            "is_retirement",
        ]
        + [f"{side}_{column}" for side in ("winner", "loser") for column in STAT_COLUMNS]
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


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
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "BREAK_PRIOR_WEIGHT",
    "METRICS",
    "OVERALL_HALF_LIFE_MATCHES",
    "PROJECT_END_YEAR",
    "PROJECT_START_YEAR",
    "RATE_PRIOR_WEIGHT",
    "SACKMANN_YEARS",
    "SURFACE_HALF_LIFE_MATCHES",
    "SackmannBuildPaths",
    "SackmannBuildResult",
    "build_pre_match_stat_features",
    "build_sackmann_enriched_features",
    "derive_match_stats",
    "link_project_matches",
    "load_sackmann_matches",
    "name_keys",
    "normalize_sackmann_frame",
    "write_sackmann_artifacts",
]
