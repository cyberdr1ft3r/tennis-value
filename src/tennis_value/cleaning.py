"""Canonical match cleaning and quality reporting."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tennis_value.orientation import (
    OrientedPlayers,
    generate_match_id,
    normalize_for_match,
    normalize_text,
    orient_players,
)

SUPPORTED_SURFACES = {"Hard", "Clay", "Grass"}
CANONICAL_COLUMNS = [
    "match_id",
    "match_date",
    "tournament",
    "tournament_normalized",
    "surface",
    "round",
    "best_of",
    "player_1",
    "player_2",
    "player_1_display_name",
    "player_2_display_name",
    "player_1_normalized",
    "player_2_normalized",
    "player_1_rank",
    "player_2_rank",
    "player_1_odds",
    "player_2_odds",
    "player_1_won",
    "is_retirement",
    "odds_source",
    "source_file",
]
DUPLICATE_COMPARE_COLUMNS = [
    column for column in CANONICAL_COLUMNS if column not in {"source_file"}
]

RejectionReason = Literal[
    "invalid_match_date",
    "missing_winner",
    "missing_loser",
    "same_player",
    "unsupported_surface",
    "walkover",
    "missing_winner_result",
    "invalid_best_of",
    "exact_duplicate",
    "conflicting_duplicate",
]


class DataQualityReport(BaseModel):
    """JSON-serializable data-quality report for canonical cleaning."""

    model_config = ConfigDict(frozen=True)

    rows_received: int
    rows_accepted: int
    rows_rejected: int
    acceptance_rate: float
    rejections_by_reason: dict[str, int] = Field(default_factory=dict)
    walkovers: int = 0
    retirements: int = 0
    unsupported_surfaces: int = 0
    missing_rankings: int = 0
    missing_odds: int = 0
    exact_duplicates: int = 0
    conflicting_duplicates: int = 0
    surface_counts: dict[str, int] = Field(default_factory=dict)
    round_counts: dict[str, int] = Field(default_factory=dict)
    source_file_counts: dict[str, int] = Field(default_factory=dict)
    original_surface_counts: dict[str, int] = Field(default_factory=dict)
    minimum_match_date: str | None = None
    maximum_match_date: str | None = None


class CleaningResult(BaseModel):
    """Canonical rows, rejected rows, and report."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    canonical_matches: pd.DataFrame
    rejected_rows: pd.DataFrame
    quality_report: DataQualityReport


def clean_matches(raw_matches: pd.DataFrame) -> CleaningResult:
    """Convert raw ingested rows into canonical, neutrally oriented matches."""
    canonical_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    original_surface_counts = Counter(
        str(value) if not _is_missing(value) else "Missing"
        for value in raw_matches.get("surface", pd.Series(dtype="object"))
    )

    for row_index, (_, row) in enumerate(raw_matches.reset_index(drop=True).iterrows()):
        row_dict = {str(key): value for key, value in row.to_dict().items()}
        rejection = _validate_raw_row(row)
        if rejection is not None:
            rejected_rows.append(_rejected_row(row_dict, rejection, row_index))
            continue

        oriented = orient_players(row["winner"], row["loser"])
        if oriented is None:
            rejected_rows.append(_rejected_row(row_dict, "same_player", row_index))
            continue

        canonical_rows.append(_canonical_row(row, oriented))

    canonical = _coerce_canonical_dtypes(pd.DataFrame(canonical_rows, columns=CANONICAL_COLUMNS))
    canonical, duplicate_rejections = _handle_duplicates(canonical)
    rejected_rows.extend(duplicate_rejections)

    rejected = pd.DataFrame(rejected_rows)
    canonical = canonical.sort_values(
        ["match_date", "tournament_normalized", "round", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    report = _build_quality_report(raw_matches, canonical, rejected, original_surface_counts)

    return CleaningResult(
        canonical_matches=canonical,
        rejected_rows=rejected,
        quality_report=report,
    )


def write_cleaning_outputs(
    result: CleaningResult,
    output_path: Path,
    report_path: Path,
    rejected_path: Path,
) -> None:
    """Write canonical matches, quality report, and rejected rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    result.canonical_matches.to_parquet(output_path, index=False)
    report_path.write_text(result.quality_report.model_dump_json(indent=2), encoding="utf-8")
    result.rejected_rows.to_csv(rejected_path, index=False)


def normalize_tournament(value: Any) -> tuple[str | None, str | None]:
    """Return display and comparison forms for tournament names."""
    display = normalize_text(value)
    if display is None:
        return None, None
    return display, normalize_for_match(display)


def normalize_surface(value: Any) -> str:
    """Normalize source surface values."""
    text = normalize_for_match(value)
    if text is None:
        return "Other"
    if text in {"hard", "hardcourt", "hard court", "indoor hard", "outdoor hard"}:
        return "Hard"
    if "hard" in text and "clay" not in text:
        return "Hard"
    if text in {"clay", "red clay"} or text.endswith(" clay"):
        return "Clay"
    if text == "grass" or text.endswith(" grass"):
        return "Grass"
    return "Other"


def normalize_round(value: Any) -> str:
    """Normalize common round names."""
    text = normalize_for_match(value)
    if text is None:
        return "Unknown"
    mapping = {
        "r128": "R128",
        "round of 128": "R128",
        "1st round": "R128",
        "first round": "R128",
        "r64": "R64",
        "round of 64": "R64",
        "2nd round": "R64",
        "second round": "R64",
        "r32": "R32",
        "round of 32": "R32",
        "3rd round": "R32",
        "third round": "R32",
        "r16": "R16",
        "round of 16": "R16",
        "4th round": "R16",
        "fourth round": "R16",
        "qf": "QF",
        "quarterfinal": "QF",
        "quarterfinals": "QF",
        "quarter finals": "QF",
        "sf": "SF",
        "semi final": "SF",
        "semi finals": "SF",
        "semifinal": "SF",
        "semifinals": "SF",
        "f": "F",
        "final": "F",
        "the final": "F",
        "rr": "RR",
        "round robin": "RR",
        "br": "BR",
        "bronze": "BR",
        "bronze medal": "BR",
        "unknown": "Unknown",
    }
    return mapping.get(text, "Unknown")


def is_walkover(row: pd.Series) -> bool:
    """Return whether source status fields indicate a walkover."""
    text = _status_text(row)
    return bool(re.search(r"\b(walk\s*-?\s*over|w\s*/?\s*o|wo)\b", text, re.IGNORECASE))


def is_retirement(row: pd.Series) -> bool:
    """Return whether source status fields indicate a retirement."""
    text = _status_text(row)
    return bool(re.search(r"\b(ret|retired|retirement|abandoned)\b", text, re.IGNORECASE))


def _validate_raw_row(row: pd.Series) -> RejectionReason | None:
    if _parse_match_date(row.get("match_date")) is None:
        return "invalid_match_date"
    if normalize_text(row.get("winner")) is None:
        return "missing_winner"
    if normalize_text(row.get("loser")) is None:
        return "missing_loser"
    if is_walkover(row):
        return "walkover"
    winner_normalized = normalize_for_match(row.get("winner"))
    loser_normalized = normalize_for_match(row.get("loser"))
    if winner_normalized is None or loser_normalized is None:
        return "missing_winner_result"
    if winner_normalized == loser_normalized:
        return "same_player"
    if normalize_surface(row.get("surface")) not in SUPPORTED_SURFACES:
        return "unsupported_surface"
    best_of = _nullable_int(row.get("best_of"))
    if best_of is not None and best_of not in {3, 5}:
        return "invalid_best_of"
    return None


def _canonical_row(row: pd.Series, oriented: OrientedPlayers) -> dict[str, Any]:
    match_date = _parse_match_date(row["match_date"])
    if match_date is None:
        msg = "canonical row requires a valid match date"
        raise ValueError(msg)
    tournament_display, tournament_normalized = normalize_tournament(row.get("tournament"))
    round_normalized = normalize_round(row.get("round"))
    best_of = _nullable_int(row.get("best_of")) or 3
    player_1_rank, player_2_rank = _oriented_pair(row, "winner_rank", "loser_rank", oriented)
    player_1_odds, player_2_odds = _oriented_pair(row, "winner_odds", "loser_odds", oriented)

    match_id = generate_match_id(
        match_date=match_date,
        tournament_normalized=tournament_normalized or "",
        round_normalized=round_normalized,
        player_1_normalized=oriented.player_1_normalized,
        player_2_normalized=oriented.player_2_normalized,
    )

    return {
        "match_id": match_id,
        "match_date": match_date,
        "tournament": tournament_display,
        "tournament_normalized": tournament_normalized,
        "surface": normalize_surface(row.get("surface")),
        "round": round_normalized,
        "best_of": best_of,
        "player_1": oriented.player_1_display,
        "player_2": oriented.player_2_display,
        "player_1_display_name": oriented.player_1_display,
        "player_2_display_name": oriented.player_2_display,
        "player_1_normalized": oriented.player_1_normalized,
        "player_2_normalized": oriented.player_2_normalized,
        "player_1_rank": player_1_rank,
        "player_2_rank": player_2_rank,
        "player_1_odds": player_1_odds,
        "player_2_odds": player_2_odds,
        "player_1_won": oriented.player_1_won,
        "is_retirement": is_retirement(row),
        "odds_source": normalize_text(row.get("odds_source")) or "Missing",
        "source_file": normalize_text(row.get("source_file")) or "",
    }


def _oriented_pair(
    row: pd.Series,
    winner_field: str,
    loser_field: str,
    oriented: OrientedPlayers,
) -> tuple[Any, Any]:
    winner_value = row.get(winner_field)
    loser_value = row.get(loser_field)
    if oriented.swapped:
        return loser_value, winner_value
    return winner_value, loser_value


def _handle_duplicates(canonical: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if canonical.empty:
        return canonical, []

    keep_indices: list[int] = []
    reject_indices: set[int] = set()
    duplicate_rejections: list[dict[str, Any]] = []

    for _, group in canonical.groupby("match_id", sort=False, dropna=False):
        indices = list(group.index)
        if len(indices) == 1:
            keep_indices.append(indices[0])
            continue

        comparable = group[DUPLICATE_COMPARE_COLUMNS].astype("string").fillna("<NA>")
        if comparable.drop_duplicates().shape[0] == 1:
            keep_indices.append(indices[0])
            for index in indices[1:]:
                reject_indices.add(index)
                duplicate_rejections.append(
                    _rejected_row(canonical.loc[index].to_dict(), "exact_duplicate", index)
                )
            continue

        for index in indices:
            reject_indices.add(index)
            duplicate_rejections.append(
                _rejected_row(canonical.loc[index].to_dict(), "conflicting_duplicate", index)
            )

    kept = canonical.loc[[index for index in keep_indices if index not in reject_indices]].copy()
    return kept.reset_index(drop=True), duplicate_rejections


def _build_quality_report(
    raw_matches: pd.DataFrame,
    canonical: pd.DataFrame,
    rejected: pd.DataFrame,
    original_surface_counts: Counter[str],
) -> DataQualityReport:
    rows_received = len(raw_matches)
    rows_accepted = len(canonical)
    rows_rejected = len(rejected)
    rejections = (
        Counter(rejected["rejection_reason"])
        if "rejection_reason" in rejected
        else Counter()
    )
    min_date = canonical["match_date"].min() if not canonical.empty else None
    max_date = canonical["match_date"].max() if not canonical.empty else None
    missing_rankings = int(
        (canonical["player_1_rank"].isna() | canonical["player_2_rank"].isna()).sum()
    ) if not canonical.empty else 0
    missing_odds = int(
        (canonical["player_1_odds"].isna() | canonical["player_2_odds"].isna()).sum()
    ) if not canonical.empty else 0

    return DataQualityReport(
        rows_received=rows_received,
        rows_accepted=rows_accepted,
        rows_rejected=rows_rejected,
        acceptance_rate=rows_accepted / rows_received if rows_received else 0.0,
        rejections_by_reason=dict(sorted(rejections.items())),
        walkovers=int(rejections.get("walkover", 0)),
        retirements=int(canonical["is_retirement"].sum()) if not canonical.empty else 0,
        unsupported_surfaces=int(rejections.get("unsupported_surface", 0)),
        missing_rankings=missing_rankings,
        missing_odds=missing_odds,
        exact_duplicates=int(rejections.get("exact_duplicate", 0)),
        conflicting_duplicates=int(rejections.get("conflicting_duplicate", 0)),
        surface_counts=_value_counts(canonical, "surface"),
        round_counts=_value_counts(canonical, "round"),
        source_file_counts=_value_counts(canonical, "source_file"),
        original_surface_counts=dict(sorted(original_surface_counts.items())),
        minimum_match_date=(
            pd.Timestamp(min_date).strftime("%Y-%m-%d") if min_date is not None else None
        ),
        maximum_match_date=(
            pd.Timestamp(max_date).strftime("%Y-%m-%d") if max_date is not None else None
        ),
    )


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _status_text(row: pd.Series) -> str:
    pieces: list[str] = []
    for field in ("status_or_comment", "status", "comment", "result", "score", "Score"):
        if field in row and not _is_missing(row[field]):
            pieces.append(str(row[field]))
    return " ".join(pieces)


def _nullable_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_match_date(value: Any) -> pd.Timestamp | None:
    if _is_missing(value):
        return None
    try:
        parsed = pd.Timestamp(value).normalize()
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def _rejected_row(row: dict[str, Any], reason: RejectionReason, row_index: int) -> dict[str, Any]:
    rejected = dict(row)
    rejected["rejection_reason"] = reason
    rejected["source_row_index"] = row_index
    return rejected


def _coerce_canonical_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    coerced = frame.copy()
    coerced["match_date"] = pd.to_datetime(coerced["match_date"], errors="coerce")
    coerced["best_of"] = pd.to_numeric(coerced["best_of"], errors="coerce").astype("Int64")
    coerced["player_1_rank"] = (
        pd.to_numeric(coerced["player_1_rank"], errors="coerce").astype("Int64")
    )
    coerced["player_2_rank"] = (
        pd.to_numeric(coerced["player_2_rank"], errors="coerce").astype("Int64")
    )
    coerced["player_1_odds"] = (
        pd.to_numeric(coerced["player_1_odds"], errors="coerce").astype("Float64")
    )
    coerced["player_2_odds"] = (
        pd.to_numeric(coerced["player_2_odds"], errors="coerce").astype("Float64")
    )
    coerced["player_1_won"] = coerced["player_1_won"].astype("bool")
    coerced["is_retirement"] = coerced["is_retirement"].astype("bool")
    text_columns = [
        column
        for column in CANONICAL_COLUMNS
        if column
        not in {
            "match_date",
            "best_of",
            "player_1_rank",
            "player_2_rank",
            "player_1_odds",
            "player_2_odds",
            "player_1_won",
            "is_retirement",
        }
    ]
    for column in text_columns:
        coerced[column] = coerced[column].astype("string")
    return coerced[CANONICAL_COLUMNS]


def quality_report_to_json(report: DataQualityReport) -> str:
    """Serialize a quality report as formatted JSON."""
    return json.dumps(report.model_dump(mode="json"), indent=2)


__all__ = [
    "CANONICAL_COLUMNS",
    "CleaningResult",
    "DataQualityReport",
    "clean_matches",
    "is_retirement",
    "is_walkover",
    "normalize_round",
    "normalize_surface",
    "normalize_tournament",
    "quality_report_to_json",
    "write_cleaning_outputs",
]
