"""Audit Tennis-Data odds source columns and paired-odds quality."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tennis_value.ingest import (
    DEFAULT_ODDS_PAIRS,
    OUTPUT_COLUMNS,
    discover_tennis_data_files,
    ingest_tennis_data,
)

BOOKMAKER_LABELS = {
    "B365": "Bet365",
    "Avg": "Average",
    "Max": "Maximum",
    "PS": "Pinnacle/PS",
    "Pinnacle": "Pinnacle/PS",
}


class OddsSourceAuditSummary(BaseModel):
    """JSON-serializable odds-source audit summary."""

    model_config = ConfigDict(frozen=True)

    created_at_utc: str
    raw_input: str
    processed_input: str | None
    current_selection_policy: list[dict[str, str]]
    selected_source: str
    source_files: list[str]
    original_odds_column_names: dict[str, list[str]] = Field(default_factory=dict)
    rows_by_source: dict[str, int] = Field(default_factory=dict)
    fallback_rows: int
    source_consistency: dict[str, Any]
    quality_flag_counts: dict[str, int] = Field(default_factory=dict)
    suspected_use_of_maximum_odds: bool
    suspected_winner_loser_pair_mismatch_rows: int
    processed_rows: int | None = None
    processed_rows_with_missing_odds: int | None = None


class OddsSourceAuditResult(BaseModel):
    """Odds-source audit artifacts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    summary: OddsSourceAuditSummary
    quality_rows: pd.DataFrame
    overround_by_source: pd.DataFrame


@dataclass(frozen=True)
class OddsAuditOutputPaths:
    """Output paths for odds-source audit artifacts."""

    summary: Path
    quality_rows: Path
    overround_by_source: Path


def audit_odds_sources(
    raw_input: Path,
    processed_input: Path | None = None,
) -> OddsSourceAuditResult:
    """Audit current raw odds-column selection and paired-odds quality."""
    source_columns = _source_odds_columns(raw_input)
    ingestion = ingest_tennis_data(raw_input)
    raw_rows = ingestion.data.copy(deep=True)
    if raw_rows.empty:
        raw_rows = pd.DataFrame(columns=OUTPUT_COLUMNS)
    quality_rows = build_odds_quality_rows(raw_rows)
    overround_by_source = build_overround_by_source(quality_rows)
    processed_rows: int | None = None
    processed_missing: int | None = None
    if processed_input is not None and processed_input.exists():
        processed = pd.read_parquet(processed_input)
        processed_rows = len(processed)
        processed_missing = int(
            (processed["player_1_odds"].isna() | processed["player_2_odds"].isna()).sum()
        ) if {"player_1_odds", "player_2_odds"}.issubset(processed.columns) else None

    rows_by_source = _value_counts(quality_rows, "odds_source")
    fallback_rows = int(quality_rows["used_fallback_source"].sum()) if not quality_rows.empty else 0
    flag_columns = [
        "overround_below_1_00",
        "overround_above_1_12",
        "either_odd_below_1_02",
        "either_odd_above_30",
        "missing_paired_odds",
        "suspected_maximum_odds",
        "suspected_winner_loser_pair_mismatch",
    ]
    quality_counts = {
        column: int(quality_rows[column].sum()) if column in quality_rows else 0
        for column in flag_columns
    }
    summary = OddsSourceAuditSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        raw_input=str(raw_input),
        processed_input=str(processed_input) if processed_input is not None else None,
        current_selection_policy=[
            {
                "winner_column": winner,
                "loser_column": loser,
                "source": source,
                "meaning": _source_meaning(source),
            }
            for winner, loser, source in DEFAULT_ODDS_PAIRS
        ],
        selected_source="fallback_hierarchy: B365W/B365L first, then AvgW/AvgL",
        source_files=sorted(source_columns),
        original_odds_column_names=source_columns,
        rows_by_source=rows_by_source,
        fallback_rows=fallback_rows,
        source_consistency=_source_consistency(quality_rows),
        quality_flag_counts=quality_counts,
        suspected_use_of_maximum_odds=bool(quality_counts["suspected_maximum_odds"] > 0),
        suspected_winner_loser_pair_mismatch_rows=quality_counts[
            "suspected_winner_loser_pair_mismatch"
        ],
        processed_rows=processed_rows,
        processed_rows_with_missing_odds=processed_missing,
    )
    return OddsSourceAuditResult(
        summary=summary,
        quality_rows=quality_rows,
        overround_by_source=overround_by_source,
    )


def build_odds_quality_rows(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Build row-level odds quality flags without discarding suspicious rows."""
    frame = raw_rows.copy(deep=True)
    if frame.empty:
        return _empty_quality_rows()
    winner_odds = pd.to_numeric(frame["winner_odds"], errors="coerce")
    loser_odds = pd.to_numeric(frame["loser_odds"], errors="coerce")
    overround = 1.0 / winner_odds + 1.0 / loser_odds
    output = frame[
        [
            "match_date",
            "tournament",
            "winner",
            "loser",
            "winner_odds",
            "loser_odds",
            "odds_source",
            "source_file",
        ]
    ].copy()
    output["year"] = pd.to_datetime(output["match_date"], errors="coerce").dt.year.astype("Int64")
    output["overround"] = overround.astype("Float64")
    output["used_fallback_source"] = output["odds_source"].astype(str).eq("Average")
    output["overround_below_1_00"] = output["overround"] < 1.00
    output["overround_above_1_12"] = output["overround"] > 1.12
    output["either_odd_below_1_02"] = (winner_odds < 1.02) | (loser_odds < 1.02)
    output["either_odd_above_30"] = (winner_odds > 30) | (loser_odds > 30)
    output["missing_paired_odds"] = winner_odds.isna() | loser_odds.isna()
    output["suspected_maximum_odds"] = output["odds_source"].astype(str).eq("Maximum")
    output["suspected_winner_loser_pair_mismatch"] = (
        (winner_odds > loser_odds) & output["odds_source"].astype(str).ne("Missing")
    )
    for column in [
        "used_fallback_source",
        "overround_below_1_00",
        "overround_above_1_12",
        "either_odd_below_1_02",
        "either_odd_above_30",
        "missing_paired_odds",
        "suspected_maximum_odds",
        "suspected_winner_loser_pair_mismatch",
    ]:
        output[column] = output[column].fillna(False).astype("bool")
    return output.reset_index(drop=True)


def build_overround_by_source(quality_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize overround distribution by source and season."""
    if quality_rows.empty:
        return pd.DataFrame(
            columns=[
                "odds_source",
                "year",
                "rows",
                "mean_overround",
                "median_overround",
                "min_overround",
                "max_overround",
                "p05_overround",
                "p95_overround",
            ]
        )
    rows: list[dict[str, Any]] = []
    valid = quality_rows[quality_rows["overround"].notna()].copy()
    for (source, year), group in valid.groupby(["odds_source", "year"], dropna=False, sort=True):
        overround = pd.to_numeric(group["overround"], errors="coerce")
        rows.append(
            {
                "odds_source": str(source),
                "year": _nullable_year(year),
                "rows": len(group),
                "mean_overround": float(overround.mean()),
                "median_overround": float(overround.median()),
                "min_overround": float(overround.min()),
                "max_overround": float(overround.max()),
                "p05_overround": float(overround.quantile(0.05)),
                "p95_overround": float(overround.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def write_odds_audit_artifacts(
    result: OddsSourceAuditResult,
    paths: OddsAuditOutputPaths,
) -> None:
    """Write odds-source audit JSON and Parquet artifacts."""
    for path in paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(
        json.dumps(_jsonable(result.summary.model_dump()), indent=2),
        encoding="utf-8",
    )
    result.quality_rows.to_parquet(paths.quality_rows, index=False)
    result.overround_by_source.to_parquet(paths.overround_by_source, index=False)


def _source_odds_columns(raw_input: Path) -> dict[str, list[str]]:
    columns_by_file: dict[str, list[str]] = {}
    for file_path in discover_tennis_data_files(raw_input):
        try:
            if file_path.suffix.casefold() == ".csv":
                columns = list(pd.read_csv(file_path, nrows=0).columns)
            else:
                columns = list(pd.read_excel(file_path, nrows=0).columns)
        except Exception:  # noqa: BLE001 - audit should keep scanning other files.
            columns_by_file[file_path.name] = []
            continue
        odds_columns = [str(column) for column in columns if _looks_like_odds_column(column)]
        columns_by_file[file_path.name] = odds_columns
    return columns_by_file


def _nullable_year(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _looks_like_odds_column(column: Any) -> bool:
    text = str(column).strip()
    normalized = text.casefold()
    if normalized in {"winner", "loser", "wrank", "lrank"}:
        return False
    suffix_match = normalized.endswith("w") or normalized.endswith("l")
    prefix_match = normalized[:3] in {"b36", "avg", "max"} or normalized[:2] == "ps"
    return suffix_match and prefix_match


def _source_meaning(source: str) -> str:
    return BOOKMAKER_LABELS.get(source, source)


def _source_consistency(quality_rows: pd.DataFrame) -> dict[str, Any]:
    if quality_rows.empty:
        return {"same_source_pairing": True, "note": "no rows available"}
    source_counts = _value_counts(quality_rows, "odds_source")
    return {
        "same_source_pairing": True,
        "note": "ingestion selects winner and loser odds as named source pairs",
        "source_counts": source_counts,
    }


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _empty_quality_rows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_date",
            "tournament",
            "winner",
            "loser",
            "winner_odds",
            "loser_odds",
            "odds_source",
            "source_file",
            "year",
            "overround",
            "used_fallback_source",
            "overround_below_1_00",
            "overround_above_1_12",
            "either_odd_below_1_02",
            "either_odd_above_30",
            "missing_paired_odds",
            "suspected_maximum_odds",
            "suspected_winner_loser_pair_mismatch",
        ]
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


__all__ = [
    "OddsAuditOutputPaths",
    "OddsSourceAuditResult",
    "OddsSourceAuditSummary",
    "audit_odds_sources",
    "build_odds_quality_rows",
    "build_overround_by_source",
    "write_odds_audit_artifacts",
]
