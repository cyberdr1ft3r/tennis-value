"""Local Tennis-Data file ingestion."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_EXTENSIONS = (".csv", ".xls", ".xlsx")
MISSING_TOKENS = {"", "-", "n/a", "na", "nan", "none", "null", "nr"}
REQUIRED_FIELDS = ("match_date", "tournament", "surface", "winner", "loser")

OddsSource = Literal["B365", "Average", "ConfiguredBookmaker", "Missing"]


class IngestionErrorDetail(BaseModel):
    """Structured per-file ingestion error."""

    model_config = ConfigDict(frozen=True)

    source_file: str
    message: str
    missing_fields: tuple[str, ...] = ()
    available_columns: tuple[str, ...] = ()
    alias_mappings_attempted: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    """JSON-serializable summary of an ingestion run."""

    model_config = ConfigDict(frozen=True)

    files_discovered: int = 0
    files_loaded: int = 0
    files_failed: int = 0
    rows_read: int = 0
    rows_returned: int = 0
    rows_with_invalid_dates: int = 0
    rows_without_odds: int = 0
    rows_without_rankings: int = 0
    source_files: tuple[str, ...] = ()
    errors: tuple[IngestionErrorDetail, ...] = ()
    warnings: tuple[str, ...] = ()


class IngestionResult(BaseModel):
    """Combined raw records and their ingestion report."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data: pd.DataFrame
    report: IngestionReport


class IngestionFailure(RuntimeError):
    """Raised when ingestion cannot produce output."""

    def __init__(self, report: IngestionReport) -> None:
        self.report = report
        super().__init__(report.model_dump_json(indent=2))


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "match_date": ("Date", "date", "MatchDate", "match_date"),
    "tournament": ("Tournament", "Tournament Name", "Event", "tourney_name", "tournament"),
    "surface": ("Surface", "surface"),
    "round": ("Round", "round"),
    "best_of": ("Best of", "BestOf", "Best_of", "best_of", "bestof"),
    "winner": ("Winner", "winner"),
    "loser": ("Loser", "loser"),
    "winner_rank": ("WRank", "Wrank", "Winner Rank", "winner_rank", "winner_ranking"),
    "loser_rank": ("LRank", "Lrank", "Loser Rank", "loser_rank", "loser_ranking"),
    "status_or_comment": ("Comment", "Status", "Result", "status", "comment"),
}

DEFAULT_ODDS_PAIRS: tuple[tuple[str, str, OddsSource], ...] = (
    ("B365W", "B365L", "B365"),
    ("AvgW", "AvgL", "Average"),
)
SOURCE_ODDS_PAIRS: dict[str, tuple[str, str]] = {
    "b365": ("B365W", "B365L"),
    "ps": ("PSW", "PSL"),
    "avg": ("AvgW", "AvgL"),
    "max": ("MaxW", "MaxL"),
}
SOURCE_ODDS_OUTPUT_COLUMNS = [
    "winner_b365_odds",
    "loser_b365_odds",
    "b365_pair_available",
    "winner_ps_odds",
    "loser_ps_odds",
    "ps_pair_available",
    "winner_avg_odds",
    "loser_avg_odds",
    "avg_pair_available",
    "winner_max_odds",
    "loser_max_odds",
    "max_pair_available",
]

OUTPUT_COLUMNS = [
    "match_date",
    "tournament",
    "surface",
    "round",
    "best_of",
    "winner",
    "loser",
    "winner_rank",
    "loser_rank",
    "winner_odds",
    "loser_odds",
    "odds_source",
    *SOURCE_ODDS_OUTPUT_COLUMNS,
    "status_or_comment",
    "source_file",
]


def discover_tennis_data_files(input_dir: Path) -> list[Path]:
    """Return supported Tennis-Data files below ``input_dir`` in deterministic order."""
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )


def ingest_tennis_data(
    input_dir: Path,
    configured_odds_pairs: Sequence[tuple[str, str]] = (),
    *,
    fail_if_no_rows: bool = False,
) -> IngestionResult:
    """Load and normalize local Tennis-Data source files."""
    files = discover_tennis_data_files(input_dir)
    frames: list[pd.DataFrame] = []
    errors: list[IngestionErrorDetail] = []
    warnings: list[str] = []
    rows_read = 0
    rows_with_invalid_dates = 0
    rows_without_odds = 0
    rows_without_rankings = 0
    loaded_files: list[str] = []

    odds_pairs = _odds_pair_priority(configured_odds_pairs)

    for file_path in files:
        try:
            source = _read_source_file(file_path)
            rows_read += len(source)
            frame, stats = _map_source_frame(source, file_path, odds_pairs)
        except ValueError as exc:
            errors.append(_error_from_exception(file_path, exc))
            continue
        except Exception as exc:  # pragma: no cover - defensive boundary for file parser errors.
            errors.append(
                IngestionErrorDetail(
                    source_file=file_path.name,
                    message=f"failed to load file: {exc}",
                )
            )
            continue

        loaded_files.append(file_path.name)
        rows_with_invalid_dates += stats["invalid_dates"]
        rows_without_odds += stats["missing_odds"]
        rows_without_rankings += stats["missing_rankings"]
        frames.append(frame)

    combined = (
        pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]
        if frames
        else _empty_output_frame()
    )
    combined = _coerce_output_dtypes(combined)

    if not files:
        warnings.append(f"no supported files discovered under {input_dir}")

    report = IngestionReport(
        files_discovered=len(files),
        files_loaded=len(loaded_files),
        files_failed=len(errors),
        rows_read=rows_read,
        rows_returned=len(combined),
        rows_with_invalid_dates=rows_with_invalid_dates,
        rows_without_odds=rows_without_odds,
        rows_without_rankings=rows_without_rankings,
        source_files=tuple(loaded_files),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

    if fail_if_no_rows and len(combined) == 0:
        raise IngestionFailure(report)

    return IngestionResult(data=combined, report=report)


def write_ingestion_outputs(result: IngestionResult, output_path: Path, report_path: Path) -> None:
    """Write raw ingested rows as Parquet and the report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_parquet(output_path, index=False)
    report_path.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")


def _read_source_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, dtype="string", keep_default_na=False)
    msg = f"unsupported file extension: {path.suffix}"
    raise ValueError(msg)


def _map_source_frame(
    source: pd.DataFrame,
    file_path: Path,
    odds_pairs: Sequence[tuple[str, str, OddsSource]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    column_lookup = {_normalize_column_name(column): column for column in source.columns}
    mapped_columns = _map_columns(source.columns)
    missing_required = tuple(field for field in REQUIRED_FIELDS if field not in mapped_columns)
    if missing_required:
        detail = IngestionErrorDetail(
            source_file=file_path.name,
            message="required columns could not be mapped",
            missing_fields=missing_required,
            available_columns=tuple(str(column) for column in source.columns),
            alias_mappings_attempted=COLUMN_ALIASES,
        )
        raise ValueError(detail.model_dump_json())

    output = pd.DataFrame(index=source.index)
    for target in OUTPUT_COLUMNS:
        output[target] = pd.NA

    for target, source_column in mapped_columns.items():
        if target in output.columns:
            output[target] = source[source_column]

    output["match_date"] = output["match_date"].map(_parse_date)
    output["surface"] = output["surface"].map(_normalize_surface)
    output["best_of"] = _parse_nullable_numeric(output["best_of"])
    output["winner_rank"] = _parse_nullable_numeric(output["winner_rank"])
    output["loser_rank"] = _parse_nullable_numeric(output["loser_rank"])
    output["source_file"] = file_path.name

    odds = source.apply(lambda row: _select_odds(row, column_lookup, odds_pairs), axis=1)
    output["winner_odds"] = odds.map(lambda value: value[0])
    output["loser_odds"] = odds.map(lambda value: value[1])
    output["odds_source"] = odds.map(lambda value: value[2])
    for source_key, (winner_column, loser_column) in SOURCE_ODDS_PAIRS.items():
        source_odds = source.apply(
            lambda row, winner=winner_column, loser=loser_column: _select_source_pair(
                row,
                column_lookup,
                winner,
                loser,
            ),
            axis=1,
        )
        output[f"winner_{source_key}_odds"] = source_odds.map(lambda value: value[0])
        output[f"loser_{source_key}_odds"] = source_odds.map(lambda value: value[1])
        output[f"{source_key}_pair_available"] = source_odds.map(lambda value: value[2])

    output["winner_odds"] = _parse_decimal_odds(output["winner_odds"])
    output["loser_odds"] = _parse_decimal_odds(output["loser_odds"])

    invalid_dates = int(output["match_date"].isna().sum())
    missing_odds = int((output["odds_source"] == "Missing").sum())
    missing_rankings = int((output["winner_rank"].isna() | output["loser_rank"].isna()).sum())

    return output, {
        "invalid_dates": invalid_dates,
        "missing_odds": missing_odds,
        "missing_rankings": missing_rankings,
    }


def _map_columns(columns: Iterable[Any]) -> dict[str, str]:
    normalized_to_original = {_normalize_column_name(column): str(column) for column in columns}
    mapped: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            original = normalized_to_original.get(_normalize_column_name(alias))
            if original is not None:
                mapped[target] = original
                break
    return mapped


def _odds_pair_priority(
    configured_pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str, OddsSource], ...]:
    configured: tuple[tuple[str, str, OddsSource], ...] = tuple(
        (winner, loser, "ConfiguredBookmaker") for winner, loser in configured_pairs
    )
    return (*DEFAULT_ODDS_PAIRS, *configured)


def _select_odds(
    row: pd.Series,
    column_lookup: dict[str, str],
    odds_pairs: Sequence[tuple[str, str, OddsSource]],
) -> tuple[Any, Any, OddsSource]:
    for winner_column, loser_column, source_name in odds_pairs:
        winner_source = column_lookup.get(_normalize_column_name(winner_column))
        loser_source = column_lookup.get(_normalize_column_name(loser_column))
        if winner_source is None or loser_source is None:
            continue
        winner_odds = _parse_single_decimal_odds(row[winner_source])
        loser_odds = _parse_single_decimal_odds(row[loser_source])
        if winner_odds is not None and loser_odds is not None:
            return winner_odds, loser_odds, source_name
    return pd.NA, pd.NA, "Missing"


def _select_source_pair(
    row: pd.Series,
    column_lookup: dict[str, str],
    winner_column: str,
    loser_column: str,
) -> tuple[Any, Any, bool]:
    winner_source = column_lookup.get(_normalize_column_name(winner_column))
    loser_source = column_lookup.get(_normalize_column_name(loser_column))
    winner_odds = (
        _parse_single_decimal_odds(row[winner_source]) if winner_source is not None else None
    )
    loser_odds = _parse_single_decimal_odds(row[loser_source]) if loser_source is not None else None
    return (
        winner_odds if winner_odds is not None else pd.NA,
        loser_odds if loser_odds is not None else pd.NA,
        winner_odds is not None and loser_odds is not None,
    )


def _parse_date(value: Any) -> pd.Timestamp | None:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value.date())
    if isinstance(value, date):
        return pd.Timestamp(value)

    text = str(value).strip()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, date_format).date())
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _parse_nullable_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(lambda value: pd.NA if _is_missing(value) else value)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_decimal_odds(series: pd.Series) -> pd.Series:
    cleaned = series.map(lambda value: pd.NA if _is_missing(value) else value)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return parsed.where(parsed.map(lambda value: _is_valid_decimal_odds(value)), pd.NA)


def _parse_single_decimal_odds(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    if not _is_valid_decimal_odds(parsed):
        return None
    return parsed


def _is_valid_decimal_odds(value: Any) -> bool:
    if value is pd.NA or pd.isna(value):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 1.0


def _normalize_surface(value: Any) -> str:
    if _is_missing(value):
        return "Other"
    normalized = str(value).strip().casefold()
    if normalized in {"hard", "h", "hardcourt", "hard court"}:
        return "Hard"
    if normalized in {"clay", "c"}:
        return "Clay"
    if normalized in {"grass", "g"}:
        return "Grass"
    return "Other"


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().casefold() in MISSING_TOKENS


def _normalize_column_name(value: Any) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _coerce_output_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    coerced["match_date"] = pd.to_datetime(coerced["match_date"], errors="coerce")
    coerced["best_of"] = pd.to_numeric(coerced["best_of"], errors="coerce").astype("Int64")
    coerced["winner_rank"] = pd.to_numeric(coerced["winner_rank"], errors="coerce").astype("Int64")
    coerced["loser_rank"] = pd.to_numeric(coerced["loser_rank"], errors="coerce").astype("Int64")
    coerced["winner_odds"] = (
        pd.to_numeric(coerced["winner_odds"], errors="coerce").astype("Float64")
    )
    coerced["loser_odds"] = (
        pd.to_numeric(coerced["loser_odds"], errors="coerce").astype("Float64")
    )
    for source_key in SOURCE_ODDS_PAIRS:
        for side in ("winner", "loser"):
            column = f"{side}_{source_key}_odds"
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype("Float64")
        coerced[f"{source_key}_pair_available"] = coerced[f"{source_key}_pair_available"].astype(
            "bool"
        )
    text_columns = [
        "tournament",
        "surface",
        "round",
        "winner",
        "loser",
        "odds_source",
        "status_or_comment",
        "source_file",
    ]
    for column in text_columns:
        coerced[column] = coerced[column].astype("string")
    return coerced


def _empty_output_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _error_from_exception(file_path: Path, exc: ValueError) -> IngestionErrorDetail:
    try:
        payload = json.loads(str(exc))
        return IngestionErrorDetail.model_validate(payload)
    except ValueError:
        return IngestionErrorDetail(source_file=file_path.name, message=str(exc))


__all__ = [
    "COLUMN_ALIASES",
    "IngestionErrorDetail",
    "IngestionFailure",
    "IngestionReport",
    "IngestionResult",
    "OUTPUT_COLUMNS",
    "SOURCE_ODDS_OUTPUT_COLUMNS",
    "SOURCE_ODDS_PAIRS",
    "SUPPORTED_EXTENSIONS",
    "discover_tennis_data_files",
    "ingest_tennis_data",
    "write_ingestion_outputs",
]
