"""Command-line interface for Tennis Value."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from tennis_value import __version__
from tennis_value.cleaning import clean_matches, write_cleaning_outputs
from tennis_value.config import EloConfig, FeatureConfig
from tennis_value.elo import add_elo_features_with_report, write_elo_outputs
from tennis_value.features import build_features_with_report, write_feature_outputs
from tennis_value.ingest import IngestionFailure, ingest_tennis_data, write_ingestion_outputs

app = typer.Typer(help="Tennis Value research tooling.")
console = Console()
DEFAULT_INGEST_INPUT = Path("data/raw/tennis_data")
DEFAULT_INGEST_OUTPUT = Path("data/processed/raw_matches.parquet")
DEFAULT_INGEST_REPORT = Path("reports/ingestion_report.json")
DEFAULT_CLEAN_INPUT = Path("data/processed/raw_matches.parquet")
DEFAULT_CLEAN_OUTPUT = Path("data/processed/matches.parquet")
DEFAULT_CLEAN_REPORT = Path("reports/data_quality.json")
DEFAULT_CLEAN_REJECTED = Path("reports/rejected_rows.csv")
DEFAULT_ELO_INPUT = Path("data/processed/matches.parquet")
DEFAULT_ELO_OUTPUT = Path("data/processed/matches_with_elo.parquet")
DEFAULT_ELO_REPORT = Path("reports/elo_quality.json")
DEFAULT_FEATURE_INPUT = Path("data/processed/matches_with_elo.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/processed/features.parquet")
DEFAULT_FEATURE_REPORT = Path("reports/feature_quality.json")


@app.callback()
def main() -> None:
    """Tennis Value command group."""


@app.command()
def version() -> None:
    """Print the installed Tennis Value version."""
    console.print(__version__)


@app.command()
def ingest(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option(
            "--input",
            help="Directory containing local Tennis-Data CSV/XLS/XLSX files.",
        ),
    ] = DEFAULT_INGEST_INPUT,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path for the combined raw-match Parquet output.",
        ),
    ] = DEFAULT_INGEST_OUTPUT,
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            help="Path for the ingestion report JSON output.",
        ),
    ] = DEFAULT_INGEST_REPORT,
) -> None:
    """Load local Tennis-Data files into a raw-match dataset."""
    try:
        result = ingest_tennis_data(input, fail_if_no_rows=True)
    except IngestionFailure as exc:
        console.print("[red]Ingestion failed: no rows returned.[/red]")
        console.print(exc.report.model_dump_json(indent=2))
        raise typer.Exit(code=1) from exc

    write_ingestion_outputs(result, output, report)
    if result.report.files_failed:
        console.print("[red]Ingestion completed with file errors.[/red]")
        console.print(result.report.model_dump_json(indent=2))
        raise typer.Exit(code=1)

    console.print(
        "Loaded "
        f"{result.report.files_loaded} file(s), "
        f"{result.report.rows_returned} row(s) -> {output}"
    )
    console.print(f"Report -> {report}")


@app.command()
def clean(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to the raw ingested Parquet dataset."),
    ] = DEFAULT_CLEAN_INPUT,
    output: Annotated[
        Path,
        typer.Option("--output", help="Path for the canonical matches Parquet output."),
    ] = DEFAULT_CLEAN_OUTPUT,
    report: Annotated[
        Path,
        typer.Option("--report", help="Path for the data-quality report JSON output."),
    ] = DEFAULT_CLEAN_REPORT,
    rejected: Annotated[
        Path,
        typer.Option("--rejected", help="Path for rejected rows CSV output."),
    ] = DEFAULT_CLEAN_REJECTED,
) -> None:
    """Clean raw ingested rows into canonical matches."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    raw_matches = pd.read_parquet(input)
    result = clean_matches(raw_matches)
    write_cleaning_outputs(result, output, report, rejected)
    console.print(
        "Accepted "
        f"{result.quality_report.rows_accepted} / "
        f"{result.quality_report.rows_received} row(s) -> {output}"
    )
    console.print(f"Report -> {report}")
    console.print(f"Rejected rows -> {rejected}")


@app.command("build-elo")
def build_elo(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to the canonical matches Parquet dataset."),
    ] = DEFAULT_ELO_INPUT,
    output: Annotated[
        Path,
        typer.Option("--output", help="Path for the Elo-enriched Parquet output."),
    ] = DEFAULT_ELO_OUTPUT,
    report: Annotated[
        Path,
        typer.Option("--report", help="Path for the Elo quality report JSON output."),
    ] = DEFAULT_ELO_REPORT,
) -> None:
    """Build pre-match overall and surface Elo features."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    matches = pd.read_parquet(input)
    try:
        result = add_elo_features_with_report(matches, EloConfig())
    except ValueError as exc:
        console.print(f"[red]Elo generation failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    write_elo_outputs(result, output, report)
    console.print(
        "Built Elo for "
        f"{result.report.rows_returned} row(s), "
        f"{result.report.eligible_updates} eligible update(s) -> {output}"
    )
    console.print(f"Report -> {report}")


@app.command("build-features")
def build_features(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to the Elo-enriched matches Parquet dataset."),
    ] = DEFAULT_FEATURE_INPUT,
    output: Annotated[
        Path,
        typer.Option("--output", help="Path for the model-ready feature Parquet output."),
    ] = DEFAULT_FEATURE_OUTPUT,
    report: Annotated[
        Path,
        typer.Option("--report", help="Path for the feature quality report JSON output."),
    ] = DEFAULT_FEATURE_REPORT,
) -> None:
    """Build rolling and static model-ready match features."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    matches = pd.read_parquet(input)
    try:
        result = build_features_with_report(matches, FeatureConfig())
    except ValueError as exc:
        console.print(f"[red]Feature generation failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    write_feature_outputs(result, output, report)
    console.print(
        "Built features for "
        f"{result.report.rows_returned} row(s), "
        f"{result.report.eligible_history_updates} eligible history update(s) -> {output}"
    )
    console.print(f"Report -> {report}")


if __name__ == "__main__":
    app()
