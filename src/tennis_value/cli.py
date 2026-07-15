"""Command-line interface for Tennis Value."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from tennis_value import __version__
from tennis_value.ingest import IngestionFailure, ingest_tennis_data, write_ingestion_outputs

app = typer.Typer(help="Tennis Value research tooling.")
console = Console()
DEFAULT_INGEST_INPUT = Path("data/raw/tennis_data")
DEFAULT_INGEST_OUTPUT = Path("data/processed/raw_matches.parquet")
DEFAULT_INGEST_REPORT = Path("reports/ingestion_report.json")


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


if __name__ == "__main__":
    app()
