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
DEFAULT_TRAIN_INPUT = Path("data/processed/features.parquet")
DEFAULT_MODEL_OUTPUT = Path("models/model_v1.joblib")
DEFAULT_METADATA_OUTPUT = Path("models/model_v1_metadata.json")
DEFAULT_PREDICTIONS_OUTPUT = Path("reports/model_v1_predictions.parquet")
DEFAULT_TRAINING_SUMMARY_OUTPUT = Path("reports/model_v1_training_summary.json")
DEFAULT_EVALUATION_PREDICTIONS = Path("reports/model_v1_predictions.parquet")
DEFAULT_EVALUATION_FEATURES = Path("data/processed/features.parquet")
DEFAULT_METRICS_OUTPUT = Path("reports/model_v1_metrics.json")
DEFAULT_COMPARISON_OUTPUT = Path("reports/model_v1_comparison.json")
DEFAULT_CALIBRATION_OUTPUT = Path("reports/model_v1_calibration.parquet")
DEFAULT_SURFACE_OUTPUT = Path("reports/model_v1_surface_metrics.parquet")
DEFAULT_CALIBRATION_PLOT = Path("reports/model_v1_calibration.png")
DEFAULT_DISTRIBUTION_PLOT = Path("reports/model_v1_probability_distribution.png")


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


@app.command()
def train(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to the model-ready feature Parquet dataset."),
    ] = DEFAULT_TRAIN_INPUT,
    model_output: Annotated[
        Path,
        typer.Option("--model-output", help="Path for the trained model joblib artifact."),
    ] = DEFAULT_MODEL_OUTPUT,
    metadata_output: Annotated[
        Path,
        typer.Option("--metadata-output", help="Path for the model metadata JSON output."),
    ] = DEFAULT_METADATA_OUTPUT,
    predictions_output: Annotated[
        Path,
        typer.Option("--predictions-output", help="Path for partitioned predictions Parquet."),
    ] = DEFAULT_PREDICTIONS_OUTPUT,
    summary_output: Annotated[
        Path,
        typer.Option("--summary-output", help="Path for the training summary JSON output."),
    ] = DEFAULT_TRAINING_SUMMARY_OUTPUT,
) -> None:
    """Train the baseline chronological probability model."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.train import train_probability_model, write_training_outputs

    features = pd.read_parquet(input)
    try:
        result = train_probability_model(
            features,
            input_dataset_path=input,
            model_output_path=model_output,
            prediction_output_path=predictions_output,
        )
    except ValueError as exc:
        console.print(f"[red]Training failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    write_training_outputs(
        result,
        model_output,
        metadata_output,
        predictions_output,
        summary_output,
    )
    console.print(
        "Trained "
        f"{result.metadata.model_version} with "
        f"{result.summary.train_rows} train, "
        f"{result.summary.validation_rows} validation, "
        f"{result.summary.test_rows} test row(s) -> {model_output}"
    )
    console.print(f"Metadata -> {metadata_output}")
    console.print(f"Predictions -> {predictions_output}")
    console.print(f"Summary -> {summary_output}")


@app.command()
def evaluate(
    predictions: Annotated[
        Path,
        typer.Option("--predictions", help="Path to model prediction Parquet output."),
    ] = DEFAULT_EVALUATION_PREDICTIONS,
    features: Annotated[
        Path | None,
        typer.Option("--features", help="Optional feature Parquet for Elo baseline join."),
    ] = DEFAULT_EVALUATION_FEATURES,
    metrics_output: Annotated[
        Path,
        typer.Option("--metrics-output", help="Path for model metrics JSON."),
    ] = DEFAULT_METRICS_OUTPUT,
    comparison_output: Annotated[
        Path,
        typer.Option("--comparison-output", help="Path for baseline comparison JSON."),
    ] = DEFAULT_COMPARISON_OUTPUT,
    calibration_output: Annotated[
        Path,
        typer.Option("--calibration-output", help="Path for calibration table Parquet."),
    ] = DEFAULT_CALIBRATION_OUTPUT,
    surface_output: Annotated[
        Path,
        typer.Option("--surface-output", help="Path for surface metrics Parquet."),
    ] = DEFAULT_SURFACE_OUTPUT,
    calibration_plot: Annotated[
        Path,
        typer.Option("--calibration-plot", help="Path for calibration PNG."),
    ] = DEFAULT_CALIBRATION_PLOT,
    distribution_plot: Annotated[
        Path,
        typer.Option("--distribution-plot", help="Path for probability distribution PNG."),
    ] = DEFAULT_DISTRIBUTION_PLOT,
) -> None:
    """Evaluate saved probability predictions."""
    if not predictions.exists():
        console.print(f"[red]Predictions file does not exist: {predictions}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.evaluate import (
        EvaluationOutputPaths,
        evaluate_predictions,
        join_elo_baseline,
        write_evaluation_artifacts,
    )

    prediction_rows = pd.read_parquet(predictions)
    should_join_elo = (
        features is not None
        and features.exists()
        and "elo_expected_player_1" not in prediction_rows
    )
    if should_join_elo:
        assert features is not None
        try:
            prediction_rows = join_elo_baseline(prediction_rows, pd.read_parquet(features))
        except ValueError as exc:
            console.print(f"[red]Elo baseline join failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc

    try:
        model_version = str(prediction_rows["model_version"].iloc[0])
        result = evaluate_predictions(prediction_rows, model_version=model_version)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    output_paths = EvaluationOutputPaths(
        metrics_output=metrics_output,
        comparison_output=comparison_output,
        calibration_output=calibration_output,
        surface_output=surface_output,
        calibration_plot=calibration_plot,
        distribution_plot=distribution_plot,
    )
    write_evaluation_artifacts(result, output_paths, predictions=prediction_rows)

    test_metrics = result.metrics_report.get("primary_test_metrics", {})
    comparison = result.comparison_report
    elo_test = comparison.get("elo_baseline", {}).get("partitions", {}).get("test", {})
    bookmaker_test = (
        comparison.get("bookmaker_no_vig_baseline", {}).get("partitions", {}).get("test", {})
    )
    console.print(f"Model version: {model_version}")
    console.print(f"Test rows: {test_metrics.get('sample_count', 'N/A')}")
    console.print(f"Test log loss: {_display_metric(test_metrics.get('log_loss'))}")
    console.print(f"Test Brier score: {_display_metric(test_metrics.get('brier_score'))}")
    console.print(f"Test accuracy: {_display_metric(test_metrics.get('accuracy'))}")
    console.print(f"Test ROC AUC: {_display_metric(test_metrics.get('roc_auc'))}")
    console.print(
        "Test calibration error: "
        f"{_display_metric(test_metrics.get('expected_calibration_error'))}"
    )
    console.print(
        "Elo log-loss improvement: "
        f"{_display_metric(elo_test.get('log_loss_improvement'))}"
    )
    console.print(
        "Bookmaker log-loss improvement: "
        f"{_display_metric(bookmaker_test.get('log_loss_improvement'))}"
    )
    console.print(f"Valid-odds test rows: {bookmaker_test.get('rows_with_valid_odds', 'N/A')}")
    console.print(f"Metrics -> {metrics_output}")
    console.print(f"Comparison -> {comparison_output}")
    console.print(f"Calibration -> {calibration_output}")
    console.print(f"Surface metrics -> {surface_output}")
    console.print(f"Calibration plot -> {calibration_plot}")
    console.print(f"Distribution plot -> {distribution_plot}")


def _display_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    app()
