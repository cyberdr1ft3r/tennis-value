"""Command-line interface for Tennis Value."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import pandas as pd
import typer
from rich.console import Console

from tennis_value import __version__
from tennis_value.cleaning import clean_matches, write_cleaning_outputs
from tennis_value.config import (
    BacktestConfig,
    BacktestPartition,
    EloConfig,
    FeatureConfig,
    ValueThresholds,
)
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
DEFAULT_VALUE_PREDICTIONS = Path("reports/model_v1_predictions.parquet")
DEFAULT_VALUE_OUTPUT = Path("reports/model_v1_value_assessments.parquet")
DEFAULT_VALUE_SUMMARY = Path("reports/model_v1_value_summary.json")
DEFAULT_BACKTEST_INPUT = Path("reports/model_v1_value_assessments.parquet")
DEFAULT_BACKTEST_BETS = Path("reports/backtest_bets.parquet")
DEFAULT_BACKTEST_SUMMARY = Path("reports/backtest_summary.json")
DEFAULT_BACKTEST_SURFACE = Path("reports/backtest_by_surface.parquet")
DEFAULT_BACKTEST_EDGE = Path("reports/backtest_by_edge_bucket.parquet")
DEFAULT_BACKTEST_ODDS = Path("reports/backtest_by_odds_bucket.parquet")
DEFAULT_BACKTEST_BANKROLL_PLOT = Path("reports/backtest_bankroll_curve.png")
DEFAULT_BACKTEST_DRAWDOWN_PLOT = Path("reports/backtest_drawdown.png")
DEFAULT_TRAIN_V2_INPUT = Path("data/processed/features.parquet")
DEFAULT_MODEL_V2_OUTPUT = Path("models/model_v2.joblib")
DEFAULT_MODEL_V2_METADATA = Path("models/model_v2_metadata.json")
DEFAULT_MODEL_V2_PREDICTIONS = Path("reports/model_v2_predictions.parquet")
DEFAULT_MODEL_V2_METRICS = Path("reports/model_v2_walk_forward_metrics.json")
DEFAULT_MODEL_V2_CORRECTIONS = Path("reports/model_v2_corrections.parquet")
DEFAULT_MODEL_V2_CALIBRATION = Path("reports/model_v2_calibration.parquet")
DEFAULT_MODEL_V2_CORRECTION_PLOT = Path("reports/model_v2_correction_distribution.png")
DEFAULT_DIAGNOSE_V2_BOOTSTRAP = Path("reports/model_v2_bootstrap_significance.json")
DEFAULT_DIAGNOSE_V2_ABLATION_METRICS = Path("reports/model_v2_ablation_metrics.parquet")
DEFAULT_DIAGNOSE_V2_ABLATION_SUMMARY = Path("reports/model_v2_ablation_summary.json")
DEFAULT_DIAGNOSE_V2_COEFFICIENTS = Path("reports/model_v2_coefficients.parquet")
DEFAULT_DIAGNOSE_V2_CORRECTION_DIAGNOSTICS = Path(
    "reports/model_v2_correction_diagnostics.parquet"
)
DEFAULT_DIAGNOSE_V2_CORRECTION_BUCKETS = Path("reports/model_v2_correction_buckets.parquet")
DEFAULT_DIAGNOSE_V2_ODDS_QUALITY = Path("reports/model_v2_odds_quality_metrics.parquet")
DEFAULT_DIAGNOSE_V2_SUMMARY = Path("reports/model_v2_diagnostic_summary.json")
DEFAULT_DIAGNOSE_V2_BOOTSTRAP_PLOT = Path("reports/model_v2_bootstrap_distribution.png")
DEFAULT_DIAGNOSE_V2_ABLATION_PLOT = Path("reports/model_v2_ablation_log_loss.png")
DEFAULT_DIAGNOSE_V2_CORRECTION_PLOT = Path("reports/model_v2_correction_performance.png")
DEFAULT_ODDS_AUDIT_SUMMARY = Path("reports/odds_source_audit.json")
DEFAULT_ODDS_AUDIT_ROWS = Path("reports/odds_quality_rows.parquet")
DEFAULT_ODDS_AUDIT_OVERROUND = Path("reports/odds_overround_by_source.parquet")
DEFAULT_ODDS_SIDE_INTEGRITY_SUMMARY = Path("reports/odds_side_integrity_summary.json")
DEFAULT_ODDS_SIDE_INTEGRITY_ROWS = Path("reports/odds_side_integrity_rows.parquet")
DEFAULT_ODDS_SIDE_MANUAL_REVIEW = Path("reports/odds_side_manual_review.csv")
DEFAULT_MODEL_V2_1_OUTPUT = Path("models/model_v2_1_form_workload.joblib")
DEFAULT_MODEL_V2_1_METADATA = Path("models/model_v2_1_form_workload_metadata.json")
DEFAULT_MODEL_V2_1_PREDICTIONS = Path("reports/model_v2_1_predictions.parquet")
DEFAULT_MODEL_V2_1_ARCHITECTURE = Path("reports/model_v2_1_architecture_metrics.parquet")
DEFAULT_MODEL_V2_1_BOOTSTRAP = Path("reports/model_v2_1_block_bootstrap.json")
DEFAULT_MODEL_V2_1_CORRECTION = Path("reports/model_v2_1_correction_direction.parquet")
DEFAULT_MODEL_V2_1_ODDS = Path("reports/model_v2_1_odds_sensitivity.parquet")
DEFAULT_MODEL_V2_1_SUMMARY = Path("reports/model_v2_1_summary.json")
DEFAULT_MODEL_V2_1_ARCHITECTURE_PLOT = Path("reports/model_v2_1_architecture_comparison.png")
DEFAULT_MODEL_V2_1_CORRECTION_PLOT = Path("reports/model_v2_1_correction_calibration.png")
DEFAULT_MARKET_ANCHOR_COVERAGE = Path("reports/market_anchor_coverage.parquet")
DEFAULT_MARKET_ANCHOR_METRICS = Path("reports/market_anchor_metrics.parquet")
DEFAULT_MARKET_ANCHOR_COMMON_ROWS = Path("reports/market_anchor_common_rows.parquet")
DEFAULT_MARKET_ANCHOR_BOOTSTRAP = Path("reports/market_anchor_block_bootstrap.json")
DEFAULT_MARKET_ANCHOR_SOURCE_DIAGNOSTICS = Path(
    "reports/market_anchor_source_diagnostics.json"
)
DEFAULT_MARKET_ANCHOR_DISAGREEMENT = Path("reports/market_anchor_probability_disagreement.parquet")
DEFAULT_MARKET_ANCHOR_SUMMARY = Path("reports/market_anchor_benchmark_summary.json")
DEFAULT_MARKET_ANCHOR_LOG_LOSS_PLOT = Path("reports/market_anchor_log_loss_comparison.png")
DEFAULT_MARKET_ANCHOR_IMPROVEMENT_PLOT = Path(
    "reports/market_anchor_improvement_comparison.png"
)


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


@app.command("assess-value")
def assess_value(
    predictions: Annotated[
        Path,
        typer.Option("--predictions", help="Path to model prediction Parquet output."),
    ] = DEFAULT_VALUE_PREDICTIONS,
    output: Annotated[
        Path,
        typer.Option("--output", help="Path for value assessment Parquet output."),
    ] = DEFAULT_VALUE_OUTPUT,
    summary: Annotated[
        Path,
        typer.Option("--summary", help="Path for value assessment summary JSON."),
    ] = DEFAULT_VALUE_SUMMARY,
    minimum_probability: Annotated[
        float | None,
        typer.Option("--minimum-probability", help="Override minimum model probability."),
    ] = None,
    minimum_edge: Annotated[
        float | None,
        typer.Option("--minimum-edge", help="Override minimum edge versus no-vig market."),
    ] = None,
    minimum_expected_value: Annotated[
        float | None,
        typer.Option("--minimum-expected-value", help="Override minimum theoretical EV."),
    ] = None,
    minimum_odds: Annotated[
        float | None,
        typer.Option("--minimum-odds", help="Override minimum decimal odds."),
    ] = None,
    maximum_odds: Annotated[
        float | None,
        typer.Option("--maximum-odds", help="Override maximum decimal odds."),
    ] = None,
) -> None:
    """Assess theoretical betting value in saved probability predictions."""
    if not predictions.exists():
        console.print(f"[red]Predictions file does not exist: {predictions}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.value import assess_predictions_with_summary, write_value_outputs

    default_thresholds = ValueThresholds()
    try:
        thresholds = ValueThresholds(
            min_model_probability=(
                default_thresholds.min_model_probability
                if minimum_probability is None
                else minimum_probability
            ),
            min_edge=default_thresholds.min_edge if minimum_edge is None else minimum_edge,
            min_expected_value=(
                default_thresholds.min_expected_value
                if minimum_expected_value is None
                else minimum_expected_value
            ),
            min_odds=default_thresholds.min_odds if minimum_odds is None else minimum_odds,
            max_odds=default_thresholds.max_odds if maximum_odds is None else maximum_odds,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid value thresholds: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        prediction_rows = pd.read_parquet(predictions)
        result = assess_predictions_with_summary(prediction_rows, thresholds)
    except (ImportError, KeyError, ValueError) as exc:
        console.print(f"[red]Value assessment failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    write_value_outputs(result, output, summary)

    top_skip_reasons = sorted(
        result.summary.skip_reason_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:5]
    console.print("Theoretical paper-value assessments")
    console.print(f"Model version: {result.summary.model_version or 'N/A'}")
    console.print(f"Rows assessed: {result.summary.rows_assessed}")
    console.print(f"Rows with valid odds: {result.summary.rows_with_valid_odds}")
    console.print(f"Recommendations: {result.summary.rows_with_recommendations}")
    console.print(f"Recommendation rate: {result.summary.recommendation_rate:.6f}")
    console.print(
        f"Average recommended odds: {_display_metric(result.summary.average_recommended_odds)}"
    )
    console.print(f"Average edge: {_display_metric(result.summary.average_recommended_edge)}")
    console.print(
        "Average expected value: "
        f"{_display_metric(result.summary.average_recommended_expected_value)}"
    )
    if top_skip_reasons:
        console.print(
            "Top skip reasons: "
            + ", ".join(f"{reason}={count}" for reason, count in top_skip_reasons)
        )
    console.print(f"Assessments -> {output}")
    console.print(f"Summary -> {summary}")


@app.command()
def backtest(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to Task 9 value assessment Parquet output."),
    ] = DEFAULT_BACKTEST_INPUT,
    bets_output: Annotated[
        Path,
        typer.Option("--bets-output", help="Path for bet ledger Parquet output."),
    ] = DEFAULT_BACKTEST_BETS,
    summary_output: Annotated[
        Path,
        typer.Option("--summary-output", help="Path for backtest summary JSON."),
    ] = DEFAULT_BACKTEST_SUMMARY,
    surface_output: Annotated[
        Path,
        typer.Option("--surface-output", help="Path for surface grouped Parquet output."),
    ] = DEFAULT_BACKTEST_SURFACE,
    edge_output: Annotated[
        Path,
        typer.Option("--edge-output", help="Path for edge-bucket grouped Parquet output."),
    ] = DEFAULT_BACKTEST_EDGE,
    odds_output: Annotated[
        Path,
        typer.Option("--odds-output", help="Path for odds-bucket grouped Parquet output."),
    ] = DEFAULT_BACKTEST_ODDS,
    bankroll_plot: Annotated[
        Path,
        typer.Option("--bankroll-plot", help="Path for bankroll curve PNG output."),
    ] = DEFAULT_BACKTEST_BANKROLL_PLOT,
    drawdown_plot: Annotated[
        Path,
        typer.Option("--drawdown-plot", help="Path for drawdown PNG output."),
    ] = DEFAULT_BACKTEST_DRAWDOWN_PLOT,
    starting_bankroll: Annotated[
        float | None,
        typer.Option("--starting-bankroll", help="Override starting paper bankroll."),
    ] = None,
    stake_fraction: Annotated[
        float | None,
        typer.Option("--stake-fraction", help="Override flat stake fraction."),
    ] = None,
    partition: Annotated[
        str | None,
        typer.Option("--partition", help="Backtest partition to simulate."),
    ] = None,
) -> None:
    """Run a historical flat-stake paper backtest."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.backtest import (
        BacktestOutputPaths,
        run_backtest,
        write_backtest_artifacts,
    )

    default_config = BacktestConfig()
    try:
        selected_partition: BacktestPartition = _parse_backtest_partition(
            default_config.partition if partition is None else partition
        )
        config = BacktestConfig(
            starting_bankroll=(
                default_config.starting_bankroll
                if starting_bankroll is None
                else starting_bankroll
            ),
            flat_stake_fraction=(
                default_config.flat_stake_fraction if stake_fraction is None else stake_fraction
            ),
            partition=selected_partition,
            retirement_policy=default_config.retirement_policy,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid backtest configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        assessments = pd.read_parquet(input)
        result = run_backtest(assessments, config)
    except (ImportError, KeyError, ValueError) as exc:
        console.print(f"[red]Backtest failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    output_paths = BacktestOutputPaths(
        bets_output=bets_output,
        summary_output=summary_output,
        surface_output=surface_output,
        edge_output=edge_output,
        odds_output=odds_output,
        bankroll_plot=bankroll_plot,
        drawdown_plot=drawdown_plot,
    )
    write_backtest_artifacts(result, output_paths)

    summary = result.summary
    console.print("Historical paper backtest")
    console.print(f"Model version: {summary.get('model_version') or 'N/A'}")
    console.print(f"Partition: {summary['partition']}")
    console.print(f"Starting bankroll: {_display_metric(summary['starting_bankroll'])}")
    console.print(f"Bets placed: {summary['bets_placed']}")
    console.print(
        f"Wins / losses / voids: {summary['wins']} / {summary['losses']} / {summary['voids']}"
    )
    console.print(f"Total staked: {_display_metric(summary['total_staked'])}")
    console.print(f"Profit or loss: {_display_metric(summary['profit_loss'])}")
    console.print(f"ROI: {_display_metric(summary['roi'])}")
    console.print(f"Ending bankroll: {_display_metric(summary['ending_bankroll'])}")
    console.print(f"Maximum drawdown: {_display_metric(summary['maximum_drawdown'])}")
    console.print(f"Average odds: {_display_metric(summary['average_odds'])}")
    console.print(f"Bets -> {bets_output}")
    console.print(f"Summary -> {summary_output}")
    console.print(f"By surface -> {surface_output}")
    console.print(f"By edge bucket -> {edge_output}")
    console.print(f"By odds bucket -> {odds_output}")
    console.print(f"Bankroll plot -> {bankroll_plot}")
    console.print(f"Drawdown plot -> {drawdown_plot}")


@app.command("train-v2")
def train_v2(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to model-ready feature Parquet dataset."),
    ] = DEFAULT_TRAIN_V2_INPUT,
    model_output: Annotated[
        Path,
        typer.Option("--model-output", help="Path for trained Model v2 joblib artifact."),
    ] = DEFAULT_MODEL_V2_OUTPUT,
    metadata_output: Annotated[
        Path,
        typer.Option("--metadata-output", help="Path for Model v2 metadata JSON."),
    ] = DEFAULT_MODEL_V2_METADATA,
    predictions_output: Annotated[
        Path,
        typer.Option("--predictions-output", help="Path for walk-forward predictions Parquet."),
    ] = DEFAULT_MODEL_V2_PREDICTIONS,
    metrics_output: Annotated[
        Path,
        typer.Option("--metrics-output", help="Path for walk-forward metrics JSON."),
    ] = DEFAULT_MODEL_V2_METRICS,
    corrections_output: Annotated[
        Path,
        typer.Option("--corrections-output", help="Path for correction diagnostics Parquet."),
    ] = DEFAULT_MODEL_V2_CORRECTIONS,
    calibration_output: Annotated[
        Path,
        typer.Option("--calibration-output", help="Path for calibration table Parquet."),
    ] = DEFAULT_MODEL_V2_CALIBRATION,
    correction_plot: Annotated[
        Path,
        typer.Option("--correction-plot", help="Path for correction-distribution PNG."),
    ] = DEFAULT_MODEL_V2_CORRECTION_PLOT,
) -> None:
    """Train the market-anchored walk-forward Model v2 experiment."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.train_v2 import (
        ModelV2OutputPaths,
        train_model_v2,
        write_model_v2_artifacts,
    )

    features = pd.read_parquet(input)
    model_v1_predictions = (
        pd.read_parquet(DEFAULT_PREDICTIONS_OUTPUT)
        if DEFAULT_PREDICTIONS_OUTPUT.exists()
        else None
    )
    try:
        result = train_model_v2(
            features,
            input_dataset_path=input,
            model_v1_predictions=model_v1_predictions,
        )
    except ValueError as exc:
        console.print(f"[red]Model v2 training failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    output_paths = ModelV2OutputPaths(
        model_output=model_output,
        metadata_output=metadata_output,
        predictions_output=predictions_output,
        metrics_output=metrics_output,
        corrections_output=corrections_output,
        calibration_output=calibration_output,
        correction_distribution_plot=correction_plot,
    )
    write_model_v2_artifacts(result, output_paths)

    console.print("Market-anchored Model v2 walk-forward results")
    console.print(
        "year | rows | v2 log loss | market log loss | improvement | "
        "v2 brier | market brier | AUC | calibration error | avg abs correction"
    )
    for fold in result.folds:
        metrics = fold.metrics
        v2_metrics = metrics["model_v2"]
        market_metrics = metrics["market"]
        correction = metrics["correction_diagnostics"]
        calibration = result.metrics_report["calibration_summary"].get(
            str(fold.evaluation_year),
            {},
        )
        console.print(
            f"{fold.evaluation_year} | "
            f"{metrics['sample_count']} | "
            f"{_display_metric(v2_metrics['log_loss'])} | "
            f"{_display_metric(market_metrics['log_loss'])} | "
            f"{_display_metric(metrics['log_loss_improvement_vs_market'])} | "
            f"{_display_metric(v2_metrics['brier_score'])} | "
            f"{_display_metric(market_metrics['brier_score'])} | "
            f"{_display_metric(v2_metrics['roc_auc'])} | "
            f"{_display_metric(calibration.get('expected_calibration_error'))} | "
            f"{_display_metric(correction['mean_absolute_correction'])}"
        )
    console.print(f"Model -> {model_output}")
    console.print(f"Metadata -> {metadata_output}")
    console.print(f"Predictions -> {predictions_output}")
    console.print(f"Metrics -> {metrics_output}")
    console.print(f"Corrections -> {corrections_output}")
    console.print(f"Calibration -> {calibration_output}")
    console.print(f"Correction plot -> {correction_plot}")


@app.command("diagnose-v2")
def diagnose_v2(
    predictions: Annotated[
        Path,
        typer.Option("--predictions", help="Path to Model v2 predictions Parquet."),
    ] = DEFAULT_MODEL_V2_PREDICTIONS,
    features: Annotated[
        Path,
        typer.Option("--features", help="Path to model-ready feature Parquet dataset."),
    ] = DEFAULT_TRAIN_V2_INPUT,
    bootstrap_samples: Annotated[
        int,
        typer.Option("--bootstrap-samples", help="Paired bootstrap sample count."),
    ] = 10_000,
) -> None:
    """Run Model v2 significance, ablation, and correction diagnostics."""
    if not predictions.exists():
        console.print(f"[red]Predictions file does not exist: {predictions}[/red]")
        raise typer.Exit(code=1)
    if not features.exists():
        console.print(f"[red]Features file does not exist: {features}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.diagnose_v2 import (
        DiagnosticOutputPaths,
        run_diagnostics,
        write_diagnostic_artifacts,
    )

    try:
        result = run_diagnostics(
            predictions=pd.read_parquet(predictions),
            features=pd.read_parquet(features),
            bootstrap_samples=bootstrap_samples,
        )
    except ValueError as exc:
        console.print(f"[red]Model v2 diagnostics failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    paths = DiagnosticOutputPaths(
        bootstrap_significance=DEFAULT_DIAGNOSE_V2_BOOTSTRAP,
        ablation_metrics=DEFAULT_DIAGNOSE_V2_ABLATION_METRICS,
        ablation_summary=DEFAULT_DIAGNOSE_V2_ABLATION_SUMMARY,
        coefficients=DEFAULT_DIAGNOSE_V2_COEFFICIENTS,
        correction_diagnostics=DEFAULT_DIAGNOSE_V2_CORRECTION_DIAGNOSTICS,
        correction_buckets=DEFAULT_DIAGNOSE_V2_CORRECTION_BUCKETS,
        odds_quality_metrics=DEFAULT_DIAGNOSE_V2_ODDS_QUALITY,
        diagnostic_summary=DEFAULT_DIAGNOSE_V2_SUMMARY,
        bootstrap_distribution_plot=DEFAULT_DIAGNOSE_V2_BOOTSTRAP_PLOT,
        ablation_log_loss_plot=DEFAULT_DIAGNOSE_V2_ABLATION_PLOT,
        correction_performance_plot=DEFAULT_DIAGNOSE_V2_CORRECTION_PLOT,
    )
    write_diagnostic_artifacts(result, paths)

    console.print("Model v2 diagnostic analysis")
    segments = result.bootstrap_significance["segments"]
    for label, segment in segments.items():
        log_loss = segment["log_loss"]
        console.print(
            f"{label}: improvement={_display_metric(log_loss['mean_improvement'])}, "
            f"95% CI=[{_display_metric(log_loss['ci_lower'])}, "
            f"{_display_metric(log_loss['ci_upper'])}], "
            f"P(v2 > market)={_display_metric(log_loss['probability_model_beats_market'])}"
        )
    best_by_year = result.ablation_summary["best_variant_by_year"]
    console.print(
        "Best ablation variant by year: "
        + ", ".join(f"{year}={item['variant']}" for year, item in best_by_year.items())
    )
    full_vs_recalibration = result.ablation_summary["full_model_v2_vs_market_recalibration"]
    console.print(
        "Full v2 beats market-only recalibration: "
        + ", ".join(
            f"{year}={item['full_beats_recalibration']}"
            for year, item in full_vs_recalibration.items()
        )
    )
    summary = result.diagnostic_summary
    console.print(
        "Best/worst correction buckets: "
        f"{summary['best_correction_bucket']['correction_bucket']} / "
        f"{summary['worst_correction_bucket']['correction_bucket']}"
    )
    console.print(f"Bootstrap significance -> {paths.bootstrap_significance}")
    console.print(f"Ablation metrics -> {paths.ablation_metrics}")
    console.print(f"Diagnostic summary -> {paths.diagnostic_summary}")


@app.command("audit-odds")
def audit_odds(
    raw_input: Annotated[
        Path,
        typer.Option("--raw-input", help="Directory containing local Tennis-Data files."),
    ] = DEFAULT_INGEST_INPUT,
    processed_input: Annotated[
        Path,
        typer.Option("--processed-input", help="Optional processed feature Parquet dataset."),
    ] = DEFAULT_TRAIN_V2_INPUT,
) -> None:
    """Audit current bookmaker odds-source selection and odds quality."""
    if not raw_input.exists():
        console.print(f"[red]Raw input directory does not exist: {raw_input}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.odds_audit import (
        OddsAuditOutputPaths,
        audit_odds_sources,
        write_odds_audit_artifacts,
    )

    result = audit_odds_sources(raw_input, processed_input)
    paths = OddsAuditOutputPaths(
        summary=DEFAULT_ODDS_AUDIT_SUMMARY,
        quality_rows=DEFAULT_ODDS_AUDIT_ROWS,
        overround_by_source=DEFAULT_ODDS_AUDIT_OVERROUND,
        side_integrity_summary=DEFAULT_ODDS_SIDE_INTEGRITY_SUMMARY,
        side_integrity_rows=DEFAULT_ODDS_SIDE_INTEGRITY_ROWS,
        manual_review_rows=DEFAULT_ODDS_SIDE_MANUAL_REVIEW,
    )
    write_odds_audit_artifacts(result, paths)
    summary = result.summary
    console.print("Odds source audit")
    console.print(f"Selection policy: {summary.selected_source}")
    console.print(
        "Rows by source: "
        + ", ".join(f"{source}={count}" for source, count in summary.rows_by_source.items())
    )
    console.print(f"Fallback rows: {summary.fallback_rows}")
    console.print(
        "Suspicious rows: "
        + ", ".join(
            f"{flag}={count}" for flag, count in summary.quality_flag_counts.items()
        )
    )
    if result.side_integrity_summary is not None:
        console.print(
            "Verified odds-side mapping failures: "
            f"{result.side_integrity_summary['total_odds_side_mapping_failures']}"
        )
    console.print(f"Summary -> {paths.summary}")
    console.print(f"Quality rows -> {paths.quality_rows}")
    console.print(f"Overround by source -> {paths.overround_by_source}")
    console.print(f"Side integrity -> {paths.side_integrity_summary}")


@app.command("train-v2-1")
def train_v2_1(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to model-ready feature Parquet dataset."),
    ] = DEFAULT_TRAIN_V2_INPUT,
    bootstrap_samples: Annotated[
        int,
        typer.Option("--bootstrap-samples", help="Paired block-bootstrap sample count."),
    ] = 10_000,
) -> None:
    """Train the focused Model v2.1 form/workload market-correction experiment."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.train_v2_1 import (
        ModelV21OutputPaths,
        train_model_v2_1,
        write_model_v2_1_artifacts,
    )

    try:
        result = train_model_v2_1(
            pd.read_parquet(input),
            input_dataset_path=input,
            bootstrap_samples=bootstrap_samples,
        )
    except ValueError as exc:
        console.print(f"[red]Model v2.1 training failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    paths = ModelV21OutputPaths(
        model_output=DEFAULT_MODEL_V2_1_OUTPUT,
        metadata_output=DEFAULT_MODEL_V2_1_METADATA,
        predictions_output=DEFAULT_MODEL_V2_1_PREDICTIONS,
        architecture_metrics=DEFAULT_MODEL_V2_1_ARCHITECTURE,
        block_bootstrap=DEFAULT_MODEL_V2_1_BOOTSTRAP,
        correction_direction=DEFAULT_MODEL_V2_1_CORRECTION,
        odds_sensitivity=DEFAULT_MODEL_V2_1_ODDS,
        summary=DEFAULT_MODEL_V2_1_SUMMARY,
        architecture_comparison_plot=DEFAULT_MODEL_V2_1_ARCHITECTURE_PLOT,
        correction_calibration_plot=DEFAULT_MODEL_V2_1_CORRECTION_PLOT,
    )
    write_model_v2_1_artifacts(result, paths)

    metrics = result.architecture_metrics
    console.print("Model v2.1 form/workload market-correction experiment")
    console.print(
        "year | raw market | recalibration | free form/workload | fixed offset | capped fixed"
    )
    for year in ("2023", "2024", "2025"):
        group = metrics[metrics["segment"].astype(str).eq(year)]
        values = {
            str(row["architecture"]): row["model_log_loss"] for _, row in group.iterrows()
        }
        raw_market = group["raw_market_log_loss"].iloc[0] if not group.empty else None
        console.print(
            f"{year} | "
            f"{_display_metric(raw_market)} | "
            f"{_display_metric(values.get('market_recalibration'))} | "
            f"{_display_metric(values.get('free_form_workload'))} | "
            f"{_display_metric(values.get('fixed_offset_form_workload'))} | "
            f"{_display_metric(values.get('fixed_offset_form_workload_capped'))}"
        )
    bootstrap = result.block_bootstrap["comparisons"]
    capped = result.summary["capped_rows"]
    capped_rate = result.summary["capped_row_rate"]
    slopes = result.summary["correction_direction_slope_by_architecture"]
    console.print("Pooled block-bootstrap vs raw market:")
    for architecture, comparisons in bootstrap.items():
        combined = comparisons["raw_market"]["combined_2023_2025"]
        console.print(
            f"{architecture}: "
            f"{_display_metric(combined['mean_log_loss_improvement'])}, "
            f"95% CI=[{_display_metric(combined['log_loss_ci_lower'])}, "
            f"{_display_metric(combined['log_loss_ci_upper'])}]"
        )
    console.print(
        "Correction-direction slope: "
        + ", ".join(f"{name}={_display_metric(value)}" for name, value in slopes.items())
    )
    console.print(f"Capped rows: {capped} ({capped_rate:.2%})")
    console.print(f"Model -> {paths.model_output}")
    console.print(f"Predictions -> {paths.predictions_output}")
    console.print(f"Summary -> {paths.summary}")


@app.command("benchmark-markets")
def benchmark_markets(
    input: Annotated[  # noqa: A002
        Path,
        typer.Option("--input", help="Path to model-ready feature Parquet dataset."),
    ] = DEFAULT_TRAIN_V2_INPUT,
    bootstrap_samples: Annotated[
        int,
        typer.Option("--bootstrap-samples", help="Paired block-bootstrap sample count."),
    ] = 10_000,
) -> None:
    """Benchmark the focused form/workload signal across bookmaker market anchors."""
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(code=1)

    from tennis_value.benchmark_markets import (
        MarketBenchmarkOutputPaths,
        run_market_benchmark,
        write_market_benchmark_artifacts,
    )

    try:
        result = run_market_benchmark(
            pd.read_parquet(input),
            bootstrap_samples=bootstrap_samples,
        )
    except ValueError as exc:
        console.print(f"[red]Market benchmark failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    paths = MarketBenchmarkOutputPaths(
        coverage=DEFAULT_MARKET_ANCHOR_COVERAGE,
        metrics=DEFAULT_MARKET_ANCHOR_METRICS,
        common_rows=DEFAULT_MARKET_ANCHOR_COMMON_ROWS,
        block_bootstrap=DEFAULT_MARKET_ANCHOR_BOOTSTRAP,
        source_diagnostics=DEFAULT_MARKET_ANCHOR_SOURCE_DIAGNOSTICS,
        probability_disagreement=DEFAULT_MARKET_ANCHOR_DISAGREEMENT,
        summary=DEFAULT_MARKET_ANCHOR_SUMMARY,
        log_loss_plot=DEFAULT_MARKET_ANCHOR_LOG_LOSS_PLOT,
        improvement_plot=DEFAULT_MARKET_ANCHOR_IMPROVEMENT_PLOT,
    )
    if DEFAULT_ODDS_SIDE_INTEGRITY_SUMMARY.exists():
        import json

        integrity = json.loads(DEFAULT_ODDS_SIDE_INTEGRITY_SUMMARY.read_text(encoding="utf-8"))
        result.summary["odds_side_mapping_failures"] = integrity.get(
            "total_odds_side_mapping_failures"
        )
    write_market_benchmark_artifacts(result, paths)

    metrics = result.metrics
    console.print("source | scope | year | rows | raw market LL | recalibration LL | "
                  "form/workload LL | improvement vs raw | improvement vs recalibration")
    for (source, scope, year), group in metrics.groupby(["source", "scope", "segment"], sort=True):
        if str(year) == "combined_2023_2025":
            continue
        raw = group[group["architecture"] == "raw_market"].iloc[0]
        recal = group[group["architecture"] == "market_recalibration"].iloc[0]
        form = group[group["architecture"] == "free_form_workload"].iloc[0]
        console.print(
            f"{source} | {scope} | {year} | {int(form['sample_count'])} | "
            f"{_display_metric(raw['log_loss'])} | "
            f"{_display_metric(recal['log_loss'])} | "
            f"{_display_metric(form['log_loss'])} | "
            f"{_display_metric(form['log_loss_improvement_vs_raw_market'])} | "
            f"{_display_metric(form['log_loss_improvement_vs_recalibration'])}"
        )
    console.print("Pooled block-bootstrap intervals:")
    for source, scopes in result.block_bootstrap["comparisons"].items():
        for scope, segments in scopes.items():
            pooled = segments["combined_2023_2025"]
            raw = pooled["versus_raw_market"]
            recal = pooled["versus_market_recalibration"]
            console.print(
                f"{source} {scope}: vs raw "
                f"[{_display_metric(raw['log_loss_ci_lower'])}, "
                f"{_display_metric(raw['log_loss_ci_upper'])}], vs recal "
                f"[{_display_metric(recal['log_loss_ci_lower'])}, "
                f"{_display_metric(recal['log_loss_ci_upper'])}]"
            )
    console.print(f"Metrics -> {paths.metrics}")
    console.print(f"Summary -> {paths.summary}")


def _display_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _parse_backtest_partition(value: str) -> BacktestPartition:
    if value in {"train", "validation", "test"}:
        return cast(BacktestPartition, value)
    msg = f"unsupported partition: {value}"
    raise ValueError(msg)


if __name__ == "__main__":
    app()
