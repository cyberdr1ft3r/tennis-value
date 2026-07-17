"""Benchmark focused Model v2.1 signal across bookmaker market anchors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_value.evaluate import (
    build_calibration_table,
    calculate_calibration_summary,
    calculate_probability_metrics,
)
from tennis_value.train_v2 import build_walk_forward_folds, prepare_model_v2_dataset
from tennis_value.train_v2_1 import (
    MODEL_V2_1_FEATURES,
    block_bootstrap_comparison,
)

MARKET_SOURCES = {
    "Bet365": ("player_1_b365_odds", "player_2_b365_odds", "b365_pair_available"),
    "Pinnacle/PS": ("player_1_ps_odds", "player_2_ps_odds", "ps_pair_available"),
    "Average": ("player_1_avg_odds", "player_2_avg_odds", "avg_pair_available"),
}
ARCHITECTURES = ("raw_market", "market_recalibration", "free_form_workload")
SCOPES = ("source_available", "common_all_sources")


@dataclass(frozen=True)
class MarketBenchmarkOutputPaths:
    """Output paths for market-anchor benchmark artifacts."""

    coverage: Path
    metrics: Path
    common_rows: Path
    block_bootstrap: Path
    source_diagnostics: Path
    probability_disagreement: Path
    summary: Path
    log_loss_plot: Path
    improvement_plot: Path


@dataclass(frozen=True)
class MarketBenchmarkResult:
    """Market-anchor benchmark artifact bundle."""

    coverage: pd.DataFrame
    metrics: pd.DataFrame
    common_rows: pd.DataFrame
    block_bootstrap: dict[str, Any]
    source_diagnostics: dict[str, Any]
    probability_disagreement: pd.DataFrame
    summary: dict[str, Any]


def run_market_benchmark(
    features: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
) -> MarketBenchmarkResult:
    """Run multi-bookmaker market-anchor benchmark."""
    input_frame = features.copy(deep=True)
    coverage = build_anchor_coverage(input_frame)
    common_ids = _common_match_ids(input_frame)
    prediction_frames: list[pd.DataFrame] = []
    for source, columns in MARKET_SOURCES.items():
        for scope in SCOPES:
            scoped = _source_frame(input_frame, source, columns, scope, common_ids)
            if scoped.empty:
                continue
            prediction_frames.append(_fit_source_scope(scoped, source, scope))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = build_market_anchor_metrics(predictions)
    bootstrap = build_market_anchor_bootstrap(predictions, samples=bootstrap_samples)
    diagnostics = build_source_diagnostics(input_frame, predictions)
    disagreement = build_probability_disagreement(input_frame, common_ids)
    summary = build_market_benchmark_summary(metrics, bootstrap, diagnostics)
    common_rows = predictions[predictions["scope"] == "common_all_sources"][
        ["match_id", "source", "scope", "evaluation_year"]
    ].drop_duplicates()
    return MarketBenchmarkResult(
        coverage=coverage,
        metrics=metrics,
        common_rows=common_rows,
        block_bootstrap=bootstrap,
        source_diagnostics=diagnostics,
        probability_disagreement=disagreement,
        summary=summary,
    )


def build_anchor_coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Report included and excluded rows by source/year/scope."""
    rows: list[dict[str, Any]] = []
    years = pd.to_datetime(features["match_date"], errors="coerce").dt.year
    common_ids = _common_match_ids(features)
    for source, (_, _, flag) in MARKET_SOURCES.items():
        available = _source_available_mask(features, flag)
        for scope in SCOPES:
            mask = available.copy()
            if scope == "common_all_sources":
                mask &= features["match_id"].isin(common_ids)
            for year in sorted(years.dropna().astype(int).unique()):
                year_mask = years == year
                rows.append(
                    {
                        "source": source,
                        "scope": scope,
                        "year": int(year),
                        "included_rows": int((mask & year_mask).sum()),
                        "excluded_rows": int((~mask & year_mask).sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_market_anchor_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate architecture metrics by source/scope/year and pooled."""
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(["source", "scope", "evaluation_year"], sort=True)
    for (source, scope, year), group in grouped:
        rows.extend(_metrics_rows(str(source), str(scope), str(year), group))
    grouped_pooled = predictions.groupby(["source", "scope"], sort=True)
    for (source, scope), group in grouped_pooled:
        rows.extend(_metrics_rows(str(source), str(scope), "combined_2023_2025", group))
    return pd.DataFrame(rows)


def build_market_anchor_bootstrap(
    predictions: pd.DataFrame,
    *,
    samples: int,
) -> dict[str, Any]:
    """Paired block bootstrap for form/workload against raw market and recalibration."""
    comparisons: dict[str, Any] = {}
    form = predictions[predictions["architecture"] == "free_form_workload"]
    for (source, scope), group in form.groupby(["source", "scope"], sort=True):
        comparisons.setdefault(str(source), {})[str(scope)] = {}
        for label, segment in _year_and_combined(group):
            comparisons[str(source)][str(scope)][label] = {
                "versus_raw_market": block_bootstrap_comparison(
                    segment,
                    candidate_probability_column="model_probability",
                    comparator_probability_column="raw_market_probability",
                    samples=samples,
                ),
                "versus_market_recalibration": block_bootstrap_comparison(
                    segment,
                    candidate_probability_column="model_probability",
                    comparator_probability_column="market_recalibration_probability",
                    samples=samples,
                ),
            }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": samples,
        "seed": 42,
        "block_definition": "evaluation_year + ISO week",
        "comparisons": comparisons,
    }


def build_source_diagnostics(features: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    """Build bookmaker source diagnostics and raw-market winners."""
    source_stats: dict[str, Any] = {}
    for source, (p1_col, p2_col, flag) in MARKET_SOURCES.items():
        available = _source_available_mask(features, flag)
        source_frame = features[available].copy()
        overround = _overround(source_frame[p1_col], source_frame[p2_col])
        source_stats[source] = {
            "valid_paired_rows": int(available.sum()),
            "missing_paired_rows": int((~available).sum()),
            "mean_overround": _nullable_float(overround.mean()),
            "median_overround": _nullable_float(overround.median()),
            "minimum_overround": _nullable_float(overround.min()),
            "maximum_overround": _nullable_float(overround.max()),
            "p01_overround": _nullable_float(overround.quantile(0.01)),
            "p05_overround": _nullable_float(overround.quantile(0.05)),
            "p25_overround": _nullable_float(overround.quantile(0.25)),
            "p50_overround": _nullable_float(overround.quantile(0.50)),
            "p75_overround": _nullable_float(overround.quantile(0.75)),
            "p95_overround": _nullable_float(overround.quantile(0.95)),
            "p99_overround": _nullable_float(overround.quantile(0.99)),
            "rows_below_overround_1_00": int((overround < 1.00).sum()),
            "rows_above_overround_1_12": int((overround > 1.12).sum()),
        }
    raw = predictions[predictions["architecture"] == "raw_market"]
    winners: dict[str, Any] = {}
    for (scope, year), group in raw.groupby(["scope", "evaluation_year"], sort=True):
        source_rows: list[dict[str, Any]] = []
        for grouped_source, source_group in group.groupby("source", sort=True):
            metrics = calculate_probability_metrics(
                source_group["actual_player_1_won"],
                source_group["model_probability"],
            )
            source_rows.append(
                {"source": str(grouped_source), "raw_market_log_loss": metrics.log_loss}
            )
        best = sorted(source_rows, key=lambda row: (row["raw_market_log_loss"], row["source"]))[0]
        winners[f"{scope}_{year}"] = {
            "source": best["source"],
            "raw_market_log_loss": float(best["raw_market_log_loss"]),
        }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_overround": source_stats,
        "lowest_raw_market_log_loss_by_year": winners,
    }


def build_probability_disagreement(features: pd.DataFrame, common_ids: set[str]) -> pd.DataFrame:
    """Report pairwise source probability differences on common rows."""
    common = features[features["match_id"].isin(common_ids)].copy()
    probabilities = {
        source: _no_vig_probability(common[p1], common[p2])
        for source, (p1, p2, _) in MARKET_SOURCES.items()
    }
    rows: list[dict[str, Any]] = []
    for left, right in (
        ("Bet365", "Pinnacle/PS"),
        ("Bet365", "Average"),
        ("Pinnacle/PS", "Average"),
    ):
        diff = probabilities[left] - probabilities[right]
        abs_diff = diff.abs()
        rows.append(
            {
                "pair": f"{left} versus {right}",
                "rows": int(diff.notna().sum()),
                "mean_signed_difference": float(diff.mean()),
                "mean_absolute_difference": float(abs_diff.mean()),
                "median_absolute_difference": float(abs_diff.median()),
                "p95_absolute_difference": float(abs_diff.quantile(0.95)),
                "maximum_absolute_difference": float(abs_diff.max()),
                "pearson_correlation": float(probabilities[left].corr(probabilities[right])),
                "spearman_correlation": float(
                    probabilities[left].corr(probabilities[right], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_market_benchmark_summary(
    metrics: pd.DataFrame,
    bootstrap: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Answer the benchmark interpretation questions."""
    answers: dict[str, Any] = {}
    for source in MARKET_SOURCES:
        for scope in SCOPES:
            pooled = bootstrap["comparisons"].get(source, {}).get(scope, {}).get(
                "combined_2023_2025",
                {},
            )
            answers[f"{source}_{scope}"] = {
                "beats_raw_market": pooled.get("versus_raw_market", {}).get(
                    "log_loss_interval_excludes_zero"
                ),
                "beats_market_recalibration": pooled.get(
                    "versus_market_recalibration",
                    {},
                ).get("log_loss_interval_excludes_zero"),
                "vs_raw_market_ci": [
                    pooled.get("versus_raw_market", {}).get("log_loss_ci_lower"),
                    pooled.get("versus_raw_market", {}).get("log_loss_ci_upper"),
                ],
                "vs_recalibration_ci": [
                    pooled.get("versus_market_recalibration", {}).get("log_loss_ci_lower"),
                    pooled.get("versus_market_recalibration", {}).get("log_loss_ci_upper"),
                ],
            }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_after_inspecting_2023_2025": True,
        "answers": answers,
        "odds_side_mapping_failures": None,
        "lowest_raw_market_log_loss_by_year": diagnostics["lowest_raw_market_log_loss_by_year"],
        "interpretation_warning": "No profitability claim; feature set was selected after review.",
    }


def write_market_benchmark_artifacts(
    result: MarketBenchmarkResult,
    paths: MarketBenchmarkOutputPaths,
) -> None:
    """Write market benchmark artifacts."""
    for path in paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result.coverage.to_parquet(paths.coverage, index=False)
    result.metrics.to_parquet(paths.metrics, index=False)
    result.common_rows.to_parquet(paths.common_rows, index=False)
    paths.block_bootstrap.write_text(
        json.dumps(_jsonable(result.block_bootstrap), indent=2),
        "utf-8",
    )
    paths.source_diagnostics.write_text(
        json.dumps(_jsonable(result.source_diagnostics), indent=2),
        "utf-8",
    )
    result.probability_disagreement.to_parquet(paths.probability_disagreement, index=False)
    paths.summary.write_text(json.dumps(_jsonable(result.summary), indent=2), "utf-8")
    _write_plot(result.metrics, paths.log_loss_plot, "log_loss", "Market Anchor Log Loss")
    _write_plot(
        result.metrics,
        paths.improvement_plot,
        "log_loss_improvement_vs_raw_market",
        "Form/Workload Improvement",
    )


def _fit_source_scope(frame: pd.DataFrame, source: str, scope: str) -> pd.DataFrame:
    prepared = prepare_model_v2_dataset(frame)
    predictions: list[pd.DataFrame] = []
    for fold in build_walk_forward_folds(prepared):
        recalibration = _pipeline(["market_logit_player_1"])
        form = _pipeline(MODEL_V2_1_FEATURES)
        recalibration.fit(fold.train[["market_logit_player_1"]], fold.train["target"])
        form.fit(fold.train[MODEL_V2_1_FEATURES], fold.train["target"])
        raw_probability = fold.evaluation["market_probability_player_1"].astype("float64")
        recal_probability = pd.Series(
            recalibration.predict_proba(fold.evaluation[["market_logit_player_1"]])[:, 1],
            index=fold.evaluation.index,
            dtype="float64",
        )
        form_probability = pd.Series(
            form.predict_proba(fold.evaluation[MODEL_V2_1_FEATURES])[:, 1],
            index=fold.evaluation.index,
            dtype="float64",
        )
        predictions.append(
            _prediction_rows(
                fold.evaluation,
                source,
                scope,
                "raw_market",
                raw_probability,
                recal_probability,
            )
        )
        predictions.append(
            _prediction_rows(
                fold.evaluation,
                source,
                scope,
                "market_recalibration",
                recal_probability,
                recal_probability,
            )
        )
        predictions.append(
            _prediction_rows(
                fold.evaluation,
                source,
                scope,
                "free_form_workload",
                form_probability,
                recal_probability,
            )
        )
    return pd.concat(predictions, ignore_index=True)


def _prediction_rows(
    evaluation: pd.DataFrame,
    source: str,
    scope: str,
    architecture: str,
    probability: pd.Series,
    recalibration_probability: pd.Series,
) -> pd.DataFrame:
    rows = evaluation[
        [
            "match_id",
            "match_date",
            "target",
            "market_probability_player_1",
            "overround",
        ]
    ].copy()
    rows["source"] = source
    rows["scope"] = scope
    rows["architecture"] = architecture
    rows["evaluation_year"] = pd.to_datetime(rows["match_date"]).dt.year.astype("int64")
    rows["iso_week"] = pd.to_datetime(rows["match_date"]).dt.isocalendar().week.astype("int64")
    rows["actual_player_1_won"] = rows["target"].astype("int64")
    rows["raw_market_probability"] = rows["market_probability_player_1"].astype("float64")
    rows["market_recalibration_probability"] = recalibration_probability.astype("float64")
    rows["model_probability"] = probability.astype("float64")
    rows["probability_correction"] = rows["model_probability"] - rows["raw_market_probability"]
    return rows.drop(columns=["target"])


def _source_frame(
    features: pd.DataFrame,
    source: str,
    columns: tuple[str, str, str],
    scope: str,
    common_ids: set[str],
) -> pd.DataFrame:
    p1_col, p2_col, flag_col = columns
    if not {p1_col, p2_col, flag_col}.issubset(features.columns):
        return pd.DataFrame(columns=features.columns)
    mask = features[flag_col].astype(bool)
    if scope == "common_all_sources":
        mask &= features["match_id"].isin(common_ids)
    frame = features.loc[mask].copy()
    frame["player_1_odds"] = frame[p1_col]
    frame["player_2_odds"] = frame[p2_col]
    frame["odds_source"] = source
    return frame


def _metrics_rows(
    source: str,
    scope: str,
    segment: str,
    group: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = group[group["architecture"] == "raw_market"].iloc[0:]
    raw_metrics = calculate_probability_metrics(
        raw["actual_player_1_won"],
        raw["model_probability"],
    )
    recal_ll = None
    form_ll = None
    for architecture, arch_group in group.groupby("architecture", sort=True):
        metrics = calculate_probability_metrics(
            arch_group["actual_player_1_won"],
            arch_group["model_probability"],
        )
        calibration = build_calibration_table(
            arch_group.assign(partition=segment),
            probability_column="model_probability",
            target_column="actual_player_1_won",
        )
        ece = calculate_calibration_summary(calibration).get(segment)
        if architecture == "market_recalibration":
            recal_ll = metrics.log_loss
        if architecture == "free_form_workload":
            form_ll = metrics.log_loss
        rows.append(
            {
                "source": source,
                "scope": scope,
                "segment": segment,
                "architecture": architecture,
                "sample_count": metrics.sample_count,
                "log_loss": metrics.log_loss,
                "brier_score": metrics.brier_score,
                "accuracy": metrics.accuracy,
                "roc_auc": metrics.roc_auc,
                "expected_calibration_error": ece.expected_calibration_error if ece else None,
                "mean_market_probability": float(arch_group["raw_market_probability"].mean()),
                "mean_model_probability": float(arch_group["model_probability"].mean()),
                "mean_absolute_correction": float(
                    arch_group["probability_correction"].abs().mean()
                ),
                "raw_market_log_loss": raw_metrics.log_loss,
                "log_loss_improvement_vs_raw_market": raw_metrics.log_loss - metrics.log_loss,
                "log_loss_improvement_vs_recalibration": None,
            }
        )
    if recal_ll is not None and form_ll is not None:
        for row in rows:
            if row["architecture"] == "free_form_workload":
                row["log_loss_improvement_vs_recalibration"] = recal_ll - form_ll
    return rows


def _pipeline(feature_names: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="median", add_indicator=True),
                                    ),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_names,
                        )
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs"),
            ),
        ]
    )


def _common_match_ids(features: pd.DataFrame) -> set[str]:
    masks = []
    for _, _, flag in MARKET_SOURCES.values():
        masks.append(_source_available_mask(features, flag))
    if not masks:
        return set()
    common = masks[0]
    for mask in masks[1:]:
        common &= mask
    return set(features.loc[common, "match_id"].astype(str))


def _source_available_mask(features: pd.DataFrame, flag: str) -> pd.Series:
    if flag in features:
        return features[flag].astype(bool)
    return pd.Series(False, index=features.index)


def _year_and_combined(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(str(year), group) for year, group in frame.groupby("evaluation_year", sort=True)]
    groups.append(("combined_2023_2025", frame))
    return groups


def _overround(player_1_odds: pd.Series, player_2_odds: pd.Series) -> pd.Series:
    return 1.0 / pd.to_numeric(player_1_odds, errors="coerce") + 1.0 / pd.to_numeric(
        player_2_odds,
        errors="coerce",
    )


def _no_vig_probability(player_1_odds: pd.Series, player_2_odds: pd.Series) -> pd.Series:
    odds_1 = pd.to_numeric(player_1_odds, errors="coerce")
    odds_2 = pd.to_numeric(player_2_odds, errors="coerce")
    raw_1 = 1.0 / odds_1
    raw_2 = 1.0 / odds_2
    return raw_1 / (raw_1 + raw_2)


def _nullable_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _write_plot(metrics: pd.DataFrame, path: Path, value_column: str, title: str) -> None:
    subset = metrics[
        (metrics["segment"] == "combined_2023_2025")
        & (metrics["architecture"] == "free_form_workload")
    ]
    labels = (subset["source"] + " " + subset["scope"]).astype(str).tolist()
    values = subset[value_column].astype(float).tolist()
    image = Image.new("RGB", (900, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 50, 850, 350), outline="black")
    draw.text((70, 18), title, fill="black")
    if not values:
        values = [0.0]
        labels = ["empty"]
    minimum = min(values + [0.0])
    maximum = max(values + [0.0])
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    zero_y = 50 + (maximum - 0.0) / (maximum - minimum) * 300
    draw.line((70, zero_y, 850, zero_y), fill="gray")
    width = 780 / len(values)
    for index, value in enumerate(values):
        x0 = 70 + index * width + 5
        x1 = x0 + width - 10
        y = 50 + (maximum - value) / (maximum - minimum) * 300
        draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill="teal")
        draw.text((x0, 360), labels[index][:14], fill="black")
    image.save(path)


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
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "ARCHITECTURES",
    "MARKET_SOURCES",
    "MarketBenchmarkOutputPaths",
    "MarketBenchmarkResult",
    "build_anchor_coverage",
    "build_market_anchor_bootstrap",
    "build_market_anchor_metrics",
    "build_probability_disagreement",
    "run_market_benchmark",
    "write_market_benchmark_artifacts",
]
