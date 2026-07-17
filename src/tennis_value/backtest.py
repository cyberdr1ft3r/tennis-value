"""Chronological flat-stake paper backtesting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from tennis_value.config import BacktestConfig
from tennis_value.odds import validate_decimal_odds
from tennis_value.value import validate_model_probability

BetResult = Literal["won", "lost", "void"]
SelectionSide = Literal["player_1", "player_2"]
SUPPORTED_PARTITIONS = {"train", "validation", "test"}
REQUIRED_COLUMNS = (
    "match_id",
    "match_date",
    "partition",
    "surface",
    "player_1",
    "player_2",
    "actual_player_1_won",
    "is_retirement",
    "has_recommendation",
    "recommended_side",
    "recommended_player",
    "recommended_probability",
    "recommended_market_probability",
    "recommended_odds",
    "recommended_edge",
    "recommended_expected_value",
    "model_version",
)


class BetSettlement(BaseModel):
    """Settlement result for one simulated paper bet."""

    model_config = ConfigDict(frozen=True)

    result: BetResult
    profit_loss: float
    stake_returned: float
    settlement_reason: str


class DrawdownResult(BaseModel):
    """Maximum peak-to-trough drawdown and the full drawdown curve."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    maximum_drawdown: float
    maximum_drawdown_percentage: float
    curve: pd.DataFrame


class BacktestResult(BaseModel):
    """Backtest ledger, summary, grouped tables, and curves."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    ledger: pd.DataFrame
    summary: dict[str, Any]
    by_surface: pd.DataFrame
    by_edge_bucket: pd.DataFrame
    by_odds_bucket: pd.DataFrame
    bankroll_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame


@dataclass(frozen=True)
class BacktestOutputPaths:
    """Output paths for Task 10 artifacts."""

    bets_output: Path
    summary_output: Path
    surface_output: Path
    edge_output: Path
    odds_output: Path
    bankroll_plot: Path
    drawdown_plot: Path


def calculate_flat_stake(bankroll: float, stake_fraction: float) -> float:
    """Calculate a flat fraction stake from the bankroll immediately before a bet."""
    bankroll_value = _finite_float(bankroll, "bankroll")
    stake_fraction_value = _finite_float(stake_fraction, "stake fraction")
    if bankroll_value <= 0:
        msg = "bankroll must be greater than zero"
        raise ValueError(msg)
    if stake_fraction_value <= 0 or stake_fraction_value > 1:
        msg = "stake fraction must be greater than zero and no greater than one"
        raise ValueError(msg)
    stake = bankroll_value * stake_fraction_value
    if not math.isfinite(stake) or stake <= 0:
        msg = "calculated stake must be finite and greater than zero"
        raise ValueError(msg)
    return stake


def settle_paper_bet(
    *,
    recommended_side: str,
    actual_player_1_won: bool,
    decimal_odds: float,
    stake: float,
    is_retirement: bool,
    retirement_policy: str,
) -> BetSettlement:
    """Settle one paper bet using the oriented result target."""
    odds = validate_decimal_odds(decimal_odds)
    stake_value = _finite_float(stake, "stake")
    if stake_value <= 0:
        msg = "stake must be greater than zero"
        raise ValueError(msg)
    if recommended_side not in {"player_1", "player_2"}:
        msg = f"invalid recommendation side: {recommended_side}"
        raise ValueError(msg)
    if retirement_policy not in {"void", "settle"}:
        msg = f"unsupported retirement policy: {retirement_policy}"
        raise ValueError(msg)
    if is_retirement and retirement_policy == "void":
        return BetSettlement(
            result="void",
            profit_loss=0.0,
            stake_returned=stake_value,
            settlement_reason="retirement_void",
        )

    won = (recommended_side == "player_1" and actual_player_1_won) or (
        recommended_side == "player_2" and not actual_player_1_won
    )
    if won:
        return BetSettlement(
            result="won",
            profit_loss=stake_value * (odds - 1.0),
            stake_returned=stake_value,
            settlement_reason="selection_won",
        )
    return BetSettlement(
        result="lost",
        profit_loss=-stake_value,
        stake_returned=0.0,
        settlement_reason="selection_lost",
    )


def calculate_drawdown(bankroll_series: pd.Series) -> DrawdownResult:
    """Calculate positive maximum drawdown from a bankroll curve including initial bankroll."""
    if bankroll_series.empty:
        msg = "bankroll series must not be empty"
        raise ValueError(msg)
    bankroll = pd.to_numeric(bankroll_series, errors="raise").astype(float)
    if not np.isfinite(bankroll).all():
        msg = "bankroll series must contain only finite values"
        raise ValueError(msg)
    running_peak = bankroll.cummax()
    drawdown = bankroll - running_peak
    drawdown_percentage = drawdown / running_peak
    curve = pd.DataFrame(
        {
            "bet_sequence": list(range(len(bankroll))),
            "bankroll": bankroll.to_numpy(),
            "running_peak": running_peak.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "drawdown_percentage": drawdown_percentage.to_numpy(),
        }
    )
    return DrawdownResult(
        maximum_drawdown=abs(float(drawdown.min())),
        maximum_drawdown_percentage=abs(float(drawdown_percentage.min())),
        curve=curve,
    )


def run_backtest(assessments: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """Run a deterministic chronological paper backtest without mutating the input frame."""
    active_config = config or BacktestConfig()
    _validate_backtest_config(active_config)
    _validate_assessment_frame(assessments)

    frame = assessments.copy(deep=True)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="raise")
    selected_partition = active_config.partition
    selected = frame[frame["partition"] == selected_partition].copy()
    selected = selected.sort_values(["match_date", "match_id"], kind="mergesort").reset_index(
        drop=True
    )

    bankroll = float(active_config.starting_bankroll)
    ledger_rows: list[dict[str, Any]] = []
    bankroll_points = [_bankroll_point(0, None, None, bankroll)]
    rows_with_recommendations = int(selected["has_recommendation"].astype(bool).sum())

    for _, row in selected.iterrows():
        if not bool(row["has_recommendation"]):
            continue
        _validate_recommended_row(row)
        stake = calculate_flat_stake(bankroll, active_config.flat_stake_fraction)
        settlement = settle_paper_bet(
            recommended_side=str(row["recommended_side"]),
            actual_player_1_won=_coerce_bool(row["actual_player_1_won"], "actual_player_1_won"),
            decimal_odds=float(row["recommended_odds"]),
            stake=stake,
            is_retirement=_coerce_bool(row["is_retirement"], "is_retirement"),
            retirement_policy=active_config.retirement_policy,
        )
        bankroll_before = bankroll
        bankroll = bankroll_before + settlement.profit_loss
        if not math.isfinite(bankroll):
            msg = "bankroll became non-finite"
            raise ValueError(msg)
        bet_sequence = len(ledger_rows) + 1
        ledger_rows.append(
            {
                "bet_sequence": bet_sequence,
                "match_id": str(row["match_id"]),
                "match_date": pd.Timestamp(row["match_date"]).strftime("%Y-%m-%d"),
                "partition": str(row["partition"]),
                "surface": str(row["surface"]),
                "player_1": str(row["player_1"]),
                "player_2": str(row["player_2"]),
                "selection_side": str(row["recommended_side"]),
                "selection_player": str(row["recommended_player"]),
                "model_version": str(row["model_version"]),
                "model_probability": _finite_float(
                    row["recommended_probability"],
                    "model probability",
                ),
                "market_probability": _finite_float(
                    row["recommended_market_probability"],
                    "market probability",
                ),
                "edge": _finite_float(row["recommended_edge"], "edge"),
                "expected_value": _finite_float(
                    row["recommended_expected_value"],
                    "expected value",
                ),
                "decimal_odds": validate_decimal_odds(row["recommended_odds"]),
                "bankroll_before": bankroll_before,
                "stake": stake,
                "result": settlement.result,
                "profit_loss": settlement.profit_loss,
                "bankroll_after": bankroll,
                "is_retirement": _coerce_bool(row["is_retirement"], "is_retirement"),
                "settlement_reason": settlement.settlement_reason,
                "edge_bucket": edge_bucket(_finite_float(row["recommended_edge"], "edge")),
                "odds_bucket": odds_bucket(validate_decimal_odds(row["recommended_odds"])),
            }
        )
        bankroll_points.append(
            _bankroll_point(
                bet_sequence,
                str(row["match_id"]),
                pd.Timestamp(row["match_date"]).strftime("%Y-%m-%d"),
                bankroll,
            )
        )

    ledger = pd.DataFrame(ledger_rows, columns=_ledger_columns())
    bankroll_curve = pd.DataFrame(bankroll_points)
    drawdown = calculate_drawdown(bankroll_curve["bankroll"])
    bankroll_curve = bankroll_curve.assign(
        running_peak=drawdown.curve["running_peak"],
        drawdown=drawdown.curve["drawdown"],
        drawdown_percentage=drawdown.curve["drawdown_percentage"],
    )

    by_surface = _group_performance(ledger, "surface")
    by_edge_bucket = _group_performance(ledger, "edge_bucket")
    by_odds_bucket = _group_performance(ledger, "odds_bucket")
    summary = _build_summary(
        frame=frame,
        selected=selected,
        ledger=ledger,
        config=active_config,
        rows_with_recommendations=rows_with_recommendations,
        drawdown=drawdown,
        by_surface=by_surface,
        by_edge_bucket=by_edge_bucket,
        by_odds_bucket=by_odds_bucket,
    )
    return BacktestResult(
        ledger=ledger,
        summary=summary,
        by_surface=by_surface,
        by_edge_bucket=by_edge_bucket,
        by_odds_bucket=by_odds_bucket,
        bankroll_curve=bankroll_curve,
        drawdown_curve=drawdown.curve,
    )


def write_backtest_artifacts(result: BacktestResult, output_paths: BacktestOutputPaths) -> None:
    """Write all Task 10 backtest artifacts."""
    for path in (
        output_paths.bets_output,
        output_paths.summary_output,
        output_paths.surface_output,
        output_paths.edge_output,
        output_paths.odds_output,
        output_paths.bankroll_plot,
        output_paths.drawdown_plot,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    result.ledger.to_parquet(output_paths.bets_output, index=False)
    result.by_surface.to_parquet(output_paths.surface_output, index=False)
    result.by_edge_bucket.to_parquet(output_paths.edge_output, index=False)
    result.by_odds_bucket.to_parquet(output_paths.odds_output, index=False)
    summary = {
        **result.summary,
        "artifact_paths": {
            "bets": str(output_paths.bets_output),
            "summary": str(output_paths.summary_output),
            "by_surface": str(output_paths.surface_output),
            "by_edge_bucket": str(output_paths.edge_output),
            "by_odds_bucket": str(output_paths.odds_output),
            "bankroll_plot": str(output_paths.bankroll_plot),
            "drawdown_plot": str(output_paths.drawdown_plot),
        },
    }
    output_paths.summary_output.write_text(_json_dumps(summary), encoding="utf-8")
    _write_line_plot(
        result.bankroll_curve["bankroll"].tolist(),
        output_paths.bankroll_plot,
        title="Historical Paper Bankroll Curve",
        y_label="Bankroll",
        reference=float(result.summary["starting_bankroll"]),
    )
    _write_line_plot(
        (result.bankroll_curve["drawdown_percentage"] * 100).tolist(),
        output_paths.drawdown_plot,
        title="Historical Paper Drawdown",
        y_label="Drawdown %",
        reference=0.0,
    )


def edge_bucket(edge: float) -> str:
    """Bucket a recommendation edge for grouped reporting."""
    value = _finite_float(edge, "edge")
    if value < 0.04:
        return "<0.04"
    if value < 0.06:
        return "0.04-0.06"
    if value < 0.08:
        return "0.06-0.08"
    if value < 0.12:
        return "0.08-0.12"
    return "0.12+"


def odds_bucket(decimal_odds: float) -> str:
    """Bucket decimal odds for grouped reporting."""
    odds = validate_decimal_odds(decimal_odds)
    if 1.50 <= odds < 1.75:
        return "1.50-1.75"
    if 1.75 <= odds < 2.00:
        return "1.75-2.00"
    if 2.00 <= odds < 2.50:
        return "2.00-2.50"
    if 2.50 <= odds <= 3.50:
        return "2.50-3.50"
    return "other"


def _validate_backtest_config(config: BacktestConfig) -> None:
    calculate_flat_stake(config.starting_bankroll, config.flat_stake_fraction)
    if config.partition not in SUPPORTED_PARTITIONS:
        msg = f"unsupported partition: {config.partition}"
        raise ValueError(msg)
    if config.retirement_policy not in {"void", "settle"}:
        msg = f"unsupported retirement policy: {config.retirement_policy}"
        raise ValueError(msg)


def _validate_assessment_frame(assessments: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in assessments.columns]
    if missing:
        msg = f"missing required backtest columns: {missing}"
        raise ValueError(msg)
    if assessments["match_id"].isna().any():
        msg = "match_id must not be missing"
        raise ValueError(msg)
    if assessments["match_id"].duplicated().any():
        msg = "duplicate match_id values are not allowed"
        raise ValueError(msg)
    try:
        pd.to_datetime(assessments["match_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        msg = "invalid match_date values"
        raise ValueError(msg) from exc
    partitions = set(assessments["partition"].dropna().astype(str))
    unsupported = sorted(partitions - SUPPORTED_PARTITIONS)
    if unsupported:
        msg = f"unsupported partitions in assessments: {unsupported}"
        raise ValueError(msg)
    for _, row in assessments.iterrows():
        if bool(row["has_recommendation"]):
            _validate_recommended_row(row)
        else:
            side = _optional_text(row["recommended_side"])
            if side not in {None, "none"}:
                msg = "has_recommendation=false row contains a recommendation side"
                raise ValueError(msg)


def _validate_recommended_row(row: pd.Series) -> None:
    side = _optional_text(row["recommended_side"])
    if side not in {"player_1", "player_2"}:
        msg = f"invalid recommendation side: {side}"
        raise ValueError(msg)
    if row["recommended_player"] is None or pd.isna(row["recommended_player"]):
        msg = "recommended rows require recommended_player"
        raise ValueError(msg)
    if side == "player_1" and str(row["recommended_player"]) != str(row["player_1"]):
        msg = "player_1 recommendation does not match player_1 name"
        raise ValueError(msg)
    if side == "player_2" and str(row["recommended_player"]) != str(row["player_2"]):
        msg = "player_2 recommendation does not match player_2 name"
        raise ValueError(msg)
    validate_model_probability(row["recommended_probability"])
    validate_model_probability(row["recommended_market_probability"])
    validate_decimal_odds(row["recommended_odds"])
    _finite_float(row["recommended_edge"], "edge")
    _finite_float(row["recommended_expected_value"], "expected value")
    _coerce_bool(row["actual_player_1_won"], "actual_player_1_won")
    _coerce_bool(row["is_retirement"], "is_retirement")


def _build_summary(
    *,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    ledger: pd.DataFrame,
    config: BacktestConfig,
    rows_with_recommendations: int,
    drawdown: DrawdownResult,
    by_surface: pd.DataFrame,
    by_edge_bucket: pd.DataFrame,
    by_odds_bucket: pd.DataFrame,
) -> dict[str, Any]:
    wins = int((ledger["result"] == "won").sum()) if not ledger.empty else 0
    losses = int((ledger["result"] == "lost").sum()) if not ledger.empty else 0
    voids = int((ledger["result"] == "void").sum()) if not ledger.empty else 0
    total_staked = float(ledger["stake"].sum()) if not ledger.empty else 0.0
    profit_loss = float(ledger["profit_loss"].sum()) if not ledger.empty else 0.0
    ending_bankroll = float(config.starting_bankroll + profit_loss)
    roi = profit_loss / total_staked if total_staked > 0 else None
    settled = wins + losses
    win_rate = wins / settled if settled else None
    warnings = [] if total_staked > 0 else ["ROI is null because no money was staked."]
    return {
        "model_version": _single_value(frame, "model_version"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": config.partition,
        "configuration": {
            "starting_bankroll": config.starting_bankroll,
            "flat_stake_fraction": config.flat_stake_fraction,
            "partition": config.partition,
            "retirement_policy": config.retirement_policy,
        },
        "rows_received": len(frame),
        "rows_in_selected_partition": len(selected),
        "rows_with_recommendations": rows_with_recommendations,
        "bets_placed": len(ledger),
        "rows_skipped": len(selected) - len(ledger),
        "starting_bankroll": float(config.starting_bankroll),
        "ending_bankroll": ending_bankroll,
        "total_staked": total_staked,
        "profit_loss": profit_loss,
        "roi": roi,
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "win_rate": win_rate,
        "average_odds": _nullable_mean(ledger, "decimal_odds"),
        "average_probability": _nullable_mean(ledger, "model_probability"),
        "average_market_probability": _nullable_mean(ledger, "market_probability"),
        "average_edge": _nullable_mean(ledger, "edge"),
        "average_expected_value": _nullable_mean(ledger, "expected_value"),
        "maximum_drawdown": drawdown.maximum_drawdown,
        "maximum_drawdown_percentage": drawdown.maximum_drawdown_percentage,
        "longest_winning_streak": _longest_streak(ledger, "won"),
        "longest_losing_streak": _longest_streak(ledger, "lost"),
        "performance_by_surface": _records(by_surface),
        "performance_by_edge_bucket": _records(by_edge_bucket),
        "performance_by_odds_bucket": _records(by_odds_bucket),
        "warnings": warnings,
        "artifact_paths": {},
    }


def _group_performance(ledger: pd.DataFrame, group_column: str) -> pd.DataFrame:
    columns = [
        group_column,
        "bets",
        "wins",
        "losses",
        "voids",
        "total_staked",
        "profit_loss",
        "roi",
        "average_odds",
        "average_edge",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key, group in ledger.groupby(group_column, sort=True, dropna=False):
        total_staked = float(group["stake"].sum())
        profit_loss = float(group["profit_loss"].sum())
        rows.append(
            {
                group_column: str(key),
                "bets": len(group),
                "wins": int((group["result"] == "won").sum()),
                "losses": int((group["result"] == "lost").sum()),
                "voids": int((group["result"] == "void").sum()),
                "total_staked": total_staked,
                "profit_loss": profit_loss,
                "roi": profit_loss / total_staked if total_staked > 0 else None,
                "average_odds": _nullable_mean(group, "decimal_odds"),
                "average_edge": _nullable_mean(group, "edge"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _longest_streak(ledger: pd.DataFrame, target: BetResult) -> int:
    longest = 0
    current = 0
    for result in ledger["result"].tolist() if not ledger.empty else []:
        if result == target:
            current += 1
            longest = max(longest, current)
        elif result in {"won", "lost"}:
            current = 0
    return longest


def _write_line_plot(
    values: list[float],
    path: Path,
    *,
    title: str,
    y_label: str,
    reference: float,
) -> None:
    from PIL import Image, ImageDraw

    width = 900
    height = 520
    margin_left = 80
    margin_top = 60
    margin_right = 40
    margin_bottom = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    draw.rectangle(
        (margin_left, margin_top, margin_left + plot_width, margin_top + plot_height),
        outline="black",
    )
    draw.text((margin_left, 20), title, fill="black")
    draw.text((20, margin_top + plot_height // 2), y_label, fill="black")
    if not values:
        values = [reference]
    min_value = min([*values, reference])
    max_value = max([*values, reference])
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    def point(index: int, value: float) -> tuple[float, float]:
        x = margin_left + (index / max(len(values) - 1, 1)) * plot_width
        y = margin_top + (max_value - value) / (max_value - min_value) * plot_height
        return x, y

    reference_y = point(0, reference)[1]
    draw.line((margin_left, reference_y, margin_left + plot_width, reference_y), fill="gray")
    points = [point(index, float(value)) for index, value in enumerate(values)]
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")
    else:
        draw.line(points, fill="blue", width=3)
    draw.text((margin_left, height - 40), "Bet sequence", fill="black")
    image.save(path)


def _bankroll_point(
    bet_sequence: int,
    match_id: str | None,
    match_date: str | None,
    bankroll: float,
) -> dict[str, Any]:
    return {
        "bet_sequence": bet_sequence,
        "match_id": match_id,
        "match_date": match_date,
        "bankroll": bankroll,
    }


def _ledger_columns() -> list[str]:
    return [
        "bet_sequence",
        "match_id",
        "match_date",
        "partition",
        "surface",
        "player_1",
        "player_2",
        "selection_side",
        "selection_player",
        "model_version",
        "model_probability",
        "market_probability",
        "edge",
        "expected_value",
        "decimal_odds",
        "bankroll_before",
        "stake",
        "result",
        "profit_loss",
        "bankroll_after",
        "is_retirement",
        "settlement_reason",
        "edge_bucket",
        "odds_bucket",
    ]


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(result):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return result


def _coerce_bool(value: Any, name: str) -> bool:
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
    msg = f"{name} must be boolean"
    raise ValueError(msg)


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _nullable_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def _single_value(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    values = sorted({str(value) for value in frame[column].dropna().unique()})
    return values[0] if len(values) == 1 else ",".join(values)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _jsonable(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if pd.isna(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(_jsonable_nested(payload), indent=2)


def _jsonable_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_nested(item) for item in value]
    return _jsonable(value)


__all__ = [
    "BacktestOutputPaths",
    "BacktestResult",
    "BetSettlement",
    "DrawdownResult",
    "calculate_drawdown",
    "calculate_flat_stake",
    "edge_bucket",
    "odds_bucket",
    "run_backtest",
    "settle_paper_bet",
    "write_backtest_artifacts",
]
