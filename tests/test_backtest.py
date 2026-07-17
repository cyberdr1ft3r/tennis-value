from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from tennis_value.backtest import (
    BacktestOutputPaths,
    calculate_drawdown,
    calculate_flat_stake,
    edge_bucket,
    odds_bucket,
    run_backtest,
    settle_paper_bet,
    write_backtest_artifacts,
)
from tennis_value.config import BacktestConfig


def _row(
    match_id: str,
    match_date: str,
    *,
    partition: str = "test",
    surface: str = "Hard",
    has_recommendation: bool = True,
    recommended_side: str = "player_1",
    actual_player_1_won: bool = True,
    is_retirement: bool = False,
    odds: float = 2.0,
    edge: float = 0.05,
) -> dict[str, object]:
    player_1 = f"{match_id} A"
    player_2 = f"{match_id} B"
    recommended_player = player_1 if recommended_side == "player_1" else player_2
    if not has_recommendation:
        recommended_side = "none"
        recommended_player = None
    return {
        "match_id": match_id,
        "match_date": match_date,
        "partition": partition,
        "surface": surface,
        "player_1": player_1,
        "player_2": player_2,
        "actual_player_1_won": actual_player_1_won,
        "is_retirement": is_retirement,
        "has_recommendation": has_recommendation,
        "recommended_side": recommended_side,
        "recommended_player": recommended_player,
        "recommended_probability": 0.60,
        "recommended_market_probability": 0.50,
        "recommended_odds": odds,
        "recommended_edge": edge,
        "recommended_expected_value": 0.20,
        "model_version": "model_v1",
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_flat_stake_calculation_and_validation() -> None:
    assert calculate_flat_stake(10_000, 0.005) == pytest.approx(50)
    assert calculate_flat_stake(10_050, 0.005) == pytest.approx(50.25)

    for bankroll in (0, -1, math.inf):
        with pytest.raises(ValueError):
            calculate_flat_stake(bankroll, 0.005)
    for fraction in (0, -0.1, 1.01, math.inf):
        with pytest.raises(ValueError):
            calculate_flat_stake(10_000, fraction)


def test_settlement_win_loss_and_void_rules() -> None:
    assert settle_paper_bet(
        recommended_side="player_1",
        actual_player_1_won=True,
        decimal_odds=2.0,
        stake=50,
        is_retirement=False,
        retirement_policy="void",
    ).profit_loss == pytest.approx(50)
    assert settle_paper_bet(
        recommended_side="player_1",
        actual_player_1_won=True,
        decimal_odds=1.8,
        stake=50,
        is_retirement=False,
        retirement_policy="void",
    ).profit_loss == pytest.approx(40)
    assert settle_paper_bet(
        recommended_side="player_1",
        actual_player_1_won=False,
        decimal_odds=2.0,
        stake=50,
        is_retirement=False,
        retirement_policy="void",
    ).profit_loss == pytest.approx(-50)
    void = settle_paper_bet(
        recommended_side="player_1",
        actual_player_1_won=True,
        decimal_odds=2.0,
        stake=50,
        is_retirement=True,
        retirement_policy="void",
    )
    assert void.result == "void"
    assert void.profit_loss == pytest.approx(0)
    assert void.stake_returned == pytest.approx(50)


@pytest.mark.parametrize(
    ("side", "actual_player_1_won", "result"),
    [
        ("player_1", True, "won"),
        ("player_1", False, "lost"),
        ("player_2", False, "won"),
        ("player_2", True, "lost"),
    ],
)
def test_player_side_result_resolution(
    side: str,
    actual_player_1_won: bool,
    result: str,
) -> None:
    settlement = settle_paper_bet(
        recommended_side=side,
        actual_player_1_won=actual_player_1_won,
        decimal_odds=2.0,
        stake=50,
        is_retirement=False,
        retirement_policy="void",
    )

    assert settlement.result == result


def test_chronology_stakes_use_updated_bankroll_and_same_date_match_id_order() -> None:
    unordered = _frame(
        [
            _row("m03", "2025-01-02", actual_player_1_won=False),
            _row("m02", "2025-01-01", odds=2.0, actual_player_1_won=True),
            _row("m01", "2025-01-01", odds=2.0, actual_player_1_won=True),
        ]
    )

    result = run_backtest(unordered)

    assert result.ledger["match_id"].tolist() == ["m01", "m02", "m03"]
    assert result.ledger["bankroll_before"].tolist() == pytest.approx([10_000, 10_050, 10_100.25])
    assert result.ledger["stake"].tolist() == pytest.approx([50, 50.25, 50.50125])

    repeated = run_backtest(unordered.sample(frac=1, random_state=7).reset_index(drop=True))
    pd.testing.assert_frame_equal(result.ledger, repeated.ledger)


def test_duplicate_match_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate match_id"):
        run_backtest(_frame([_row("m01", "2025-01-01"), _row("m01", "2025-01-02")]))


def test_one_bet_per_match_and_no_recommendation_skips() -> None:
    result = run_backtest(
        _frame(
            [
                _row("m01", "2025-01-01", has_recommendation=False),
                _row("m02", "2025-01-02"),
            ]
        )
    )

    assert len(result.ledger) == 1
    assert result.ledger["match_id"].tolist() == ["m02"]
    assert result.summary["rows_skipped"] == 1
    assert result.ledger["match_id"].is_unique


def test_contradictory_recommendation_fields_fail() -> None:
    bad = _row("m01", "2025-01-01", recommended_side="player_1")
    bad["recommended_player"] = bad["player_2"]

    with pytest.raises(ValueError, match="does not match"):
        run_backtest(_frame([bad]))

    bad_no_rec = _row("m02", "2025-01-02", has_recommendation=False)
    bad_no_rec["recommended_side"] = "player_1"
    with pytest.raises(ValueError, match="has_recommendation=false"):
        run_backtest(_frame([bad_no_rec]))


def test_accounting_roi_and_void_win_rate() -> None:
    result = run_backtest(
        _frame(
            [
                _row("m01", "2025-01-01", odds=2.0, actual_player_1_won=True),
                _row("m02", "2025-01-02", odds=2.0, actual_player_1_won=False),
                _row("m03", "2025-01-03", odds=2.0, is_retirement=True),
            ]
        )
    )

    ledger = result.ledger
    assert result.summary["voids"] == 1
    assert result.summary["win_rate"] == pytest.approx(0.5)
    assert result.summary["ending_bankroll"] == pytest.approx(
        result.summary["starting_bankroll"] + ledger["profit_loss"].sum()
    )
    assert result.summary["total_staked"] == pytest.approx(ledger["stake"].sum())
    assert result.summary["roi"] == pytest.approx(
        ledger["profit_loss"].sum() / ledger["stake"].sum()
    )


def test_drawdown_calculation_includes_initial_bankroll() -> None:
    rising = calculate_drawdown(pd.Series([10_000, 10_100, 10_200]))
    assert rising.maximum_drawdown == pytest.approx(0)
    assert rising.maximum_drawdown_percentage == pytest.approx(0)

    drawdown = calculate_drawdown(pd.Series([10_000, 10_500, 9_450, 9_900]))
    assert drawdown.maximum_drawdown == pytest.approx(1_050)
    assert drawdown.maximum_drawdown_percentage == pytest.approx(0.10)
    assert drawdown.curve["bet_sequence"].tolist() == [0, 1, 2, 3]


def test_voids_neither_extend_nor_break_streaks() -> None:
    result = run_backtest(
        _frame(
            [
                _row("m01", "2025-01-01", actual_player_1_won=True),
                _row("m02", "2025-01-02", is_retirement=True),
                _row("m03", "2025-01-03", actual_player_1_won=True),
                _row("m04", "2025-01-04", actual_player_1_won=False),
                _row("m05", "2025-01-05", actual_player_1_won=False),
            ]
        )
    )

    assert result.summary["longest_winning_streak"] == 2
    assert result.summary["longest_losing_streak"] == 2


def test_grouped_metrics_and_buckets() -> None:
    result = run_backtest(
        _frame(
            [
                _row("m01", "2025-01-01", surface="Hard", odds=1.6, edge=0.05),
                _row("m02", "2025-01-02", surface="Clay", odds=1.9, edge=0.07),
                _row("m03", "2025-01-03", surface="Grass", odds=2.2, edge=0.10),
                _row("m04", "2025-01-04", surface="Hard", odds=3.0, edge=0.13),
            ]
        )
    )

    assert edge_bucket(0.03) == "<0.04"
    assert edge_bucket(0.05) == "0.04-0.06"
    assert odds_bucket(1.6) == "1.50-1.75"
    assert odds_bucket(3.0) == "2.50-3.50"
    assert set(result.by_surface["surface"]) == {"Clay", "Grass", "Hard"}
    assert result.by_surface["bets"].sum() == 4
    assert result.by_edge_bucket["total_staked"].sum() == pytest.approx(
        result.ledger["stake"].sum()
    )
    assert result.by_odds_bucket["profit_loss"].sum() == pytest.approx(
        result.ledger["profit_loss"].sum()
    )


def test_partition_filtering_default_and_override() -> None:
    frame = _frame(
        [
            _row("tr01", "2023-01-01", partition="train"),
            _row("va01", "2024-01-01", partition="validation"),
            _row("te01", "2025-01-01", partition="test"),
        ]
    )

    default = run_backtest(frame)
    validation = run_backtest(frame, BacktestConfig(partition="validation"))

    assert default.summary["partition"] == "test"
    assert default.ledger["match_id"].tolist() == ["te01"]
    assert validation.ledger["match_id"].tolist() == ["va01"]

    bad = frame.copy()
    bad.loc[0, "partition"] = "future"
    with pytest.raises(ValueError, match="unsupported partitions"):
        run_backtest(bad)


def test_artifacts_are_written_to_configured_paths() -> None:
    output_dir = Path(".tmp-task10-artifacts")
    paths = BacktestOutputPaths(
        bets_output=output_dir / "bets.parquet",
        summary_output=output_dir / "summary.json",
        surface_output=output_dir / "surface.parquet",
        edge_output=output_dir / "edge.parquet",
        odds_output=output_dir / "odds.parquet",
        bankroll_plot=output_dir / "bankroll.png",
        drawdown_plot=output_dir / "drawdown.png",
    )
    for path in paths.__dict__.values():
        path.unlink(missing_ok=True)

    result = run_backtest(_frame([_row("m01", "2025-01-01")]))
    write_backtest_artifacts(result, paths)

    assert pd.read_parquet(paths.bets_output)["match_id"].tolist() == ["m01"]
    assert json.loads(paths.summary_output.read_text(encoding="utf-8"))["bets_placed"] == 1
    assert not pd.read_parquet(paths.surface_output).empty
    assert not pd.read_parquet(paths.edge_output).empty
    assert not pd.read_parquet(paths.odds_output).empty
    assert paths.bankroll_plot.exists() and paths.bankroll_plot.stat().st_size > 0
    assert paths.drawdown_plot.exists() and paths.drawdown_plot.stat().st_size > 0


def test_determinism_input_integrity_and_clear_errors() -> None:
    frame = _frame([_row("m01", "2025-01-01"), _row("m02", "2025-01-02", odds=1.8)])
    original = frame.copy(deep=True)

    first = run_backtest(frame)
    second = run_backtest(frame)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(first.ledger, second.ledger)
    comparable_first = {k: v for k, v in first.summary.items() if k != "created_at_utc"}
    comparable_second = {k: v for k, v in second.summary.items() if k != "created_at_utc"}
    assert comparable_first == comparable_second

    with pytest.raises(ValueError, match="missing required"):
        run_backtest(frame.drop(columns=["recommended_odds"]))

    malformed = frame.copy()
    malformed.loc[0, "recommended_probability"] = math.inf
    with pytest.raises(ValueError, match="model probability"):
        run_backtest(malformed)
