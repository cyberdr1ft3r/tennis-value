"""Create a deterministic model-ready training fixture for local Task 7 demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(match_id: str, match_date: str, target: bool, value: float) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(match_date),
        "surface": "Hard",
        "player_1": f"Fixture {match_id} A",
        "player_2": f"Fixture {match_id} B",
        "player_1_won": target,
        "player_1_odds": 1.8,
        "player_2_odds": 2.0,
        "is_retirement": False,
        "overall_elo_diff": value,
        "surface_elo_diff": value / 2,
        "elo_expected_player_1": max(0.05, min(0.95, 0.5 + value / 100)),
        "log_rank_diff": None if match_id.endswith("3") else value / 10,
        "recent_10_win_rate_diff": value / 100,
        "surface_recent_10_win_rate_diff": value / 100,
        "days_since_last_match_diff": None if match_id.endswith("4") else value,
        "matches_last_14d_diff": int(value) % 3,
        "history_count_min": int(abs(value)) % 5,
        "best_of_5": 0,
        "surface_clay": 0,
        "surface_grass": 0,
    }


def build_fixture() -> pd.DataFrame:
    """Return a deterministic feature frame with all configured partitions populated."""
    rows = [
        _row("tr01", "2020-01-15", True, 1.0),
        _row("tr02", "2021-03-20", False, 2.0),
        _row("tr03", "2022-06-10", True, 3.0),
        _row("tr04", "2023-12-31", False, 4.0),
        _row("va01", "2024-01-01", True, 5.0),
        _row("va02", "2024-07-15", False, 6.0),
        _row("te01", "2025-01-01", True, 7.0),
        _row("te02", "2025-12-31", False, 8.0),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training_fixture_features.parquet"),
        help="Path for the generated model-ready feature fixture.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_fixture().to_parquet(args.output, index=False)
    print(f"Wrote {len(build_fixture())} row(s) -> {args.output}")


if __name__ == "__main__":
    main()
