from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tennis_value.odds_audit import (
    OddsAuditOutputPaths,
    audit_odds_sources,
    build_odds_quality_rows,
    write_odds_audit_artifacts,
)


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "tournament": "T",
                "surface": "Hard",
                "round": "R32",
                "best_of": 3,
                "winner": "A",
                "loser": "B",
                "winner_rank": 1,
                "loser_rank": 2,
                "winner_odds": 1.5,
                "loser_odds": 2.7,
                "odds_source": "B365",
                "status_or_comment": "Completed",
                "source_file": "2024.csv",
            },
            {
                "match_date": "2024-01-02",
                "tournament": "T",
                "surface": "Hard",
                "round": "R32",
                "best_of": 3,
                "winner": "C",
                "loser": "D",
                "winner_rank": 3,
                "loser_rank": 4,
                "winner_odds": 31.0,
                "loser_odds": 1.01,
                "odds_source": "Average",
                "status_or_comment": "Completed",
                "source_file": "2024.csv",
            },
        ]
    )


def test_odds_quality_flags_and_source_pair_consistency() -> None:
    rows = build_odds_quality_rows(_raw_rows())

    assert rows.loc[0, "odds_source"] == "B365"
    assert bool(rows.loc[1, "used_fallback_source"])
    assert bool(rows.loc[1, "either_odd_below_1_02"])
    assert bool(rows.loc[1, "either_odd_above_30"])
    assert bool(rows.loc[1, "large_market_reference_disagreement"])


def test_audit_identifies_current_source_columns_and_fallback_policy(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2024-01-01",
                "Tournament": "T",
                "Surface": "Hard",
                "Round": "R32",
                "Winner": "A",
                "Loser": "B",
                "B365W": "",
                "B365L": "",
                "AvgW": "1.9",
                "AvgL": "1.9",
            }
        ]
    ).to_csv(raw_dir / "2024.csv", index=False)

    result = audit_odds_sources(raw_dir)

    assert result.summary.selected_source == "fallback_hierarchy: B365W/B365L first, then AvgW/AvgL"
    assert result.summary.rows_by_source == {"Average": 1}
    assert result.summary.fallback_rows == 1
    assert result.summary.source_consistency["same_source_pairing"] is True
    assert result.summary.original_odds_column_names["2024.csv"] == [
        "B365W",
        "B365L",
        "AvgW",
        "AvgL",
    ]


def test_odds_audit_artifacts_have_stable_schema(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2024-01-01",
                "Tournament": "T",
                "Surface": "Hard",
                "Winner": "A",
                "Loser": "B",
                "B365W": "1.5",
                "B365L": "2.7",
            }
        ]
    ).to_csv(raw_dir / "2024.csv", index=False)
    paths = OddsAuditOutputPaths(
        summary=tmp_path / "audit.json",
        quality_rows=tmp_path / "rows.parquet",
        overround_by_source=tmp_path / "overround.parquet",
    )

    write_odds_audit_artifacts(audit_odds_sources(raw_dir), paths)

    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert "current_selection_policy" in summary
    assert {"overround", "missing_paired_odds"}.issubset(pd.read_parquet(paths.quality_rows))
    assert {"odds_source", "year", "rows"}.issubset(pd.read_parquet(paths.overround_by_source))
