from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tennis_value.ingest import (
    IngestionReport,
    discover_tennis_data_files,
    ingest_tennis_data,
    write_ingestion_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _valid_rows() -> list[dict[str, object]]:
    return [
        {
            "Date": "2024-01-02",
            "Tournament": "Brisbane",
            "Surface": "Hard",
            "Round": "R32",
            "Best of": "3",
            "Winner": "Player A",
            "Loser": "Player B",
            "WRank": "10",
            "LRank": "20",
            "B365W": "1.80",
            "B365L": "2.05",
            "AvgW": "1.78",
            "AvgL": "2.08",
            "Comment": "Completed",
        },
        {
            "Date": "03/01/2024",
            "Tournament": "Brisbane",
            "Surface": "Clay",
            "Round": "R16",
            "Best of": "3",
            "Winner": "Player C",
            "Loser": "Player D",
            "WRank": "NR",
            "LRank": "-",
            "B365W": "N/A",
            "B365L": "",
            "AvgW": "1.60",
            "AvgL": "2.30",
            "Comment": "Retired",
        },
    ]


def test_file_discovery_is_recursive_and_filters_supported_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(input_dir / "2024.csv", _valid_rows())
    _write_csv(input_dir / "nested" / "2025.csv", _valid_rows())
    (input_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    files = discover_tennis_data_files(input_dir)

    assert [path.name for path in files] == ["2025.csv", "2024.csv"] or sorted(
        path.name for path in files
    ) == ["2024.csv", "2025.csv"]


def test_csv_loading_alias_mapping_source_file_and_odds_priority(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(input_dir / "2024.csv", _valid_rows())

    result = ingest_tennis_data(input_dir)

    assert result.report.files_loaded == 1
    assert result.report.rows_returned == 2
    assert result.data.loc[0, "source_file"] == "2024.csv"
    assert result.data.loc[0, "match_date"] == pd.Timestamp("2024-01-02")
    assert result.data.loc[0, "surface"] == "Hard"
    assert result.data.loc[0, "winner_odds"] == 1.8
    assert result.data.loc[0, "loser_odds"] == 2.05
    assert result.data.loc[0, "odds_source"] == "B365"
    assert result.data.loc[1, "odds_source"] == "Average"
    assert pd.isna(result.data.loc[1, "winner_rank"])
    assert result.report.rows_without_rankings == 1


def test_xlsx_loading_and_combining_multiple_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(input_dir / "2024.csv", _valid_rows())
    xlsx_path = input_dir / "2025.xlsx"
    pd.DataFrame(
        [
            {
                "Date": "06/10/2025",
                "Tournament": "Queens",
                "Surface": "Grass",
                "Round": "R32",
                "BestOf": "3",
                "Winner": "Player E",
                "Loser": "Player F",
                "WRank": "30",
                "LRank": "40",
                "AvgW": "1.55",
                "AvgL": "2.60",
                "Status": "Completed",
            }
        ]
    ).to_excel(xlsx_path, index=False)

    result = ingest_tennis_data(input_dir)

    assert result.report.files_loaded == 2
    assert result.report.rows_returned == 3
    assert set(result.data["source_file"]) == {"2024.csv", "2025.xlsx"}
    assert "status_or_comment" in result.data.columns


def test_missing_odds_invalid_dates_surface_other_and_nullable_numeric(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(
        input_dir / "2024.csv",
        [
            {
                "Date": "bad-date",
                "Tournament": "Indoor Cup",
                "Surface": "Carpet",
                "Winner": "Player A",
                "Loser": "Player B",
                "WRank": "",
                "LRank": "NR",
                "B365W": "1.00",
                "B365L": "abc",
            }
        ],
    )

    result = ingest_tennis_data(input_dir)

    assert pd.isna(result.data.loc[0, "match_date"])
    assert result.data.loc[0, "surface"] == "Other"
    assert pd.isna(result.data.loc[0, "winner_rank"])
    assert pd.isna(result.data.loc[0, "winner_odds"])
    assert result.data.loc[0, "odds_source"] == "Missing"
    assert result.report.rows_with_invalid_dates == 1
    assert result.report.rows_without_odds == 1
    assert result.report.rows_without_rankings == 1


def test_configured_same_bookmaker_odds_pair_is_used(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(
        input_dir / "2024.csv",
        [
            {
                "Date": "2024-01-02",
                "Tournament": "Brisbane",
                "Surface": "Hard",
                "Winner": "Player A",
                "Loser": "Player B",
                "CustomW": "1.90",
                "CustomL": "1.95",
            }
        ],
    )

    result = ingest_tennis_data(input_dir, configured_odds_pairs=(("CustomW", "CustomL"),))

    assert result.data.loc[0, "winner_odds"] == 1.9
    assert result.data.loc[0, "loser_odds"] == 1.95
    assert result.data.loc[0, "odds_source"] == "ConfiguredBookmaker"


def test_missing_required_columns_produce_helpful_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(input_dir / "bad.csv", [{"Date": "2024-01-02", "Winner": "Player A"}])

    result = ingest_tennis_data(input_dir)

    assert result.report.files_failed == 1
    assert result.report.errors[0].source_file == "bad.csv"
    assert "tournament" in result.report.errors[0].missing_fields
    assert "Date" in result.report.errors[0].available_columns
    assert "match_date" in result.report.errors[0].alias_mappings_attempted


def test_report_json_serialization_and_output_writing(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(input_dir / "2024.csv", _valid_rows())
    output_path = tmp_path / "processed" / "raw_matches.parquet"
    report_path = tmp_path / "reports" / "ingestion_report.json"

    result = ingest_tennis_data(input_dir)
    write_ingestion_outputs(result, output_path, report_path)

    assert output_path.exists()
    assert report_path.exists()
    loaded_report = IngestionReport.model_validate(json.loads(report_path.read_text()))
    assert loaded_report.rows_returned == 2


def test_repeated_ingestion_is_deterministic(tmp_path: Path) -> None:
    input_dir = tmp_path / "tennis_data"
    _write_csv(input_dir / "2024.csv", _valid_rows())

    first = ingest_tennis_data(input_dir)
    second = ingest_tennis_data(input_dir)

    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.report == second.report


def test_empty_input_dir_returns_warning(tmp_path: Path) -> None:
    result = ingest_tennis_data(tmp_path / "missing")

    assert result.report.files_discovered == 0
    assert result.report.warnings
    assert result.data.empty
