from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.fetch_sackmann_data import fetch_sackmann_files, year_list


def test_year_list_is_deterministic() -> None:
    assert year_list(2015, 2018) == [2015, 2016, 2017, 2018]


def test_year_list_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="start_year"):
        year_list(2025, 2015)


def test_existing_valid_file_is_not_downloaded(tmp_path: Path) -> None:
    output = tmp_path / "raw"
    output.mkdir()
    existing = output / "atp_matches_2020.csv"
    existing.write_bytes(b"header\nrow\n")
    calls: list[str] = []

    results = fetch_sackmann_files(
        start_year=2020,
        end_year=2020,
        output_dir=output,
        manifest_path=tmp_path / "manifest.json",
        fetcher=lambda url, timeout: calls.append(url) or b"new",
    )

    assert calls == []
    assert results[0].download_status == "existing_valid"
    assert results[0].byte_size == len(b"header\nrow\n")


def test_force_redownloads_and_writes_checksum_manifest(tmp_path: Path) -> None:
    output = tmp_path / "raw"
    output.mkdir()
    (output / "atp_matches_2020.csv").write_bytes(b"old")
    manifest = tmp_path / "manifest.json"

    results = fetch_sackmann_files(
        start_year=2020,
        end_year=2020,
        output_dir=output,
        manifest_path=manifest,
        force=True,
        fetcher=lambda url, timeout: b"new payload",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert results[0].download_status == "downloaded"
    assert results[0].sha256 == payload["files"][0]["SHA-256"]
    assert (output / "atp_matches_2020.csv").read_bytes() == b"new payload"


def test_empty_download_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty download"):
        fetch_sackmann_files(
            start_year=2020,
            end_year=2020,
            output_dir=tmp_path / "raw",
            manifest_path=tmp_path / "manifest.json",
            fetcher=lambda url, timeout: b"",
        )
