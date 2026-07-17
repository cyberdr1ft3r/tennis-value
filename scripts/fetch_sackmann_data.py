"""Download Jeff Sackmann ATP yearly match-stat files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2025
DEFAULT_OUTPUT_DIR = Path("data/raw/sackmann")
DEFAULT_MANIFEST = Path("reports/sackmann_fetch_manifest.json")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRIES = 3

Fetcher = Callable[[str, float], bytes]


@dataclass(frozen=True)
class FetchResult:
    """One manifest row for a Sackmann file fetch."""

    year: int
    filename: str
    source: str
    download_status: str
    byte_size: int
    sha256: str
    download_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "filename": self.filename,
            "source": self.source,
            "download status": self.download_status,
            "byte size": self.byte_size,
            "SHA-256": self.sha256,
            "download timestamp": self.download_timestamp,
        }


def year_list(start_year: int, end_year: int) -> list[int]:
    """Return deterministic inclusive season years."""
    if start_year > end_year:
        msg = "start_year must be less than or equal to end_year"
        raise ValueError(msg)
    return list(range(start_year, end_year + 1))


def fetch_sackmann_files(
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    force: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    fetcher: Fetcher | None = None,
) -> list[FetchResult]:
    """Download yearly ATP match files and write a JSON manifest."""
    if retries <= 0:
        msg = "retries must be greater than zero"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    active_fetcher = fetcher or _urlopen_fetch
    results: list[FetchResult] = []
    for year in year_list(start_year, end_year):
        filename = f"atp_matches_{year}.csv"
        source = f"{BASE_URL}/{filename}"
        destination = output_dir / filename
        timestamp = _utc_timestamp()
        if destination.exists() and destination.stat().st_size > 0 and not force:
            payload = destination.read_bytes()
            results.append(
                FetchResult(
                    year=year,
                    filename=filename,
                    source=source,
                    download_status="existing_valid",
                    byte_size=len(payload),
                    sha256=_sha256(payload),
                    download_timestamp=timestamp,
                )
            )
            continue

        payload = _fetch_with_retries(source, timeout_seconds, retries, active_fetcher)
        if not payload:
            msg = f"empty download rejected for {source}"
            raise ValueError(msg)
        _atomic_write(destination, payload)
        results.append(
            FetchResult(
                year=year,
                filename=filename,
                source=source,
                download_status="downloaded",
                byte_size=len(payload),
                sha256=_sha256(payload),
                download_timestamp=timestamp,
            )
        )
    manifest_path.write_text(
        json.dumps(
            {
                "source_attribution": (
                    "Jeff Sackmann tennis_atp ATP match files. Data is subject to the "
                    "source repository license and attribution requirements."
                ),
                "files": [result.as_dict() for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def _fetch_with_retries(
    url: str,
    timeout_seconds: float,
    retries: int,
    fetcher: Fetcher,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fetcher(url, timeout_seconds)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.2)
    msg = f"failed to download {url}: {last_error}"
    raise RuntimeError(msg) from last_error


def _urlopen_fetch(url: str, timeout_seconds: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        return response.read()


def _atomic_write(destination: Path, payload: bytes) -> None:
    with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(destination)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    """CLI wrapper for direct script use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = fetch_sackmann_files(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output,
        manifest_path=args.manifest,
        force=args.force,
    )
    print(f"Fetched or verified {len(results)} Sackmann file(s).")


if __name__ == "__main__":
    main()
