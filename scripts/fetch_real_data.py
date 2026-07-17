"""Download the real ATP 2020-2025 season spreadsheets used by Tennis Value.

The files are fetched from a public GitHub mirror of Tennis-Data season
spreadsheets and stored under ``data/raw/tennis_data`` by default. Raw data is
ignored by Git and must remain local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

DEFAULT_YEARS: Final[tuple[int, ...]] = tuple(range(2020, 2026))
MIN_YEAR: Final[int] = 2020
MAX_YEAR: Final[int] = 2025
BASE_URL: Final[str] = (
    "https://raw.githubusercontent.com/"
    "nickdatak/Tennis-Match-Predictions/main/data/{year}.xlsx"
)
SOURCE_REPOSITORY: Final[str] = (
    "https://github.com/nickdatak/Tennis-Match-Predictions"
)
UPSTREAM_SOURCE: Final[str] = "https://www.tennis-data.co.uk/alldata.php"
USER_AGENT: Final[str] = "tennis-value-data-fetcher/1.0"
XLSX_MAGIC: Final[bytes] = b"PK\x03\x04"


@dataclass(frozen=True)
class DownloadRecord:
    year: int
    filename: str
    source_url: str
    bytes: int
    sha256: str
    status: str


def parse_years(values: Iterable[str]) -> tuple[int, ...]:
    years: list[int] = []
    for value in values:
        try:
            year = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid year: {value!r}") from exc
        if not MIN_YEAR <= year <= MAX_YEAR:
            raise argparse.ArgumentTypeError(
                f"Year {year} is outside the supported range {MIN_YEAR}-{MAX_YEAR}."
            )
        years.append(year)
    return tuple(sorted(set(years)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_xlsx(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Downloaded file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {path}")
    with path.open("rb") as handle:
        magic = handle.read(len(XLSX_MAGIC))
    if magic != XLSX_MAGIC:
        raise ValueError(
            f"Downloaded file is not a valid XLSX/ZIP container: {path}"
        )


def download_file(
    *,
    url: str,
    destination: Path,
    timeout: float,
    retries: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Unexpected HTTP status {response.status} for {url}"
                    )
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{destination.name}.",
                    suffix=".part",
                    dir=destination.parent,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    shutil.copyfileobj(response, temporary)

            validate_xlsx(temporary_path)
            os.replace(temporary_path, destination)
            return
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts: {last_error}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ATP 2020-2025 season spreadsheets for Tennis Value."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/tennis_data"),
        help="Destination directory (default: data/raw/tennis_data).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=[str(year) for year in DEFAULT_YEARS],
        metavar="YEAR",
        help="Years to download, limited to 2020-2025.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing season files instead of validating and keeping them.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts per file (default: 3).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        years = parse_years(args.years)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")
    if args.retries <= 0:
        parser.error("--retries must be greater than zero.")

    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[DownloadRecord] = []

    print(f"Destination: {output_dir}")
    print(f"Seasons: {', '.join(str(year) for year in years)}")

    for year in years:
        filename = f"{year}.xlsx"
        destination = output_dir / filename
        source_url = BASE_URL.format(year=year)

        try:
            if destination.exists() and not args.force:
                validate_xlsx(destination)
                status = "kept-existing"
            else:
                print(f"Downloading {year}...")
                download_file(
                    url=source_url,
                    destination=destination,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                status = "downloaded"

            record = DownloadRecord(
                year=year,
                filename=filename,
                source_url=source_url,
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                status=status,
            )
            records.append(record)
            print(
                f"  {filename}: {record.bytes:,} bytes, "
                f"sha256={record.sha256[:12]}..., {status}"
            )
        except Exception as exc:  # noqa: BLE001 - CLI must report per-file failure.
            print(f"ERROR: {year}: {exc}", file=sys.stderr)
            return 1

    manifest = {
        "upstream_source": UPSTREAM_SOURCE,
        "mirror_repository": SOURCE_REPOSITORY,
        "records": [asdict(record) for record in records],
    }
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest: {manifest_path}")
    print("Download complete. Raw files remain local and are ignored by Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
