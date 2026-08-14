#!/usr/bin/env python3
"""Download and normalize the QM9 CSV used by the Buhito benchmark.

The default source is the official DeepChem QM9 CSV. The generated file is
written under ``data/`` and is intentionally excluded from Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Optional
from urllib.request import Request, urlopen

DEFAULT_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
)


def download(url: str, destination: Path, timeout: int) -> str:
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "buhito-qm9-preparer/1"})
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def normalize_csv(source: Path, output: Path, limit: Optional[int]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {source}.")

        by_lower = {name.lower(): name for name in reader.fieldnames}
        smiles_source = by_lower.get("smiles")
        if smiles_source is None:
            raise ValueError(
                f"QM9 CSV must contain a SMILES column; found {reader.fieldnames}."
            )

        fieldnames = [
            "smiles" if name == smiles_source else name
            for name in reader.fieldnames
        ]
        temporary = output.with_suffix(output.suffix + ".tmp")
        rows_written = 0
        with temporary.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                normalized = {
                    ("smiles" if key == smiles_source else key): value
                    for key, value in row.items()
                }
                writer.writerow(normalized)
                rows_written += 1
                if limit is not None and rows_written >= limit:
                    break
        temporary.replace(output)
    return rows_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an external QM9 CSV with a lowercase 'smiles' column."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qm9/qm9_processed.csv"),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Normalize a local CSV instead of downloading the default source.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Write only the first N rows for a quick benchmark fixture.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sha256",
        default=None,
        help="Optional expected SHA-256 of the downloaded raw CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"{output} already exists. Use --force to replace it."
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    with tempfile.TemporaryDirectory(prefix="buhito-qm9-") as temp_directory:
        temporary_source = Path(temp_directory) / "qm9.csv"
        if args.source_csv is not None:
            local_source = args.source_csv.expanduser().resolve()
            if not local_source.exists():
                raise FileNotFoundError(local_source)
            shutil.copyfile(local_source, temporary_source)
            raw_sha256 = hashlib.sha256(temporary_source.read_bytes()).hexdigest()
            source_description = str(local_source)
        else:
            raw_sha256 = download(args.url, temporary_source, args.timeout)
            source_description = args.url

        if args.sha256 and raw_sha256.lower() != args.sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch: expected {args.sha256}, got {raw_sha256}."
            )

        rows = normalize_csv(temporary_source, output, args.limit)

    print(f"Source: {source_description}")
    print(f"Raw SHA-256: {raw_sha256}")
    print(f"Rows written: {rows}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
