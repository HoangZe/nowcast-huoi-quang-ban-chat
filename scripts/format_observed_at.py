#!/usr/bin/env python3
"""
Convert the observed_at column in rainfall CSV files from
"YYYY-MM-DD HH:MM:SS" to "YYYY-MM-DDTHH:MM:SS.000" (ISO 8601 with milliseconds).

Usage:
    python scripts/format_observed_at.py dataset/hourly_rain_data/Ta-Hua-rainfall.csv
    python scripts/format_observed_at.py dataset/hourly_rain_data/*.csv
"""

import csv
import re
import sys
from pathlib import Path


def transform_timestamp(ts: str) -> str:
    """
    Transform "2026-07-15 14:00:00" into "2026-07-15T14:00:00.000".
    If the timestamp already contains 'T', return it unchanged.
    """
    ts = ts.strip()
    if not ts:
        return ts
    # If already in the target format, skip
    if "T" in ts:
        return ts
    # Match "YYYY-MM-DD HH:MM:SS" and replace the space with 'T', append ".000"
    match = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", ts)
    if match:
        return f"{match.group(1)}T{match.group(2)}.000"
    # If it doesn't match the expected pattern, return as-is
    return ts


def process_file(filepath: str) -> None:
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}", file=sys.stderr)
        return

    # Read all rows
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "observed_at" not in reader.fieldnames:
            print(f"Skipping {filepath}: no 'observed_at' column", file=sys.stderr)
            return
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Transform the observed_at column
    changed = 0
    for row in rows:
        original = row["observed_at"]
        transformed = transform_timestamp(original)
        if transformed != original:
            row["observed_at"] = transformed
            changed += 1

    if changed == 0:
        print(f"No changes needed in {filepath}")
        return

    # Write back to the same file
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {changed} rows in {filepath}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python format_observed_at.py <file1.csv> [file2.csv ...]", file=sys.stderr)
        sys.exit(1)

    for arg in sys.argv[1:]:
        # Support glob patterns
        for p in Path().glob(arg):
            process_file(str(p))


if __name__ == "__main__":
    main()