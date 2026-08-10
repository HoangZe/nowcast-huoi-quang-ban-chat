#!/usr/bin/env python3
"""Engineer seasonal features from time_update into the Bản Chát inflow CSV.

This script performs two in-place modifications on
dataset/ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv:

1. hour: replaced with the HH:MM:SS hour component extracted from time_update
   (the existing column currently holds all-zero placeholders).
2. day_of_year_sin / day_of_year_cos: appended as new columns, encoding the
   day-of-year of each time_update as a cyclic sin/cos pair in [-1, 1].

The file is streamed row-by-row, so it is safe for large CSVs (no full
in-memory load of the entire dataset). All other columns and rows are preserved.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
DAYS_PER_YEAR = 365.25


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_arguments() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=(
            root
            / "dataset"
            / "ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv"
        ),
        help="Path to the CSV to modify in place.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Text encoding used to read/write the CSV (default: utf-8-sig).",
    )
    return parser.parse_args()


def hour_of_timestamp(timestamp: datetime) -> str:
    """Return the HH:MM:SS hour component of a timestamp."""
    return timestamp.strftime("%H:%M:%S")


def day_of_year_cyclic(timestamp: datetime) -> tuple[float, float]:
    """Return (sin, cos) encoding of the day-of-year in [-1, 1]."""
    day_of_year = timestamp.timetuple().tm_yday
    angle = 2.0 * math.pi * day_of_year / DAYS_PER_YEAR
    return round(math.sin(angle), 10), round(math.cos(angle), 10)


def engineer_features(
    reader: csv.DictReader,
    writer: csv.DictWriter,
) -> tuple[int, int]:
    """Rewrite each row, updating hour and adding the cyclic day-of-year.

    Returns (rows_written, errors) counts.
    """
    rows_written = 0
    errors = 0
    for row_number, row in enumerate(reader, start=2):
        raw_timestamp = row.get("time_update")
        if not raw_timestamp:
            print(f"Row {row_number}: missing time_update; skipping.")
            errors += 1
            continue
        try:
            timestamp = datetime.strptime(raw_timestamp.strip(), TIMESTAMP_FORMAT)
        except (TypeError, ValueError) as error:
            print(f"Row {row_number}: invalid time_update {raw_timestamp!r}: {error}")
            errors += 1
            continue

        row["hour"] = hour_of_timestamp(timestamp)
        day_of_year_sin, day_of_year_cos = day_of_year_cyclic(timestamp)
        row["day_of_year_sin"] = day_of_year_sin
        row["day_of_year_cos"] = day_of_year_cos
        writer.writerow(row)
        rows_written += 1
    return rows_written, errors


def main() -> None:
    arguments = parse_arguments()
    data_path = arguments.data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Write to a temporary sibling file, then atomically replace the original.
    temporary_path = data_path.with_name(data_path.name + ".tmp")
    try:
        with (
            data_path.open(encoding=arguments.encoding, newline="") as source,
            temporary_path.open(encoding=arguments.encoding, newline="", mode="w") as target,
        ):
            reader = csv.DictReader(source)
            original_fieldnames = list(reader.fieldnames or [])
            if "hour" not in original_fieldnames:
                raise ValueError("CSV is missing the 'hour' column.")
            if "time_update" not in original_fieldnames:
                raise ValueError("CSV is missing the 'time_update' column.")

            new_fieldnames = original_fieldnames + [
                "day_of_year_sin",
                "day_of_year_cos",
            ]
            writer = csv.DictWriter(target, fieldnames=new_fieldnames)
            writer.writeheader()
            rows_written, errors = engineer_features(reader, writer)

        temporary_path.replace(data_path)
    except Exception:
        # Clean up the temporary file on any failure so we never leave a
        # half-written file alongside the original.
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print(f"Engineered {rows_written} rows, {errors} error(s).")
    print(f"Updated columns: hour, day_of_year_sin, day_of_year_cos")
    print(f"Wrote: {data_path}")


if __name__ == "__main__":
    main()