#!/usr/bin/env python3
"""Convert comma decimal separators to period decimal separators in a CSV.

The dataset `ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv`
stores numeric values with a comma as the decimal separator (European locale),
e.g. `"473,42"` or `"0,05001904267"`.  This script rewrites those values to use
a period so the file is compatible with parsers that expect a period decimal
separator (Python's ``float()``, pandas, etc.).

Only the numeric data columns are touched; timestamp columns such as
``time_update`` and ``time_changed`` already use a period and are left unchanged.

Usage::

    python scripts/convert_comma_to_period_decimals.py
    python scripts/convert_comma_to_period_decimals.py --input path/to/in.csv --output path/to/out.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Columns whose values may use a comma as the decimal separator.
DECIMAL_COLUMNS = ["water_level_m", "inflow_m3s", "precipitation_mm"]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_arguments() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "dataset" / "ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv",
        help="Input CSV with comma decimal separators.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to overwriting --input.",
    )
    return parser.parse_args()


def convert_value(value: str) -> str:
    """Replace a comma decimal separator with a period, preserving non-numeric text."""
    if value is None:
        return value
    return value.replace(",", ".")


def main() -> None:
    arguments = parse_arguments()
    input_path: Path = arguments.input
    output_path: Path = arguments.output if arguments.output is not None else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read all rows, convert the decimal columns, then write back.
    with input_path.open(encoding="utf-8-sig", newline="") as data_file:
        reader = csv.DictReader(data_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = [c for c in DECIMAL_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"Input CSV is missing expected columns: {missing}")
        rows = list(reader)

    converted = 0
    for row in rows:
        for column in DECIMAL_COLUMNS:
            original = row[column]
            if original is None:
                continue
            if "," in original:
                row[column] = convert_value(original)
                converted += 1

    # Write to a temp file first, then replace, so a failure mid-write
    # does not corrupt the original when output == input.
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(output_path)

    print(f"Converted {converted} comma-decimal value(s) across {len(rows)} row(s).")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()