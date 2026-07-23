#!/usr/bin/env python3
"""Inspect the cleaned full inflow dataset without loading it outside this script."""

import csv
from collections import Counter
from datetime import datetime

INPUT_FILE = "dataset/huoi_quang_all_history_cleaned_full.csv"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)


def summarize_row(row):
    return {
        "time_update": row.get("time_update"),
        "inflow_m3s": row.get("inflow_m3s"),
        "lake_info_id": row.get("lake_info_id"),
        "lake_name": row.get("lake_name"),
    }


def main():
    rows = []
    gap_counts = Counter()
    missing_inflow = 0

    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as data_file:
        reader = csv.DictReader(data_file)
        fieldnames = reader.fieldnames or []
        previous_timestamp = None

        for row in reader:
            rows.append(summarize_row(row))
            if not row.get("inflow_m3s", "").strip():
                missing_inflow += 1
            current_timestamp = parse_timestamp(row["time_update"])
            if previous_timestamp is not None:
                gap_hours = (current_timestamp - previous_timestamp).total_seconds() / 3600
                gap_counts[round(gap_hours, 4)] += 1
            previous_timestamp = current_timestamp

    print(f"Dataset: {INPUT_FILE}")
    print(f"Columns ({len(fieldnames)}): {fieldnames}")
    print(f"Rows: {len(rows)}")
    print(f"Missing inflow_m3s: {missing_inflow}")
    print(f"First 3 rows: {rows[:3]}")
    print(f"Last 3 rows: {rows[-3:]}")
    print(f"Unique gap durations (hours): {sorted(gap_counts)}")
    print(f"Gap counts: {dict(sorted(gap_counts.items()))}")


if __name__ == "__main__":
    main()
