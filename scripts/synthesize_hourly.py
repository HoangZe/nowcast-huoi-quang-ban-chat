#!/usr/bin/env python3
"""
Script 2: Synthesize hourly inflow data for 2-hour gaps.

This script reads huoi_quang_all_history.csv and creates huoi_quang_all_history_copy.csv
with new data points inserted between every pair of adjacent records that are exactly
2 hours apart.

For each 2-hour gap, a new record is created at the midpoint (1 hour after the earlier
record) with:
  - time_update: the hour between the two original timestamps
  - lake_info_id: 47
  - lake_name: "Huội Quảng"
  - inflow_m3s: average of the two adjacent inflow_m3s values
  - All other columns: 0

IMPORTANT: This script ONLY synthesizes data for gaps that are EXACTLY 2 hours.
Irregular gaps (larger or smaller) are left untouched and passed through as-is.
"""

import csv
import os
from datetime import datetime, timedelta

INPUT_FILE = "../dataset/ban_chat_all_history.csv"
OUTPUT_FILE = "../dataset/ban_chat_all_history_copy.csv"

EXPECTED_GAP_HOURS = 2

# Columns that should be set to 0 for synthesized records
ZERO_COLUMNS = [
    "hour",
    "water_level_m",
    "total_outflow_m3s",
    "powerhouse_outflow_m3s",
    "spillway_outflow_m3s",
    "hours_with_data",
    "note",
    "time_changed",
]


def parse_timestamp(ts_str: str) -> datetime:
    """Parse a timestamp string like '2015-12-01T17:00:00.000' into a datetime object."""
    return datetime.strptime(ts_str.strip(), "%Y-%m-%dT%H:%M:%S.%f")


def format_timestamp(dt: datetime) -> str:
    """Format a datetime back to the CSV timestamp format."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def main():
    rows_written = 0
    rows_synthesized = 0

    with open(INPUT_FILE, mode="r", encoding="utf-8-sig") as infile, \
         open(OUTPUT_FILE, mode="w", encoding="utf-8-sig", newline="") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        prev_row = None
        prev_ts = None

        for row in reader:
            current_ts = parse_timestamp(row["time_update"])

            if prev_row is not None:
                gap_hours = (current_ts - prev_ts).total_seconds() / 3600.0

                # Only synthesize for EXACTLY 2-hour gaps
                if abs(gap_hours - EXPECTED_GAP_HOURS) < 0.001:
                    # Create a new record at the midpoint (1 hour after prev_ts)
                    mid_ts = prev_ts + timedelta(hours=1)

                    new_row = {}
                    # Copy all fields from the previous row as a starting point
                    for col in fieldnames:
                        new_row[col] = ""

                    # Set the required fields for the synthesized record
                    new_row["lake_info_id"] = "45"
                    new_row["lake_name"] = "Bản Chát"
                    new_row["time_update"] = format_timestamp(mid_ts)

                    # Calculate average inflow
                    prev_inflow = float(prev_row["inflow_m3s"]) if prev_row["inflow_m3s"] else 0.0
                    curr_inflow = float(row["inflow_m3s"]) if row["inflow_m3s"] else 0.0
                    avg_inflow = (prev_inflow + curr_inflow) / 2.0
                    new_row["inflow_m3s"] = str(avg_inflow)

                    # Set all other columns to 0
                    for col in ZERO_COLUMNS:
                        new_row[col] = "0"

                    writer.writerow(new_row)
                    rows_synthesized += 1

                # Write the current (original) row
                writer.writerow(row)
                rows_written += 1
            else:
                # First row: just write it as-is
                writer.writerow(row)
                rows_written += 1

            prev_row = row
            prev_ts = current_ts

    print(f"Original rows written: {rows_written}")
    print(f"Synthesized rows inserted: {rows_synthesized}")
    print(f"Total rows in output: {rows_written + rows_synthesized}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()