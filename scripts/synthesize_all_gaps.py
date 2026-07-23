#!/usr/bin/env python3
"""
Script 3: Synthesize data for ALL gaps (both 2-hour and irregular) in the
huoi_quang_all_history dataset, writing the result directly to
huoi_quang_all_history_copy.csv.

This script reads huoi_quang_all_history.csv and creates a new
huoi_quang_all_history_copy.csv with synthesized data points inserted
between EVERY pair of adjacent records, regardless of gap size.

For each gap of N hours between two records:
  - (N - 1) new records are synthesized at each hour between the two timestamps
  - Each synthesized record has:
      - time_update: the exact hour between the two original timestamps
      - lake_info_id: 47
      - lake_name: "Huội Quảng"
      - inflow_m3s: the AVERAGE of the two adjacent inflow_m3s values
        (ALL synthesized records in the same gap get the SAME average value)
      - All other columns: 0

Examples:
  - 2-hour gap (e.g., 07:00 → 09:00): 1 record at 08:00 with avg inflow
  - 6-hour gap (e.g., 01:00 → 07:00): 5 records at 02:00,03:00,04:00,05:00,06:00
    all with the same avg inflow of the 01:00 and 07:00 values
  - Cross-day gap (e.g., 23:00 → next day 05:00): records at 00:00,01:00,...
    with correct date handling
"""

import csv
from datetime import datetime, timedelta

INPUT_FILE = "../dataset/ban_chat_all_history_synthesized_2-hour-gaps.csv"
OUTPUT_FILE = "../dataset/ban_chat_all_history_synthesized_full.csv"

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

                # Only synthesize if gap is > 0 hours (should always be true)
                if gap_hours > 0:
                    # Number of records to synthesize = gap_hours - 1
                    # (e.g., 2h gap -> 1 record, 6h gap -> 5 records)
                    num_to_synthesize = int(round(gap_hours)) - 1

                    if num_to_synthesize > 0:
                        # Calculate average inflow from the two bounding records
                        prev_inflow = float(prev_row["inflow_m3s"]) if prev_row["inflow_m3s"] else 0.0
                        curr_inflow = float(row["inflow_m3s"]) if row["inflow_m3s"] else 0.0
                        avg_inflow = (prev_inflow + curr_inflow) / 2.0

                        # Synthesize records at each hour between prev_ts and current_ts
                        for hour_offset in range(1, num_to_synthesize + 1):
                            mid_ts = prev_ts + timedelta(hours=hour_offset)

                            new_row = {}
                            for col in fieldnames:
                                new_row[col] = ""

                            new_row["lake_info_id"] = "45"
                            new_row["lake_name"] = "Bản Chát"
                            new_row["time_update"] = format_timestamp(mid_ts)
                            new_row["inflow_m3s"] = str(avg_inflow)

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