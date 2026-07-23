#!/usr/bin/env python3
"""
Script 1: Spot gaps larger than 1 hour in the ban_chat_all_history dataset.

This script reads the CSV file and identifies gaps between consecutive data points
that are larger than 1 hour. It reports:
  - The row numbers of the two adjacent records
  - The timestamps of both records
  - The gap duration in hours

This helps you understand where data is missing so you can
decide on a synthesizing tactic for those cases.
"""

import csv
from datetime import datetime

INPUT_FILE = "dataset/ban_chat_all_history.csv"

# Expected gap in hours
EXPECTED_GAP_HOURS = 1

def parse_timestamp(ts_str: str) -> datetime:
    """Parse a timestamp string like '2015-12-01T17:00:00.000' into a datetime object."""
    return datetime.strptime(ts_str.strip(), "%Y-%m-%dT%H:%M:%S.%f")

def main():
    gaps_found = []
    total_rows = 0

    with open(INPUT_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        prev_row = None
        prev_ts = None

        for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
            total_rows = row_num
            current_ts = parse_timestamp(row["time_update"])

            if prev_row is not None:
                gap_hours = (current_ts - prev_ts).total_seconds() / 3600.0

                if gap_hours > EXPECTED_GAP_HOURS + 0.001:  # tolerance for floating point
                    gaps_found.append({
                        "prev_row": row_num - 1,
                        "curr_row": row_num,
                        "prev_timestamp": prev_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                        "curr_timestamp": current_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                        "gap_hours": gap_hours,
                    })

            prev_row = row
            prev_ts = current_ts

    # Print summary
    print(f"Total rows processed (excluding header): {total_rows - 1}")
    print(f"Total irregular gaps found: {len(gaps_found)}")
    print()

    if gaps_found:
        print(f"{'Prev Row':>8} | {'Curr Row':>8} | {'Previous Timestamp':<26} | {'Current Timestamp':<26} | {'Gap (hrs)':>10}")
        print("-" * 90)
        for g in gaps_found:
            print(f"{g['prev_row']:>8} | {g['curr_row']:>8} | {g['prev_timestamp']:<26} | {g['curr_timestamp']:<26} | {g['gap_hours']:>10.3f}")

        print()
        print(f"Total gaps larger than {EXPECTED_GAP_HOURS}h: {len(gaps_found)}")

        # Show unique gap durations
        gap_durations = sorted(set(g["gap_hours"] for g in gaps_found))
        print(f"\nUnique gap durations (hours): {gap_durations}")
    else:
        print(f"All gaps are exactly {EXPECTED_GAP_HOURS}h. No gaps larger than {EXPECTED_GAP_HOURS}h found.")


if __name__ == "__main__":
    main()