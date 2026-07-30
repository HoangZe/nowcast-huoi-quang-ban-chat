#!/usr/bin/env python3
"""
Script to check for irregular time gaps in the Ban-Chat-satellite.csv dataset
(after conversion to hourly data).

A gap of exactly 1 hour is considered normal. Any gap larger than 1 hour
is flagged as irregular.

Output:
  - Summary statistics
  - Table of all irregular gaps
  - List of unique gap durations for analysis
"""

import csv
from datetime import datetime
from collections import Counter

INPUT_FILE = "dataset/hourly_rain_data/Ban-Chat-satellite.csv"
EXPECTED_GAP_HOURS = 1.0
TOLERANCE = 0.001  # floating-point tolerance


def parse_timestamp(ts_str: str) -> datetime:
    """Parse a timestamp string like '2016-01-01 07:00:00' into a datetime object."""
    return datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")


def main():
    gaps_found = []
    total_rows = 0

    with open(INPUT_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        prev_ts = None

        for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
            total_rows = row_num
            current_ts = parse_timestamp(row["datetime"])

            if prev_ts is not None:
                gap_hours = (current_ts - prev_ts).total_seconds() / 3600.0

                # Flag if gap is larger than 1 hour (beyond tolerance)
                if gap_hours > EXPECTED_GAP_HOURS + TOLERANCE:
                    gaps_found.append({
                        "prev_row": row_num - 1,
                        "curr_row": row_num,
                        "prev_timestamp": prev_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "curr_timestamp": current_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "gap_hours": gap_hours,
                    })

            prev_ts = current_ts

    # ── Print summary ──────────────────────────────────────────────
    print("=" * 80)
    print(f"  Dataset: {INPUT_FILE}")
    print(f"  Total rows processed (excluding header): {total_rows - 1}")
    print(f"  Expected gap between records: {EXPECTED_GAP_HOURS} hour(s)")
    print(f"  Irregular gaps (> {EXPECTED_GAP_HOURS}h) found: {len(gaps_found)}")
    print("=" * 80)
    print()

    if not gaps_found:
        print("  ✓ All gaps are exactly 1 hour. No irregularities found.")
        return

    # ── Print table ────────────────────────────────────────────────
    header = f"{'Prev Row':>8} | {'Curr Row':>8} | {'Previous Timestamp':<26} | {'Current Timestamp':<26} | {'Gap (hrs)':>10}"
    print(header)
    print("-" * len(header))
    for g in gaps_found:
        print(f"{g['prev_row']:>8} | {g['curr_row']:>8} | {g['prev_timestamp']:<26} | {g['curr_timestamp']:<26} | {g['gap_hours']:>10.3f}")

    print()

    # ── Statistics ─────────────────────────────────────────────────
    gap_durations = sorted(set(g["gap_hours"] for g in gaps_found))
    print(f"  Unique irregular gap durations (hours): {gap_durations}")
    print()

    # Count by duration
    duration_counts = Counter(g["gap_hours"] for g in gaps_found)
    print(f"  Breakdown by gap duration:")
    print(f"  {'Gap (hours)':>14} | {'Count':>6}")
    print(f"  {'-'*14}-+-{'-'*6}")
    for dur in sorted(duration_counts):
        print(f"  {dur:>14.3f} | {duration_counts[dur]:>6}")

    print()
    print(f"  Total irregular gaps: {len(gaps_found)}")


if __name__ == "__main__":
    main()