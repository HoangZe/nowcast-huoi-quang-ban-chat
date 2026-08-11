#!/usr/bin/env python3
"""
Fill zero values in the inflow_m3s column of
ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv.

For every row where inflow_m3s == 0, the value is replaced with the average
of the nearest non-zero inflow_m3s value before it and the nearest non-zero
inflow_m3s value after it (computed from the ORIGINAL data). This handles
both isolated zeros and runs of consecutive zeros: all zeros in a run get
the same boundary average (last non-zero before the run, first non-zero
after the run).

The script modifies the dataset file in place.
"""

import csv
import os
import sys

INPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "dataset",
    "ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv",
)
INFLOW_COL = "inflow_m3s"


def main():
    input_path = os.path.abspath(INPUT_FILE)

    if not os.path.isfile(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Read all rows
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if INFLOW_COL not in fieldnames:
        print(f"ERROR: Column '{INFLOW_COL}' not found in header.")
        sys.exit(1)

    n = len(rows)

    # Extract original inflow values (float or None for non-numeric)
    original = []
    for row in rows:
        try:
            original.append(float(row[INFLOW_COL]))
        except (TypeError, ValueError):
            original.append(None)

    replaced = 0
    examples = []

    for i in range(n):
        value = original[i]
        if value is None or value != 0:
            continue

        # Find nearest non-zero value BEFORE this row in the ORIGINAL data
        before = None
        j = i - 1
        while j >= 0:
            v = original[j]
            if v is not None and v != 0:
                before = v
                break
            j -= 1

        # Find nearest non-zero value AFTER this row in the ORIGINAL data
        after = None
        j = i + 1
        while j < n:
            v = original[j]
            if v is not None and v != 0:
                after = v
                break
            j += 1

        if before is None or after is None:
            # No valid neighbor on one side; cannot fill
            continue

        new_value = (before + after) / 2.0
        rows[i][INFLOW_COL] = repr(new_value)
        replaced += 1

        if len(examples) < 5:
            examples.append(
                {
                    "row": i + 2,  # +2 for header and 0-based index
                    "before": before,
                    "after": after,
                    "new": new_value,
                }
            )

    # Write back to the same file
    with open(input_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"File: {input_path}")
    print(f"Total data rows: {n}")
    print(f"Zero values replaced: {replaced}")
    print("\nExample replacements (row number, before, after, new value):")
    for ex in examples:
        print(
            f"  Row {ex['row']}: ({ex['before']} + {ex['after']}) / 2 = {ex['new']}"
        )


if __name__ == "__main__":
    main()