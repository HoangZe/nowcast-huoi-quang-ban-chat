#!/usr/bin/env python3
"""
Process Ban-Chat-satellite.csv:
1. Shift datetime from UTC to GMT+7 (add 7 hours)
2. Convert half-hourly data to hourly by summing precipitation_mm of each
   hourly mark (e.g., 08:00:00) with its half-hourly mark (e.g., 08:30:00).
   The result takes the hourly timestamp.

Usage: python scripts/process_rainfall_hourly.py
"""

import pandas as pd
import os

INPUT_PATH = "dataset/hourly_rain_data/Ban-Chat-satellite.csv"
OUTPUT_PATH = "dataset/hourly_rain_data/Ban-Chat-satellite.csv"  # same file, in-place

def main():
    # 1. Read CSV
    df = pd.read_csv(INPUT_PATH)
    print(f"Original rows: {len(df)}")

    # 2. Parse datetime column (currently UTC)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # 3. Shift from UTC to GMT+7 (add 7 hours)
    df["datetime"] = df["datetime"] + pd.Timedelta(hours=7)

    # 4. Convert half-hourly to hourly by summing precipitation_mm
    #    Strategy: group by the hourly floor of each timestamp.
    #    For a timestamp like "2026-07-20 02:30:00", floor gives "2026-07-20 02:00:00".
    #    For "2026-07-20 02:00:00", floor gives "2026-07-20 02:00:00".
    #    Both rows in the same hour group get summed.
    df["hour_bin"] = df["datetime"].dt.floor("h")
    df_hourly = df.groupby("hour_bin", as_index=False).agg(
        precipitation_mm=("precipitation_mm", "sum")
    )
    df_hourly.rename(columns={"hour_bin": "datetime"}, inplace=True)

    # 5. Drop any rows where precipitation_mm is NaN (shouldn't happen)
    df_hourly.dropna(subset=["precipitation_mm"], inplace=True)

    # 6. Format datetime as string without timezone suffix
    df_hourly["datetime"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Hourly rows: {len(df_hourly)}")
    print(f"First 5 rows:\n{df_hourly.head()}")
    print(f"Last 5 rows:\n{df_hourly.tail()}")

    # 7. Write back to the same CSV file
    df_hourly.to_csv(OUTPUT_PATH, index=False)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()