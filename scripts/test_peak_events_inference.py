#!/usr/bin/env python3
"""Test inference on peak flood events using a saved LSTM checkpoint.

Loads src/models/lstm_inflow.pth (trained by src/train_lstm_inflow.py, a
2-feature inflow+precipitation model), detects peak flood events in the
dataset, runs direct 6-hour inference across each event window, reports
per-category / per-horizon metrics, and saves predicted-vs-actual plots in
the same 2x3 (t+1..t+6) style as the training script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

# Keep Matplotlib's cache outside the repository and in a writable location.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nowcast-huoi-quang-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
EXPECTED_INTERVAL = timedelta(hours=1)
EPSILON = 1e-8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "lstm_inflow.pth"
OUTPUT_DIR = PROJECT_ROOT / "src" / "plots" / "peak_events"
RESULTS_JSON = PROJECT_ROOT / "scripts" / "peak_event_inference_results.json"

# Peak event classification thresholds (m3/s)
SMALL_FLOOR = 850.0
BIG_FLOOR = 2100.0
MAJOR_FLOOR = 3000.0

# Event window around each peak (hours before / after the peak index)
PRE_HOURS = 72
POST_HOURS = 48


@dataclass(frozen=True)
class ModelConfig:
    lookback_hours: int = 48
    horizon_hours: int = 6
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2


class InflowLSTM(nn.Module):
    """A direct multi-output LSTM for multivariate inflow histories."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float, horizon_hours: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, horizon_hours)

    def forward(self, features: Tensor) -> Tensor:
        sequence_output, _ = self.lstm(features)
        return self.output(sequence_output[:, -1, :])


def parse_locale_float(value: str) -> float:
    """Parse a numeric string that may use a comma as the decimal separator."""
    return float(value.strip().replace(",", "."))


def load_inflow_series(data_path: Path) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    timestamps: list[datetime] = []
    inflows: list[float] = []
    precipitations: list[float] = []

    with data_path.open(encoding="utf-8-sig", newline="") as data_file:
        reader = csv.DictReader(data_file)
        required_columns = {"time_update", "inflow_m3s", "precipitation_mm"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            timestamp = datetime.strptime(row["time_update"].strip(), TIMESTAMP_FORMAT)
            try:
                inflow = parse_locale_float(row["inflow_m3s"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid inflow_m3s on row {row_number}: {row['inflow_m3s']!r}") from error
            if not np.isfinite(inflow):
                raise ValueError(f"Non-finite inflow_m3s on row {row_number}: {inflow}")
            try:
                precipitation = parse_locale_float(row["precipitation_mm"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid precipitation_mm on row {row_number}: {row['precipitation_mm']!r}") from error
            if not np.isfinite(precipitation):
                raise ValueError(f"Non-finite precipitation_mm on row {row_number}: {precipitation}")
            timestamps.append(timestamp)
            inflows.append(inflow)
            precipitations.append(precipitation)

    if not timestamps:
        raise ValueError("Dataset contains no rows.")
    return (
        timestamps,
        np.asarray(inflows, dtype=np.float64),
        np.asarray(precipitations, dtype=np.float64),
    )


def find_hourly_segments(timestamps: list[datetime]) -> list[tuple[int, int]]:
    """Return half-open row-index ranges separated by any non-hourly transition."""
    segments: list[tuple[int, int]] = []
    segment_start = 0
    for index in range(1, len(timestamps)):
        if timestamps[index] - timestamps[index - 1] != EXPECTED_INTERVAL:
            segments.append((segment_start, index))
            segment_start = index
    segments.append((segment_start, len(timestamps)))
    return segments


def detect_peaks(inflow_values: np.ndarray) -> np.ndarray:
    """Return indices of peak flood events (inflow >= 850 m3/s)."""
    if _HAS_SCIPY:
        peak_indices, _ = find_peaks(
            inflow_values,
            height=SMALL_FLOOR,
            distance=24,
            prominence=50,
        )
        return np.asarray(peak_indices, dtype=int)
    # Fallback: simple local maxima above the small-flood floor.
    peak_indices = []
    for i in range(1, len(inflow_values) - 1):
        if (
            inflow_values[i] >= SMALL_FLOOR
            and inflow_values[i] >= inflow_values[i - 1]
            and inflow_values[i] >= inflow_values[i + 1]
        ):
            if not peak_indices or (i - peak_indices[-1]) >= 12:
                peak_indices.append(i)
    return np.asarray(peak_indices, dtype=int)


def classify_peak(peak_value: float) -> str:
    if peak_value >= MAJOR_FLOOR:
        return "major_and_large"
    elif peak_value >= BIG_FLOOR:
        return "big"
    elif peak_value >= SMALL_FLOOR:
        return "small"
    return "below_threshold"


def load_checkpoint(model_path: Path) -> tuple[dict, ModelConfig, dict]:
    """Load the checkpoint and return (raw, config, scaler)."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config", {})
    config = ModelConfig(
        lookback_hours=int(raw_config.get("lookback_hours", 48)),
        horizon_hours=int(raw_config.get("horizon_hours", 6)),
        hidden_size=int(raw_config.get("hidden_size", 256)),
        num_layers=int(raw_config.get("num_layers", 2)),
        dropout=float(raw_config.get("dropout", 0.2)),
    )
    scaler = checkpoint.get("scaler", {})
    return checkpoint, config, scaler


def split_of_index(index: int, train_end: int, val_end: int) -> str:
    """Return the split ('train'/'validation'/'test') for a row index."""
    if index < train_end:
        return "train"
    elif index < val_end:
        return "validation"
    return "test"


def build_model(config: ModelConfig, input_size: int) -> InflowLSTM:
    return InflowLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        horizon_hours=config.horizon_hours,
    )


def inverse_scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def metric_summary(actual: np.ndarray, predicted: np.ndarray, mase_scale: float) -> dict[str, dict[str, float]]:
    def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        absolute_error = np.abs(y_pred - y_true)
        squared_error = np.square(y_pred - y_true)
        observed_variance = np.sum(np.square(y_true - np.mean(y_true)))
        nse = 1.0 - np.sum(squared_error) / (observed_variance + EPSILON)
        return {
            "mae": float(np.mean(absolute_error)),
            "rmse": float(np.sqrt(np.mean(squared_error))),
            "smape_percent": float(
                np.mean(200.0 * absolute_error / (np.abs(y_true) + np.abs(y_pred) + EPSILON))
            ),
            "mase": float(np.mean(absolute_error) / mase_scale),
            "nse": float(nse),
        }

    summary = {"overall": metrics_for(actual, predicted)}
    for horizon_index in range(actual.shape[1]):
        summary[f"t+{horizon_index + 1}"] = metrics_for(
            actual[:, horizon_index], predicted[:, horizon_index]
        )
    return summary


def print_metrics(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(
        f"{'Horizon':<10} {'MAE':>12} {'RMSE':>12} {'sMAPE (%)':>14} "
        f"{'MASE':>12} {'NSE':>10}"
    )
    for horizon, values in summary.items():
        print(
            f"{horizon:<10} {values['mae']:>12.4f} {values['rmse']:>12.4f} "
            f"{values['smape_percent']:>14.4f} {values['mase']:>12.4f} "
            f"{values['nse']:>10.4f}"
        )


def save_event_plot(
    plot_path: Path,
    timestamps: list[datetime],
    event: dict,
    target_times: list[list[datetime]],
    actual: np.ndarray,
    predicted: np.ndarray,
    config: ModelConfig,
) -> None:
    """Save a 2x3 (t+1..t+6) predicted-vs-actual plot for one peak event."""
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    for horizon_index, axis in enumerate(axes.flat):
        times = target_times[horizon_index]
        axis.plot(times, actual[:, horizon_index], label="Actual", linewidth=1.5)
        axis.plot(times, predicted[:, horizon_index], label="Predicted", linewidth=1.25)
        axis.set_title(f"BẢN CHÁT t+{horizon_index + 1}")
        axis.set_ylabel("Inflow (m³/s)")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
        if horizon_index == 0:
            axis.legend()
    figure.suptitle(
        f"BẢN CHÁT - LSTM peak event {event['category']} "
        f"(peak {event['peak_value']:.1f} m³/s @ {event['timestamp']:%Y-%m-%d %H:%M})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "validation", "test"],
        default="all",
        help=(
            "Which peak events to run inference on. 'test' runs only on the "
            "held-out test split (leakage-free). 'all' runs on every detected "
            "peak event. Default: all."
        ),
    )
    args = parser.parse_args()
    split_filter = args.split

    print("=" * 80)
    print("PEAK EVENT INFERENCE TEST")
    print("=" * 80)
    print(f"Split filter: {split_filter}")

    # 1. Load checkpoint and inspect architecture
    checkpoint, config, scaler = load_checkpoint(MODEL_PATH)
    print(f"\nCheckpoint: {MODEL_PATH}")
    print(f"Config: {asdict(config)}")

    # Load the checkpoint's authoritative split boundaries for leakage filtering.
    split_boundaries = checkpoint.get("split_boundaries", {})
    ckpt_train_end = int(split_boundaries.get("train_end_row_exclusive", 0))
    ckpt_val_end = int(split_boundaries.get("validation_end_row_exclusive", 0))
    print(
        f"Checkpoint splits: train=[0,{ckpt_train_end}), "
        f"validation=[{ckpt_train_end},{ckpt_val_end}), "
        f"test=[{ckpt_val_end},...)"
    )

    # Determine input size from the saved LSTM layer weight shape.
    state = checkpoint["model_state_dict"]
    lstm_weight = state.get("lstm.weight_ih_l0")
    if lstm_weight is None:
        raise ValueError("Checkpoint has no lstm.weight_ih_l0; cannot determine input size.")
    input_size = int(lstm_weight.shape[1])
    print(f"Detected LSTM input size: {input_size} (1=inflow only, 2=inflow+precipitation)")

    # 2. Load dataset
    timestamps, inflows, precipitations = load_inflow_series(DATA_PATH)
    print(f"\nDataset rows: {len(inflows)}")
    print(f"Time range: {timestamps[0]}  ->  {timestamps[-1]}")

    # 3. Build model and load weights
    device = torch.device("cpu")
    model = build_model(config, input_size).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded on {device}.")

    # 4. Scaling (use the scaler saved in the checkpoint)
    if input_size == 2:
        inflow_scaler = scaler.get("inflow", {})
        precip_scaler = scaler.get("precipitation", {})
        inflow_mean = float(inflow_scaler.get("mean", 0.0))
        inflow_std = float(inflow_scaler.get("std", 1.0))
        precip_mean = float(precip_scaler.get("mean", 0.0))
        precip_std = float(precip_scaler.get("std", 1.0))
        scaled_inflows = ((inflows - inflow_mean) / inflow_std).astype(np.float32)
        scaled_precipitations = ((precipitations - precip_mean) / precip_std).astype(np.float32)
        scaled_features = np.column_stack([scaled_inflows, scaled_precipitations])
    else:
        scaler_mean = float(scaler.get("mean", 0.0))
        scaler_std = float(scaler.get("std", 1.0))
        inflow_mean = scaler_mean
        inflow_std = scaler_std
        scaled_inflows = ((inflows - scaler_mean) / scaler_std).astype(np.float32)
        scaled_features = scaled_inflows.reshape(-1, 1)

    print(f"Inflow scaler: mean={inflow_mean:.4f}, std={inflow_std:.4f}")

    # 5. MASE scale (24-hour seasonal naive on the full series, matching training)
    seasonal_period = 24
    mase_scale = float(
        np.mean(np.abs(inflows[seasonal_period:] - inflows[:-seasonal_period]))
    )
    if mase_scale <= EPSILON:
        mase_scale = 1.0
    print(f"MASE scale (24h seasonal naive): {mase_scale:.4f}")

    # 6. Detect peak events
    peak_indices = detect_peaks(inflows)
    print(f"\nDetected peak events (inflow >= {SMALL_FLOOR:.0f} m3/s): {len(peak_indices)}")

    events = []
    for idx in peak_indices:
        peak_value = float(inflows[idx])
        events.append({
            "index": int(idx),
            "timestamp": timestamps[idx],
            "peak_value": peak_value,
            "category": classify_peak(peak_value),
            "split": split_of_index(int(idx), ckpt_train_end, ckpt_val_end),
        })

    # Filter events by the requested split (leakage control).
    if split_filter != "all":
        before = len(events)
        events = [e for e in events if e["split"] == split_filter]
        print(
            f"Filtered to split '{split_filter}': {len(events)} / {before} peak events "
            f"(leakage-free if 'test')."
        )
    else:
        from collections import Counter
        split_counts = Counter(e["split"] for e in events)
        print(
            f"Split distribution: train={split_counts.get('train', 0)}, "
            f"validation={split_counts.get('validation', 0)}, "
            f"test={split_counts.get('test', 0)}"
        )

    # 7. Run inference across each event window
    lookback = config.lookback_hours
    horizon = config.horizon_hours
    n_rows = len(inflows)

    # Collect per-event prediction data
    event_predictions = []  # list of dicts with per-horizon arrays
    all_actual = []
    all_predicted = []

    for event in events:
        peak_idx = event["index"]
        # Event window: [peak - PRE_HOURS, peak + POST_HOURS]
        window_start = max(0, peak_idx - PRE_HOURS)
        window_end = min(n_rows, peak_idx + POST_HOURS + 1)

        # Valid prediction starts: need lookback before and horizon after
        valid_starts = []
        for s in range(window_start, window_end):
            if s + lookback + horizon <= n_rows:
                valid_starts.append(s)
        if not valid_starts:
            continue

        # Build feature batch for all valid starts
        feature_batch = np.stack(
            [scaled_features[s : s + lookback] for s in valid_starts]
        )
        with torch.no_grad():
            output = model(torch.as_tensor(feature_batch, dtype=torch.float32).to(device))
            pred_scaled = output.cpu().numpy()

        pred = inverse_scale(pred_scaled, inflow_mean, inflow_std)
        # Actual targets: inflows[s + lookback : s + lookback + horizon]
        actual = np.stack(
            [inflows[s + lookback : s + lookback + horizon] for s in valid_starts]
        )

        # Target times per horizon
        target_times = [
            [timestamps[s + lookback + h] for s in valid_starts] for h in range(horizon)
        ]

        event["valid_starts"] = valid_starts
        event["target_times"] = target_times
        event["actual"] = actual
        event["predicted"] = pred

        event_predictions.append(event)
        all_actual.append(actual)
        all_predicted.append(pred)

    if not event_predictions:
        raise RuntimeError("No valid peak events with full inference windows found.")

    all_actual = np.concatenate(all_actual, axis=0)
    all_predicted = np.concatenate(all_predicted, axis=0)
    print(f"Total inference samples across all events: {all_actual.shape[0]}")

    # 8. Metrics per category and overall
    categories = ["small", "big", "major_and_large"]
    category_metrics = {}
    for cat in categories:
        cat_actual = np.concatenate(
            [e["actual"] for e in event_predictions if e["category"] == cat], axis=0
        ) if any(e["category"] == cat for e in event_predictions) else None
        if cat_actual is None or len(cat_actual) == 0:
            category_metrics[cat] = None
            continue
        cat_predicted = np.concatenate(
            [e["predicted"] for e in event_predictions if e["category"] == cat], axis=0
        )
        category_metrics[cat] = metric_summary(cat_actual, cat_predicted, mase_scale)

    overall_metrics = metric_summary(all_actual, all_predicted, mase_scale)

    print("\n" + "=" * 80)
    print("OVERALL PEAK-EVENT METRICS (all categories combined)")
    print("=" * 80)
    print_metrics("Overall peak-event metrics", overall_metrics)

    for cat in categories:
        if category_metrics[cat] is None:
            print(f"\nCategory '{cat}': no events with valid inference windows.")
            continue
        print("\n" + "=" * 80)
        print(f"CATEGORY: {cat}")
        print("=" * 80)
        print_metrics(f"Peak-event metrics - {cat}", category_metrics[cat])

    # 9. Save per-event prediction results to JSON
    results = {
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "config": asdict(config),
        "input_size": input_size,
        "split_filter": split_filter,
        "split_boundaries": {
            "train_end_row_exclusive": ckpt_train_end,
            "validation_end_row_exclusive": ckpt_val_end,
        },
        "scaler": {
            "inflow": {"mean": inflow_mean, "std": inflow_std},
        },
        "peak_thresholds": {
            "small_floor": SMALL_FLOOR,
            "big_floor": BIG_FLOOR,
            "major_floor": MAJOR_FLOOR,
        },
        "event_window_hours": {"pre": PRE_HOURS, "post": POST_HOURS},
        "overall_metrics": overall_metrics,
        "category_metrics": {
            cat: (category_metrics[cat] if category_metrics[cat] is not None else None)
            for cat in categories
        },
        "events": [],
    }
    for e in event_predictions:
        results["events"].append({
            "index": e["index"],
            "timestamp": e["timestamp"].isoformat(timespec="seconds"),
            "peak_value": e["peak_value"],
            "category": e["category"],
            "split": e["split"],
            "n_samples": int(len(e["valid_starts"])),
            "actual": e["actual"].tolist(),
            "predicted": e["predicted"].tolist(),
        })

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_JSON}")

    # 10. Save plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for e in event_predictions:
        safe_ts = e["timestamp"].strftime("%Y%m%d_%H%M")
        plot_path = OUTPUT_DIR / f"peak_{e['category']}_{safe_ts}_peak{e['peak_value']:.0f}.png"
        save_event_plot(
            plot_path,
            timestamps,
            e,
            e["target_times"],
            e["actual"],
            e["predicted"],
            config,
        )
        print(f"Plot saved: {plot_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()