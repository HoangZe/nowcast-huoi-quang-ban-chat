#!/usr/bin/env python3
"""Train a gap-aware, direct six-hour residual TCN inflow forecaster.

The model predicts the six hourly *changes* in inflow instead of the six
absolute inflow values:

    delta_k = Q(t+k) - Q(t+k-1),  k = 1..6

where Q(t) is the last observed value in the lookback window.  Absolute
forecasts are rebuilt by cumulative summation:

    Q_hat(t+k) = Q(t) + sum_{j=1..k} delta_hat_j

This residual parameterization removes the trivial persistence shortcut that
makes a direct absolute-value model lag observed peaks.

Only contiguous hourly observations are used in an input/target window.  A
non-hourly timestamp transition starts a new segment, so no model sample can
learn across a missing or irregular period.

Training runs for exactly 1000 epochs (no early stopping).  The best
checkpoint is selected on validation residual loss by default; absolute-inflow
validation MAE is also logged every epoch, and --selection-metric absolute can
be used to compare the two criteria.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# Keep Matplotlib's cache outside the repository and in a writable location.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nowcast-huoi-quang-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
EXPECTED_INTERVAL = timedelta(hours=1)
EPSILON = 1e-8


@dataclass(frozen=True)
class TrainingConfig:
    lookback_hours: int = 48
    horizon_hours: int = 6
    hidden_size: int = 128
    num_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 1000
    seed: int = 42
    selection_metric: str = "residual"


class ResidualForecastDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Generate direct multi-horizon step-change samples.

    Each sample returns:
      - scaled model features for the lookback window,
      - the six standardized step-to-step inflow changes,
      - the last observed inflow (the anchor used for reconstruction).
    """

    def __init__(
        self,
        scaled_features: np.ndarray,
        raw_inflows: np.ndarray,
        starts: np.ndarray,
        lookback_hours: int,
        horizon_hours: int,
        delta_std: float,
    ) -> None:
        self.features = torch.as_tensor(scaled_features, dtype=torch.float32)
        self.inflows = torch.as_tensor(raw_inflows, dtype=torch.float32)
        self.starts = torch.as_tensor(starts, dtype=torch.long)
        self.lookback_hours = lookback_hours
        self.horizon_hours = horizon_hours
        self.delta_std = float(delta_std)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor]:
        start = int(self.starts[item])
        history_end = start + self.lookback_hours
        future = self.inflows[history_end : history_end + self.horizon_hours]
        anchor = self.inflows[history_end - 1]
        previous = torch.cat((anchor.view(1), future[:-1]))
        residual = (future - previous) / self.delta_std
        return self.features[start:history_end], residual, anchor


class Chomp1d(nn.Module):
    """Remove the right-side padding that makes a convolution causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, inputs: Tensor) -> Tensor:
        if self.chomp_size == 0:
            return inputs
        return inputs[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else None
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in (self.conv1, self.conv2, self.downsample):
            if module is None:
                continue
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.dropout1(self.relu1(self.chomp1(self.conv1(inputs))))
        output = self.dropout2(self.relu2(self.chomp2(self.conv2(output))))
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu1(output + residual)


class InflowResidualTCN(nn.Module):
    """Causal dilated TCN that predicts standardized step-to-step inflow changes."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        horizon_hours: int,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("TCN kernel_size must be at least 2.")
        if num_layers < 1:
            raise ValueError("TCN num_layers must be at least 1.")

        blocks: list[nn.Module] = []
        for layer_index in range(num_layers):
            blocks.append(
                TemporalBlock(
                    input_channels=input_size if layer_index == 0 else hidden_size,
                    output_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon_hours),
        )
        # Two dilated convolutions per block, with dilations 1, 2, 4, ...
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2**num_layers - 1)

    def forward(self, features: Tensor) -> Tensor:
        # features: (batch, lookback, feature_size) -> (batch, feature_size, lookback)
        sequence = self.tcn(features.transpose(1, 2))
        # With a sufficiently large receptive field, the final causal step
        # summarizes the complete lookback window.
        return self.head(sequence[:, :, -1])


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_arguments() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root / "dataset" / "ban_chat_all_history_synthesized_full_with_satellite_rainfall.csv",
        help="Input CSV containing time_update, inflow_m3s, and precipitation_mm columns.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "src" / "models" / "tcn_inflow_residual_ban_chat.pth",
        help="Checkpoint path to overwrite with the selected model.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=root / "src" / "plots" / "tcn_residual_test_predictions_ban_chat_6h.png",
        help="Versioned test prediction visualization path.",
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--channels", type=int, default=TrainingConfig.hidden_size)
    parser.add_argument("--num-blocks", type=int, default=TrainingConfig.num_layers)
    parser.add_argument("--kernel-size", type=int, default=TrainingConfig.kernel_size)
    parser.add_argument("--dropout", type=float, default=TrainingConfig.dropout)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument(
        "--selection-metric",
        choices=("residual", "absolute"),
        default=TrainingConfig.selection_metric,
        help=(
            "Checkpoint selection criterion: 'residual' uses validation SmoothL1 "
            "in standardized delta space; 'absolute' uses reconstructed inflow MAE."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
    ):
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")
    return torch.device("cpu")


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

        previous_timestamp: datetime | None = None
        for row_number, row in enumerate(reader, start=2):
            timestamp = datetime.strptime(row["time_update"].strip(), TIMESTAMP_FORMAT)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"Timestamps must be strictly increasing; row {row_number} is {timestamp}."
                )
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
            previous_timestamp = timestamp

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


def sample_starts_by_split(
    segments: Iterable[tuple[int, int]],
    row_count: int,
    config: TrainingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Create starts whose complete horizon target belongs to one time split."""

    train_boundary = int(row_count * 0.70)
    validation_boundary = int(row_count * 0.85)
    split_starts: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    minimum_length = config.lookback_hours + config.horizon_hours

    for segment_start, segment_end in segments:
        if segment_end - segment_start < minimum_length:
            continue
        for start in range(segment_start, segment_end - minimum_length + 1):
            target_start = start + config.lookback_hours
            target_end = target_start + config.horizon_hours
            if target_end <= train_boundary:
                split_starts["train"].append(start)
            elif target_start >= train_boundary and target_end <= validation_boundary:
                split_starts["validation"].append(start)
            elif target_start >= validation_boundary and target_end <= row_count:
                split_starts["test"].append(start)

    starts = {
        split: np.asarray(indices, dtype=np.int64) for split, indices in split_starts.items()
    }
    boundaries = {
        "train_end_row_exclusive": train_boundary,
        "validation_end_row_exclusive": validation_boundary,
        "test_end_row_exclusive": row_count,
    }
    return starts, boundaries


def compute_training_delta_std(
    inflows: np.ndarray,
    segments: Iterable[tuple[int, int]],
    train_end: int,
) -> float:
    """Standardize deltas with a scale estimated only inside the training split.

    Differences are computed only within contiguous hourly segments, so a
    non-hourly transition never contributes a spurious jump.
    """

    difference_blocks: list[np.ndarray] = []
    for segment_start, segment_end in segments:
        end = min(segment_end, train_end)
        if end - segment_start >= 2:
            difference_blocks.append(np.diff(inflows[segment_start:end]))

    if not difference_blocks:
        raise ValueError("Training split has too few contiguous hourly transitions.")
    differences = np.concatenate(difference_blocks)
    delta_std = float(np.std(differences))
    if delta_std <= EPSILON:
        raise ValueError("Training inflow deltas have zero variance; cannot normalize.")
    return delta_std


def make_loader(
    dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return reconstructed inflow, residuals, and residual SmoothL1 loss.

    Predicted standardized deltas are converted back to m³/s and cumulatively
    added to the last observed inflow of each lookback window.
    """

    model.eval()
    predicted_inflow: list[np.ndarray] = []
    actual_inflow: list[np.ndarray] = []
    predicted_residual: list[np.ndarray] = []
    actual_residual: list[np.ndarray] = []
    residual_loss_sum = 0.0
    residual_element_count = 0

    with torch.no_grad():
        for features, target_residual, anchor in loader:
            features = features.to(device)
            target_residual = target_residual.to(device)
            anchor = anchor.to(device)

            output_residual = model(features)
            residual_loss_sum += F.smooth_l1_loss(
                output_residual, target_residual, reduction="sum"
            ).item()
            residual_element_count += target_residual.numel()

            predicted_inflow.append(
                (
                    anchor.unsqueeze(1)
                    + torch.cumsum(output_residual * delta_std, dim=1)
                )
                .cpu()
                .numpy()
            )
            actual_inflow.append(
                (
                    anchor.unsqueeze(1)
                    + torch.cumsum(target_residual * delta_std, dim=1)
                )
                .cpu()
                .numpy()
            )
            predicted_residual.append(output_residual.cpu().numpy())
            actual_residual.append(target_residual.cpu().numpy())

    residual_loss = residual_loss_sum / max(residual_element_count, 1)
    return (
        np.concatenate(predicted_inflow, axis=0),
        np.concatenate(actual_inflow, axis=0),
        np.concatenate(predicted_residual, axis=0),
        np.concatenate(actual_residual, axis=0),
        residual_loss,
    )


def metric_summary(
    actual: np.ndarray, predicted: np.ndarray, mase_scale: float
) -> dict[str, dict[str, float]]:
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


def residual_metric_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    delta_std: float,
) -> dict[str, dict[str, float]]:
    """Summarize step-change errors in m³/s (inputs are standardized deltas)."""

    actual_delta = actual * delta_std
    predicted_delta = predicted * delta_std

    def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        absolute_error = np.abs(y_pred - y_true)
        squared_error = np.square(y_pred - y_true)
        return {
            "mae": float(np.mean(absolute_error)),
            "rmse": float(np.sqrt(np.mean(squared_error))),
        }

    summary = {"overall": metrics_for(actual_delta, predicted_delta)}
    for horizon_index in range(actual.shape[1]):
        summary[f"t+{horizon_index + 1}"] = metrics_for(
            actual_delta[:, horizon_index], predicted_delta[:, horizon_index]
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


def print_residual_metrics(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"{'Horizon':<10} {'MAE (m³/s)':>14} {'RMSE (m³/s)':>15}")
    for horizon, values in summary.items():
        print(
            f"{horizon:<10} {values['mae']:>14.4f} {values['rmse']:>15.4f}"
        )


def snapshot_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Keep a CPU copy so checkpointing is device-independent."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def persistence_predictions(
    inflows: np.ndarray,
    starts: np.ndarray,
    lookback_hours: int,
    horizon_hours: int,
) -> np.ndarray:
    anchors = inflows[starts + lookback_hours - 1]
    return np.repeat(anchors[:, None], horizon_hours, axis=1)


def save_prediction_plot(
    plot_path: Path,
    timestamps: list[datetime],
    inflows: np.ndarray,
    test_starts: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    config: TrainingConfig,
) -> None:
    """Save a six-panel plot for the first contiguous test-anchor interval."""

    if len(test_starts) == 0:
        raise ValueError("Cannot plot predictions without test samples.")

    plot_count = 1
    while plot_count < min(168, len(test_starts)) and test_starts[plot_count] == test_starts[plot_count - 1] + 1:
        plot_count += 1

    persistence = inflows[test_starts[:plot_count] + config.lookback_hours - 1]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    for horizon_index, axis in enumerate(axes.flat):
        target_indices = test_starts[:plot_count] + config.lookback_hours + horizon_index
        target_times = [timestamps[index] for index in target_indices]
        axis.plot(target_times, actual[:plot_count, horizon_index], label="Actual", linewidth=1.5)
        axis.plot(
            target_times,
            predicted[:plot_count, horizon_index],
            label="Predicted",
            linewidth=1.25,
        )
        axis.plot(
            target_times,
            persistence,
            label="Persistence (last observed)" if horizon_index == 0 else None,
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )
        axis.set_title(f"BẢN CHÁT t+{horizon_index + 1}")
        axis.set_ylabel("Inflow (m³/s)")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
        if horizon_index == 0:
            axis.legend()
    figure.suptitle(
        "BẢN CHÁT - TCN residual test forecasts for the first contiguous test interval",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig(
        max_epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        hidden_size=arguments.channels,
        num_layers=arguments.num_blocks,
        kernel_size=arguments.kernel_size,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        selection_metric=arguments.selection_metric,
    )
    if config.max_epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch-size must both be positive.")
    if config.hidden_size < 1 or config.num_layers < 1:
        raise ValueError("channels and num-blocks must both be positive.")
    if config.kernel_size < 2:
        raise ValueError("kernel-size must be at least 2.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if config.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive.")
    if config.selection_metric not in {"residual", "absolute"}:
        raise ValueError("selection-metric must be 'residual' or 'absolute'.")

    set_seed(config.seed)
    timestamps, inflows, precipitations = load_inflow_series(arguments.data_path)
    segments = find_hourly_segments(timestamps)
    split_starts, boundaries = sample_starts_by_split(segments, len(inflows), config)
    empty_splits = [name for name, starts in split_starts.items() if len(starts) == 0]
    if empty_splits:
        raise ValueError(
            "No valid contiguous samples for split(s): "
            f"{', '.join(empty_splits)}. Check the data cadence or shorten the lookback."
        )

    train_end = boundaries["train_end_row_exclusive"]
    training_inflows = inflows[:train_end]
    training_precipitations = precipitations[:train_end]
    inflow_mean = float(np.mean(training_inflows))
    inflow_std = float(np.std(training_inflows))
    precipitation_mean = float(np.mean(training_precipitations))
    precipitation_std = float(np.std(training_precipitations))
    if inflow_std <= EPSILON:
        raise ValueError("Training inflow values have zero variance; cannot normalize.")
    if precipitation_std <= EPSILON:
        raise ValueError("Training precipitation values have zero variance; cannot normalize.")
    scaled_inflows = ((inflows - inflow_mean) / inflow_std).astype(np.float32)
    scaled_precipitations = ((precipitations - precipitation_mean) / precipitation_std).astype(np.float32)
    scaled_features = np.column_stack([scaled_inflows, scaled_precipitations])

    delta_std = compute_training_delta_std(inflows, segments, train_end)

    seasonal_period = 24
    if len(training_inflows) <= seasonal_period:
        raise ValueError("Training period is too short to calculate the 24-hour MASE scale.")
    mase_scale = float(
        np.mean(np.abs(training_inflows[seasonal_period:] - training_inflows[:-seasonal_period]))
    )
    if mase_scale <= EPSILON:
        raise ValueError("The 24-hour seasonal-naive MASE scale is zero.")

    datasets = {
        name: ResidualForecastDataset(
            scaled_features,
            inflows,
            starts,
            config.lookback_hours,
            config.horizon_hours,
            delta_std,
        )
        for name, starts in split_starts.items()
    }
    loaders = {
        "train": make_loader(datasets["train"], config.batch_size, shuffle=True),
        "validation": make_loader(datasets["validation"], config.batch_size, shuffle=False),
        "test": make_loader(datasets["test"], config.batch_size, shuffle=False),
    }

    device = resolve_device(arguments.device)
    model = InflowResidualTCN(
        input_size=scaled_features.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        horizon_hours=config.horizon_hours,
    ).to(device)
    if model.receptive_field < config.lookback_hours:
        raise ValueError(
            f"TCN receptive field ({model.receptive_field} hours) is shorter than the "
            f"{config.lookback_hours}-hour lookback; increase --num-blocks."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.SmoothL1Loss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(f"Dataset rows: {len(inflows)}")
    print(f"Hourly segments: {len(segments)} (non-hourly boundaries: {len(segments) - 1})")
    print(
        "Samples: "
        + ", ".join(f"{name}={len(starts)}" for name, starts in split_starts.items())
    )
    print(
        f"Device: {device}; inflow scaler mean={inflow_mean:.4f}, std={inflow_std:.4f}; "
        f"precipitation scaler mean={precipitation_mean:.4f}, std={precipitation_std:.4f}"
    )
    print(
        f"Model: InflowResidualTCN; parameters={parameter_count:,}; "
        f"receptive_field={model.receptive_field}h; delta_std={delta_std:.4f} m³/s"
    )
    print(
        f"Target: step-to-step residuals; reconstruction=Q(t)+cumsum(delta); "
        f"selection_metric={config.selection_metric}; epochs={config.max_epochs}"
    )

    best_residual_loss = float("inf")
    best_residual_epoch = 0
    best_residual_state: dict[str, Tensor] | None = None
    best_absolute_mae = float("inf")
    best_absolute_epoch = 0
    best_absolute_state: dict[str, Tensor] | None = None
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "validation_residual_loss": [],
        "validation_residual_mae_m3s": [],
        "validation_absolute_mae": [],
    }

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        training_loss = 0.0
        training_items = 0
        for features, target_residual, _anchor in loaders["train"]:
            features = features.to(device)
            target_residual = target_residual.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss = loss_function(output, target_residual)
            loss.backward()
            optimizer.step()
            batch_size = features.size(0)
            training_loss += loss.item() * batch_size
            training_items += batch_size

        (
            validation_prediction,
            validation_actual,
            validation_residual_prediction,
            validation_residual_actual,
            validation_residual_loss,
        ) = predict(model, loaders["validation"], device, delta_std)
        validation_absolute_mae = float(
            np.mean(np.abs(validation_prediction - validation_actual))
        )
        validation_residual_mae_m3s = float(
            np.mean(
                np.abs(
                    (validation_residual_prediction - validation_residual_actual)
                    * delta_std
                )
            )
        )
        epoch_train_loss = training_loss / training_items

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["validation_residual_loss"].append(validation_residual_loss)
        history["validation_residual_mae_m3s"].append(validation_residual_mae_m3s)
        history["validation_absolute_mae"].append(validation_absolute_mae)

        print(
            f"Epoch {epoch:04d}/{config.max_epochs}: "
            f"train_huber={epoch_train_loss:.6f}, "
            f"validation_residual_loss={validation_residual_loss:.6f}, "
            f"validation_residual_mae={validation_residual_mae_m3s:.4f} m³/s, "
            f"validation_inflow_mae={validation_absolute_mae:.4f} m³/s"
        )

        if validation_residual_loss < best_residual_loss:
            best_residual_loss = validation_residual_loss
            best_residual_epoch = epoch
            best_residual_state = snapshot_state_dict(model)
        if validation_absolute_mae < best_absolute_mae:
            best_absolute_mae = validation_absolute_mae
            best_absolute_epoch = epoch
            best_absolute_state = snapshot_state_dict(model)

    if config.selection_metric == "residual":
        best_epoch = best_residual_epoch
        best_state = best_residual_state
        selected_validation_metric = best_residual_loss
    else:
        best_epoch = best_absolute_epoch
        best_state = best_absolute_state
        selected_validation_metric = best_absolute_mae

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint.")
    model.load_state_dict(best_state)

    (
        validation_prediction,
        validation_actual,
        validation_residual_prediction,
        validation_residual_actual,
        validation_residual_loss,
    ) = predict(model, loaders["validation"], device, delta_std)
    (
        test_prediction,
        test_actual,
        test_residual_prediction,
        test_residual_actual,
        test_residual_loss,
    ) = predict(model, loaders["test"], device, delta_std)

    validation_metrics = metric_summary(validation_actual, validation_prediction, mase_scale)
    test_metrics = metric_summary(test_actual, test_prediction, mase_scale)
    validation_residual_metrics = residual_metric_summary(
        validation_residual_actual, validation_residual_prediction, delta_std
    )
    test_residual_metrics = residual_metric_summary(
        test_residual_actual, test_residual_prediction, delta_std
    )
    validation_persistence = persistence_predictions(
        inflows,
        split_starts["validation"],
        config.lookback_hours,
        config.horizon_hours,
    )
    test_persistence = persistence_predictions(
        inflows,
        split_starts["test"],
        config.lookback_hours,
        config.horizon_hours,
    )
    validation_persistence_metrics = metric_summary(
        validation_actual, validation_persistence, mase_scale
    )
    test_persistence_metrics = metric_summary(
        test_actual, test_persistence, mase_scale
    )

    arguments.model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": snapshot_state_dict(model),
        "candidate_models": {
            "residual": {
                "epoch": best_residual_epoch,
                "validation_residual_loss": best_residual_loss,
                "model_state_dict": best_residual_state,
            },
            "absolute": {
                "epoch": best_absolute_epoch,
                "validation_absolute_mae": best_absolute_mae,
                "model_state_dict": best_absolute_state,
            },
        },
        "config": asdict(config),
        "architecture": {
            "name": "InflowResidualTCN",
            "input_size": int(scaled_features.shape[1]),
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "kernel_size": config.kernel_size,
            "dropout": config.dropout,
            "horizon_hours": config.horizon_hours,
            "receptive_field": model.receptive_field,
        },
        "target": {
            "type": "step_to_step_residual",
            "definition": "delta_k = Q(t+k) - Q(t+k-1), k=1..horizon; Q(t) is the last lookback value",
            "reconstruction": "Q_hat(t+k) = Q(t) + cumsum(delta_hat_1..delta_hat_k)",
            "residual_scaler": {"std": delta_std, "mean_centered": False},
        },
        "scaler": {
            "inflow": {"mean": inflow_mean, "std": inflow_std},
            "precipitation": {"mean": precipitation_mean, "std": precipitation_std},
            "residual": {"std": delta_std, "mean_centered": False},
        },
        "split_boundaries": {
            **boundaries,
            "train_end_timestamp": timestamps[train_end - 1].isoformat(timespec="milliseconds"),
            "validation_end_timestamp": timestamps[
                boundaries["validation_end_row_exclusive"] - 1
            ].isoformat(timespec="milliseconds"),
        },
        "dataset": {
            "path": str(arguments.data_path),
            "row_count": len(inflows),
            "segment_count": len(segments),
            "non_hourly_boundary_count": len(segments) - 1,
            "sample_counts": {name: int(len(starts)) for name, starts in split_starts.items()},
        },
        "training": {
            "device": str(device),
            "max_epochs": config.max_epochs,
            "epochs_completed": config.max_epochs,
            "selection_metric": config.selection_metric,
            "selected_best_epoch": best_epoch,
            "selected_best_validation_metric": selected_validation_metric,
            "best_residual_epoch": best_residual_epoch,
            "best_validation_residual_loss": best_residual_loss,
            "best_absolute_epoch": best_absolute_epoch,
            "best_validation_absolute_mae": best_absolute_mae,
            "loss": "SmoothL1Loss",
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
        },
        "history": history,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "validation_residual": validation_residual_metrics,
            "test_residual": test_residual_metrics,
            "validation_residual_loss": validation_residual_loss,
            "test_residual_loss": test_residual_loss,
        },
        "baselines": {
            "persistence": {
                "validation": validation_persistence_metrics,
                "test": test_persistence_metrics,
            }
        },
    }
    torch.save(checkpoint, arguments.model_path)
    save_prediction_plot(
        arguments.plot_path,
        timestamps,
        inflows,
        split_starts["test"],
        test_actual,
        test_prediction,
        config,
    )

    print(
        f"\nSelected checkpoint: epoch {best_epoch} by {config.selection_metric} "
        f"(metric={selected_validation_metric:.6f})"
    )
    print(
        f"Best residual checkpoint: epoch {best_residual_epoch}, "
        f"validation residual loss={best_residual_loss:.6f}"
    )
    print(
        f"Best absolute checkpoint: epoch {best_absolute_epoch}, "
        f"validation inflow MAE={best_absolute_mae:.4f} m³/s"
    )
    print(f"Model saved to: {arguments.model_path}")
    print(f"Plot saved to: {arguments.plot_path}")
    print_metrics("Validation metrics (absolute inflow)", validation_metrics)
    print_metrics("Test metrics (absolute inflow)", test_metrics)
    print_residual_metrics("Validation metrics (step-change residuals)", validation_residual_metrics)
    print_residual_metrics("Test metrics (step-change residuals)", test_residual_metrics)
    print(
        f"\nPersistence test baseline: MAE={test_persistence_metrics['overall']['mae']:.4f}, "
        f"RMSE={test_persistence_metrics['overall']['rmse']:.4f}, "
        f"MASE={test_persistence_metrics['overall']['mase']:.4f}, "
        f"NSE={test_persistence_metrics['overall']['nse']:.4f}"
    )


if __name__ == "__main__":
    main()
