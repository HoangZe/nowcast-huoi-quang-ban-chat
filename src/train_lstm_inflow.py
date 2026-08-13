#!/usr/bin/env python3
"""Train a gap-aware six-hour inflow forecaster using an LSTM encoder-decoder.

The model is a Seq2Seq LSTM with Bahdanau (additive) attention: an encoder LSTM
reads the lookback window and retains every hidden state; a decoder LSTM emits
one forecast per horizon (t+1 ... t+6), attending back over the encoder states
to pull in the relevant past rainfall/inflow moment for each step. The decoder
attends only over the *past* window, so no future information leaks into the
forecasts. During training it is teacher-forced; during inference it free-runs.

The default loss is a differentiable peak-weighted NSE (nse_peak) that
emphasizes high-flow samples to counter peak attenuation. It can be switched to
the original Huber loss with --loss huber for A/B comparison.

Only contiguous hourly observations are used in an input/target window.  A
non-hourly timestamp transition starts a new segment, so no model sample can
learn across a missing or irregular period.
"""

from __future__ import annotations

import argparse
import copy
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
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    seed: int = 42


class SeriesForecastDataset(Dataset[tuple[Tensor, Tensor]]):
    """Generate direct multi-horizon samples from pre-validated start indices."""

    def __init__(
        self,
        scaled_features: np.ndarray,
        scaled_targets: np.ndarray,
        starts: np.ndarray,
        lookback_hours: int,
        horizon_hours: int,
    ) -> None:
        self.features = torch.as_tensor(scaled_features, dtype=torch.float32)
        self.targets = torch.as_tensor(scaled_targets, dtype=torch.float32)
        self.starts = torch.as_tensor(starts, dtype=torch.long)
        self.lookback_hours = lookback_hours
        self.horizon_hours = horizon_hours

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        start = int(self.starts[item])
        history_end = start + self.lookback_hours
        features = self.features[start:history_end]
        target = self.targets[history_end : history_end + self.horizon_hours]
        return features, target


class InflowLSTM(nn.Module):
    """A direct multi-output LSTM for multivariate inflow histories."""

    def __init__(
        self, hidden_size: int, num_layers: int, dropout: float, horizon_hours: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, horizon_hours)

    def forward(self, features: Tensor) -> Tensor:
        sequence_output, _ = self.lstm(features)
        return self.output(sequence_output[:, -1, :])


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention over encoder hidden states.

    At each decoder step, a context vector is computed as a weighted sum of the
    encoder hidden states, where the weights are learned scores that indicate
    which past moments (e.g. a rainfall burst) are most relevant.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # Project the encoder and decoder hidden states into a common space.
        self.encoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> tuple[Tensor, Tensor]:
        # encoder_outputs: (batch, lookback, hidden)
        aligned_encoder = torch.tanh(self.encoder_projection(encoder_outputs))
        # decoder_hidden: (batch, hidden) -> unsqueeze for broadcasting
        aligned_decoder = torch.tanh(
            self.decoder_projection(decoder_hidden).unsqueeze(1)
        )
        # Raw alignment scores, then softmax over the lookback axis.
        scores = self.score(aligned_encoder + aligned_decoder).squeeze(-1)  # (batch, lookback)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden)
        return context, weights


class Seq2SeqAttentionLSTM(nn.Module):
    """An LSTM encoder-decoder with Bahdanau attention for multi-horizon inflow.

    The encoder LSTM reads the lookback window and retains every hidden state.
    The decoder LSTM then produces one forecast per horizon (t+1 ... t+6), and at
    each step uses attention to pull in the relevant past rainfall/inflow moment
    from the encoder states. The decoder attends only over the *past* window:
    no future information is leaked into the forecasts.

    During training the decoder is teacher-forced with the true target values;
    during inference it free-runs on its own predictions.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon_hours: int,
        feature_size: int = 2,
    ) -> None:
        super().__init__()
        self.horizon_hours = horizon_hours
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = BahdanauAttention(hidden_size)
        # Decoder input: the previous inflow value / SOS token concatenated with
        # the attention context vector.
        self.decoder = nn.LSTM(
            input_size=hidden_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)
        # Learned start-of-sequence token (scalar, same scale as the targets).
        self.start_token = nn.Parameter(torch.zeros(1))

    def forward(self, features: Tensor, target: Tensor | None = None) -> Tensor:
        batch_size = features.size(0)
        device = features.device
        # features: (batch, lookback, feature_size)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(features)

        # Initialize the decoder state from the encoder's final state.
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        last_hidden = decoder_hidden[-1]  # (batch, hidden), top layer for attention

        context, _ = self.attention(last_hidden, encoder_outputs)

        # First decoder input: learned SOS token + initial context.
        start_value = self.start_token.view(1, 1).expand(batch_size, 1)
        decoder_input = torch.cat([start_value, context], dim=-1)  # (batch, hidden+1)

        outputs: list[Tensor] = []
        for step in range(self.horizon_hours):
            output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input.unsqueeze(1),
                (decoder_hidden, decoder_cell),
            )
            output = output[:, 0, :]  # (batch, hidden)
            prediction = self.output(self.dropout(output))[:, 0]  # (batch,)
            outputs.append(prediction)

            # Build the next decoder input.
            last_hidden = decoder_hidden[-1]
            context, _ = self.attention(last_hidden, encoder_outputs)
            next_value = target[:, step] if target is not None else prediction
            decoder_input = torch.cat(
                [next_value.unsqueeze(1), context], dim=-1
            )

        return torch.stack(outputs, dim=1)


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
        default=root / "src" / "models" / "lstm_inflow_ban_chat.pth",
        help="Checkpoint path to overwrite with the selected model.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=root / "src" / "plots" / "lstm_test_predictions_ban_chat_6h.png",
        help="Versioned test prediction visualization path.",
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument(
        "--loss",
        type=str,
        choices=("huber", "nse_peak"),
        default="nse_peak",
        help="Loss function: 'huber' (SmoothL1) or 'nse_peak' (peak-weighted NSE).",
    )
    parser.add_argument(
        "--peak-weight",
        type=float,
        default=3.0,
        help="Peak emphasis for the nse_peak loss (weights in [1, 1+peak_weight]).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """Create starts whose complete six-row target belongs to one time split."""

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


def make_loader(dataset: Dataset[tuple[Tensor, Tensor]], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def nse_peak_loss(
    prediction: Tensor,
    target: Tensor,
    peak_weight: float = 3.0,
) -> Tensor:
    """Differentiable peak-weighted NSE loss.

    Minimizing this maximizes a weighted NSE. Samples are weighted continuously
    by their target magnitude, so high-flow (peak) samples are emphasized while
    every sample still contributes. The weights lie in [1, 1 + peak_weight].

    loss = 1 - NSE_weighted
         = sum(w * (target - pred)^2) / sum(w * (target - target_mean)^2)
    """
    differences = target - prediction
    squared_error = differences * differences
    weights = 1.0 + peak_weight * (
        target.abs() / (target.abs().max() + EPSILON)
    )
    weighted_numerator = (weights * squared_error).sum()
    weighted_denominator = (weights * (target - target.mean()) ** 2).sum()
    nse = 1.0 - weighted_numerator / (weighted_denominator + EPSILON)
    return 1.0 - nse


def inverse_scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, mean: float, std: float
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, target in loader:
            output = model(features.to(device)).cpu().numpy()
            predictions.append(output)
            targets.append(target.numpy())
    predicted = inverse_scale(np.concatenate(predictions, axis=0), mean, std)
    actual = inverse_scale(np.concatenate(targets, axis=0), mean, std)
    return predicted, actual


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


def save_prediction_plot(
    plot_path: Path,
    timestamps: list[datetime],
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
        axis.set_title(f"BẢN CHÁT t+{horizon_index + 1}")
        axis.set_ylabel("Inflow (m³/s)")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
        if horizon_index == 0:
            axis.legend()
    figure.suptitle("BẢN CHÁT - LSTM test forecasts for the first contiguous test interval", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig(
        max_epochs=arguments.epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )
    if config.max_epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must all be positive.")

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

    seasonal_period = 24
    if len(training_inflows) <= seasonal_period:
        raise ValueError("Training period is too short to calculate the 24-hour MASE scale.")
    mase_scale = float(
        np.mean(np.abs(training_inflows[seasonal_period:] - training_inflows[:-seasonal_period]))
    )
    if mase_scale <= EPSILON:
        raise ValueError("The 24-hour seasonal-naive MASE scale is zero.")

    datasets = {
        name: SeriesForecastDataset(scaled_features, scaled_inflows, starts, config.lookback_hours, config.horizon_hours)
        for name, starts in split_starts.items()
    }
    loaders = {
        "train": make_loader(datasets["train"], config.batch_size, shuffle=True),
        "validation": make_loader(datasets["validation"], config.batch_size, shuffle=False),
        "test": make_loader(datasets["test"], config.batch_size, shuffle=False),
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Seq2SeqAttentionLSTM(
        config.hidden_size, config.num_layers, config.dropout, config.horizon_hours
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if arguments.loss == "huber":
        loss_function = nn.SmoothL1Loss()
        loss_label = "huber"
    else:
        loss_function = lambda pred, target: nse_peak_loss(
            pred, target, peak_weight=arguments.peak_weight
        )
        loss_label = f"nse_peak(w={arguments.peak_weight:g})"

    print(f"Dataset rows: {len(inflows)}")
    print(f"Hourly segments: {len(segments)} (non-hourly boundaries: {len(segments) - 1})")
    print(
        "Samples: "
        + ", ".join(f"{name}={len(starts)}" for name, starts in split_starts.items())
    )
    print(f"Model: Seq2SeqAttentionLSTM; loss: {loss_label}")
    print(f"Device: {device}; inflow scaler mean={inflow_mean:.4f}, std={inflow_std:.4f}; precipitation scaler mean={precipitation_mean:.4f}, std={precipitation_std:.4f}")

    best_validation_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        training_loss = 0.0
        training_items = 0
        for features, target in loaders["train"]:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # Teacher forcing: pass the true target so the decoder conditions on
            # the observed inflow at each step during training.
            output = model(features, target)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            batch_size = features.size(0)
            training_loss += loss.item() * batch_size
            training_items += batch_size

        validation_prediction, validation_actual = predict(
            model, loaders["validation"], device, inflow_mean, inflow_std
        )
        validation_mae = float(np.mean(np.abs(validation_prediction - validation_actual)))
        print(
            f"Epoch {epoch:03d}/{config.max_epochs}: "
            f"train_{loss_label}={training_loss / training_items:.6f}, validation_mae={validation_mae:.4f}"
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint.")
    model.load_state_dict(best_state)

    validation_prediction, validation_actual = predict(
        model, loaders["validation"], device, inflow_mean, inflow_std
    )
    test_prediction, test_actual = predict(model, loaders["test"], device, inflow_mean, inflow_std)
    validation_metrics = metric_summary(validation_actual, validation_prediction, mase_scale)
    test_metrics = metric_summary(test_actual, test_prediction, mase_scale)

    arguments.model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "scaler": {
            "inflow": {"mean": inflow_mean, "std": inflow_std},
            "precipitation": {"mean": precipitation_mean, "std": precipitation_std},
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
            "best_epoch": best_epoch,
            "best_validation_mae": best_validation_mae,
            "selection_metric": "validation_mae",
            "model": "Seq2SeqAttentionLSTM",
            "loss": loss_label,
            "loss_type": arguments.loss,
            "peak_weight": arguments.peak_weight,
        },
        "metrics": {"validation": validation_metrics, "test": test_metrics},
    }
    torch.save(checkpoint, arguments.model_path)
    save_prediction_plot(
        arguments.plot_path,
        timestamps,
        split_starts["test"],
        test_actual,
        test_prediction,
        config,
    )

    print(f"\nBest checkpoint: epoch {best_epoch}, validation MAE {best_validation_mae:.4f}")
    print(f"Model saved to: {arguments.model_path}")
    print(f"Plot saved to: {arguments.plot_path}")
    print_metrics("Validation metrics", validation_metrics)
    print_metrics("Test metrics", test_metrics)


if __name__ == "__main__":
    main()
