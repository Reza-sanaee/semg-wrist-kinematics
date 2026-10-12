"""Prediction postprocessing, quantitative evaluation, and plotting."""

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

AXIS_NAMES = ("Roll", "Pitch", "Yaw")


def validate_samples(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 3:
        raise ValueError("Samples must have matching shapes (N, 3).")
    if len(y_true) < 2 or not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Samples must contain at least two finite observations.")
    return y_true, y_pred


def postprocess_predictions(raw_predictions, sigma=10, gain=1.35):
    """Return smoothed predictions and gain-calibrated, clipped predictions."""
    raw_predictions = np.asarray(raw_predictions)
    smoothed = np.zeros_like(raw_predictions)
    for axis in range(3):
        smoothed[:, axis] = gaussian_filter1d(raw_predictions[:, axis], sigma=sigma)
    return smoothed, np.clip(smoothed * gain, -1.0, 1.0)


def calculate_metrics(y_true, y_pred):
    """Compute RMSE and Pearson correlation independently for each axis."""
    y_true, y_pred = validate_samples(y_true, y_pred)
    rows = []
    for axis, name in enumerate(AXIS_NAMES):
        rmse = float(np.sqrt(np.mean((y_true[:, axis] - y_pred[:, axis]) ** 2)))
        if np.std(y_true[:, axis]) > 0 and np.std(y_pred[:, axis]) > 0:
            correlation = float(pearsonr(y_true[:, axis], y_pred[:, axis]).statistic)
        else:
            correlation = 0.0
        rows.append({"axis": name, "rmse": rmse, "pearson_cc": correlation})
    return rows


def save_metrics(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["axis", "rmse", "pearson_cc"])
        writer.writeheader()
        writer.writerows(rows)


def plot_predictions(y_true, y_pred, path=None, num_samples=None, show=False):
    """Plot complete samples or a requested prefix using English labels."""
    if show:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        figure = Figure(figsize=(15, 10))
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 1, sharex=True)

    y_true, y_pred = validate_samples(y_true, y_pred)
    if num_samples is not None:
        y_true, y_pred = y_true[:num_samples], y_pred[:num_samples]
    for axis, name in enumerate(AXIS_NAMES):
        axes[axis].plot(
            y_true[:, axis], label="Actual value (smoothed target)",
            color="black", linewidth=2.5, alpha=0.7,
        )
        axes[axis].plot(
            y_pred[:, axis], label="Predicted value", color="red",
            linestyle="--", linewidth=1.5,
        )
        axes[axis].set_title(f"{name} axis")
        axes[axis].set_ylabel("Normalized amplitude (−1 to +1)")
        axes[axis].legend(loc="upper right")
        axes[axis].grid(True, linestyle=":", alpha=0.6)
        axes[axis].set_ylim(-1.2, 1.2)
    axes[-1].set_xlabel("Time windows")
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
    if show:
        plt.show()
    return figure
