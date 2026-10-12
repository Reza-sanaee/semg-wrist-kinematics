"""Additional diagnostics of saved samples and the supplied model package.

These diagnostics describe the submitted assets. They do not retrain the model,
recover missing pre-calibration predictions, or establish deployment performance.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

from .evaluation import AXIS_NAMES, calculate_metrics, save_metrics, validate_samples


def detailed_metrics(y_true, y_pred, activity_threshold=0.05):
    """Add error, zero-baseline, saturation, and activity diagnostics per axis."""
    y_true, y_pred = validate_samples(y_true, y_pred)
    if activity_threshold < 0:
        raise ValueError("The activity threshold must be nonnegative.")
    rows = calculate_metrics(y_true, y_pred)
    for axis, row in enumerate(rows):
        actual, predicted = y_true[:, axis], y_pred[:, axis]
        error = predicted - actual
        active = np.abs(actual) > activity_threshold
        denominator = np.sum((actual - np.mean(actual)) ** 2)
        row.update({
            "n_samples": len(actual),
            "mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error)),
            "p95_absolute_error": float(np.percentile(np.abs(error), 95)),
            "max_absolute_error": float(np.max(np.abs(error))),
            "r_squared": float(1 - np.sum(error ** 2) / denominator)
            if denominator > 0 else None,
            "zero_baseline_rmse": float(np.sqrt(np.mean(actual ** 2))),
            "active_samples": int(active.sum()),
            "active_rmse": float(np.sqrt(np.mean(error[active] ** 2)))
            if active.any() else None,
            "near_rest_samples": int((~active).sum()),
            "near_rest_rmse": float(np.sqrt(np.mean(error[~active] ** 2)))
            if (~active).any() else None,
            "saturated_predictions": int(
                np.isclose(np.abs(predicted), 1, atol=1e-7, rtol=0).sum()
            ),
            "activity_threshold": activity_threshold,
        })
    return rows


def inspect_model_package(path):
    """Read Keras JSON metadata and shapes without loading TensorFlow."""
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        config = json.loads(archive.read("config.json"))
    layers = []
    for entry in config["config"]["layers"]:
        properties = entry["config"]
        layers.append({
            "class": entry["class_name"],
            "name": properties.get("name"),
            "input_shape": entry.get("build_config", {}).get("input_shape"),
            "filters": properties.get("filters"),
            "units": properties.get("units"),
            "kernel_size": properties.get("kernel_size"),
            "dilation_rate": properties.get("dilation_rate"),
            "rate": properties.get("rate"),
        })
    return {"metadata": metadata, "layers": layers, "compile_config": config.get("compile_config")}


def inspect_recording(path, window_size=600, step=100):
    """Describe the raw recording and count windows crossing repetition labels.

    The counts follow the source movement selection and final-sample assignment.
    No source data are modified and no preprocessing is fitted here.
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("Window size and step must be positive.")
    recording = loadmat(path)
    for key in ("emg", "restimulus", "rerepetition"):
        if key not in recording:
            raise ValueError(f"Missing recording variable: {key}")
    emg = np.asarray(recording["emg"])
    labels = np.asarray(recording["restimulus"]).reshape(-1)
    reps = np.asarray(recording["rerepetition"]).reshape(-1)
    if not (len(emg) == len(labels) == len(reps)):
        raise ValueError("Recording arrays have inconsistent lengths.")
    indices = np.flatnonzero(np.isin(labels, (0, 9, 10, 13, 14, 15, 16)))
    selected_reps = reps[indices]
    windows = mixed_repetition_windows = gap_windows = 0
    for start in range(0, len(indices) - window_size, step):
        selected = indices[start : start + window_size]
        windows += 1
        if np.unique(selected_reps[start : start + window_size]).size > 1:
            mixed_repetition_windows += 1
        if np.any(np.diff(selected) != 1):
            gap_windows += 1
    codes, counts = np.unique(labels, return_counts=True)
    return {
        "input_file": str(Path(path).resolve()),
        "emg_shape": list(emg.shape),
        "finite_emg": bool(np.isfinite(emg).all()),
        "movement_counts": {str(int(k)): int(v) for k, v in zip(codes, counts)},
        "selected_samples": len(indices),
        "selected_sequence_gaps": int(np.sum(np.diff(indices) != 1)),
        "total_selected_windows": windows,
        "mixed_repetition_windows": mixed_repetition_windows,
        "windows_crossing_discarded_time_gaps": gap_windows,
        "window_size": window_size,
        "step": step,
    }


def analyze_samples(samples_dir, output_dir, model_path=None, activity_threshold=0.05):
    """Export reproducible numeric diagnostics and three English figures."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    samples_dir, output_dir = Path(samples_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true = np.load(samples_dir / "y_true_sample.npy")
    pred = np.load(samples_dir / "y_pred_sample.npy")
    true, pred = validate_samples(true, pred)
    rows = detailed_metrics(true, pred, activity_threshold)
    summary = {
        "sample_count": len(true),
        "units": "normalized target amplitude",
        "prediction_stage": "calibrated and clipped exported predictions",
        "activity_definition": f"abs(target) > {activity_threshold}, per axis",
        "axis_metrics": rows,
        "joint_mean_absolute_error": float(np.mean(np.abs(pred - true))),
    }
    if model_path is not None:
        summary["model_package"] = inspect_model_package(model_path)
    (output_dir / "sample_diagnostics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_metrics(calculate_metrics(true, pred), output_dir / "calibrated_sample_metrics.csv")

    figure = Figure(figsize=(13, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    for axis, name in enumerate(AXIS_NAMES):
        error = pred[:, axis] - true[:, axis]
        axes[axis, 0].plot(error, color="#a43b42", linewidth=0.9)
        axes[axis, 0].axhline(0, color="#293e48", linewidth=0.8)
        axes[axis, 0].set_title(f"{name} residuals")
        axes[axis, 0].set_ylabel("Prediction minus target")
        axes[axis, 0].grid(alpha=0.2)
        axes[axis, 1].hist(error, bins=50, color="#497481", edgecolor="white")
        axes[axis, 1].set_title(f"{name} error distribution")
        axes[axis, 1].set_ylabel("Observation count")
    axes[-1, 0].set_xlabel("Saved window index")
    axes[-1, 1].set_xlabel("Normalized error")
    figure.tight_layout()
    figure.savefig(output_dir / "residual_diagnostics.png", dpi=200)
    figure.savefig(output_dir / "residual_diagnostics.svg")

    figure = Figure(figsize=(12, 4.2))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis, name in enumerate(AXIS_NAMES):
        axes[axis].scatter(true[:, axis], pred[:, axis], s=5, alpha=0.22, color="#497481")
        axes[axis].plot([-1, 1], [-1, 1], "--", color="#a43b42")
        axes[axis].set_title(name)
        axes[axis].set_xlabel("Smoothed target")
        axes[axis].set_ylabel("Calibrated prediction")
        axes[axis].set_xlim(-1.1, 1.1)
        axes[axis].set_ylim(-1.1, 1.1)
        axes[axis].set_aspect("equal", adjustable="box")
        axes[axis].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_scatter.png", dpi=200)
    figure.savefig(output_dir / "prediction_scatter.svg")

    figure = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    positions = np.arange(3)
    axes[0].bar(positions - 0.18, [r["rmse"] for r in rows], 0.36,
                label="Calibrated prediction", color="#497481")
    axes[0].bar(positions + 0.18, [r["zero_baseline_rmse"] for r in rows], 0.36,
                label="Zero baseline", color="#adbcc2")
    axes[1].bar(positions - 0.18, [r["active_rmse"] for r in rows], 0.36,
                label="Active target", color="#a43b42")
    axes[1].bar(positions + 0.18, [r["near_rest_rmse"] for r in rows], 0.36,
                label="Near-rest target", color="#c5d5dc")
    for ax in axes:
        ax.set_xticks(positions, AXIS_NAMES)
        ax.set_ylabel("RMSE in normalized amplitude")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_title("Comparison with the constant zero baseline")
    axes[1].set_title(f"Activity-conditioned errors  threshold {activity_threshold}")
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_activity_comparison.png", dpi=200)
    figure.savefig(output_dir / "baseline_activity_comparison.svg")
    return summary
