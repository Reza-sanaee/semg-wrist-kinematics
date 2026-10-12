"""Command-line entry points for evaluation, training, and 3D replay."""

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Continuous wrist kinematics estimation using sEMG"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser(
        "evaluate-samples", help="Evaluate the preserved exported predictions"
    )
    samples.add_argument(
        "--samples-dir", type=Path, default=PROJECT_ROOT / "results/samples"
    )
    samples.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/recomputed"
    )
    samples.add_argument("--plot", action="store_true")
    analysis = commands.add_parser("analyze-samples", help="Create extended saved-sample diagnostics")
    analysis.add_argument(
        "--samples-dir", type=Path, default=PROJECT_ROOT / "results/samples"
    )
    analysis.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/extended_analysis"
    )
    analysis.add_argument("--activity-threshold", type=float, default=0.05)
    audit = commands.add_parser("inspect-recording", help="Audit an available raw MATLAB recording")
    audit.add_argument("recording", type=Path)
    audit.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/recording_audit.json")
    audit.add_argument("--window-size", type=int, default=600)
    audit.add_argument("--step", type=int, default=100)
    replay = commands.add_parser("visualize", help="Open the interactive 3D replay")
    replay.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "results/samples")
    replay.add_argument("--fps", type=int, default=40)
    for name in ("train", "predict"):
        sub = commands.add_parser(name, help=f"{name.capitalize()} with the NinaPro raw recording")
        sub.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
        sub.add_argument("--subject", type=int, default=1)
        sub.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/new_run")
        sub.add_argument(
            "--model", type=Path, default=PROJECT_ROOT / "models/paper_model_best.keras"
        )
        if name == "train":
            sub.add_argument(
                "--checkpoint", type=Path,
                default=PROJECT_ROOT / "models/phase4_new_run.keras",
            )
    args = parser.parse_args()
    if args.command == "analyze-samples":
        from .diagnostics import analyze_samples
        summary = analyze_samples(
            args.samples_dir, args.output_dir,
            PROJECT_ROOT / "models/paper_model_best.keras", args.activity_threshold,
        )
        print(f"Analyzed {summary['sample_count']} saved observations.")
        return
    if args.command == "inspect-recording":
        from .diagnostics import inspect_recording
        summary = inspect_recording(args.recording, args.window_size, args.step)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    if args.command == "visualize":
        from .visualization import animate_samples
        animate_samples(args.samples_dir, args.fps)
        return
    import numpy as np
    from .evaluation import (
        calculate_metrics, plot_predictions, postprocess_predictions, save_metrics,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate-samples":
        y_true = np.load(args.samples_dir / "y_true_sample.npy")
        y_pred = np.load(args.samples_dir / "y_pred_sample.npy")
        rows = calculate_metrics(y_true, y_pred)
        save_metrics(rows, args.output_dir / "calibrated_sample_metrics.csv")
        if args.plot:
            plot_predictions(y_true, y_pred, args.output_dir / "calibrated_predictions.png")
    else:
        from .preprocessing import prepare_data
        config = ExperimentConfig(
            data_dir=args.data_dir, subject=args.subject,
            output_dir=args.output_dir, model_path=args.model,
        )
        data = prepare_data(config)
        if args.command == "train":
            from .model import train_model
            _, history = train_model(data, config, args.checkpoint)
            (args.output_dir / "training_history.json").write_text(
                json.dumps(history.history, indent=2), encoding="utf-8"
            )
            model_path = args.checkpoint
        else:
            model_path = args.model
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
        raw = model.predict(data.X_test)
        smoothed, calibrated = postprocess_predictions(
            raw, config.prediction_sigma, config.gain_factor
        )
        save_metrics(
            calculate_metrics(data.y_test, smoothed),
            args.output_dir / "pre_calibration_metrics.csv",
        )
        rows = calculate_metrics(data.y_test, calibrated)
        save_metrics(rows, args.output_dir / "calibrated_metrics.csv")
        np.save(args.output_dir / "y_true_sample.npy", data.y_test)
        np.save(args.output_dir / "y_pred_sample.npy", calibrated)
        np.save(args.output_dir / "y_pred_smoothed.npy", smoothed)
        plot_predictions(data.y_test, calibrated, args.output_dir / "predictions.png")
    for row in rows:
        print(
            f"{row['axis']}: RMSE {row['rmse']:.4f}, "
            f"Pearson CC {row['pearson_cc']:.4f}"
        )


if __name__ == "__main__":
    main()
