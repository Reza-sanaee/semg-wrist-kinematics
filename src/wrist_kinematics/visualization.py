"""Interactive 3D replay of the supplied target and prediction trajectories."""

import math
from pathlib import Path

import numpy as np

from .evaluation import validate_samples


def animate_samples(samples_dir: Path, fps=40):
    """Replay normalized angles using the original illustrative 90-degree scale."""
    from vpython import box, canvas, color, cylinder, label, rate, vector

    if fps <= 0:
        raise ValueError("Frames per second must be positive.")
    y_true = np.load(Path(samples_dir) / "y_true_sample.npy")
    y_pred = np.load(Path(samples_dir) / "y_pred_sample.npy")
    y_true, y_pred = validate_samples(y_true, y_pred)
    scene = canvas(
        title="NinaPro DB2 Continuous Wrist Kinematics Prediction",
        width=1100, height=650, background=vector(0.15, 0.15, 0.15),
    )
    scene.camera.pos = vector(0, 3, 10)
    for position in (-4, 2):
        cylinder(
            pos=vector(position, 0, 0), axis=vector(1, 0, 0),
            radius=0.6, color=color.gray(0.5),
        )
    actual_hand = box(
        pos=vector(-2, 0, 0), size=vector(4, 0.8, 2),
        color=vector(0.2, 0.6, 1), opacity=0.9,
    )
    predicted_hand = box(
        pos=vector(4, 0, 0), size=vector(4, 0.8, 2), color=color.green, opacity=0.9,
    )
    label(
        pos=vector(-3, 3, 0), text="Actual hand", box=False,
        color=vector(0.2, 0.6, 1), height=18,
    )
    label(
        pos=vector(3, 3, 0), text="Predicted hand", box=False,
        color=color.green, height=18,
    )
    info = label(
        pos=vector(0, 5, 0), text="Initializing…", box=False, height=18, color=color.white,
    )
    error_info = label(
        pos=vector(0, 4, 0), text="", box=False, height=16, color=color.orange,
    )

    def apply_rotation(obj, roll, pitch, yaw):
        obj.axis = vector(4, 0, 0)
        obj.up = vector(0, 1, 0)
        max_angle = math.pi / 2
        obj.rotate(angle=roll * max_angle, axis=vector(1, 0, 0))
        obj.rotate(angle=yaw * max_angle, axis=vector(0, 1, 0))
        obj.rotate(angle=pitch * max_angle, axis=vector(0, 0, 1))

    print("Animation started in the browser.")
    for actual, predicted in zip(y_true, y_pred):
        rate(fps)
        apply_rotation(actual_hand, *actual)
        apply_rotation(predicted_hand, *predicted)
        error = float(np.mean(np.abs(actual - predicted)))
        if error > 0.4:
            predicted_hand.color = color.red
        elif error > 0.2:
            predicted_hand.color = color.orange
        else:
            predicted_hand.color = color.green
        roll, pitch, yaw = predicted
        info.text = f"Predicted: Roll {roll: .2f}   Pitch {pitch: .2f}   Yaw {yaw: .2f}"
        error_info.text = f"Mean instantaneous error: {error: .2f}"
    print("Animation finished.")
