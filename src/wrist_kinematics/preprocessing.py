"""Preprocessing equivalent to the original Phase 4 notebook.

Filtering, smoothing, movement selection, and window assignment deliberately
retain the source experiment's offline behavior. See TECHNICAL_NOTES.md for
limitations of this procedure when assessing generalization and causality.
"""

from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig

MOVEMENT_TO_RPY = {
    9: (-1.0, 0.0, 0.0),
    10: (1.0, 0.0, 0.0),
    15: (0.0, 1.0, 0.0),
    16: (0.0, -1.0, 0.0),
    13: (0.0, 0.0, -1.0),
    14: (0.0, 0.0, 1.0),
}


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Filter along time and return the absolute envelope, as in the source."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
    return np.abs(filtfilt(b, a, data, axis=0))


def map_to_rpy_smooth(labels, sigma=200):
    """Map movement codes to normalized targets, then smooth each axis."""
    labels = np.asarray(labels).flatten()
    rpy = np.zeros((len(labels), 3))
    for code, target in MOVEMENT_TO_RPY.items():
        rpy[labels == code] = target
    result = np.zeros_like(rpy)
    for axis in range(3):
        result[:, axis] = gaussian_filter1d(rpy[:, axis], sigma=sigma)
    return result


def create_windows(X, Y, window_size, step):
    """Keep the original exclusive upper bound and final-sample targets."""
    windows_X, windows_Y = [], []
    for start in range(0, len(X) - window_size, step):
        windows_X.append(X[start : start + window_size])
        windows_Y.append(Y[start + window_size - 1])
    return np.array(windows_X), np.array(windows_Y)


def prepare_data(config: ExperimentConfig) -> PreparedData:
    """Load one subject and reproduce the original repetition-based split."""
    path = (
        config.data_dir
        / f"DB2_s{config.subject}"
        / f"S{config.subject}_E1_A1.mat"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"NinaPro input file not found: {path}. "
            "Place the authorized dataset in data/ or provide --data-dir."
        )
    recording = loadmat(path)
    for key in ("emg", "restimulus", "rerepetition"):
        if key not in recording:
            raise ValueError(f"Missing MATLAB variable {key!r} in {path}")
    raw_emg = recording["emg"]
    action_codes = recording["restimulus"]
    repetitions = recording["rerepetition"]
    if raw_emg.ndim != 2 or raw_emg.shape[1] != 12:
        raise ValueError("The experiment requires 12 EMG channels.")
    if not (len(raw_emg) == len(action_codes) == len(repetitions)):
        raise ValueError("EMG, movement labels, and repetition labels differ in length.")

    mask = np.isin(action_codes, config.movement_codes).flatten()
    emg = raw_emg[mask, :]
    actions = action_codes[mask]
    reps = repetitions[mask]
    envelope = butter_lowpass_filter(
        np.abs(emg), config.cutoff_hz, config.sampling_rate, order=4
    )
    training_mask = np.isin(reps, config.train_repetitions).flatten()
    if not training_mask.any():
        raise ValueError("No samples match the configured training repetitions.")
    scaler = StandardScaler().fit(envelope[training_mask])
    scaled = scaler.transform(envelope)
    labels = map_to_rpy_smooth(actions, sigma=config.label_sigma)
    X_train, y_train, X_test, y_test = [], [], [], []
    for start in range(0, len(scaled) - config.window_size, config.step):
        end = start + config.window_size
        window = scaled[start:end]
        target = labels[end - 1]
        repetition = reps[end - 1].item()
        if repetition in config.train_repetitions:
            X_train.append(window)
            y_train.append(target)
        elif repetition in config.test_repetitions:
            X_test.append(window)
            y_test.append(target)
    if not X_train or not X_test:
        raise ValueError("The configured windowing produced an empty train or test set.")
    return PreparedData(
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test), scaler,
    )
