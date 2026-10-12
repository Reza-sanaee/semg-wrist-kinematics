"""Checks for newly added diagnostics, including source-pipeline equivalence."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_kinematics.config import ExperimentConfig, PROJECT_ROOT
from wrist_kinematics.diagnostics import (
    detailed_metrics, inspect_model_package, inspect_recording,
)
from wrist_kinematics.model import train_model
from wrist_kinematics.preprocessing import prepare_data


class ExtendedDiagnosticsTests(unittest.TestCase):
    def test_known_errors_and_zero_baseline(self):
        true = np.array([[-1, 0, 1], [0, 1, 0], [1, 0, -1]], dtype=float)
        pred = true.copy()
        pred[:, 0] *= -1
        rows = detailed_metrics(true, pred)
        self.assertAlmostEqual(rows[0]["rmse"], np.sqrt(8 / 3))
        self.assertAlmostEqual(rows[0]["r_squared"], -3)
        self.assertAlmostEqual(rows[0]["pearson_cc"], -1)
        self.assertEqual(rows[1]["rmse"], 0)
        self.assertEqual(rows[2]["rmse"], 0)
        self.assertEqual(rows[1]["active_samples"], 1)

    def test_constant_target_has_no_defined_r_squared(self):
        rows = detailed_metrics(np.zeros((5, 3)), np.ones((5, 3)))
        for row in rows:
            self.assertIsNone(row["r_squared"])
            self.assertIsNone(row["active_rmse"])
            self.assertEqual(row["near_rest_samples"], 5)

    def test_saved_model_architecture_and_metadata(self):
        package = inspect_model_package(PROJECT_ROOT / "models/paper_model_best.keras")
        convs = [p for p in package["layers"] if p["class"] == "Conv1D"]
        self.assertEqual([p["filters"] for p in convs], [32, 64])
        self.assertEqual([p["dilation_rate"] for p in convs], [[1], [1]])
        lstm = [p for p in package["layers"] if p["class"] == "LSTM"]
        self.assertEqual(len(lstm), 1)
        self.assertEqual(lstm[0]["units"], 64)
        self.assertEqual(package["metadata"]["keras_version"], "3.14.0")

    def test_supplied_checkpoint_cannot_be_overwritten(self):
        config = ExperimentConfig()
        with self.assertRaises(ValueError):
            train_model(None, config, config.model_path)

    def test_missing_raw_recording_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                prepare_data(ExperimentConfig(data_dir=Path(folder)))

    def test_recording_audit_detects_removed_gaps_and_mixed_repetitions(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "recording.mat"
            labels = np.array([9, 9, 9, 99, 99, 10, 10, 10, 10, 10]).reshape(-1, 1)
            reps = np.array([1, 1, 1, 0, 0, 2, 2, 2, 2, 2]).reshape(-1, 1)
            scipy.io.savemat(file, {
                "emg": np.zeros((10, 12)), "restimulus": labels, "rerepetition": reps,
            })
            audit = inspect_recording(file, window_size=4, step=1)
            self.assertEqual(audit["selected_samples"], 8)
            self.assertEqual(audit["selected_sequence_gaps"], 1)
            self.assertGreater(audit["mixed_repetition_windows"], 0)
            self.assertGreater(audit["windows_crossing_discarded_time_gaps"], 0)

    def test_complete_preprocessing_matches_source_on_a_synthetic_recording(self):
        # Independent transcription of the executed source notebook operations.
        # Synthetic data exercise removed movements and train/test boundaries.
        rng = np.random.default_rng(7)
        emg = rng.normal(size=(7200, 12))
        actions = np.tile(np.repeat([0, 9, 99, 10, 15, 16, 13, 14], 150), 6)
        repetitions = np.repeat(np.arange(1, 7), 1200).reshape(-1, 1)
        actions = actions.reshape(-1, 1)
        with tempfile.TemporaryDirectory() as folder:
            subject_folder = Path(folder) / "DB2_s1"
            subject_folder.mkdir()
            file = subject_folder / "S1_E1_A1.mat"
            scipy.io.savemat(file, {
                "emg": emg, "restimulus": actions, "rerepetition": repetitions,
            })
            config = ExperimentConfig(data_dir=Path(folder))
            actual = prepare_data(config)

        mask = np.isin(actions, [9, 10, 13, 14, 15, 16, 0]).flatten()
        rectified = np.abs(emg[mask, :])
        b, a = butter(4, 5 / (0.5 * 2000), btype="low", analog=False)
        envelope = np.abs(filtfilt(b, a, rectified, axis=0))
        reps = repetitions[mask]
        scaler = StandardScaler().fit(envelope[np.isin(reps, [1, 3, 4, 6]).flatten()])
        scaled = scaler.transform(envelope)
        selected = actions[mask].flatten()
        labels = np.zeros((len(selected), 3))
        for code, target in [(9, [-1, 0, 0]), (10, [1, 0, 0]),
                             (15, [0, 1, 0]), (16, [0, -1, 0]),
                             (13, [0, 0, -1]), (14, [0, 0, 1])]:
            labels[selected == code] = target
        for axis in range(3):
            labels[:, axis] = gaussian_filter1d(labels[:, axis], sigma=400)
        expected = [[], [], [], []]
        for start in range(0, len(scaled) - 600, 100):
            end = start + 600
            index = 0 if reps[end - 1, 0] in [1, 3, 4, 6] else 2
            expected[index].append(scaled[start:end])
            expected[index + 1].append(labels[end - 1])
        for result, reference in zip(
            [actual.X_train, actual.y_train, actual.X_test, actual.y_test], expected
        ):
            np.testing.assert_allclose(result, np.array(reference), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
