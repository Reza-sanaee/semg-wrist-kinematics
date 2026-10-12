"""Numerical regression tests for behavior retained from the source notebook."""

import unittest
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_kinematics.config import PROJECT_ROOT
from wrist_kinematics.evaluation import (
    calculate_metrics, postprocess_predictions, validate_samples,
)
from wrist_kinematics.preprocessing import (
    butter_lowpass_filter, create_windows, map_to_rpy_smooth,
)


class RegressionTests(unittest.TestCase):
    def test_filter_matches_original_equation(self):
        values = np.random.default_rng(5).normal(size=(1800, 12))
        b, a = butter(4, 5 / (0.5 * 2000), btype="low", analog=False)
        expected = np.abs(filtfilt(b, a, values, axis=0))
        np.testing.assert_allclose(butter_lowpass_filter(values, 5, 2000), expected)

    def test_label_mapping_preserves_sign_and_order(self):
        labels = np.repeat([0, 9, 10, 15, 16, 13, 14], 90)
        expected = np.zeros((len(labels), 3))
        for code, target in [(9, [-1, 0, 0]), (10, [1, 0, 0]),
                             (15, [0, 1, 0]), (16, [0, -1, 0]),
                             (13, [0, 0, -1]), (14, [0, 0, 1])]:
            expected[labels == code] = target
        for axis in range(3):
            expected[:, axis] = gaussian_filter1d(expected[:, axis], sigma=5)
        np.testing.assert_allclose(map_to_rpy_smooth(labels, sigma=5), expected)

    def test_window_bounds_and_target_alignment(self):
        X = np.arange(36).reshape(12, 3)
        Y = X / 36
        windows, labels = create_windows(X, Y, 4, 2)
        self.assertEqual(windows.shape, (4, 4, 3))
        np.testing.assert_array_equal(windows[-1], X[6:10])
        np.testing.assert_array_equal(labels, Y[[3, 5, 7, 9]])

    def test_postprocessing_matches_original_and_clips(self):
        raw = np.column_stack([np.linspace(-2, 2, 100)] * 3).astype(np.float32)
        smooth, calibrated = postprocess_predictions(raw, sigma=10, gain=1.35)
        expected = np.zeros_like(raw)
        for axis in range(3):
            expected[:, axis] = gaussian_filter1d(raw[:, axis], sigma=10)
        np.testing.assert_array_equal(smooth, expected)
        np.testing.assert_array_equal(calibrated, np.clip(expected * 1.35, -1, 1))
        self.assertEqual(float(calibrated.min()), -1)
        self.assertEqual(float(calibrated.max()), 1)

    def test_preserved_samples_are_finite_and_bounded(self):
        true = np.load(PROJECT_ROOT / "results/samples/y_true_sample.npy")
        pred = np.load(PROJECT_ROOT / "results/samples/y_pred_sample.npy")
        validate_samples(true, pred)
        self.assertEqual(true.shape, (4211, 3))
        self.assertLessEqual(float(np.max(np.abs(pred))), 1)
        rows = calculate_metrics(true, pred)
        for axis, row in enumerate(rows):
            self.assertAlmostEqual(row["rmse"], np.sqrt(np.mean((true[:, axis] - pred[:, axis]) ** 2)))
            self.assertAlmostEqual(row["pearson_cc"], np.corrcoef(true[:, axis], pred[:, axis])[0, 1])

    def test_mismatched_or_nonfinite_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_samples(np.zeros((5, 3)), np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            validate_samples(np.full((5, 3), np.nan), np.zeros((5, 3)))


if __name__ == "__main__":
    unittest.main()
