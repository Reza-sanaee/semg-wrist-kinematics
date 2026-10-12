# Technical Notes on the Submitted Research

These notes accompany the complete translation. They distinguish statements in the original report from behavior observable in the supplied notebook and assets. They do not replace or shorten the report.

## Target labels and measured kinematics

The report discusses Cyberglove II kinematic recordings and describes the evaluation targets as actual wrist angles. The supplied Phase 4 code loads `emg`, `restimulus`, and `rerepetition`; it does not load a glove-angle variable. It maps movement codes 9, 10, 13, 14, 15, and 16 to normalized Roll, Pitch, and Yaw values and smooths those targets with a Gaussian filter. Therefore, the implementation evaluates agreement with movement-label-derived surrogate trajectories, rather than demonstrating direct regression against measured continuous Cyberglove wrist angles. The reconstructed Figure 4.1 labels these values as smoothed targets.

## Published metrics and saved predictions

The original notebook calculates RMSE and Pearson correlation after Gaussian smoothing of the predictions with sigma 10, but before multiplying by the gain factor 1.35 and clipping to [−1, +1]. The exported `y_pred_sample.npy` contains the calibrated and clipped predictions. Accordingly, metrics recomputed from these saved arrays differ from the published values. The original reported values remain in the report and presentation. `results/reported_metrics.csv` records them, while `results/recomputed/calibrated_sample_metrics.csv` records the separate evaluation of the preserved exported arrays. Pre-calibration predictions were not supplied as a separate array, and clipping cannot be uniquely inverted to reconstruct them.

## Correlation and percentage accuracy

Section 4.1 of the original report interprets a Pearson correlation of 0.9634 as more than 96 percent of angular behavior being correctly predicted. This statement is retained in the translation. Pearson correlation is an association measure and is not a percentage accuracy metric or a direct measure of explained variance.

## Architecture

The report describes dilated convolutions in Phase 3 and mentions them again in the final discussion. The supplied Phase 4 notebook and trained model contain two Conv1D layers with dilation rate 1, followed by a unidirectional LSTM with 64 units. The organized code preserves this final model. It does not add dilated layers or a bidirectional LSTM to the supplied Phase 4 implementation. The Phase 1–3 implementation files were not included in the archive; their development and results are retained through the report, figures, and presentation.

## Data splitting and validation

StandardScaler is fitted on training-repetition samples only, which is consistent with the source's intended protection against scaling leakage. However, the original pipeline filters and smooths the concatenated selected time series before window assignment. Each window is assigned to training or test according to the repetition of its final sample, without requiring every sample in the window to belong to the same split. Movement filtering also concatenates samples that may not have been adjacent in the original recording. These operations can mix boundaries or permit information from other partitions to affect preprocessing.

In addition, the source uses `X_test` and `y_test` as `validation_data` for checkpoint selection, EarlyStopping, and ReduceLROnPlateau. The final evaluation therefore does not use a test set isolated from model selection. The reorganization preserves this behavior to avoid silently changing the experiment or implying that the published results were reproduced under a different protocol. A future methodological revision should split independent training, validation, and test data before split-specific preprocessing and windowing.

## Offline processing and real-time claims

The source uses `filtfilt` and centered Gaussian smoothing, which depend on samples on both sides of a time point. These operations are offline and noncausal. The report's discussion of real-time control describes a proposed application; the supplied pipeline does not establish causal streaming performance, end-to-end latency, or deployment on embedded hardware.

## Counts and dates retained from the sources

The movement-category table lists 17, 23, and 9 movements, but gives a total of 50. The three listed counts sum to 49. Both the listed counts and the stated total are preserved in the translation.

The Phase 2 report describes more than 107,000 training windows and approximately 52,000 test windows. The corresponding original slide instead describes more than 52,000 training windows. Each source's wording is retained in its English counterpart.

The report cover is dated Mordad 1405 (July–August 2026), whereas the presentation is dated Khordad 1405 (May–June 2026). These source dates are preserved separately. Neither date has been replaced with the date of translation.

## Delivery decisions

The report's Word formatting has been rebuilt for left-to-right English text, with regenerated contents, figure lists, table lists, captions, and page references. All substantive source paragraphs, all four original tables, and all twelve original scientific figures are retained. Chapters 6–8 add verification, numerical evaluation, and proposed research designs, bringing the report to fifteen tables and fifteen figures. References follow the supplementary chapters. Empty template paragraphs, decorative page-label artifacts, and the obsolete `KBDFA.DLL` keyboard-support file are excluded from the clean delivery.

Figures that contained Persian text were reconstructed with English labels. The plots for Phases 1–3 and the wrist and dilation illustrations retain the supplied English graphics. The final comparison plot is redrawn from all 4,211 unchanged exported target and calibrated prediction observations. Figures 3.7 and 3.8 remain conceptual illustrations, not new experimental measurements. The expanded twenty-two-slide presentation retains all fifteen original topics and adds seven supplementary slides with additional analyses. Text and result tables are editable.

## Archived execution inconsistency

The stored preprocessing output reports 8,689 training windows. At the current batch size of 64, this would require 136 batches per epoch, but the archived training log reports 198. Earlier retained outputs or a changed data or batch configuration may explain the discrepancy. The source outputs have been preserved, and Section 6.5 discusses why they cannot establish a single consistent execution of the current notebook. The best-checkpoint and stopping events in the archived log are epoch 35 and epoch 50, respectively.
