# Validation of the Expanded English Project

## Translation completeness

All substantive paragraphs in Chapters 1–5, the abstract, all four original table datasets, the twelve original scientific figures, and all seven references are retained. A paragraph-level coverage check confirms all 195 source body elements, including table cells and figure placements. Original front-matter labels have been translated; the contents and figure/table lists have been regenerated rather than retaining obsolete Persian page numbers.

The report contains eight chapters, fifteen figures, and fifteen tables. Chapters 6–8 clearly identify their supplementary evidence and proposed experiments. The expanded presentation retains the original fifteen topics and adds seven supplementary slides. English text checks found no Persian characters in the report's document XML, the PDF text, or the presentation's slide text.

## Numerical checks performed

All thirteen tests passed:

1. Butterworth filtering agrees with the original operation.
2. Movement-code mapping preserves target signs and axis order.
3. Window bounds and final-sample target alignment match the source.
4. Gaussian prediction smoothing, multiplication by 1.35, and clipping match the source.
5. The 4,211 saved sample pairs are finite, match in shape, and have bounded predictions.
6. Mismatched and nonfinite samples are rejected.
7. Known prediction cases give the expected RMSE, R squared, and correlation.
8. Constant-target statistics identify undefined R squared and absent active observations.
9. Saved model inspection verifies convolution filters, dilation rates, LSTM units, and Keras metadata.
10. The supplied checkpoint cannot be used as the destination of a new training run.
11. Missing raw data produce an explicit error.
12. Recording inspection detects synthetic sequence gaps and mixed repetition windows.
13. Complete preprocessing matches an independent transcription of the source operations exactly on a synthetic twelve-channel MATLAB recording.

The saved-sample evaluation command and the extended-analysis command both completed. RMSE values for the saved calibrated predictions are 0.1180974156 for Roll, 0.0554271089 for Pitch, and 0.1058633302 for Yaw. The original pre-calibration metrics remain separately available in the report and `results/reported_metrics.csv`.

## Model and asset integrity

The saved model configuration and archived summary identify 45,923 total parameters, including 45,731 trainable parameters and 192 nontrainable parameters. Model, sample-array, and video SHA-256 hashes match the original archive. Supplied research papers and the offline database page were preserved. The final model package is inspected without claiming a TensorFlow inference rerun.

## Document and presentation checks

The Word report was exported through the installed Microsoft Word application, including updates to contents and page references. PDF pages were rendered with Poppler and visually inspected. The slide deck was rendered and inspected, and its package integrity, geometry, declared fonts, two native result tables, and first-party import validation passed. These checks verify file usability and layout; they do not constitute a clinical or deployment evaluation.

## Remaining experimental limits

The raw NinaPro `.mat` recording is absent, so the original data preparation and training were not rerun on the real recording. TensorFlow training, GPU behavior, physical sensor acquisition, and interactive VPython playback have not been verified in this preparation environment. Earlier phase implementations were not supplied. The additional research designs in Chapter 8 are proposals; no new model-training, subject-generalization, or deployment result is invented.
