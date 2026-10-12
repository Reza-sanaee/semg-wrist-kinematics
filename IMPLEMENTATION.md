# Continuous Wrist Kinematics Estimation Using sEMG

**Bachelor's degree project — Reza Sanaeinejad**  
**Supervisor — Dr. Mehran Jahed**  
**Department of Electrical Engineering, Sharif University of Technology**

This folder contains the complete formal English translation of the submitted project, an expanded academic report, and an organized implementation of the supplied Phase 4 experiment. All five original chapters, the abstract, research explanations, seven references, twelve scientific figures, and four original tables are retained. Three supplementary chapters add implementation verification, extended evaluation, and research design. The final report contains eight chapters, fifteen figures, and fifteen tables. The presentation retains every original slide topic and adds seven supplementary slides, giving twenty-two slides in total. The original archive remains outside this delivery folder.

## Folder structure

```text
BC_Project_English/
├── report/                 Full English report in Word and PDF
├── presentation/           English presentation and its complete text
├── figures/                All 12 report figures with English labels
│   └── source/             Editable SVG sources for reconstructed figures
├── src/wrist_kinematics/    Organized Python implementation
├── notebooks/              Documented Phase 4 workflow
├── models/                 Unmodified supplied trained Keras model
├── results/
│   ├── samples/            Unmodified target and prediction arrays
│   ├── video/              Original demonstration recording
│   ├── recomputed/         Metrics computed from the supplied saved arrays
│   └── extended_analysis/  Residual, scatter, activity, and baseline diagnostics
├── references/             Supplied English papers and offline DB2 page
├── data/                   Instructions for obtaining the missing raw data
└── tests/                  Regression checks for the reorganized implementation
```

## Setup

From this folder, create a Python environment and install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[training,visualization,notebook]"
```

For evaluating the saved arrays only, install the core package with `python -m pip install -e .`. Dependency specifications are provided, but a fully locked training environment was not included in the original submission.

`requirements-analysis-checked.txt` records the scientific package versions used for the completed sample analysis and thirteen tests. It documents the verified analysis environment; it is not presented as a reconstruction of the original TensorFlow training environment.

## Evaluate the preserved predictions

```powershell
python -m wrist_kinematics evaluate-samples --plot
```

This evaluates the supplied **calibrated and clipped** predictions. The report's published metrics were calculated in the source notebook **before** calibration. Both sets of results are retained with clear provenance; consult `TECHNICAL_NOTES.md` before comparing them.

## Interactive 3D demonstration

```powershell
python -m wrist_kinematics visualize --fps 40
```

The animation opens in a browser through VPython. The normalized outputs are mapped to an illustrative ±90° rotation range, following the original demonstration; this is not a measured physical angle calibration. The original recorded video is in `results/video/`.

## Extended evaluation and recording audit

```powershell
python -m wrist_kinematics analyze-samples
python -m wrist_kinematics inspect-recording "data\DB2_s1\S1_E1_A1.mat"
```

The first command generates the additional analyses in Chapter 7: MAE, bias, error percentiles, R squared, activity-conditioned RMSE, a constant zero baseline, clipping counts, and three figures. It also inspects the saved model's JSON configuration without TensorFlow. The second command requires a raw recording and identifies selected-sequence gaps and windows containing multiple repetition labels or crossing discarded intervals. It does not change the data or the original training procedure.

## Run the Phase 4 experiment

The archive did not include the raw NinaPro recording. Follow `data/README.md`, then use:

```powershell
python -m wrist_kinematics train --data-dir "D:\path\to\NinaPro"
python -m wrist_kinematics predict --data-dir "D:\path\to\NinaPro"
```

Training saves to a new checkpoint, `models/phase4_new_run.keras`, and does not overwrite the supplied model. Each new run exports both pre-calibration and calibrated metrics, prediction arrays, and a comparison figure. The notebook provides the same workflow with explanatory English Markdown.

## Implementation and provenance

The reorganization removes unused imports and the original hard-coded `D:/bcs_project/` path, separates reusable functions from execution, and adds input validation and configurable paths. The original movement mapping, filtering order, Gaussian smoothing, window boundaries, repetition assignment, network architecture, training settings, gain factor, and clipping are retained.

The trained model, saved arrays, video, and reference papers are preserved. `results/preserved_assets.json` records SHA-256 checksums for the model, arrays, and video. `results/original_training_log.txt` retains the original notebook's textual execution outputs. Technical inconsistencies in the submitted report and code are documented separately rather than silently altering the research.

## Checks

```powershell
python -m unittest discover -s tests -v
```

Thirteen numerical regression and diagnostic tests passed, including a complete preprocessing comparison with the source operations on a synthetic MATLAB recording. See `VALIDATION.md` for the checks actually performed and their limits. Raw-data training and interactive 3D playback require dependencies and data beyond the supplied archive.
