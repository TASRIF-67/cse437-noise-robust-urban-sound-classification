# Project Status and Command Reference

This document summarizes the completed work and the commands needed to reproduce,
validate, and inspect it. Run every command from the repository root:
`F:\cse437-project`.

## Completed so far

- Created the Python project structure, configuration hierarchy, reusable package,
  scripts, tests, and Git ignore rules.
- Created and verified the local `.venv` environment with Python 3.13 and the
  CUDA-enabled PyTorch stack for the NVIDIA GTX 1660 Ti.
- Implemented configuration composition and validation, path handling, device
  selection, deterministic seeding, logging, and experiment-directory utilities.
- Downloaded all 16 UrbanSound8K Parquet shards and retained them under
  `data/raw/UrbanSound8K_parquet/` as the reproducible source snapshot.
- Reconstructed the original 8,732 WAV files without decoding or resampling.
- Created the standard `data/raw/UrbanSound8K/audio/fold1` through `fold10`
  structure and installed the official metadata CSV.
- Implemented a reusable UrbanSound8K adapter with strict metadata, label,
  filename, fold, and missing-file validation.
- Implemented dataset inspection and JSON/CSV report generation.
- Verified 8,732 readable files, zero missing files, zero unreadable files,
  10 classes, 10 folds, and approximately 8.75 hours of audio.
- Added fixture tests for dataset validation, inspection, report storage, and
  resumable lossless Parquet-to-WAV reconstruction. The current suite has 60
  passing tests.
- Implemented Step 1 of audio preprocessing: validated waveform loading that
  preserves source channels, frames, and sample rate as a float32 PyTorch tensor.
- Implemented Step 2 of audio preprocessing: deterministic channel averaging to
  mono while preserving dtype, device, frame count, and gradient flow.
- Implemented sample-rate normalization to the configured 22,050 Hz target.
- Implemented exact four-second waveforms using trailing zero padding, seeded
  random training crops, and deterministic center evaluation crops.
- Implemented standardized log-Mel extraction with fixed `[1, 64, 173]` output.
- Added a lazy PyTorch dataset wrapper that returns model-ready features, labels,
  split provenance, and original audio properties.
- Added a preprocessing CLI that saves numerical reports and waveform/log-Mel
  plots from real dataset samples.
- Added reusable MFCC extraction for exploratory analysis and a future classical
  baseline.
- Created and executed `notebooks/01_urbansound8k_eda.ipynb` against the complete
  dataset inventory. It generated seven research figures and five CSV/JSON outputs.

## Current dataset findings

| Finding | Result | Why it matters |
|---|---:|---|
| Samples | 8,732 | Matches the official metadata inventory |
| Classes | 10 | Confirms the classification target space |
| Official folds | 10 | Used for leakage-aware train/validation/test splits |
| Missing files | 0 | Dataset paths are complete |
| Unreadable files | 0 | Every WAV header can be opened |
| Total duration | 8.75 hours | Useful for runtime and storage planning |
| Source sample rates | 8–192 kHz | Preprocessing must resample consistently |
| Class imbalance ratio | 2.67 | Macro metrics should accompany accuracy |
| Clips shorter than 4 seconds | 16.0% | These clips require the configured padding policy |
| Source channels | 739 mono, 7,993 stereo | Mono conversion is necessary for consistent input |

The configured split uses folds 1–8 for training, fold 9 for validation, and
fold 10 for testing.

## Environment commands

These setup commands are only needed on a fresh machine. All packages are
installed inside `.venv`. Create the environment, activate it, and then use the
short `python ...` commands shown below.

| Command | What it does |
|---|---|
| `python -m venv .venv` | Creates an isolated Python environment in the repository. |
| `.\.venv\Scripts\Activate.ps1` | Activates `.venv` in PowerShell. The prompt should begin with `(.venv)`. |
| `python -m pip install --upgrade pip` | Updates `pip` inside the active `.venv`. |
| `python -m pip install -r requirements-cuda.txt` | Installs the matched CUDA PyTorch stack and all shared dependencies. Use this on the verified NVIDIA machine. |
| `python -m pip install -r requirements-cpu.txt` | CPU-only alternative for a machine without a compatible NVIDIA GPU. Do not install both profiles. |
| `python -m pip install -e .` | Installs the project package in editable mode so scripts import from `src/`. |
| `python -m pip check` | Reports broken or incompatible installed dependencies. |

Activate the environment at the beginning of each new PowerShell session:

```powershell
.\.venv\Scripts\Activate.ps1
```

All remaining commands assume the terminal prompt begins with `(.venv)`.

## Dataset acquisition commands

The actual dataset is already present locally. These commands are for reproducing
or resuming acquisition on another machine.

### Preferred Hugging Face snapshot

```powershell
python -c "import os; os.environ['HF_XET_HIGH_PERFORMANCE']='1'; from huggingface_hub import snapshot_download; snapshot_download(repo_id='danavery/urbansound8K', repo_type='dataset', local_dir='data/raw/UrbanSound8K_parquet', allow_patterns=['data/*.parquet', 'UrbanSound8K.csv', 'README.md'], max_workers=1)"
```

This downloads the 16 Parquet shards, official metadata, and mirror README. It is
safe to rerun: completed files and cached partial chunks are reused.

```powershell
python scripts\prepare_urbansound8k_from_parquet.py
```

This reconstructs the embedded WAV bytes into the configured `audio/fold*`
layout. It rejects missing shards, duplicate or unsafe filenames, fold conflicts,
metadata mismatches, and non-WAV payloads. Rerunning it verifies existing files
instead of rewriting matching data.

### Kaggle archive fallback

```powershell
python scripts\download_urbansound8k.py --workers 16
```

This is a resumable byte-range downloader for the Kaggle archive mirror. It keeps
independent part files and validates the final archive size and ZIP CRCs. The
preferred Parquet workflow above was used for the current local dataset because
it performed better on this connection.

## Configuration and dataset checks

| Command | What it does |
|---|---|
| `python scripts\check_config.py` | Composes the development YAML configuration, validates it, resolves paths, and reports the selected CPU/CUDA device. |
| `python scripts\inspect_dataset.py` | Validates all metadata and scans every WAV header, then writes JSON/CSV reports. |
| `python scripts\inspect_dataset.py --skip-audio-scan` | Performs a faster metadata-only inspection. |
| `python scripts\inspect_dataset.py --split train` | Inspects only configured training folds 1–8. Replace `train` with `validation` or `test` when needed. |
| `python scripts\inspect_dataset.py --no-progress` | Runs the full inspection without the progress bar, useful for saved logs or automation. |

Inspection outputs are written to:

```text
results/metrics/dataset_inspection/summary.json
results/metrics/dataset_inspection/class_distribution.csv
results/metrics/dataset_inspection/fold_distribution.csv
results/metrics/dataset_inspection/sample_rate_distribution.csv
results/metrics/dataset_inspection/sample_inventory.csv
```

Runtime logs are written to `logs/dataset_inspection.log`.

## Preprocessing CLI commands

| Command | What it does |
|---|---|
| `python scripts\inspect_preprocessing.py` | Processes 12 evenly spaced test samples in evaluation mode and saves three plots. |
| `python scripts\inspect_preprocessing.py --split train --mode training --num-samples 24 --num-plots 4 --seed 42` | Exercises reproducible random training crops on 24 samples and saves four plots. |
| `python scripts\inspect_preprocessing.py --split validation --num-samples 8 --num-plots 0 --no-progress` | Checks eight validation samples without plots or a progress bar. |
| `python scripts\inspect_preprocessing.py --output-directory results\my_preprocessing_check` | Writes the inspection artifacts to a custom directory. |
| `python scripts\inspect_preprocessing.py --help` | Displays every CLI option and its description. |

Default preprocessing outputs are:

```text
results/preprocessing_inspection/summary.json
results/preprocessing_inspection/sample_outputs.csv
results/preprocessing_inspection/<sample-id>.png
```

The current real-data smoke run processed mono and stereo files at 44.1, 48, and
96 kHz. Additional end-to-end checks at 8 kHz and 192 kHz produced the same finite
fixed output shape: `[1, 64, 173]`.

## EDA notebook commands

| Command | What it does |
|---|---|
| `python -m jupyter lab notebooks\01_urbansound8k_eda.ipynb` | Opens the executed EDA notebook interactively in JupyterLab. |
| `python -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_urbansound8k_eda.ipynb --ExecutePreprocessor.timeout=600` | Re-executes every notebook cell and refreshes stored outputs and figures. |

The executed notebook saves:

```text
results/figures/eda/01_class_distribution.png
results/figures/eda/02_fold_distribution.png
results/figures/eda/03_duration_distribution.png
results/figures/eda/04_recording_characteristics.png
results/figures/eda/05_representative_waveforms.png
results/figures/eda/06_representative_log_mel.png
results/figures/eda/07_waveform_logmel_mfcc.png
results/metrics/eda/eda_summary.json
results/metrics/eda/class_distribution.csv
results/metrics/eda/class_by_fold.csv
results/metrics/eda/duration_summary_by_class.csv
results/metrics/eda/representative_samples.csv
```

## Development verification commands

| Command | What it does |
|---|---|
| `python -m pytest -v` | Runs the complete automated test suite with individual test names. Current expected result: 60 passed. |
| `python -m compileall -q src scripts tests` | Compiles project Python files to catch syntax errors without running experiments. |
| `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"` | Displays the PyTorch build and confirms whether CUDA and the GPU are available. |

## Completed preprocessing phase

1. Load WAV waveforms safely. **Completed.**
2. Convert multi-channel audio to mono. **Completed.**
3. Resample every sample to the configured target rate. **Completed.**
4. Pad or crop audio deterministically to four seconds. **Completed.**
5. Generate log-Mel spectrograms with consistent dimensions. **Completed.**
6. Integrate preprocessing with dataset records and DataLoader batching. **Completed.**
7. Test rate conversion, channels, lengths, reproducibility, numerical stability,
   feature shapes, and batching. **Completed.**
8. Run the real-data CLI and save inspection artifacts. **Completed.**

## Completed EDA phase

1. Verify complete dataset integrity and summarize numerical findings. **Completed.**
2. Analyze class, fold, duration, sample-rate, and channel distributions. **Completed.**
3. Select deterministic representative examples from all ten classes. **Completed.**
4. Visualize waveforms and standardized Log-Mel representations. **Completed.**
5. Implement and visualize reusable MFCC features. **Completed.**
6. Save publication-ready figures and supporting CSV/JSON data. **Completed.**
7. Execute every notebook cell inside `.venv` with no errors. **Completed.**

Controlled noise corruption and augmentation are the next phase, followed by
models, training, and robustness evaluation.
