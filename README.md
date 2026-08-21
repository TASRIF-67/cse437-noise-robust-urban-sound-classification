# Noise-Robust Urban Sound Classification

This repository supports a university data-science research project investigating
how background noise affects environmental sound classification and whether
training-time audio augmentation improves model robustness.

The initial dataset is UrbanSound8K. Dataset-specific behavior is isolated behind
an adapter so that future datasets such as ESC-50, FSD50K, GTZAN, and custom
single-label audio datasets can reuse the same preprocessing, augmentation,
training, and evaluation pipeline.

> **Current status:** The repository foundation, UrbanSound8K adapter, dataset
> validation, resumable acquisition, lossless reconstruction, and inspection
> reporting are implemented and tested. The complete audio preprocessing pipeline
> and model-ready PyTorch dataset integration are also implemented and verified.
> The executed UrbanSound8K EDA notebook, research figures, numerical summaries,
> Log-Mel examples, and MFCC examples are complete. Controlled noise corruption
> and augmentation are the next phase.

## Research questions

1. Does training-time audio augmentation reduce the loss of classification
   performance when test audio is corrupted by background noise?
2. Is the effect of augmentation consistent across CNN, CRNN, and ResNet18
   architectures?

The final experiment will compare baseline and augmented versions of all three
architectures under clean, 20 dB, 10 dB, and 0 dB test conditions. A model is
trained once per training condition and evaluated at every configured noise level;
models are not retrained for individual SNR values.

## Repository structure

```text
configs/                       Experiment settings grouped by responsibility
data/                          Local datasets, noise audio, and derived data
notebooks/                     Exploration and report-oriented analysis only
src/urban_sound_robustness/    Reusable installable Python package
  datasets/                    Dataset contracts and dataset-specific adapters
  audio/                       Loading, length handling, and feature extraction
  augmentation/                Waveform and spectrogram augmentations
  models/                      CNN, CRNN, ResNet18, and the model factory
  training/                    Training, validation, and checkpoint management
  evaluation/                  Metrics and controlled robustness evaluation
  visualization/               Reusable research plots
  utils/                       Configuration, logging, seeds, paths, and devices
scripts/                       Small command-line entry points
experiments/                   Per-run configurations, histories, and summaries
checkpoints/                   Generated model checkpoints
logs/                          Runtime and TensorBoard logs
results/                       Metrics, predictions, matrices, and figures
tests/                         Focused tests of project-owned logic
```

Large and generated contents under `data/`, `experiments/`, `checkpoints/`,
`logs/`, and `results/` are ignored by Git. Their placeholder files keep the
expected directory layout visible in a fresh clone.

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the CUDA profile verified for the GTX 1660 Ti:

```bash
python -m pip install -r requirements-cuda.txt
python -m pip install -e .
```

For development on a machine without an NVIDIA GPU, replace the first command
with:

```bash
python -m pip install -r requirements-cpu.txt
```

The profiles pin matching PyTorch, torchvision, and torchaudio builds. The shared
`requirements.txt` intentionally excludes PyTorch so a broad version constraint
cannot silently replace a CUDA build with a CPU-only package.

Editable installation allows scripts and notebooks to import
`urban_sound_robustness` consistently without modifying `PYTHONPATH`.

## Dataset setup

UrbanSound8K is not committed to this repository. The reproducible acquisition
flow downloads a public Parquet mirror containing the original WAV bytes and then
reconstructs the default layout without decoding or resampling:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='danavery/urbansound8K', repo_type='dataset', local_dir='data/raw/UrbanSound8K_parquet')"
python scripts\prepare_urbansound8k_from_parquet.py
python scripts\inspect_dataset.py
```

The resulting paths are:

```text
data/raw/UrbanSound8K/audio/fold1/...
data/raw/UrbanSound8K/audio/fold10/...
data/raw/UrbanSound8K/metadata/UrbanSound8K.csv
```

The initial split reserves folds 1-8 for training, fold 9 for validation, and
fold 10 for testing. These assignments are configurable and preserve the official
fold boundaries to reduce source-recording leakage.

The latest full local inspection verified all 8,732 metadata rows and WAV headers:
zero files were missing or unreadable, the audio totals 8.75 hours, and source
sample rates range from 8 kHz to 192 kHz. Class counts range from 374 (`gun_shot`)
to 1,000, so later evaluation will report macro-averaged metrics alongside
accuracy. Generated inspection artifacts are stored under
`results/metrics/dataset_inspection/`.

External noise belongs under `data/external_noise/`. It is used only as an
augmentation or corruption source and never as a target class.

## Exploratory data analysis

The executed notebook `notebooks/01_urbansound8k_eda.ipynb` analyzes all 8,732
inventory records and decodes ten deterministic representative class examples.
It reports class/fold balance, durations, recording characteristics, waveforms,
Log-Mel features, and MFCCs. Open it after activating `.venv`:

```powershell
python -m jupyter lab notebooks\01_urbansound8k_eda.ipynb
```

Publication-ready PNG files are saved under `results/figures/eda/`; underlying
CSV and JSON summaries are saved under `results/metrics/eda/`.

## Configuration

Configuration files are grouped by concern:

- `dataset`: metadata schema, paths, class information, and official folds.
- `audio`: sample rate, clip duration, Log-Mel parameters, and MFCC count.
- `augmentation`: baseline and robustness-oriented transformation choices.
- `model`: architecture-specific settings.
- `training`: optimizer, batch size, reproducibility, and checkpoint behavior.
- `evaluation`: deterministic SNR conditions and metric settings.
- `paths`: repository-relative storage locations.
- `experiment`: future composed run configurations.

Experiment manifests select one file from each component group and may apply small
nested overrides. Validate the development manifest with:

```bash
python scripts/check_config.py
```

The checker composes all referenced YAML files, validates cross-section assumptions,
resolves repository-relative paths, and reports the selected CPU or CUDA device.
The development manifest uses two epochs, a batch size of four, and no DataLoader
workers so future smoke tests remain inexpensive.

## Planned workflow

```text
Configuration -> dataset adapter -> audio preprocessing -> baseline/augmented training
              -> saved checkpoint -> clean/noisy evaluation -> research analysis
```

Implementation proceeds phase by phase: utilities, UrbanSound8K integration,
audio preprocessing, and EDA are complete; deterministic noise, augmentation,
models, training, evaluation, and final analysis follow.

## Reproducibility and leakage safeguards

The utility layer seeds Python, NumPy, PyTorch, CUDA, and future DataLoader workers.
It can request deterministic PyTorch behavior, create timestamped experiment IDs,
prevent output-directory reuse, and save resolved configuration and environment
snapshots. Training augmentation will never be applied to validation or test
samples. Controlled noisy test variants will later use a separate fixed corruption
seed so every model receives the same corrupted samples.

## Foundation verification

Run the focused utility suite inside the active virtual environment:

```bash
python -m pytest
```

The current tests cover the foundation utilities, UrbanSound8K validation,
inspection reports, lossless reconstruction, waveform loading, mono conversion,
resampling, duration normalization, log-Mel extraction, numerical stability,
MFCC extraction, and DataLoader batching. Use `python -m pytest -v` to see every
test name. The current expected result is 60 passed.

## Extending the project

Adding a dataset should require a new adapter under `datasets/` and a dataset YAML
file. Adding a model should require a model module plus a small factory registration.
Neither extension should require changes to generic training, evaluation,
augmentation, metrics, or visualization code.

Detailed usage, command examples, and results documentation will be expanded as
their corresponding development phases are implemented.
