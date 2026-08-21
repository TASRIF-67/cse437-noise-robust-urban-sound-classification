# Command-line scripts

`check_config.py` validates and summarizes a composed experiment configuration.
Run it from the repository root after activating `.venv`:

```bash
python scripts/check_config.py
```

`inspect_dataset.py` validates the configured dataset, checks lightweight audio
headers, and saves JSON/CSV summaries:

```bash
python scripts/inspect_dataset.py
```

Use `--skip-audio-scan` for metadata-only inspection or `--split train` to inspect
one configured fold group. Full dataset inspection remains the default.

`inspect_preprocessing.py` runs the complete configured preprocessing pipeline on
a bounded set of real samples, verifies fixed tensor shapes and finite values, and
saves CSV/JSON metrics plus waveform/log-Mel plots:

```powershell
python scripts\inspect_preprocessing.py
```

Useful variants:

```powershell
python scripts\inspect_preprocessing.py --split train --mode training --num-samples 24 --num-plots 4 --seed 42
python scripts\inspect_preprocessing.py --split validation --num-samples 8 --num-plots 0 --no-progress
python scripts\inspect_preprocessing.py --help
```

Evaluation mode uses deterministic center cropping. Training mode uses seeded
random cropping. Outputs default to `results/preprocessing_inspection/` and the
runtime log is `logs/preprocessing_inspection.log`.

`inspect_snr.py` applies the configured clean, 20 dB, 10 dB, and 0 dB
conditions to one real, deterministically preprocessed dataset clip:

```powershell
python scripts\inspect_snr.py
```

The default command uses seeded white noise only to verify the SNR mathematics
and generate listening examples. Supply a real external-noise file with:

```powershell
python scripts\inspect_snr.py --noise-file data\external_noise\example.wav
python scripts\inspect_snr.py --split validation --sample-index 10 --seed 2025
python scripts\inspect_snr.py --help
```

Outputs default to `results/snr_inspection/`: one PCM-24 WAV per condition,
`condition_results.csv`, `summary.json`, and `waveform_comparison.png`. One
shared peak-safety factor preserves SNR and relative amplitude across the WAVs.

`train.py` is the shared baseline/augmented training entry point. A tiny
real-audio integration run is:

```powershell
python scripts\train.py configs\experiment\development.yaml --epochs 1 --max-train-samples 8 --max-validation-samples 4 --num-workers 0 --run-label smoke
```

A full configured run removes the smoke limits:

```powershell
python scripts\train.py configs\experiment\cnn_baseline.yaml --run-label run01
```

The command creates isolated experiment, checkpoint, log, and metric directories.
Use `python scripts\train.py --help` for model-independent overrides.

The reproducible dataset-acquisition flow downloads the public Hugging Face
Parquet snapshot, then reconstructs its embedded WAV bytes into the configured
layout:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='danavery/urbansound8K', repo_type='dataset', local_dir='data/raw/UrbanSound8K_parquet')"
python scripts/prepare_urbansound8k_from_parquet.py
python scripts/inspect_dataset.py
```

`download_urbansound8k.py` is a resumable multi-range fallback for Kaggle's full
archive. It validates archive size and ZIP CRCs before accepting the download.

Full checkpoint benchmarking and robustness orchestration will be added when the
six expensive research runs begin. The reusable metrics and robustness-analysis
modules required by those commands are already implemented.
