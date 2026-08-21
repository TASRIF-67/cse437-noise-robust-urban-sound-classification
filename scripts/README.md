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

Later phases will add `run_eda.py`, `train.py`, `evaluate.py`, and
`evaluate_robustness.py`. Scripts delegate reusable work to the installable package
instead of containing the core pipeline themselves.
