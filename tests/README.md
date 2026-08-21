# Tests

The current suite tests configuration loading and validation, path resolution,
device selection, deterministic seeds, logging, experiment IDs, directory isolation,
reproducibility snapshots, UrbanSound8K metadata integrity, fold filtering, missing
files, corrupt audio, header inspection, result serialization, and validated
channels-first waveform loading. Deterministic mono conversion tests cover channel
averaging, input immutability, dtype preservation, gradient flow, and invalid
waveform rejection. The suite also covers up/downsampling, exact padding and
cropping, seeded random crops, log-Mel shapes, silence stability, composed
preprocessing, reusable MFCC extraction, lazy dataset loading, DataLoader
collation, exact target-SNR mixing, silence policies, short/long noise handling,
recursive noise discovery, deterministic per-sample corruption, independently
switchable augmentation, validation/test augmentation bypass, CNN/CRNN/ResNet18
output shapes, loss/backpropagation, trainer history, checkpoints, classification
metrics, result storage, and robustness summaries. The current suite contains
110 passing tests. Run it verbosely
with:

```bash
python -m pytest -v
```

The real-audio smoke run is intentionally a CLI verification rather than a unit
test because the downloaded UrbanSound8K files are not committed to Git.
