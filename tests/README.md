# Tests

The current suite tests configuration loading and validation, path resolution,
device selection, deterministic seeds, logging, experiment IDs, directory isolation,
reproducibility snapshots, UrbanSound8K metadata integrity, fold filtering, missing
files, corrupt audio, header inspection, result serialization, and validated
channels-first waveform loading. Deterministic mono conversion tests cover channel
averaging, input immutability, dtype preservation, gradient flow, and invalid
waveform rejection. The suite also covers up/downsampling, exact padding and
cropping, seeded random crops, log-Mel shapes, silence stability, composed
preprocessing, reusable MFCC extraction, lazy dataset loading, and DataLoader
collation. The current suite contains 60 passing tests. Run it verbosely
with:

```bash
python -m pytest -v
```

Later phases will add SNR mixing, deterministic corruption, augmentation switches,
model output dimensions, and end-to-end smoke-training tests.
