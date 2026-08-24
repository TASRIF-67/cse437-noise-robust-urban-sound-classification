# Phase 3 Project Log and Paper Evidence Guide

Project: **Noise-Robust Urban Sound Classification: Evaluating the Effect of
Background Noise and Audio Augmentation on Deep Learning Models**

Phase status: **Core Phase 3 completed**

Phase 3 completion date: **2026-08-24**

This document is the permanent record of the Phase 3 implementation,
experiments, results, generated evidence, and remaining report-preparation work.
All reported test results use the held-out UrbanSound8K fold 10. Smoke-run
metrics are excluded.

## 1. Research objective

The project investigates how controlled background noise affects urban sound
classification and whether training-time audio augmentation improves robustness.

The two research questions are:

1. Does training-time audio augmentation reduce the loss of classification
   performance when test audio is corrupted by background noise?
2. Is the augmentation effect consistent across CNN, CRNN, and ResNet18
   architectures?

The hypotheses were:

- H1: augmented models will show smaller accuracy and macro-F1 degradation as
  the test SNR decreases.
- H2: the size of the robustness improvement will vary by architecture.

## 2. Phase 3 scope completed

Phase 3 implemented and executed the complete experimental methodology:

- Configurable baseline and augmented training conditions.
- Three common-interface classifiers: CNN, CRNN with BiGRU, and ResNet18.
- Six primary full training runs:
  - CNN baseline
  - CNN augmented
  - CRNN baseline
  - CRNN augmented
  - ResNet18 baseline
  - ResNet18 augmented
- Shared training, validation, checkpointing, early stopping, logging, and
  TensorBoard infrastructure.
- Atomic recovery checkpoints and exact same-run resume after power loss.
- Controlled held-out-noise evaluation at clean, 20 dB, 10 dB, and 0 dB.
- Full 6-model by 4-condition matrix: 24 primary test results.
- Accuracy, macro precision, macro recall, macro F1, per-class metrics,
  confusion matrices, and per-sample predictions.
- Accuracy/macro-F1 drops, retention, SNR slopes, and normalized SNR AUC.
- Cross-model aggregation, augmentation-effect tables, rankings, and five
  final robustness figures.
- Strict evaluation checks for full-run best checkpoints, fold isolation,
  noise-split isolation, matching protocols, identical samples, and identical
  deterministic corruptions across models.
- Automated verification: 121 tests passed.

## 3. Dataset and experimental split

Dataset: UrbanSound8K

| Item | Value |
|---|---:|
| Total samples | 8,732 |
| Number of classes | 10 |
| Training folds | 1-8 |
| Validation fold | 9 |
| Test fold | 10 |
| Training samples | 7,079 |
| Validation samples | 816 |
| Test samples | 837 |

The official folds were preserved instead of constructing a random split. This
reduces leakage from related source recordings appearing in multiple subsets.
Fold 10 remained untouched during model selection and was used only for final
evaluation.

Classes:

1. air_conditioner
2. car_horn
3. children_playing
4. dog_bark
5. drilling
6. engine_idling
7. gun_shot
8. jackhammer
9. siren
10. street_music

## 4. Shared audio preprocessing

Every architecture received the same standardized input.

| Setting | Value |
|---|---|
| Target sample rate | 22,050 Hz |
| Clip duration | 4.0 seconds |
| Channels | Mono |
| Short clips | Trailing zero padding |
| Training crop | Seeded random crop |
| Validation/test crop | Deterministic center crop |
| Representation | Log-Mel spectrogram |
| FFT/window length | 1,024 samples |
| Hop length | 512 samples |
| Mel bands | 64 |
| Power | 2.0 |
| Dynamic-range limit | 80 dB |
| Mel scale | HTK |
| Normalization | Per-example standardization |
| Model input shape | [batch, 1, 64, 173] |

The processing order was:

raw waveform -> mono conversion -> resampling -> four-second normalization ->
training-only waveform augmentation -> Log-Mel conversion -> training-only
spectrogram augmentation -> standardization -> model.

Validation and test data bypass all training augmentation.

## 5. Training conditions

### 5.1 Baseline condition

All robustness-oriented waveform and spectrogram augmentations were disabled.

### 5.2 Augmented condition

Only training samples received augmentation.

| Augmentation | Probability | Range/limit |
|---|---:|---|
| Time shift | 0.5 | Maximum 20% of clip length |
| Random gain | 0.5 | -6 dB to +6 dB |
| Background noise | 0.5 | Random SNR from 0 to 20 dB |
| Frequency mask | 0.5 | Up to 8 Mel bins |
| Time mask | 0.5 | Up to 16 frames |
| Pitch shift | Disabled | Not used |
| Time stretch | Disabled | Not used |

Training noise came only from:

`data/external_noise/MS-SNSD/noise_train`

Final evaluation noise came only from:

`data/external_noise/MS-SNSD/noise_test`

This separation prevents evaluation-noise leakage into training.

## 6. Model implementations

All models accept a one-channel Log-Mel tensor and return ten class logits.

| Model | Main design | Trainable parameters |
|---|---|---:|
| CNN | Three convolutional stages with channels 32/64/128, adaptive pooling, dropout 0.3 | 94,186 |
| CRNN | CNN channels 32/64/128, one bidirectional GRU layer, hidden size 128, dropout 0.3 | 293,610 |
| ResNet18 | Single-channel torchvision ResNet18, non-pretrained, dropout 0.2 | 11,175,370 |

ResNet18 was intentionally run without ImageNet pretraining. Therefore, its
results represent training from scratch on audio spectrograms and do not include
external image-domain knowledge.

## 7. Shared training configuration

| Setting | Value |
|---|---|
| Random seed | 42 |
| Deterministic mode | Enabled |
| Maximum epochs | 50 |
| Batch size | 16 for CNN/CRNN; 8 for ResNet18 |
| DataLoader workers | 2 |
| Mixed precision | Enabled on CUDA |
| Gradient accumulation | 1 |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor/patience | 0.5 / 3 epochs |
| Selection metric | Validation macro F1 |
| Early-stopping patience | 8 epochs |
| Checkpoints | Atomic best.pt and last.pt |
| Experiment tracking | Logs, CSV history, configuration/environment snapshots, TensorBoard |

Macro F1 was used for checkpoint selection because UrbanSound8K is not perfectly
class-balanced and accuracy alone can overrepresent larger classes.

## 8. Hardware and software environment

| Item | Recorded value |
|---|---|
| Operating system | Windows 10, build 19045 |
| Python | 3.13.1 |
| PyTorch | 2.11.0+cu126 |
| CUDA runtime | 12.6 |
| cuDNN | 91002 |
| GPU | NVIDIA GeForce GTX 1660 Ti |
| GPU memory | 6,441,992,192 bytes (approximately 6 GB) |
| GPU compute capability | 7.5 |
| Environment | Repository-local `.venv` |

## 9. Completed training runs

The best epoch is the epoch with the highest validation macro F1. The process
continued until early stopping, so epochs completed can be later than the best
epoch.

| Experiment | Parameters | Best epoch | Epochs completed | Best validation macro F1 | Validation accuracy | Runtime | Resumed after interruption |
|---|---:|---:|---:|---:|---:|---:|---|
| CNN baseline | 94,186 | 21 | 29 | 0.8305 | 0.8162 | 63.7 min | Yes |
| CNN augmented | 94,186 | 21 | 29 | 0.8110 | 0.8002 | 174.0 min | No |
| CRNN baseline | 293,610 | 13 | 21 | 0.8015 | 0.7904 | 64.3 min | No |
| CRNN augmented | 293,610 | 15 | 23 | 0.8010 | 0.7941 | 139.6 min | No |
| ResNet18 baseline | 11,175,370 | 9 | 17 | 0.7760 | 0.7696 | 73.5 min | No |
| ResNet18 augmented | 11,175,370 | 20 | 28 | 0.8086 | 0.7929 | 90.2 min | Yes |

Sum of the completed-process runtimes recorded in the six final summaries:
approximately 10.1 hours. This is not a precise wall-clock total because time
spent inside interrupted, unfinished epochs may not be retained by resumed-run
summaries.

The successful CNN-baseline and ResNet18-augmented resumes demonstrated that
the recovery system could continue from the latest complete epoch after local
electricity interruptions.

## 10. Final robustness-evaluation protocol

Each selected `best.pt` checkpoint was evaluated once on all 837 test samples
under four conditions:

1. clean
2. 20 dB SNR
3. 10 dB SNR
4. 0 dB SNR

Important controls:

- Test split: fold 10 only.
- Evaluation sample limit: none.
- Number of held-out noise files: 51.
- Evaluation noise source: MS-SNSD `noise_test`.
- Corruption seed: 2025, independent from the training seed.
- Noise scaling: mean-square signal/noise power.
- The same sample receives the same selected noise file and segment across
  models and SNR conditions.
- Requested and achieved SNR, noise path, selection seed, target, prediction,
  sample ID, fold, and condition were saved for every prediction.
- The evaluator rejected `last.pt`, smoke checkpoints, mismatched manifests,
  and overlapping training/evaluation noise banks.
- The aggregator verified identical sample IDs, target order, corruption
  assignments, and achieved SNR values across all six model evaluations.

## 11. Metric definitions

- Accuracy: correctly classified samples divided by all samples.
- Macro precision/recall/F1: unweighted mean of the ten per-class values.
- Accuracy drop: clean accuracy minus condition accuracy.
- Macro-F1 drop: clean macro F1 minus condition macro F1.
- Retention: condition metric divided by the corresponding clean metric.
- Slope per dB: ordinary least-squares slope over the finite 0, 10, and 20 dB
  points.
- Normalized SNR AUC: trapezoidal area over the 0-20 dB noisy curve divided by
  the observed SNR range. It remains on the same 0-1 scale as the metric.

The normalized AUC excludes the clean point because clean audio does not have a
finite SNR.

## 12. Complete fold-10 test results

### 12.1 Accuracy

| Architecture | Training | Clean | 20 dB | 10 dB | 0 dB |
|---|---|---:|---:|---:|---:|
| CNN | Baseline | 0.7121 | 0.6703 | 0.6057 | 0.4504 |
| CNN | Augmented | 0.7288 | 0.7168 | 0.6989 | 0.5496 |
| CRNN | Baseline | 0.7838 | 0.7491 | 0.7085 | 0.4994 |
| CRNN | Augmented | 0.7419 | 0.7491 | 0.7384 | 0.6141 |
| ResNet18 | Baseline | 0.7395 | 0.6977 | 0.5938 | 0.4421 |
| ResNet18 | Augmented | 0.7575 | 0.7575 | 0.7264 | 0.6081 |

### 12.2 Macro F1

| Architecture | Training | Clean | 20 dB | 10 dB | 0 dB |
|---|---|---:|---:|---:|---:|
| CNN | Baseline | 0.7323 | 0.6624 | 0.5857 | 0.4230 |
| CNN | Augmented | 0.7469 | 0.7270 | 0.7071 | 0.5481 |
| CRNN | Baseline | **0.8037** | 0.7632 | 0.7134 | 0.4952 |
| CRNN | Augmented | 0.7574 | 0.7634 | **0.7535** | 0.6186 |
| ResNet18 | Baseline | 0.7617 | 0.7050 | 0.5774 | 0.4249 |
| ResNet18 | Augmented | 0.7781 | **0.7783** | 0.7432 | **0.6384** |

Best macro F1 by condition:

- Clean: CRNN baseline, 0.8037.
- 20 dB: ResNet18 augmented, 0.7783.
- 10 dB: CRNN augmented, 0.7535.
- 0 dB: ResNet18 augmented, 0.6384.

## 13. Robustness summary

| Model | Clean macro F1 | 0 dB macro F1 | Macro-F1 SNR AUC | AUC rank |
|---|---:|---:|---:|---:|
| ResNet18 augmented | 0.7781 | 0.6384 | **0.7258** | 1 |
| CRNN augmented | 0.7574 | 0.6186 | 0.7222 | 2 |
| CNN augmented | 0.7469 | 0.5481 | 0.6724 | 3 |
| CRNN baseline | **0.8037** | 0.4952 | 0.6713 | 4 |
| ResNet18 baseline | 0.7617 | 0.4249 | 0.5712 | 5 |
| CNN baseline | 0.7323 | 0.4230 | 0.5642 | 6 |

ResNet18 augmented was the best overall robustness model. CRNN baseline was the
best clean model, demonstrating that clean accuracy and noise robustness are not
the same objective.

## 14. Measured augmentation effects

Values below are augmented minus baseline for the same architecture.

| Architecture | Clean accuracy delta | Clean macro-F1 delta | 0 dB accuracy delta | 0 dB macro-F1 delta | Macro-F1 AUC delta |
|---|---:|---:|---:|---:|---:|
| CNN | +0.0167 | +0.0146 | +0.0992 | +0.1252 | +0.1082 |
| CRNN | -0.0418 | -0.0464 | +0.1147 | +0.1234 | +0.0509 |
| ResNet18 | +0.0179 | +0.0163 | +0.1661 | +0.2135 | +0.1546 |

Interpretation:

- Augmentation improved the noisy-curve AUC for every architecture.
- The largest AUC improvement occurred for ResNet18: +0.1546.
- At 0 dB, macro F1 improved by 0.1252 for CNN, 0.1234 for CRNN, and 0.2135
  for ResNet18.
- CNN and ResNet18 also gained slightly on clean data.
- CRNN lost 0.0464 clean macro F1 but gained 0.1234 at 0 dB, showing a
  clean-versus-robustness trade-off.
- The augmented variants occupy all top three robustness-AUC ranks.

## 15. Per-class findings at 0 dB

For ResNet18, augmentation improved 0 dB F1 for all ten classes. Its largest
gains were:

| Class | Baseline F1 | Augmented F1 | Delta |
|---|---:|---:|---:|
| gun_shot | 0.2703 | 0.8358 | +0.5656 |
| car_horn | 0.6122 | 0.8852 | +0.2730 |
| dog_bark | 0.4881 | 0.7253 | +0.2372 |
| engine_idling | 0.1308 | 0.3289 | +0.1981 |
| siren | 0.3883 | 0.5467 | +0.1583 |

The augmented ResNet18 still found engine_idling, air_conditioner, siren, and
street_music relatively difficult at 0 dB. These classes should be emphasized
in the later confusion/error analysis.

## 16. Answers to the research questions

### RQ1

**Yes, within this held-out-fold experiment.** Training-time augmentation
reduced noise-related degradation for CNN, CRNN, and ResNet18. Every augmented
model had a higher normalized macro-F1 SNR AUC and a higher 0 dB macro F1 than
its architecture-matched baseline.

### RQ2

**No, the magnitude and clean-data trade-off were not identical across
architectures.** ResNet18 received the largest overall benefit. CNN improved
more moderately. CRNN became more robust but sacrificed clean macro F1.

The findings support both H1 and H2 for the completed experiment. They should
not yet be described as statistically significant because confidence intervals
and repeated training seeds have not been performed.

## 17. Report tables to include

Recommended main-paper tables:

1. Dataset split and sample counts.
2. Shared preprocessing and training configuration.
3. Model architecture summary and parameter counts.
4. Complete 6-by-4 macro-F1 matrix, optionally including accuracy in the same
   table or a separate table.
5. Clean, 0 dB, normalized AUC, and augmentation-delta summary.
6. Selected 0 dB per-class results or largest improvements.

Place the full per-class precision/recall/F1 tables in an appendix because 240
rows are too large for the main paper.

## 18. Figures available now

### Phase 3 result figures

The following are ready for direct inclusion in LaTeX:

1. `results/analysis/final_robustness/figures/macro_f1_robustness_curves.png`
   - Primary result figure; compares clean, 20 dB, 10 dB, and 0 dB macro F1.
2. `results/analysis/final_robustness/figures/accuracy_robustness_curves.png`
   - Accuracy counterpart to the macro-F1 curve.
3. `results/analysis/final_robustness/figures/macro_f1_augmentation_effects.png`
   - Direct baseline-versus-augmented effect at every condition.
4. `results/analysis/final_robustness/figures/macro_f1_robustness_auc.png`
   - Compact overall robustness ranking.
5. `results/analysis/final_robustness/figures/zero_db_per_class_augmentation_effects.png`
   - Class-specific behavior under the hardest noise condition.

### Earlier project figures useful in the paper

- `results/figures/eda/01_class_distribution.png`
- `results/figures/eda/02_fold_distribution.png`
- `results/figures/eda/03_duration_distribution.png`
- `results/figures/eda/04_recording_characteristics.png`
- `results/figures/eda/05_representative_waveforms.png`
- `results/figures/eda/06_representative_log_mel.png`
- `results/figures/eda/07_waveform_logmel_mfcc.png`
- `results/snr_inspection/waveform_comparison.png`

Recommended main-text selection:

- One compact dataset figure, preferably class distribution.
- One representative waveform/Log-Mel figure.
- One method/pipeline diagram.
- Macro-F1 robustness curves.
- Macro-F1 augmentation effects or robustness AUC.
- One 0 dB per-class or confusion-matrix figure.

Avoid including every available plot in the main paper. Put secondary figures
in an appendix.

The `results/` directory is intentionally Git-ignored because generated
artifacts can be large. Before submitting or moving the paper to another
machine, copy every selected PNG and supporting CSV/JSON file into the LaTeX
project or a separate archived result package. Do not assume a Git clone will
contain the generated figures.

Stable copies of the selected figures, the canonical naming manifest, and a
missing-file LaTeX placeholder macro are maintained under `paper/`. New paper
sections should follow `paper/CHATGPT_LATEX_INSTRUCTIONS.md`.

## 19. Figures still recommended before submission

These do not make the core Phase 3 incomplete, but they would strengthen the
paper:

1. A clean method diagram from waveform through preprocessing, augmentation,
   model, and four-condition evaluation.
2. Training/validation loss and validation macro-F1 curves for the six runs.
3. Normalized confusion-matrix heatmaps for the most informative comparisons,
   especially baseline versus augmented ResNet18 at clean and 0 dB.
4. A sample-level error-analysis figure showing the most frequent class
   confusions and predictions corrected by augmentation.
5. Paired bootstrap 95% confidence intervals for model/augmentation differences.

## 20. Minimal LaTeX figure pattern

Copy the selected PNG files into the LaTeX project's figure directory or point
the graphics path to the repository.

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/macro_f1_robustness_curves.png}
    \caption{Macro-F1 performance under controlled background-noise
    conditions. Each model is trained once and evaluated using identical
    deterministic test corruptions.}
    \label{fig:macro_f1_robustness}
\end{figure}
```

Use `booktabs` for tables and `siunitx` if decimal alignment is needed.
Report metrics consistently to three or four decimal places.

## 21. Suggested paper structure

1. Abstract
   - Problem, dataset, six models, four test conditions, and strongest result.
2. Introduction
   - Motivation, research gap, RQ1/RQ2, and contributions.
3. Related work
   - Environmental sound classification, Log-Mel models, audio augmentation,
     and noise-robust evaluation.
4. Dataset and exploratory analysis
   - UrbanSound8K, folds, class balance, audio characteristics.
5. Methodology
   - Preprocessing, models, augmentation, training, controlled noise, metrics.
6. Experimental setup
   - Split, environment, hyperparameters, checkpoint selection, reproducibility.
7. Results
   - Main 24-result table, robustness curves, AUC, per-class findings.
8. Discussion
   - Why augmentation helped, architecture differences, CRNN trade-off,
     practical implications.
9. Limitations and threats to validity
10. Conclusion and future work

## 22. Citation checklist

The final bibliography should include primary sources for:

- UrbanSound8K and its official fold organization.
- The original ResNet architecture.
- GRU-based recurrent networks.
- SpecAugment or the source used to motivate time/frequency masking.
- MS-SNSD and its intended noise/speech data usage.
- Any PyTorch, torchaudio, librosa, or scikit-learn software citations required
  by the university's citation policy.

Verify author names, titles, venues, years, and DOI/URLs from the original
papers or official dataset pages before adding the BibTeX records.

## 23. Limitations and claims boundary

The report must state these limitations honestly:

- Only one official fold was used for final testing; full 10-fold
  cross-validation was not performed.
- Each training configuration currently has one completed training seed.
- No bootstrap confidence intervals or hypothesis tests have been reported yet.
- Robustness was tested with one held-out noise collection containing 51 files.
- The noisy curve contains three finite SNR values: 0, 10, and 20 dB.
- ResNet18 was non-pretrained; pretrained or modern audio-transformer models
  were not compared.
- The study demonstrates a controlled empirical result on this split, not a
  universal claim about all environmental audio or all background noise.

For a course report, the current experiment is complete and meaningful. For a
publication-strength claim, repeated seeds, uncertainty estimates, broader
noise/dataset validation, and possibly full fold-based evaluation are advised.

## 24. Reproducibility commands

All commands assume that `.venv` is active.

Run the automated suite:

```powershell
python -m pytest -v
```

Train a new full model:

```powershell
python scripts\train.py configs\experiment\cnn_baseline.yaml --run-label run02
```

Resume an interrupted run:

```powershell
python scripts\train.py configs\experiment\cnn_baseline.yaml --resume checkpoints\<experiment-id>\last.pt
```

Evaluate a selected full-run checkpoint:

```powershell
python scripts\evaluate_robustness.py configs\experiment\cnn_baseline.yaml checkpoints\<experiment-id>\best.pt
```

Aggregate all six final evaluations:

```powershell
python scripts\aggregate_results.py
```

The completed aggregate command produced six model summaries, 24 condition
results, and 240 per-class results.

## 25. Evidence and artifact locations

| Evidence | Location |
|---|---|
| Experiment manifests | `configs/experiment/` |
| Component configurations | `configs/dataset/`, `configs/audio/`, `configs/augmentation/`, `configs/model/`, `configs/training/`, `configs/evaluation/` |
| Resolved run configurations/environments | `experiments/<experiment-id>/` |
| Best and recovery checkpoints | `checkpoints/<experiment-id>/` |
| Training histories and validation outputs | `results/metrics/<experiment-id>/` |
| Final condition-level evaluations | `results/robustness/<experiment-id>/test/` |
| Final aggregate tables | `results/analysis/final_robustness/` |
| Final robustness figures | `results/analysis/final_robustness/figures/` |
| EDA figures and tables | `results/figures/eda/`, `results/metrics/eda/` |
| Source implementation | `src/urban_sound_robustness/` |
| CLI entry points | `scripts/` |
| Automated tests | `tests/` |

Important aggregate files:

- `analysis_summary.json`
- `master_condition_metrics.csv`
- `model_robustness_summary.csv`
- `condition_augmentation_effects.csv`
- `robustness_augmentation_effects.csv`
- `master_per_class_metrics.csv`
- `per_class_augmentation_effects.csv`
- `aggregation_protocol.json`

## 26. Phase 3 completion statement

The core Phase 3 methodology is complete. All required models were implemented,
all six baseline/augmented configurations were trained, all selected checkpoints
were evaluated on the held-out fold under identical controlled noise, and the
24 primary results were validated and aggregated.

The next work is report-oriented analysis: confidence intervals, confusion/error
analysis, selection of final figures, and LaTeX paper writing. Those tasks
strengthen the interpretation and presentation but do not indicate that the
core Phase 3 implementation or experiment matrix is unfinished.
