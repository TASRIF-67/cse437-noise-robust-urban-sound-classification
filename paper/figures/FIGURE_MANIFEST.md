# Figure Manifest

Status meanings:

- **READY**: the canonical file exists in the paper directory.
- **PLACEHOLDER**: keep the canonical path in LaTeX; the placeholder macro will
  render a box until the real asset is created.
- **APPENDIX**: ready or planned, but normally secondary to the main narrative.

Never create a blank or invalid PNG/PDF to represent a placeholder. A planned
figure must remain absent until a valid asset exists.

## Phase 2: dataset and audio understanding

| Canonical filename | Status | Suggested label | Suggested placement |
|---|---|---|---|
| `fig_p02_01_class_distribution.png` | READY | `fig:class-distribution` | Dataset section |
| `fig_p02_02_fold_distribution.png` | READY | `fig:fold-distribution` | Appendix |
| `fig_p02_03_duration_distribution.png` | READY | `fig:duration-distribution` | Appendix |
| `fig_p02_04_recording_characteristics.png` | READY | `fig:recording-characteristics` | Appendix |
| `fig_p02_05_representative_waveforms.png` | READY | `fig:representative-waveforms` | Appendix |
| `fig_p02_06_representative_log_mel.png` | READY | `fig:representative-log-mel` | Dataset/method section |
| `fig_p02_07_waveform_logmel_mfcc.png` | READY | `fig:audio-representations` | Appendix |
| `fig_p02_08_snr_waveform_comparison.png` | READY | `fig:snr-waveform-comparison` | Method section |

## Phase 3: training and robustness experiment

| Canonical filename | Status | Suggested label | Suggested placement |
|---|---|---|---|
| `fig_p03_01_experiment_pipeline.pdf` | PLACEHOLDER | `fig:experiment-pipeline` | Method section |
| `fig_p03_02_training_validation_curves.png` | PLACEHOLDER | `fig:training-curves` | Experimental setup/appendix |
| `fig_p03_03_macro_f1_robustness_curves.png` | READY | `fig:macro-f1-robustness` | Main results |
| `fig_p03_04_accuracy_robustness_curves.png` | READY, APPENDIX | `fig:accuracy-robustness` | Results/appendix |
| `fig_p03_05_macro_f1_augmentation_effects.png` | READY | `fig:augmentation-effects` | Main results |
| `fig_p03_06_macro_f1_robustness_auc.png` | READY | `fig:robustness-auc` | Main results |
| `fig_p03_07_zero_db_per_class_effects.png` | READY | `fig:zero-db-class-effects` | Results/discussion |
| `fig_p03_08_confusion_matrix_comparison.png` | PLACEHOLDER | `fig:confusion-comparison` | Results/discussion |

## Phase 4: statistical and error analysis

| Canonical filename | Status | Suggested label | Suggested placement |
|---|---|---|---|
| `fig_p04_01_augmentation_error_transitions.png` | PLACEHOLDER | `fig:error-transitions` | Error analysis |
| `fig_p04_02_paired_bootstrap_confidence_intervals.png` | PLACEHOLDER | `fig:bootstrap-confidence-intervals` | Statistical analysis |
| `fig_p04_03_class_confusion_summary.png` | PLACEHOLDER | `fig:class-confusion-summary` | Error analysis |
| `fig_p04_04_failure_case_examples.png` | PLACEHOLDER | `fig:failure-examples` | Discussion/appendix |
| `fig_p04_05_final_model_comparison.png` | PLACEHOLDER | `fig:final-model-comparison` | Conclusion/summary |

## Recommended main-paper subset

Use a compact selection unless the university template permits many figures:

1. `fig_p02_01_class_distribution.png`
2. `fig_p02_06_representative_log_mel.png`
3. `fig_p03_01_experiment_pipeline.pdf`
4. `fig_p03_03_macro_f1_robustness_curves.png`
5. `fig_p03_05_macro_f1_augmentation_effects.png`
6. `fig_p03_06_macro_f1_robustness_auc.png`
7. `fig_p03_08_confusion_matrix_comparison.png`

Move secondary figures to the appendix rather than deleting their manifest
entries or changing their filenames.
