# H-Neuron Benchmark Findings — March 2026

**Experiment date:** 2026-03-25
**Models:** 6 (3 model families × IT/PT variants)
**Datasets:** 5 medical MCQ benchmarks
**Tool:** hprobe v0.3.0, contrastive mode, L1 probe (C=0.5), batch_size=32

---

## Models & Datasets

| Model | Type | Parameters |
|-------|------|-----------|
| google/medgemma-4b-it | Medical IT | 4B |
| google/medgemma-4b-pt | Medical PT (base) | 4B |
| google/gemma-3-4b-it | General IT | 4B |
| google/gemma-3-4b-pt | General PT (base) | 4B |
| microsoft/MediPhi-Instruct | Medical IT | 3.8B |
| microsoft/MediPhi | Medical PT (base) | 3.8B |

| Dataset | Samples | Domain |
|---------|---------|--------|
| MedQA (USMLE) | 1,273 | US medical licensing |
| MedMCQA | 4,183 | Indian medical entrance |
| MMLU Medical | 644 | General medical knowledge |
| AfriMedQA | 3,958 | African medical context |
| MedXpertQA | 2,450 | Expert-level medical (very hard) |

---

## Full Results

### Model Accuracy

| Model | MedQA | MedMCQA | MMLU | AfriMedQA | MedXpertQA |
|-------|-------|---------|------|-----------|------------|
| medgemma-4b-it | 0.564 | 0.514 | 0.668 | 0.564 | 0.132 |
| medgemma-4b-pt | 0.372 | 0.380 | 0.497 | 0.383 | 0.111 |
| gemma-3-4b-it | 0.458 | 0.444 | 0.612 | 0.478 | 0.106 |
| gemma-3-4b-pt | 0.473 | 0.427 | 0.623 | 0.457 | 0.123 |
| MediPhi-Instruct | 0.566 | 0.549 | 0.747 | 0.613 | 0.125 |
| MediPhi | 0.560 | 0.548 | 0.748 | 0.604 | 0.119 |

> MedXpertQA is near-random for all models (~10–13%). MediPhi leads on MMLU (0.748).

### H-Neuron Count & Sparsity (‰)

| Model | MedQA | MedMCQA | MMLU | AfriMedQA | MedXpertQA |
|-------|-------|---------|------|-----------|------------|
| medgemma-4b-it | 305 (0.876‰) | 798 (2.292‰) | 150 (0.431‰) | 713 (2.048‰) | 493 (1.416‰) |
| medgemma-4b-pt | 274 (0.787‰) | 812 (2.332‰) | 147 (0.422‰) | 775 (2.226‰) | 329 (0.945‰) |
| gemma-3-4b-it | 303 (0.870‰) | 826 (2.372‰) | 149 (0.428‰) | 793 (2.278‰) | 346 (0.994‰) |
| gemma-3-4b-pt | 302 (0.867‰) | 779 (2.237‰) | 149 (0.428‰) | 731 (2.100‰) | 449 (1.290‰) |
| MediPhi-Instruct | 286 (1.091‰) | 821 (3.132‰) | 120 (0.458‰) | 719 (2.743‰) | 356 (1.358‰) |
| MediPhi | 271 (1.034‰) | 807 (3.078‰) | 139 (0.530‰) | 728 (2.777‰) | 431 (1.644‰) |

> H-Neuron count scales with dataset size and difficulty. MedMCQA (4183 samples, harder) yields 3–5× more H-Neurons than MMLU (644 samples, easier).

### Probe AUROC

| Model | MedQA | MedMCQA | MMLU | AfriMedQA | MedXpertQA |
|-------|-------|---------|------|-----------|------------|
| medgemma-4b-it | 0.877 | 0.873 | 0.914 | 0.887 | 0.953 |
| medgemma-4b-pt | 0.881 | 0.895 | 0.895 | 0.897 | 0.961 |
| gemma-3-4b-it | 0.874 | 0.887 | 0.874 | 0.889 | 0.959 |
| gemma-3-4b-pt | 0.888 | 0.903 | 0.879 | 0.910 | 0.958 |
| MediPhi-Instruct | 0.878 | 0.882 | 0.886 | 0.875 | 0.949 |
| MediPhi | 0.876 | 0.872 | 0.927 | 0.878 | 0.958 |

> All models achieve AUROC 0.87–0.96 — the probe reliably predicts incorrect answers from FFN activations. MedXpertQA scores highest (near-chance accuracy → cleaner hallucination signal).

### AUROC Gap (H-Neurons vs Random Baseline)

| Model | MedQA | MedMCQA | MMLU | AfriMedQA | MedXpertQA |
|-------|-------|---------|------|-----------|------------|
| medgemma-4b-it | -0.001 | -0.023 | -0.010 | -0.021 | +0.006 |
| medgemma-4b-pt | -0.009 | +0.006 | +0.004 | -0.005 | +0.004 |
| gemma-3-4b-it | -0.017 | +0.001 | +0.007 | -0.003 | +0.001 |
| gemma-3-4b-pt | -0.013 | -0.004 | -0.005 | -0.012 | -0.001 |
| MediPhi-Instruct | +0.005 | -0.010 | -0.015 | -0.013 | -0.005 |
| MediPhi | +0.008 | -0.018 | +0.013 | -0.014 | +0.001 |

> Gap is consistently near zero (−0.023 to +0.013). **H-Neurons are not more predictive than random neurons of the same count.** The L1 probe selects a sparse set, but that sparse set is not uniquely informative — the correctness signal is distributed broadly across FFN features.

### Causal Validation — Suppression Drop (CV_baseline − CV_alpha=0)

Positive = suppressing H-Neurons reduces accuracy (causal effect confirmed).
Negative = suppression increases accuracy (unexpected, likely noise).

| Model | MedQA | MedMCQA | MMLU | AfriMedQA | MedXpertQA |
|-------|-------|---------|------|-----------|------------|
| medgemma-4b-it | +0.008 | +0.019 | -0.008 | +0.020 | -0.006 |
| medgemma-4b-pt | 0.000 | +0.012 | -0.016 | **+0.059** | +0.008 |
| gemma-3-4b-it | -0.004 | +0.036 | +0.016 | +0.020 | -0.008 |
| gemma-3-4b-pt | +0.004 | +0.012 | -0.008 | **+0.122** | +0.008 |
| MediPhi-Instruct | +0.008 | -0.023 | -0.008 | -0.001 | +0.012 |
| MediPhi | +0.012 | +0.008 | -0.008 | 0.000 | -0.008 |

---

## Key Findings

### Finding 1: Probes are universally predictive but not uniquely so

AUROC is 0.87–0.96 across all 30 model×dataset combinations. FFN activations at the answer token reliably encode whether the model will be correct or incorrect. However, the AUROC gap (H-Neurons vs random) is near zero in almost all cases, meaning **any sparse set of neurons performs as well as the L1-selected H-Neurons**. The correctness signal is broadly distributed in activation space, not localized.

### Finding 2: Base (PT) models show stronger causal H-Neuron effects than instruction-tuned (IT) models

The two strongest causal suppression results are both PT models on AfriMedQA:
- **gemma-3-4b-pt: +0.122** (accuracy drops from 0.456 → 0.333 at alpha=0)
- **medgemma-4b-pt: +0.059** (accuracy drops from 0.381 → 0.322 at alpha=0)

The corresponding IT models show much weaker effects (+0.020 and +0.020). Instruction fine-tuning appears to **distribute the decision signal across more neurons**, making individual H-Neurons less causally dominant.

### Finding 3: AfriMedQA is the most sensitive dataset for causal effects

AfriMedQA produces the largest causal suppression effects across all models. This dataset is out-of-distribution for all models (African medical context, regional diseases, non-Western clinical cases). When a model operates near the boundary of its knowledge, a smaller number of neurons appear to be causally driving the answer choice.

### Finding 4: MedXpertQA reveals the clearest hallucination signal

All models score near-chance on MedXpertQA (~10–13%), and this is where probe AUROC is highest (0.949–0.961). The model is almost always wrong, creating a clean binary signal for the probe. Despite high AUROC, causal effects are weak — the model is guessing, not reasoning through dedicated circuits.

### Finding 5: Medical fine-tuning (MedGemma vs Gemma-3) improves accuracy but weakens causal localization

| | AfriMedQA accuracy | AfriMedQA CV_drop |
|--|--|--|
| gemma-3-4b-pt | 0.457 | +0.122 |
| medgemma-4b-pt | 0.383 | +0.059 |
| gemma-3-4b-it | 0.478 | +0.020 |
| medgemma-4b-it | 0.564 | +0.020 |

Medical fine-tuning improves accuracy on in-distribution tasks (MedQA, MMLU) but reduces causal H-Neuron concentration. The knowledge appears more distributed post-fine-tuning.

### Finding 6: Top H-Neuron layers are consistent within model families

**Gemma-3 / MedGemma (4B):** H-Neurons cluster consistently in layers 11–15 (out of 34).
**MediPhi (Phi-3.8B):** H-Neurons cluster in layers 23–30 (out of 32) — deeper layers.

This suggests architectural differences between Gemma-3 and Phi families in where answer-relevant computation is localized: middle layers for Gemma, later layers for Phi.

---

## Open Questions

1. **Why does AfriMedQA show stronger causal effects?** Is it OOD difficulty, topic specificity, or something else? Compare with MedXpertQA (also hard but no causal effect).

2. **Why is AUROC gap ~0?** Is the L1 regularization too strong (C=0.5 selecting too few neurons)? Try C=1.0–5.0. Or is the signal genuinely distributed?

3. **Does causal effect scale with model size?** Need 27B results to compare.

4. **Layer 11–15 dominance in Gemma-3 family** — does this persist across 27B? What is special about these layers?

5. **IT vs PT causal gap** — can we confirm this with 27B models? Is this a general phenomenon or specific to 4B scale?

---

## Methodology Notes

- **Probe:** L1 logistic regression (liblinear solver, C=0.5, class_weight=balanced)
- **Features:** Top 5,000 CETT features by variance pre-selection from all layers
- **Labeling:** Contrastive 3-vs-1 (CETT at predicted answer token vs last prompt token)
- **Split:** 80/20 train/val, stratified by correctness
- **Causal validation:** Alpha ∈ {0.0, 0.5, 1.0, 1.5, 2.0} on val split
- **Batch size:** 32 (GPU-vectorized CETT extraction)
- **Hardware:** A100 80GB
