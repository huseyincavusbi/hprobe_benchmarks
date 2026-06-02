# H-Neuron Experiment Results — June 2, 2026

## Setup
- **Model:** google/gemma-3-4b-it (gemma-3-4b, 34 layers, 348,160 features)
- **Pipeline:** Open-ended generation → consistency filter (10 draws) → CETT extraction → L1 logistic regression
- **Judge:** Rule-based (normalize_answer + uncertainty filter)
- **Configuration:** C=1, 2000 samples, batch-size 256, gen-batch-size 96, max-new-tokens 40

## Standalone Detection Results

| Dataset | H-Neurons | Ratio | AUROC | AUROC Gap | Bal Acc | Paper Gap |
|---------|-----------|-------|-------|-----------|---------|-----------|
| TriviaQA | 10 | 0.029‰ | 0.820 | **+0.404** | 0.731 | +0.149 |
| BioASQ | 8 | 0.023‰ | 0.974 | **+0.474** | 0.916 | +0.150 |
| NQ-Open | 4 | 0.011‰ | 0.628 | +0.128 | 0.629 | +0.110 |
| NonExist | — | — | — | — | — | — |

*NonExist standalone failed: 0 faithful samples. All 1000 responses judged hallucinatory. Transfer approach needed (train on TQA, test on NonExist).*

## Causal Validation

α-scaling of H-Neuron down_proj weights. α=0 (suppress), α=1 (baseline), α=2 (amplify).

| Dataset | HN | α=0 (suppress) | α=1 (baseline) | α=2 (amplify) | Random baseline |
|---------|-----|---------------|---------------|---------------|----------------|
| TriviaQA | 10 | 0.400 | 0.450 | 0.450 | 0.440 |
| BioASQ | 8 | 0.550 | 0.550 | 0.520 | 0.530 |
| NQ-Open | 4 | 0.510 | 0.540 | 0.540 | 0.550 |

*Note: Causal validation run on val_subset=100 samples. Random baseline uses same number of random neurons from same layers.*

*Previous result with top-k 5000 (C=10, 3 HN on BioASQ 200 samples): suppression improved correctness from 0.481 → 0.611 (+13pp). Current top-k 0 results show neutral/negative causal effects — more neurons (8-10) may be overly disruptive when all zeroed.*

## L2 Collinearity Check

Compares L1-selected neurons against L2 weight ranking. Low overlap = L1 picks not dominant in L2 → evidence of collinearity.

| Dataset | L1 HN | L2 overlap ratio | Weight concentration | L2 AUROC |
|---------|-------|-----------------|---------------------|----------|
| TriviaQA | 10 | 0.10 (1/10) | 0.002 | — |
| BioASQ | 8 | 0.25 (2/8) | 0.002 | — |
| NQ-Open | 4 | 0.00 (0/4) | 0.001 | — |

*Interpretation: L1-selected H-Neurons are not the dominant features in L2 ranking — suggests collinearity with other correlated features.*

## Bootstrap Stability Check

5 random seeds, bootstrapped L1 fits. Jaccard similarity between neuron sets.

| Dataset | Jaccard mean | Jaccard min | Jaccard max | Stable |
|---------|-------------|------------|------------|--------|
| TriviaQA | 0.47 | — | — | Partial |
| BioASQ | 0.69 | — | — | Moderate |
| NQ-Open | 0.45 | — | — | Low |

*Interpretation: L1 neuron selection is moderately unstable across bootstraps. BioASQ shows highest consistency.*

## Feature Correlation Analysis

For each H-Neuron, max |Pearson r| with any other feature in X_train. High correlation (>0.7) = from a correlated cluster, not unique.

| Dataset | H-Neurons | High corr (>0.7) | Max r values |
|---------|-----------|-----------------|--------------|
| TriviaQA | 10 | 7/10 | 0.613–0.937 |
| BioASQ | 8 | 8/8 | 0.758–0.973 |
| NQ-Open | 4 | 4/4 | 0.883–0.927 |

*Interpretation: 19 of 22 H-Neurons have high correlations with other features. H-Neurons are from correlated neuron clusters — they predict hallucination, but are not uniquely encoding it. This confirms Josh's L1 collinearity concern.*

## Previous Results (top-k 5000, C=1) — May 23

From earlier experiments with top-k 5000 variance pre-selection:

| Run | HN | AUROC | Gap | Bal Acc | Samples |
|-----|-----|-------|-----|---------|---------|
| TriviaQA standalone | 5 | 0.740 | +0.154 | 0.670 | 1134 |
| NQ-Open standalone | 3 | 0.690 | +0.332 | 0.647 | 702 |
| BioASQ standalone | 6 | 0.982 | +0.405 | 0.911 | 1228 |
| TQA→NQ-Open transfer | 5 | 0.745 | +0.245 | 0.691 | — |
| TQA→BioASQ transfer | 5 | 0.848 | +0.348 | 0.775 | — |

*These were at paper-comparable sample size. Transfers beat paper (paper: TQA→NQ-Open +0.110, TQA→BioASQ +0.150).*

## Summary

1. **Detection works** — AUROC gaps beat paper on all datasets (2.7-3.2x on TriviaQA and BioASQ)
2. **Causal validation is mixed** — top-k 5000 showed positive effect (+13pp), top-k 0 shows neutral/negative 
3. **L1 collinearity confirmed** — H-Neurons are from correlated clusters, not unique (19/22 have high correlation with other features)
4. **Stability is moderate** — Jaccard 0.45-0.69 across bootstraps
5. **NonExist needs transfer** — standalone fails (no faithful baseline), paper uses TQA→NonExist transfer
6. **HN count mystery** — 4-10 H-Neurons with all 348K features vs paper's ~35 at C=1