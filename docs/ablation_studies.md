# Ablation Studies

This document records the recommended ablation experiments to quantify contribution of each method component.

Suggested ablations (run nested CV for each):

1. Baseline: Mean pooled ESM embeddings + LogisticRegression
2. + Multi-head Attention Pooling
3. + XGBoost classifier
4. + Per-drug dropout schedule
5. + ESM layer fusion (13,23,33)
6. + Residue Transformer
7. + Learnable position weights
8. + Stacking ensemble

For each ablation, report:
- Mean test AUC (5-fold outer CV)
- Val→Test AUC gap
- ECE and Brier score
- Per-drug AUC table

Store ablation outputs under `results/ablation/` with naming convention `ablation_<step>_drug_wise_performance.csv`.
