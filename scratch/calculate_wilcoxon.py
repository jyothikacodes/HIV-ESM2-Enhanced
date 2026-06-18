import pandas as pd
from scipy import stats
import numpy as np

print("--- Checking results/baseline_vs_enhanced.csv ---")
df1 = pd.read_csv("results/baseline_vs_enhanced.csv")
df1_valid = df1.dropna(subset=['enhanced_auroc', 'baseline_auroc'])
if len(df1_valid) >= 5:
    stat, pval = stats.wilcoxon(df1_valid['enhanced_auroc'], df1_valid['baseline_auroc'], alternative='greater')
    print(f"Wilcoxon alternative='greater' (enhanced > baseline): stat={stat}, p-value={pval:.4e}")
    stat, pval_two = stats.wilcoxon(df1_valid['enhanced_auroc'], df1_valid['baseline_auroc'])
    print(f"Wilcoxon two-sided: stat={stat}, p-value={pval_two:.4e}")

print("\n--- Checking results/improved_pipeline/baseline_comparison.csv ---")
df2 = pd.read_csv("results/improved_pipeline/baseline_comparison.csv")
df2_valid = df2.dropna(subset=['improved_test_auc', 'baseline_test_auc'])
if len(df2_valid) >= 5:
    stat, pval = stats.wilcoxon(df2_valid['improved_test_auc'], df2_valid['baseline_test_auc'], alternative='greater')
    print(f"Wilcoxon alternative='greater' (improved > baseline): stat={stat}, p-value={pval:.4e}")
    stat, pval_two = stats.wilcoxon(df2_valid['improved_test_auc'], df2_valid['baseline_test_auc'])
    print(f"Wilcoxon two-sided: stat={stat}, p-value={pval_two:.4e}")
