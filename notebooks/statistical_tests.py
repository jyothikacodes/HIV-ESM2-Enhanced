# ══════════════════════════════════════════════════════════════════════════════
# Statistical hypothesis testing for HIV drug resistance model comparisons
# ══════════════════════════════════════════════════════════════════════════════

from scipy import stats
import numpy as np
import pandas as pd

print('=' * 70)
print('STATISTICAL SIGNIFICANCE TESTS')
print('=' * 70)

# ── 1. Paired Wilcoxon: ESM-2 (logistic) vs XGBoost baseline ────────────
print('\n--- 1. ESM-2 vs XGBoost Baseline (paired Wilcoxon signed-rank) ---')
drugs_common = sorted(set(esm2_results.keys()) & set(baseline_results.keys()))
esm2_aucs = np.array([esm2_results[d]['auc'] for d in drugs_common])
base_aucs = np.array([baseline_results[d]['auc'] for d in drugs_common])
stat, p_wilcox = stats.wilcoxon(esm2_aucs, base_aucs)
print(f'  n drugs: {len(drugs_common)}')
print(f'  ESM-2 mean AUC: {esm2_aucs.mean():.4f}')
print(f'  Baseline mean AUC: {base_aucs.mean():.4f}')
print(f'  Wilcoxon statistic: {stat:.1f}, p = {p_wilcox:.6f}')
print(f'  ESM-2 wins: {(esm2_aucs > base_aucs).sum()}/{len(drugs_common)} drugs')

# Per-drug differences
print('\n  Per-drug AUC differences (ESM-2 - Baseline):')
for d, e, b in zip(drugs_common, esm2_aucs, base_aucs):
    diff = e - b
    print(f'    {d:>4s}: {diff:+.4f}  (ESM-2={e:.4f}, Baseline={b:.4f})')

# ── 2. DeLong tests: ESM-2 vs baseline per drug ─────────────────────────
print('\n--- 2. Per-drug DeLong tests (ESM-2 vs Baseline) ---')
from src.evaluation import delong_test

delong_rows = []
for d in drugs_common:
    y_true = esm2_results[d]['y_true']
    y_esm2 = esm2_results[d]['y_pred']
    y_base = baseline_results[d]['y_pred']
    z, p = delong_test(y_true, y_esm2, y_base)
    sig = '*' if p < 0.05 else ''
    print(f'  {d:>4s}: z={z:+.3f}, p={p:.4f} {sig}')
    delong_rows.append({'drug': d, 'z': z, 'p': p, 'significant': p < 0.05})

delong_df = pd.DataFrame(delong_rows)
n_sig = delong_df['significant'].sum()
print(f'\n  Significant at p<0.05: {n_sig}/{len(delong_df)} drugs')

# ── 3. PLM pairwise comparisons (Wilcoxon) ───────────────────────────────
print('\n--- 3. Pairwise PLM comparisons (paired Wilcoxon) ---')

# Reconstruct per-drug AUCs from plm_df (from notebook 07 results)
# If plm_df not in scope, build from the classifier comparison cell
try:
    plm_names = plm_df['plm'].unique()
except NameError:
    print('  plm_df not in scope — skipping PLM pairwise tests')
    print('  (Run notebook 07 cells first, or load results/robustness/plm_comparison_results.csv)')
    plm_names = []

if len(plm_names) > 0:
    for i, plm_a in enumerate(plm_names):
        for plm_b in plm_names[i+1:]:
            a_aucs = plm_df[plm_df['plm'] == plm_a].set_index('drug')['auc']
            b_aucs = plm_df[plm_df['plm'] == plm_b].set_index('drug')['auc']
            common = a_aucs.index.intersection(b_aucs.index)
            if len(common) < 5:
                continue
            stat, p = stats.wilcoxon(a_aucs[common].values, b_aucs[common].values)
            diff = a_aucs[common].mean() - b_aucs[common].mean()
            print(f'  {plm_a} vs {plm_b}: mean diff = {diff:+.4f}, Wilcoxon p = {p:.4f}')

# ── 4. Bootstrap 95% CIs on mean AUC ────────────────────────────────────
print('\n--- 4. Bootstrap 95% CIs on mean AUC (1000 iterations) ---')

def bootstrap_mean_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(means, 100 * alpha), np.percentile(means, 100 * (1 - alpha))

# ESM-2
lo, hi = bootstrap_mean_ci(esm2_aucs)
print(f'  ESM-2 (logistic):  mean={esm2_aucs.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]')

# Baseline
lo, hi = bootstrap_mean_ci(base_aucs)
print(f'  Baseline (XGBoost): mean={base_aucs.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]')

# PLMs
if len(plm_names) > 0:
    for plm in plm_names:
        aucs = plm_df[plm_df['plm'] == plm]['auc'].values
        lo, hi = bootstrap_mean_ci(aucs)
        print(f'  {plm:>10s}:          mean={aucs.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]')

# ── 5. Per-drug bootstrap CIs for ESM-2 ─────────────────────────────────
print('\n--- 5. Per-drug bootstrap 95% CIs (ESM-2 logistic) ---')
from src.evaluation import bootstrap_auc

for d in drugs_common:
    y_true = esm2_results[d]['y_true']
    y_pred = esm2_results[d]['y_pred']
    point, lo, hi = bootstrap_auc(y_true, y_pred, n_bootstrap=1000)
    print(f'  {d:>4s}: AUC = {esm2_results[d]["auc"]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]')

# ── 6. Subtype-stratified bootstrap CIs ──────────────────────────────────
print('\n--- 6. Subtype-stratified mean AUC with bootstrap 95% CIs ---')
try:
    for subtype in strat_df['subtype'].unique():
        sub = strat_df[strat_df['subtype'] == subtype]['auc'].dropna().values
        if len(sub) >= 3:
            lo, hi = bootstrap_mean_ci(sub)
            print(f'  {subtype:>12s}: mean={sub.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(sub)})')
except NameError:
    print('  strat_df not in scope — run subtype analysis cells first')

# ── 7. Temporal holdout vs CV comparison ─────────────────────────────────
print('\n--- 7. Temporal holdout vs cross-validation (paired Wilcoxon) ---')
try:
    # Match drugs between CV and temporal
    temp_aucs = temp_df.set_index('drug')['auc']
    cv_aucs = pd.Series({d: esm2_results[d]['auc'] for d in esm2_results})
    common_t = temp_aucs.index.intersection(cv_aucs.index)
    if len(common_t) >= 5:
        stat, p = stats.wilcoxon(cv_aucs[common_t].values, temp_aucs[common_t].values)
        diff = cv_aucs[common_t].mean() - temp_aucs[common_t].mean()
        print(f'  CV mean: {cv_aucs[common_t].mean():.4f}, Temporal mean: {temp_aucs[common_t].mean():.4f}')
        print(f'  Difference: {diff:+.4f}, Wilcoxon p = {p:.4f}')
except NameError:
    print('  temp_df not in scope — run temporal holdout cells first')

print('\n' + '=' * 70)
print('DONE — copy these results into the manuscript where needed')
print('=' * 70)
