#!/usr/bin/env python3
"""
Biological Baseline Experiments on Real HIVDB Data for Initial Panel (ATV, TDF, EFV).

Executes:
1. Binary Mutation Baseline (Logistic Regression & Gradient Boosting) under nested CV.
2. Continuous log10(FC) Regression Baseline (Ridge & Gradient Boosting) under nested CV.
3. Metrics: ROC-AUC, PR-AUC, Balanced Accuracy, Brier Score, ECE, R^2, RMSE, Spearman rho.
4. Generates 95% bootstrap confidence intervals.
5. Saves fold assignments and out-of-fold predictions to results/biological_baseline/.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, r2_score, root_mean_squared_error
)
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import HXB2_PROTEASE, HXB2_RT

OUT_DIR = REPO_ROOT / "results" / "biological_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRUG_CONFIGS = [
    {
        "drug": "ATV",
        "class": "PI",
        "file": REPO_ROOT / "data/raw/PI_DataSet.txt",
        "gene": "PR",
        "pos_len": 99,
        "ref": HXB2_PROTEASE,
        "binary_cutoff": 3.0,
    },
    {
        "drug": "TDF",
        "class": "NRTI",
        "file": REPO_ROOT / "data/raw/NRTI_DataSet.txt",
        "gene": "RT",
        "pos_len": 240,
        "ref": HXB2_RT[:240],
        "binary_cutoff": 1.5,
    },
    {
        "drug": "EFV",
        "class": "NNRTI",
        "file": REPO_ROOT / "data/raw/NNRTI_DataSet.txt",
        "gene": "RT",
        "pos_len": 240,
        "ref": HXB2_RT[:240],
        "binary_cutoff": 3.0,
    },
]


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return low, high


def run_drug_baseline(cfg):
    drug = cfg["drug"]
    df = pd.read_csv(cfg["file"], sep="\t")
    pos_cols = [f"P{i}" for i in range(1, cfg["pos_len"] + 1)]
    ref = cfg["ref"]
    cutoff = cfg["binary_cutoff"]
    
    # Filter valid fold-change measurements
    s = pd.to_numeric(df[drug], errors="coerce")
    valid_mask = s.notna()
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    raw_fc = s[valid_mask].values
    log_fc = np.log10(np.clip(raw_fc, 1e-2, 1e4))
    y_binary = (raw_fc >= cutoff).astype(int)
    
    N = len(df_valid)
    print(f"\n=================================================================")
    print(f"RUNNING BIOLOGICAL BASELINES: {drug} ({cfg['class']})")
    print(f"Sample Count N = {N} | Resistant (Cutoff {cutoff}x): {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
    print(f"=================================================================")
    
    # Outer 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_prob_lr = np.zeros(N)
    oof_prob_gb = np.zeros(N)
    oof_pred_log_fc_ridge = np.zeros(N)
    oof_pred_log_fc_gb = np.zeros(N)
    fold_assignments = np.zeros(N, dtype=int)
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df_valid, y_binary)):
        fold_assignments[te_idx] = fold
        tr_df = df_valid.iloc[tr_idx]
        te_df = df_valid.iloc[te_idx]
        y_tr, y_te = y_binary[tr_idx], y_binary[te_idx]
        log_fc_tr = log_fc[tr_idx]
        
        # Build training fold mutation vocabulary (first-listed AA convention)
        tr_muts = set()
        for _, row in tr_df[pos_cols].iterrows():
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*'] and val[0].isalpha():
                    aa = val[0].upper()
                    if aa != ref[i]:
                        tr_muts.add((i, aa))
        mut_list = sorted(list(tr_muts))
        mut_to_idx = {m: idx for idx, m in enumerate(mut_list)}
        
        # Binary mutation matrices
        X_tr = np.zeros((len(tr_df), len(mut_list)), dtype=np.float32)
        X_te = np.zeros((len(te_df), len(mut_list)), dtype=np.float32)
        
        for r_idx, (_, row) in enumerate(tr_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*'] and val[0].isalpha():
                    aa = val[0].upper()
                    if (i, aa) in mut_to_idx:
                        X_tr[r_idx, mut_to_idx[(i, aa)]] = 1.0
        for r_idx, (_, row) in enumerate(te_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*'] and val[0].isalpha():
                    aa = val[0].upper()
                    if (i, aa) in mut_to_idx:
                        X_te[r_idx, mut_to_idx[(i, aa)]] = 1.0
                        
        # 1. Binary Mutation Logistic Regression
        clf_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tr, y_tr)
        oof_prob_lr[te_idx] = clf_lr.predict_proba(X_te)[:, 1]
        
        # 2. Binary Mutation Gradient Boosting
        clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_tr, y_tr)
        oof_prob_gb[te_idx] = clf_gb.predict_proba(X_te)[:, 1]
        
        # 3. Continuous log10(FC) Ridge Regression
        reg_ridge = Ridge(alpha=10.0, random_state=42).fit(X_tr, log_fc_tr)
        oof_pred_log_fc_ridge[te_idx] = reg_ridge.predict(X_te)
        
        # 4. Continuous log10(FC) Gradient Boosting
        reg_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_tr, log_fc_tr)
        oof_pred_log_fc_gb[te_idx] = reg_gb.predict(X_te)

    # Compute binary evaluation metrics
    auc_lr = float(roc_auc_score(y_binary, oof_prob_lr))
    auc_lr_ci = bootstrap_ci(roc_auc_score, y_binary, oof_prob_lr)
    ap_lr = float(average_precision_score(y_binary, oof_prob_lr))
    bal_acc_lr = float(balanced_accuracy_score(y_binary, (oof_prob_lr >= 0.5).astype(int)))
    brier_lr = float(brier_score_loss(y_binary, oof_prob_lr))
    ece_lr = compute_ece(y_binary, oof_prob_lr)
    
    auc_gb = float(roc_auc_score(y_binary, oof_prob_gb))
    auc_gb_ci = bootstrap_ci(roc_auc_score, y_binary, oof_prob_gb)
    
    # Compute continuous evaluation metrics
    r2_ridge = float(r2_score(log_fc, oof_pred_log_fc_ridge))
    rmse_ridge = float(root_mean_squared_error(log_fc, oof_pred_log_fc_ridge))
    rho_ridge, _ = stats.spearmanr(log_fc, oof_pred_log_fc_ridge)
    
    r2_gb = float(r2_score(log_fc, oof_pred_log_fc_gb))
    rmse_gb = float(root_mean_squared_error(log_fc, oof_pred_log_fc_gb))
    rho_gb, _ = stats.spearmanr(log_fc, oof_pred_log_fc_gb)
    
    # Map continuous regression predictions to binary decisions
    log_cutoff = np.log10(cutoff)
    ord_bal_acc_ridge = float(balanced_accuracy_score(y_binary, (oof_pred_log_fc_ridge >= log_cutoff).astype(int)))
    ord_bal_acc_gb = float(balanced_accuracy_score(y_binary, (oof_pred_log_fc_gb >= log_cutoff).astype(int)))
    
    print(f"Results for {drug}:")
    print(f"  [Binary LR Baseline] ROC-AUC: {auc_lr:.4f} (95% CI: [{auc_lr_ci[0]:.4f}, {auc_lr_ci[1]:.4f}])")
    print(f"                       PR-AUC: {ap_lr:.4f} | BalAcc: {bal_acc_lr:.4f} | Brier: {brier_lr:.4f} | ECE: {ece_lr:.4f}")
    print(f"  [Binary GB Baseline] ROC-AUC: {auc_gb:.4f} (95% CI: [{auc_gb_ci[0]:.4f}, {auc_gb_ci[1]:.4f}])")
    print(f"  [Continuous Ridge]   R^2: {r2_ridge:.4f} | RMSE: {rmse_ridge:.4f} | Spearman rho: {rho_ridge:.4f} | Mapped BalAcc: {ord_bal_acc_ridge:.4f}")
    print(f"  [Continuous GB]      R^2: {r2_gb:.4f} | RMSE: {rmse_gb:.4f} | Spearman rho: {rho_gb:.4f} | Mapped BalAcc: {ord_bal_acc_gb:.4f}")
    
    # Save out-of-fold predictions
    pred_df = pd.DataFrame({
        "SeqID": df_valid["SeqID"],
        "fold": fold_assignments,
        "y_true_binary": y_binary,
        "prob_lr": np.round(oof_prob_lr, 5),
        "prob_gb": np.round(oof_prob_gb, 5),
        "true_log10_fc": np.round(log_fc, 4),
        "pred_log10_fc_ridge": np.round(oof_pred_log_fc_ridge, 4),
        "pred_log10_fc_gb": np.round(oof_pred_log_fc_gb, 4),
    })
    pred_path = OUT_DIR / f"{drug.lower()}_oof_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved OOF predictions: {pred_path}")
    
    return {
        "drug": drug,
        "drug_class": cfg["class"],
        "n_samples": N,
        "binary_cutoff": cutoff,
        "resistant_pct": round(y_binary.mean() * 100, 2),
        "binary_lr_auc": round(auc_lr, 4),
        "binary_lr_auc_ci_low": round(auc_lr_ci[0], 4),
        "binary_lr_auc_ci_high": round(auc_lr_ci[1], 4),
        "binary_lr_pr_auc": round(ap_lr, 4),
        "binary_lr_bal_acc": round(bal_acc_lr, 4),
        "binary_lr_brier": round(brier_lr, 4),
        "binary_lr_ece": round(ece_lr, 4),
        "binary_gb_auc": round(auc_gb, 4),
        "continuous_ridge_r2": round(r2_ridge, 4),
        "continuous_ridge_rmse": round(rmse_ridge, 4),
        "continuous_ridge_spearman": round(rho_ridge, 4),
        "continuous_ridge_mapped_bal_acc": round(ord_bal_acc_ridge, 4),
        "continuous_gb_r2": round(r2_gb, 4),
        "continuous_gb_rmse": round(rmse_gb, 4),
        "continuous_gb_spearman": round(rho_gb, 4),
        "continuous_gb_mapped_bal_acc": round(ord_bal_acc_gb, 4),
    }


def main():
    summary_records = []
    for cfg in DRUG_CONFIGS:
        res = run_drug_baseline(cfg)
        summary_records.append(res)
        
    summary_df = pd.DataFrame(summary_records)
    summary_path = OUT_DIR / "biological_baseline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved Biological Baseline Summary: {summary_path}")


if __name__ == "__main__":
    main()
