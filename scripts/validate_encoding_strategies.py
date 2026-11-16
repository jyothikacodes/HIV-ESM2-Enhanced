#!/usr/bin/env python3
"""
Empirical Comparison of Sequence Encoding Strategies on Biological HIVDB Data:
1. First-listed amino-acid convention (Standard Farquhar benchmark).
2. Ambiguity-aware encoding (Multi-hot representation of all alleles in mixtures).
3. Training-fold consensus replacement (Imputing training-fold mode for mixtures/anomalies).
"""

import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import HXB2_PROTEASE, HXB2_RT

def test_encoding_strategies(filepath, drug, gene, pos_len, threshold):
    df = pd.read_csv(filepath, sep='\t')
    pos_cols = [f'P{i}' for i in range(1, pos_len + 1)]
    ref = HXB2_PROTEASE if gene == 'PR' else HXB2_RT[:pos_len]
    
    # Filter valid drug labels
    s = pd.to_numeric(df[drug], errors='coerce')
    valid = s.notna()
    df_valid = df[valid].copy().reset_index(drop=True)
    y = (s[valid].values >= threshold).astype(int)
    N = len(df_valid)
    
    print(f"\n=======================================================")
    print(f"Target Drug: {drug} | Gene: {gene} | N={N} | Resistant: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"=======================================================")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_auc = {'first_listed': [], 'ambiguity_multi_hot': [], 'training_consensus': []}
    results_brier = {'first_listed': [], 'ambiguity_multi_hot': [], 'training_consensus': []}
    
    for tr_idx, te_idx in skf.split(df_valid, y):
        tr_df = df_valid.iloc[tr_idx]
        te_df = df_valid.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        # -------------------------------------------------------------
        # Strategy 1: First-listed amino acid
        # -------------------------------------------------------------
        tr_muts_1 = set()
        for _, row in tr_df[pos_cols].iterrows():
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*'] and val[0].isalpha():
                    aa = val[0].upper()
                    if aa != ref[i]:
                        tr_muts_1.add((i, aa))
        mut_list_1 = sorted(list(tr_muts_1))
        mut_to_idx_1 = {m: idx for idx, m in enumerate(mut_list_1)}
        
        X_tr_1 = np.zeros((len(tr_df), len(mut_list_1)), dtype=np.float32)
        X_te_1 = np.zeros((len(te_df), len(mut_list_1)), dtype=np.float32)
        for r_idx, (_, row) in enumerate(tr_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*'] and val[0].isalpha():
                    aa = val[0].upper()
                    if (i, aa) in mut_to_idx_1:
                        X_tr_1[r_idx, mut_to_idx_1[(i, aa)]] = 1.0
        for r_idx, (_, row) in enumerate(te_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*'] and val[0].isalpha():
                    aa = val[0].upper()
                    if (i, aa) in mut_to_idx_1:
                        X_te_1[r_idx, mut_to_idx_1[(i, aa)]] = 1.0
                        
        clf1 = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tr_1, y_tr)
        p1 = clf1.predict_proba(X_te_1)[:, 1]
        results_auc['first_listed'].append(roc_auc_score(y_te, p1))
        results_brier['first_listed'].append(brier_score_loss(y_te, p1))
        
        # -------------------------------------------------------------
        # Strategy 2: Ambiguity-aware (multi-hot for all alleles in mixture)
        # -------------------------------------------------------------
        tr_muts_2 = set()
        for _, row in tr_df[pos_cols].iterrows():
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*']:
                    for char in val:
                        if char.isalpha() and char.upper() != ref[i]:
                            tr_muts_2.add((i, char.upper()))
        mut_list_2 = sorted(list(tr_muts_2))
        mut_to_idx_2 = {m: idx for idx, m in enumerate(mut_list_2)}
        
        X_tr_2 = np.zeros((len(tr_df), len(mut_list_2)), dtype=np.float32)
        X_te_2 = np.zeros((len(te_df), len(mut_list_2)), dtype=np.float32)
        for r_idx, (_, row) in enumerate(tr_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*']:
                    for char in val:
                        if char.isalpha() and (i, char.upper()) in mut_to_idx_2:
                            X_tr_2[r_idx, mut_to_idx_2[(i, char.upper())]] = 1.0
        for r_idx, (_, row) in enumerate(te_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*']:
                    for char in val:
                        if char.isalpha() and (i, char.upper()) in mut_to_idx_2:
                            X_te_2[r_idx, mut_to_idx_2[(i, char.upper())]] = 1.0
                            
        clf2 = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tr_2, y_tr)
        p2 = clf2.predict_proba(X_te_2)[:, 1]
        results_auc['ambiguity_multi_hot'].append(roc_auc_score(y_te, p2))
        results_brier['ambiguity_multi_hot'].append(brier_score_loss(y_te, p2))
        
        # -------------------------------------------------------------
        # Strategy 3: Training-fold consensus replacement
        # -------------------------------------------------------------
        pos_consensus = {}
        for i, col in enumerate(pos_cols):
            counts = Counter()
            for val in tr_df[col]:
                val_s = str(val).strip()
                if len(val_s) == 1 and val_s.isalpha():
                    counts[val_s.upper()] += 1
            pos_consensus[i] = counts.most_common(1)[0][0] if counts else ref[i]
            
        tr_muts_3 = set()
        for _, row in tr_df[pos_cols].iterrows():
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*']:
                    aa = val[0].upper() if len(val) == 1 else pos_consensus[i]
                    if aa != ref[i]:
                        tr_muts_3.add((i, aa))
        mut_list_3 = sorted(list(tr_muts_3))
        mut_to_idx_3 = {m: idx for idx, m in enumerate(mut_list_3)}
        
        X_tr_3 = np.zeros((len(tr_df), len(mut_list_3)), dtype=np.float32)
        X_te_3 = np.zeros((len(te_df), len(mut_list_3)), dtype=np.float32)
        for r_idx, (_, row) in enumerate(tr_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*']:
                    aa = val[0].upper() if len(val) == 1 else pos_consensus[i]
                    if (i, aa) in mut_to_idx_3:
                        X_tr_3[r_idx, mut_to_idx_3[(i, aa)]] = 1.0
        for r_idx, (_, row) in enumerate(te_df[pos_cols].iterrows()):
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip()
                if val and val not in ['-', '.', '~', '#', '*']:
                    aa = val[0].upper() if len(val) == 1 else pos_consensus[i]
                    if (i, aa) in mut_to_idx_3:
                        X_te_3[r_idx, mut_to_idx_3[(i, aa)]] = 1.0
                        
        clf3 = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tr_3, y_tr)
        p3 = clf3.predict_proba(X_te_3)[:, 1]
        results_auc['training_consensus'].append(roc_auc_score(y_te, p3))
        results_brier['training_consensus'].append(brier_score_loss(y_te, p3))
        
    for strat in ['first_listed', 'ambiguity_multi_hot', 'training_consensus']:
        mean_auc = np.mean(results_auc[strat])
        std_auc = np.std(results_auc[strat])
        mean_brier = np.mean(results_brier[strat])
        print(f"  {strat:22s} -> AUC: {mean_auc:.4f} +/- {std_auc:.4f} | Brier: {mean_brier:.4f}")

if __name__ == "__main__":
    test_encoding_strategies(REPO_ROOT / 'data/raw/PI_DataSet.txt', 'ATV', 'PR', 99, 3.0)
    test_encoding_strategies(REPO_ROOT / 'data/raw/NRTI_DataSet.txt', 'TDF', 'RT', 240, 1.5)
    test_encoding_strategies(REPO_ROOT / 'data/raw/NNRTI_DataSet.txt', 'EFV', 'RT', 240, 3.0)
