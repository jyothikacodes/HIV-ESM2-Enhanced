"""
Subtype robustness analysis for HIV-1 drug resistance models.

Evaluates performance across HIV-1 genetic clades (Subtype B vs Non-B subtypes: C, A, CRF01_AE, etc.).
Quantifies the subtype generalization gap to verify whether models overfit clade B polymorphisms.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def evaluate_by_subtype(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subtypes: Union[List[str], np.ndarray],
    min_samples: int = 15
) -> pd.DataFrame:
    """
    Compute performance metrics stratified by HIV-1 subtype.
    
    Args:
        y_true: True binary resistance labels.
        y_pred: Predicted probabilities.
        subtypes: Array of subtype annotations (e.g. 'B', 'C', 'CRF01_AE').
        min_samples: Minimum samples required per subtype for evaluation.
        
    Returns:
        DataFrame summarizing sample counts, positive prevalence, AUC, AP, and Brier score.
    """
    subtypes_arr = np.array(subtypes, dtype=str)
    unique_subtypes = np.unique(subtypes_arr)
    
    records = []
    
    for st in unique_subtypes:
        idx = np.where(subtypes_arr == st)[0]
        y_t = y_true[idx]
        y_p = y_pred[idx]
        
        valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
        y_t_val = y_t[valid].astype(int)
        y_p_val = y_p[valid]
        
        n_samples = len(y_t_val)
        if n_samples < min_samples:
            continue
            
        n_resistant = int(np.sum(y_t_val == 1))
        n_susceptible = int(np.sum(y_t_val == 0))
        
        if n_resistant == 0 or n_susceptible == 0:
            auc = np.nan
            ap = np.nan
        else:
            auc = float(roc_auc_score(y_t_val, y_p_val))
            ap = float(average_precision_score(y_t_val, y_p_val))
            
        brier = float(brier_score_loss(y_t_val, y_p_val))
        
        records.append({
            "subtype": st,
            "n_samples": n_samples,
            "n_resistant": n_resistant,
            "n_susceptible": n_susceptible,
            "prevalence": n_resistant / n_samples,
            "auc": auc,
            "average_precision": ap,
            "brier_score": brier,
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("n_samples", ascending=False).reset_index(drop=True)
    return df


def compute_clade_b_vs_non_b_gap(
    subtype_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute performance gap between dominant Subtype B and aggregated Non-B clades.
    """
    if subtype_df.empty or "subtype" not in subtype_df.columns:
        return {"subtype_b_auc": np.nan, "non_b_auc": np.nan, "subtype_generalization_gap": np.nan}
        
    b_row = subtype_df[subtype_df["subtype"] == "B"]
    non_b_rows = subtype_df[subtype_df["subtype"] != "B"]
    
    b_auc = b_row["auc"].values[0] if not b_row.empty else np.nan
    
    if not non_b_rows.empty and non_b_rows["auc"].notna().any():
        # Weighted average AUC across non-B clades
        valid_non_b = non_b_rows.dropna(subset=["auc"])
        total_non_b_samples = valid_non_b["n_samples"].sum()
        if total_non_b_samples > 0:
            non_b_auc = float(np.sum(valid_non_b["auc"] * valid_non_b["n_samples"]) / total_non_b_samples)
        else:
            non_b_auc = np.nan
    else:
        non_b_auc = np.nan
        
    gap = b_auc - non_b_auc if (not np.isnan(b_auc) and not np.isnan(non_b_auc)) else np.nan
    
    return {
        "subtype_b_auc": b_auc,
        "non_b_auc": non_b_auc,
        "subtype_generalization_gap": gap,
    }
