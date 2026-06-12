"""
Subtype reconstruction and stratified analysis for HIV drug resistance.

This module provides functions for:
- Reconstructing full AA sequences from HIVDB position columns + HXB2 reference
- Assigning subtypes via the Stanford Sierra GraphQL API (sierrapy)
- Running subtype-stratified performance analysis
- Temporal holdout split for quasi-independent validation

These analyses evaluate whether model performance is robust across
HIV-1 subtypes and generalises to temporally distinct sequence sets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data_processing import (
    PI_DRUGS, NRTI_DRUGS, NNRTI_DRUGS,
    get_drug_class,
    reconstruct_sequences_from_positions
)
from .feature_engineering import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE


# ── Sequence reconstruction ─────────────────────────────────────────────────


def reconstruct_all_datasets(
    data_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Reconstruct sequences for all drug classes from raw HIVDB files.

    Args:
        data_dir: Directory containing PI_dataset.txt, NRTI_dataset.txt, NNRTI_dataset.txt

    Returns:
        Dict mapping drug class -> DataFrame with columns [SeqID, sequence, drug cols...]
    """
    references = {
        'PI': HIV_PROTEASE_REFERENCE,
        'NRTI': HIV_RT_REFERENCE,
        'NNRTI': HIV_RT_REFERENCE,
    }

    datasets = {}
    for drug_class, ref in references.items():
        filepath = data_dir / f'{drug_class}_dataset.txt'
        if not filepath.exists():
            # Try alternate naming
            filepath = data_dir / f'{drug_class}_genopheno.csv'
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping {drug_class}")
            continue

        df = pd.read_csv(filepath, sep='\t')
        sequences = reconstruct_sequences_from_positions(df, ref)

        # Build output DataFrame with SeqID, sequence, and drug columns
        drug_list = {
            'PI': PI_DRUGS, 'NRTI': NRTI_DRUGS, 'NNRTI': NNRTI_DRUGS
        }[drug_class]

        out_df = pd.DataFrame({
            'SeqID': df['SeqID'].values,
            'sequence': sequences,
            'drug_class': drug_class,
        })

        # Add drug fold-change columns
        for drug in drug_list:
            if drug in df.columns:
                out_df[drug] = df[drug].values

        datasets[drug_class] = out_df

    return datasets


# ── Subtype assignment ───────────────────────────────────────────────────────

def assign_subtypes_via_sequence_similarity(
    sequences: List[str],
    seq_ids: List[str],
    drug_class: str = 'PI'
) -> pd.DataFrame:
    """
    Assign HIV-1 subtypes using a simple reference-distance approach.

    This serves as a fallback when the Sierra API is not available.
    Uses Hamming distance to a panel of subtype consensus sequences.

    For the full Sierra API approach, use assign_subtypes_sierra() instead.

    Args:
        sequences: List of amino acid sequences
        seq_ids: List of sequence identifiers
        drug_class: Drug class (determines which protein)

    Returns:
        DataFrame with columns [SeqID, subtype, distance]
    """
    # Subtype B consensus (HXB2) is the reference; most sequences will be B
    # For a proper analysis, load consensus sequences for major subtypes
    # This is a simplified placeholder — the Sierra API is preferred
    print("  Warning: using simplified subtype assignment (Hamming distance to HXB2)")
    print("  For production use, prefer assign_subtypes_sierra()")

    reference = HIV_PROTEASE_REFERENCE if drug_class == 'PI' else HIV_RT_REFERENCE
    ref_len = len(reference)

    results = []
    for seq_id, seq in zip(seq_ids, sequences):
        # Count mismatches from HXB2 (subtype B reference)
        mismatches = sum(
            1 for i in range(min(len(seq), ref_len))
            if seq[i] != reference[i]
        )
        distance = mismatches / ref_len

        # Simple heuristic: low distance -> likely subtype B
        # This is a rough proxy; proper subtyping needs phylogenetic analysis
        if distance < 0.05:
            subtype = 'B'
        elif distance < 0.10:
            subtype = 'B_divergent'
        else:
            subtype = 'non-B'

        results.append({
            'SeqID': seq_id,
            'subtype': subtype,
            'distance_from_B': distance,
            'n_mutations': mismatches
        })

    return pd.DataFrame(results)


def assign_subtypes_sierra(
    fasta_path: Path,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Assign subtypes using the Stanford Sierra GraphQL API via sierrapy.

    Requires: pip install sierrapy

    Note: This requires NUCLEOTIDE sequences. If you only have amino acid
    sequences reconstructed from position columns, use
    assign_subtypes_via_sequence_similarity() as a fallback.

    Args:
        fasta_path: Path to FASTA file with nucleotide sequences
        output_path: Optional path to save results as CSV

    Returns:
        DataFrame with columns [SeqID, subtype, subtype_distance]
    """
    try:
        import subprocess
        import json

        # Run sierrapy
        cmd = [
            'sierrapy', 'sequences', str(fasta_path),
            '-o', '/dev/stdout',
            '-q', 'bestMatchingSubtype { display subtype { name } distancePcnt }'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"sierrapy failed: {result.stderr}")

        data = json.loads(result.stdout)

        rows = []
        for entry in data:
            seq_id = entry.get('inputSequence', {}).get('header', 'unknown')
            subtype_info = entry.get('bestMatchingSubtype', {})
            rows.append({
                'SeqID': seq_id,
                'subtype': subtype_info.get('display', 'unknown'),
                'subtype_distance': subtype_info.get('distancePcnt', None)
            })

        df = pd.DataFrame(rows)

        if output_path:
            df.to_csv(output_path, index=False)

        return df

    except FileNotFoundError:
        raise RuntimeError(
            "sierrapy not found. Install with: pip install sierrapy\n"
            "Or use assign_subtypes_via_sequence_similarity() as a fallback."
        )


# ── Subtype-stratified evaluation ────────────────────────────────────────────

def subtype_stratified_evaluation(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    subtypes: pd.Series,
    drugs: List[str],
    model_type: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate model performance stratified by HIV subtype.

    Trains on all data using cross-validation, then reports AUC
    separately for each subtype group.

    Args:
        X: Feature matrix (n_samples, n_features)
        phenotypes: DataFrame with drug resistance labels
        subtypes: Series of subtype assignments (aligned with X rows)
        drugs: List of drug names
        model_type: Classifier type
        n_splits: CV folds
        random_state: Random seed

    Returns:
        DataFrame with columns [drug, subtype, auc, n_samples, n_resistant]
    """
    from .models import per_drug_training

    # First, get full cross-validated predictions
    print("Running full cross-validation...")
    full_results = per_drug_training(
        X, phenotypes, drugs,
        model_type=model_type,
        n_splits=n_splits,
        random_state=random_state
    )

    # Now stratify results by subtype
    rows = []
    unique_subtypes = subtypes.unique()

    for drug in drugs:
        if drug not in full_results:
            continue

        res = full_results[drug]
        y_true = res['y_true']
        y_pred = res['y_pred']

        # Get valid mask (same as used in per_drug_training)
        label_col = f"{drug}_class2" if f"{drug}_class2" in phenotypes.columns else drug
        valid_mask = ~np.isnan(phenotypes[label_col].values)
        valid_subtypes = subtypes[valid_mask].values

        for subtype in unique_subtypes:
            mask = valid_subtypes == subtype
            n = mask.sum()

            if n < 10:
                continue

            y_true_sub = y_true[mask]
            y_pred_sub = y_pred[mask]
            n_resistant = int(y_true_sub.sum())

            if len(np.unique(y_true_sub)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_true_sub, y_pred_sub)

            rows.append({
                'drug': drug,
                'subtype': subtype,
                'auc': auc,
                'n_samples': n,
                'n_resistant': n_resistant,
                'n_susceptible': n - n_resistant,
                'prevalence': n_resistant / n
            })

    return pd.DataFrame(rows)


# ── Temporal holdout ─────────────────────────────────────────────────────────

def create_temporal_split(
    df: pd.DataFrame,
    seq_id_col: str = 'SeqID',
    cutoff_quantile: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a temporal holdout split based on SeqID ordering.

    SeqIDs in HIVDB are approximately chronologically ordered
    (higher IDs = more recent submissions). We use this as a proxy
    for temporal splitting when exact dates aren't available.

    Args:
        df: DataFrame with SeqID column
        seq_id_col: Name of sequence ID column
        cutoff_quantile: Fraction of data for training (by SeqID order)

    Returns:
        Tuple of (train_indices, test_indices) as numpy arrays
    """
    seq_ids = df[seq_id_col].values
    cutoff = np.quantile(seq_ids, cutoff_quantile)

    train_mask = seq_ids <= cutoff
    test_mask = seq_ids > cutoff

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"  Temporal split: train={len(train_idx)} (SeqID <= {cutoff:.0f}), "
          f"test={len(test_idx)} (SeqID > {cutoff:.0f})")

    return train_idx, test_idx


def temporal_holdout_evaluation(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_type: str = 'logistic',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate model on temporal holdout split.

    Args:
        X: Feature matrix
        phenotypes: DataFrame with drug resistance labels
        drugs: List of drug names
        train_idx: Training set indices
        test_idx: Test set indices
        model_type: Classifier type
        random_state: Random seed

    Returns:
        DataFrame with per-drug results on temporal holdout
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    rows = []

    for drug in drugs:
        label_col = f"{drug}_class2" if f"{drug}_class2" in phenotypes.columns else drug
        if label_col not in phenotypes.columns:
            continue

        y = phenotypes[label_col].values

        # Get valid samples within each split
        train_valid = train_idx[~np.isnan(y[train_idx])]
        test_valid = test_idx[~np.isnan(y[test_idx])]

        X_train = X[train_valid]
        y_train = y[train_valid].astype(int)
        X_test = X[test_valid]
        y_test = y[test_valid].astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Skipping {drug}: insufficient class diversity in split")
            continue

        # Train and predict
        if model_type == 'logistic':
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(
                max_iter=1000, class_weight='balanced',
                random_state=random_state
            )
            model.fit(X_train_s, y_train)
            y_pred = model.predict_proba(X_test_s)[:, 1]
        elif model_type == 'xgboost':
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0,
                random_state=random_state, eval_metric='auc',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred)

        rows.append({
            'drug': drug,
            'auc': auc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_resistant_train': int(y_train.sum()),
            'n_resistant_test': int(y_test.sum()),
        })

        print(f"  {drug}: AUC = {auc:.4f} (train={len(y_train)}, test={len(y_test)})")

    return pd.DataFrame(rows)
