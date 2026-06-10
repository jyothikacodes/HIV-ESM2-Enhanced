"""
Enhanced SHAP explainability for HIV drug resistance prediction.

This module builds on top of the existing compute_shap_values() from
interpretability.py to map SHAP values to residue positions, extract
global residue-level importance, and visualize the results.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .interpretability import (
    compute_shap_values,
    PI_DRM_POSITIONS,
    NRTI_DRM_POSITIONS,
    NNRTI_DRM_POSITIONS
)


def compute_shap_with_residue_mapping(
    model,
    X: np.ndarray,
    reference: str,
    n_background: int = 100,
    random_state: int = 42,
    positions: Optional[List[int]] = None
) -> Dict:
    """
    Compute SHAP values and map them to residue positions.

    Args:
        model: Trained classifier (e.g., LogisticRegression, XGBClassifier)
        X: Feature matrix of shape (n_samples, n_features)
        reference: Reference sequence (e.g., HIV_PROTEASE_REFERENCE)
        n_background: Number of background samples for SHAP
        random_state: Random seed
        positions: Explicit list of 1-indexed residue positions.
                   If None, determined automatically from feature dimensionality.

    Returns:
        Dict containing:
        - 'shap_values': Raw SHAP values array/list
        - 'residue_importance': Array of shape (seq_len,) containing global importance per residue
        - 'top_residues': DataFrame with ranked residue importances
    """
    # 1. Compute raw SHAP values using the existing interpretability module function
    raw_shap = compute_shap_values(model, X, n_background=n_background, random_state=random_state)
    
    # Handle SHAP output format differences:
    # KernelExplainer / TreeExplainer can return lists of arrays (one per class).
    # For binary classification, we care about the positive class (index 1).
    if isinstance(raw_shap, list):
        if len(raw_shap) == 2:
            shap_values = raw_shap[1]  # positive class
        elif len(raw_shap) == 1:
            shap_values = raw_shap[0]
        else:
            # Multi-class: default to average absolute shap values across all classes
            shap_values = np.mean([np.abs(s) for s in raw_shap], axis=0)
    else:
        shap_values = raw_shap

    n_samples, n_features = shap_values.shape
    seq_len = len(reference)

    # 2. Determine mapping from feature index to residue position (1-indexed)
    feature_to_residue = {}
    
    if positions is not None:
        # User provided explicit positions (e.g. for sparse mutation encoding)
        for i, pos in enumerate(positions):
            feature_to_residue[i] = pos
    elif n_features == seq_len:
        # Binary mutation encoding for the whole reference
        for i in range(n_features):
            feature_to_residue[i] = i + 1
    elif n_features > seq_len and n_features % seq_len == 0:
        # Flattened per-residue embeddings: n_features = seq_len * embed_dim
        embed_dim = n_features // seq_len
        for i in range(n_features):
            feature_to_residue[i] = (i // embed_dim) + 1
    else:
        # Fallback: assume features are embedding dimensions and cannot map to residue positions directly
        # We will map to 1..n_features and print a warning
        import warnings
        warnings.warn(
            f"Cannot determine feature-to-residue mapping for n_features={n_features} "
            f"and seq_len={seq_len}. Treating each feature as its own position."
        )
        for i in range(n_features):
            feature_to_residue[i] = i + 1

    # 3. Aggregate SHAP importances per residue position
    # We use mean absolute SHAP value as the feature importance metric
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # (n_features,)
    mean_shap = np.mean(shap_values, axis=0)              # (n_features,)

    residue_abs_sum = np.zeros(seq_len + 1)  # 1-indexed array
    residue_mean_sum = np.zeros(seq_len + 1)
    residue_counts = np.zeros(seq_len + 1)

    for i in range(n_features):
        pos = feature_to_residue.get(i, -1)
        if 1 <= pos <= seq_len:
            residue_abs_sum[pos] += mean_abs_shap[i]
            residue_mean_sum[pos] += mean_shap[i]
            residue_counts[pos] += 1

    # Average over features per position where counts > 1 (e.g. embeddings)
    residue_importance = np.zeros(seq_len)
    residue_direction = np.zeros(seq_len)
    
    for pos in range(1, seq_len + 1):
        idx = pos - 1
        if residue_counts[pos] > 0:
            # Sum for absolute magnitude, mean for direction
            residue_importance[idx] = residue_abs_sum[pos]
            residue_direction[idx] = residue_mean_sum[pos] / residue_counts[pos]

    # Create top residues DataFrame
    top_residues = get_top_contributing_residues(shap_values, reference, positions=positions)

    return {
        'shap_values': raw_shap,
        'residue_importance': residue_importance,
        'top_residues': top_residues
    }


def get_top_contributing_residues(
    shap_values: np.ndarray,
    reference: str,
    top_k: int = 20,
    positions: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Aggregate SHAP values to residue-level and rank the positions.

    Args:
        shap_values: Array of shape (n_samples, n_features)
        reference: Reference sequence
        top_k: Number of top positions to return
        positions: Optional explicit list of 1-indexed positions

    Returns:
        DataFrame containing ranked residue importances
    """
    # Handle list-based shap values (e.g. multi-class)
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        elif len(shap_values) == 1:
            shap_values = shap_values[0]
        else:
            shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)

    n_samples, n_features = shap_values.shape
    seq_len = len(reference)

    # Reconstruct mapping
    if positions is not None:
        feature_to_pos = {i: pos for i, pos in enumerate(positions)}
    elif n_features == seq_len:
        feature_to_pos = {i: i + 1 for i in range(n_features)}
    elif n_features > seq_len and n_features % seq_len == 0:
        embed_dim = n_features // seq_len
        feature_to_pos = {i: (i // embed_dim) + 1 for i in range(n_features)}
    else:
        feature_to_pos = {i: i + 1 for i in range(n_features)}

    # Calculate mean absolute SHAP and mean SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)

    # Aggregate by position
    pos_data = {}
    for i in range(n_features):
        pos = feature_to_pos.get(i, -1)
        if pos == -1:
            continue
        if pos not in pos_data:
            pos_data[pos] = {'abs_shap': [], 'raw_shap': []}
        pos_data[pos]['abs_shap'].append(mean_abs_shap[i])
        pos_data[pos]['raw_shap'].append(mean_shap[i])

    rows = []
    for pos, values in pos_data.items():
        if 1 <= pos <= seq_len:
            ref_aa = reference[pos - 1]
        else:
            ref_aa = '?'
            
        # Sum of absolute values represents total magnitude of position's influence
        abs_sum = np.sum(values['abs_shap'])
        # Mean raw shap represents overall direction of influence
        mean_raw = np.mean(values['raw_shap'])
        direction = 1 if mean_raw >= 0 else -1

        rows.append({
            'position': pos,
            'amino_acid': ref_aa,
            'mean_abs_shap': abs_sum,
            'mean_shap': mean_raw,
            'direction': direction
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
    return df.head(top_k)


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get global feature importance ranking.

    Args:
        shap_values: Array of shape (n_samples, n_features) or list
        feature_names: Optional list of names for the features

    Returns:
        DataFrame with columns: feature, mean_abs_shap
    """
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        elif len(shap_values) == 1:
            shap_values = shap_values[0]
        else:
            shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })
    
    return df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def plot_shap_residue_importance(
    residue_importance: np.ndarray,
    reference: str,
    drug_class: str = 'PI',
    drug: Optional[str] = None,
    top_k_highlight: int = 10,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot residue-level SHAP importance, highlighting known DRMs.

    Args:
        residue_importance: Array of shape (seq_len,)
        reference: Reference sequence
        drug_class: 'PI', 'NRTI', or 'NNRTI'
        drug: Specific drug name (for drug-specific highlights if available)
        top_k_highlight: Number of top contributing residues to annotate
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    seq_len = len(residue_importance)
    positions = np.arange(1, seq_len + 1)

    # 1. Resolve DRM positions to highlight
    known_drms = set()
    if drug_class == 'PI':
        known_drms.update(PI_DRM_POSITIONS.get('major', set()))
        known_drms.update(PI_DRM_POSITIONS.get('minor', set()))
    elif drug_class == 'NRTI':
        known_drms.update(NRTI_DRM_POSITIONS.get('major', set()))
        known_drms.update(NRTI_DRM_POSITIONS.get('TAMs', set()))
        known_drms.update(NRTI_DRM_POSITIONS.get('other', set()))
    elif drug_class == 'NNRTI':
        known_drms.update(NNRTI_DRM_POSITIONS.get('major', set()))
        known_drms.update(NNRTI_DRM_POSITIONS.get('minor', set()))

    # 2. Draw barplot
    colors = ['#e74c3c' if pos in known_drms else '#3498db' for pos in positions]
    bars = ax.bar(positions, residue_importance, color=colors, alpha=0.8, width=0.8)

    # 3. Highlight top K residue positions
    top_indices = np.argsort(residue_importance)[::-1][:top_k_highlight]
    for idx in top_indices:
        pos = idx + 1
        val = residue_importance[idx]
        ref_aa = reference[idx] if idx < len(reference) else '?'
        
        # Label above bar
        ax.annotate(
            f"{ref_aa}{pos}",
            xy=(pos, val),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=8, fontweight='bold',
            rotation=45 if top_k_highlight > 8 else 0
        )

    # Legend & styling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label=f'Known {drug_class} DRM'),
        Patch(facecolor='#3498db', alpha=0.8, label='Other Position')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    title = f"Residue-Level SHAP Importance ({drug_class})"
    if drug:
        title += f" - {drug}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Residue Position (1-indexed)", fontsize=12)
    ax.set_ylabel("Mean |SHAP| Value", fontsize=12)
    ax.set_xlim(0, seq_len + 1)
    
    # Grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()

    plt.tight_layout()
    return fig
