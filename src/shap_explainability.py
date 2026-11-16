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
    positions: Optional[List[int]] = None,
    per_residue_embeddings: Optional[Union[np.ndarray, List[np.ndarray]]] = None
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
        per_residue_embeddings: Optional per-residue embeddings array of shape
                   (n_samples, seq_len, embed_dim) or list of (seq_len, embed_dim).
                   Required to project 1280-dim pooled embeddings back to residues.

    Returns:
        Dict containing:
        - 'shap_values': Raw SHAP values array/list
        - 'residue_importance': Array of shape (seq_len,) containing global importance per residue
        - 'top_residues': DataFrame with ranked residue importances (strictly 1..seq_len)
    """
    import warnings

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
    is_pooled_latent = False
    
    if positions is not None:
        # User provided explicit positions (e.g. for sparse mutation encoding)
        for i, pos in enumerate(positions):
            if 1 <= pos <= seq_len:
                feature_to_residue[i] = pos
    elif n_features == seq_len:
        # Binary mutation encoding for the whole reference or per-residue scores
        for i in range(n_features):
            feature_to_residue[i] = i + 1
    elif n_features > seq_len and n_features % seq_len == 0:
        # Flattened per-residue embeddings: n_features = seq_len * embed_dim
        embed_dim = n_features // seq_len
        for i in range(n_features):
            feature_to_residue[i] = (i // embed_dim) + 1
    else:
        is_pooled_latent = True

    residue_importance = np.zeros(seq_len)

    if is_pooled_latent:
        if per_residue_embeddings is not None:
            # Project pooled 1280-dim SHAP attributions back onto residue tokens
            # using per-residue token embeddings: contrib(j) = sum_d |phi_d| * (|e_{j,d}| / sum_k |e_{k,d}|)
            n_emb_samples = len(per_residue_embeddings)
            abs_shap = np.abs(shap_values[:n_emb_samples])  # (N, embed_dim)
            sample_residue_scores = np.zeros((n_emb_samples, seq_len))

            for s_idx in range(n_emb_samples):
                token_emb = np.asarray(per_residue_embeddings[s_idx])  # (seq_len, embed_dim)
                if token_emb.shape[0] >= seq_len and token_emb.shape[1] == n_features:
                    token_slice = token_emb[:seq_len]  # (seq_len, embed_dim)
                    token_abs = np.abs(token_slice)
                    col_sums = np.sum(token_abs, axis=0, keepdims=True) + 1e-12
                    token_weights = token_abs / col_sums  # (seq_len, embed_dim)
                    sample_residue_scores[s_idx] = np.dot(token_weights, abs_shap[s_idx])
                else:
                    # Fallback to token L2 norm
                    norms = np.linalg.norm(token_emb[:seq_len], axis=1)
                    sample_residue_scores[s_idx] = norms * np.mean(abs_shap[s_idx])

            residue_importance = np.mean(sample_residue_scores, axis=0)
        else:
            warnings.warn(
                f"Feature dimension ({n_features}) represents latent PLM embedding dimensions, "
                f"not residue positions for reference sequence of length {seq_len}. "
                "Residue-level SHAP requires per_residue_embeddings or explicit positions. "
                "To inspect embedding dimensions, use get_global_feature_importance().",
                UserWarning
            )
            residue_importance = np.zeros(seq_len)
    else:
        # 3. Aggregate SHAP importances per residue position
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        residue_abs_sum = np.zeros(seq_len + 1)
        residue_counts = np.zeros(seq_len + 1)

        for i in range(n_features):
            pos = feature_to_residue.get(i, -1)
            if 1 <= pos <= seq_len:
                residue_abs_sum[pos] += mean_abs_shap[i]
                residue_counts[pos] += 1

        for pos in range(1, seq_len + 1):
            idx = pos - 1
            if residue_counts[pos] > 0:
                residue_importance[idx] = residue_abs_sum[pos]

    # Create top residues DataFrame strictly within 1..seq_len
    top_residues = get_top_contributing_residues(
        shap_values,
        reference,
        positions=positions,
        residue_importance=residue_importance
    )

    return {
        'shap_values': raw_shap,
        'residue_importance': residue_importance,
        'top_residues': top_residues
    }


def get_top_contributing_residues(
    shap_values: np.ndarray,
    reference: str,
    top_k: int = 20,
    positions: Optional[List[int]] = None,
    residue_importance: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Aggregate SHAP values to residue-level and rank the positions.
    Guarantees valid 1-indexed residue positions strictly within 1..len(reference)
    and maps only to genuine amino acids from the reference sequence (no '?' values).

    Args:
        shap_values: Array of shape (n_samples, n_features)
        reference: Reference sequence
        top_k: Number of top positions to return
        positions: Optional explicit list of 1-indexed positions
        residue_importance: Optional precomputed residue importance array of shape (seq_len,)

    Returns:
        DataFrame containing ranked residue importances
    """
    seq_len = len(reference)

    if residue_importance is not None and len(residue_importance) == seq_len:
        # Build directly from precomputed residue importance
        rows = []
        for pos in range(1, seq_len + 1):
            idx = pos - 1
            rows.append({
                'position': pos,
                'amino_acid': reference[idx],
                'mean_abs_shap': float(residue_importance[idx]),
                'mean_shap': float(residue_importance[idx]),
                'direction': 1
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        return df.head(top_k)

    # Handle list-based shap values (e.g. multi-class)
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        elif len(shap_values) == 1:
            shap_values = shap_values[0]
        else:
            shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)

    n_samples, n_features = shap_values.shape

    # Reconstruct mapping
    feature_to_pos = {}
    if positions is not None:
        for i, pos in enumerate(positions):
            if 1 <= pos <= seq_len:
                feature_to_pos[i] = pos
    elif n_features == seq_len:
        for i in range(n_features):
            feature_to_pos[i] = i + 1
    elif n_features > seq_len and n_features % seq_len == 0:
        embed_dim = n_features // seq_len
        for i in range(n_features):
            feature_to_pos[i] = (i // embed_dim) + 1
    else:
        # Features are latent embedding dimensions (e.g. 1280), NOT sequence positions.
        # Do not fabricate false sequence positions.
        import warnings
        warnings.warn(
            f"Cannot map {n_features} embedding dimensions to {seq_len} residues without explicit positions. "
            "Returning empty residue ranking. Use get_global_feature_importance() instead.",
            UserWarning
        )
        return pd.DataFrame(columns=['position', 'amino_acid', 'mean_abs_shap', 'mean_shap', 'direction'])

    # Calculate mean absolute SHAP and mean SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)

    # Aggregate by position strictly within 1..seq_len
    pos_data = {}
    for i in range(n_features):
        pos = feature_to_pos.get(i, -1)
        if pos < 1 or pos > seq_len:
            continue
        if pos not in pos_data:
            pos_data[pos] = {'abs_shap': [], 'raw_shap': []}
        pos_data[pos]['abs_shap'].append(mean_abs_shap[i])
        pos_data[pos]['raw_shap'].append(mean_shap[i])

    rows = []
    for pos, values in pos_data.items():
        ref_aa = reference[pos - 1]
        abs_sum = float(np.sum(values['abs_shap']))
        mean_raw = float(np.mean(values['raw_shap']))
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
    Get global feature importance ranking. For pooled PLM embeddings, feature names
    are accurately identified as Embedding_Dim_N instead of misleading sequence positions.

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

    n_features = shap_values.shape[1]
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    if feature_names is None:
        if n_features == 1280:
            feature_names = [f"Embedding_Dim_{i}" for i in range(n_features)]
        else:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
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
