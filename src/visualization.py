"""
Visualization utilities for HIV drug resistance analysis.

This module provides functions for:
- ROC curves per drug
- Drug comparison bar charts
- Attention heatmaps
- Calibration plots
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.size'] = 13


def plot_roc_curves(
    results: Dict,
    title: str = "ROC Curves by Drug",
    figsize: Tuple[int, int] = (16, 14),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple drugs.

    Args:
        results: Dictionary with 'y_true' and 'y_pred' for each drug
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_drugs = len(results)
    n_cols = 3
    n_rows = (n_drugs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_drugs > 1 else [axes]

    for idx, (drug, res) in enumerate(results.items()):
        ax = axes[idx]
        # Skip or annotate drugs without predictions
        if not isinstance(res, dict) or 'y_true' not in res or 'y_pred' not in res:
            ax.text(0.5, 0.5, 'No predictions', ha='center', va='center', fontsize=12)
            ax.set_title(f'{drug} (skipped)')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue

        try:
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_pred'])
        except Exception:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
            ax.set_title(f'{drug} (insufficient)')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue

        auc = res.get('auc', np.nan)

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{drug}')
        ax.legend(loc='lower right')

    # Hide empty subplots
    for idx in range(n_drugs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    return fig


def plot_drug_comparison(
    esm2_results: Dict,
    baseline_results: Dict,
    title: str = "ESM-2 vs Baseline AUC Comparison",
    figsize: Tuple[int, int] = (18, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing ESM-2 and baseline AUCs by drug.

    Args:
        esm2_results: ESM-2 model results
        baseline_results: Baseline model results
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    drugs = sorted(set(esm2_results.keys()) & set(baseline_results.keys()))

    esm2_aucs = [esm2_results[d]['auc'] for d in drugs]
    baseline_aucs = [baseline_results[d]['auc'] for d in drugs]

    x = np.arange(len(drugs))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width/2, baseline_aucs, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, esm2_aucs, width, label='ESM-2', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('AUC-ROC', fontsize=14)
    ax.set_xlabel('Drug', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0.8, 1.0])

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, rotation=90)

    # Add mean lines
    ax.axhline(y=np.mean(baseline_aucs), color='#3498db', linestyle='--', alpha=0.5, label=f'Baseline mean: {np.mean(baseline_aucs):.3f}')
    ax.axhline(y=np.mean(esm2_aucs), color='#e74c3c', linestyle='--', alpha=0.5, label=f'ESM-2 mean: {np.mean(esm2_aucs):.3f}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    return fig


def plot_attention_heatmap(
    attention_differential: np.ndarray,
    drm_positions: set,
    seq_len: int,
    drug: str,
    drug_class: str,
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention differential with DRM position highlights.

    Args:
        attention_differential: Attention difference (resistant - susceptible)
        drm_positions: Set of known DRM positions (1-indexed)
        seq_len: Sequence length to plot
        drug: Drug name
        drug_class: Drug class
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(1, seq_len + 1)
    diff_plot = attention_differential[:seq_len]

    # Color by direction
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff_plot]
    bars = ax.bar(positions, diff_plot, color=colors, width=1.0, edgecolor='none')

    # Highlight DRM positions
    for pos in drm_positions:
        if pos <= seq_len:
            ax.axvline(x=pos, color='green', alpha=0.3, linewidth=2, zorder=0)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Sequence Position', fontsize=13)
    ax.set_ylabel('Attention Differential\n(Resistant - Susceptible)', fontsize=13)
    ax.set_title(f'{drug_class}/{drug}: ESM-2 Attention Profile\n(Green lines = known DRM positions)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, seq_len + 1)

    # Mark top 5 positions
    top5_idx = np.argsort(np.abs(diff_plot))[-5:]
    for idx in top5_idx:
        pos = idx + 1
        val = diff_plot[idx]
        marker = '*' if pos in drm_positions else 'o'
        ax.annotate(f'{marker}{pos}', (pos, val),
                   textcoords="offset points",
                   xytext=(0, 5 if val > 0 else -10),
                   ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).

    Args:
        y_true: True labels
        y_pred_raw: Raw predicted probabilities
        y_pred_calibrated: Calibrated probabilities (optional)
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def compute_calibration_curve(y_true, y_pred, n_bins):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins[1:-1])

        bin_means = []
        bin_true_probs = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(y_pred[mask].mean())
                bin_true_probs.append(y_true[mask].mean())
                bin_counts.append(mask.sum())

        return np.array(bin_means), np.array(bin_true_probs), np.array(bin_counts)

    # Raw predictions
    ax = axes[0]
    bin_means, bin_true_probs, bin_counts = compute_calibration_curve(y_true, y_pred_raw, n_bins)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.bar(bin_means, bin_true_probs, width=0.08, alpha=0.7, label='Model')
    ax.scatter(bin_means, bin_true_probs, s=bin_counts/10, c='red', alpha=0.5)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Raw Predictions')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()

    # Calibrated predictions (if provided)
    ax = axes[1]
    if y_pred_calibrated is not None:
        bin_means, bin_true_probs, bin_counts = compute_calibration_curve(y_true, y_pred_calibrated, n_bins)

        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.bar(bin_means, bin_true_probs, width=0.08, alpha=0.7, label='Model')
        ax.scatter(bin_means, bin_true_probs, s=bin_counts/10, c='red', alpha=0.5)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('After Platt Scaling')
    else:
        ax.text(0.5, 0.5, 'No calibrated\npredictions', ha='center', va='center', fontsize=14)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    return fig


def plot_drm_enrichment(
    validation_df: pd.DataFrame,
    figsize: Tuple[int, int] = (20, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot DRM enrichment validation results.

    Args:
        validation_df: DataFrame with enrichment results
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    top20_df = validation_df[validation_df['top_k'] == 20]

    # 1. Enrichment by drug class
    ax = axes[0]
    sns.boxplot(data=top20_df, x='drug_class', y='enrichment_ratio', ax=ax, palette='Set2')
    ax.axhline(y=1, color='red', linestyle='--', label='Expected by chance')
    ax.set_xlabel('Drug Class', fontsize=14)
    ax.set_ylabel('Enrichment Ratio', fontsize=14)
    ax.set_title('DRM Enrichment in Top-20\nAttention Positions', fontsize=14, fontweight='bold')
    ax.legend()

    # 2. Enrichment vs top-k
    ax = axes[1]
    for dc in validation_df['drug_class'].unique():
        dc_df = validation_df[validation_df['drug_class'] == dc]
        means = dc_df.groupby('top_k')['enrichment_ratio'].mean()
        stds = dc_df.groupby('top_k')['enrichment_ratio'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', label=dc, capsize=3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Top-k Positions', fontsize=14)
    ax.set_ylabel('Mean Enrichment Ratio', fontsize=14)
    ax.set_title('DRM Enrichment vs\nSelection Threshold', fontsize=14, fontweight='bold')
    ax.legend()

    # 3. P-value distribution
    ax = axes[2]
    p_values = top20_df['p_value'].values
    ax.hist(p_values, bins=20, edgecolor='white', alpha=0.7, color='#3498db')
    ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_xlabel("P-value (Fisher's Exact Test)", fontsize=14)
    ax.set_ylabel('Number of Drugs', fontsize=14)
    ax.set_title('Statistical Significance of\nDRM Enrichment', fontsize=14, fontweight='bold')
    ax.legend()

    n_sig = (p_values < 0.05).sum()
    ax.annotate(f'{n_sig}/{len(p_values)} significant\n(p<0.05)',
               xy=(0.95, 0.95), xycoords='axes fraction',
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    return fig


def plot_model_comparison_heatmap(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap comparing model performance across drugs.

    Args:
        comparison_df: DataFrame from compare_models()
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    pivot_df = comparison_df.pivot(index='drug', columns='model', values='auc')

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
               vmin=0.85, vmax=1.0, ax=ax, cbar_kws={'label': 'AUC-ROC'})

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Drug', fontsize=14)
    ax.set_title('Model Performance Comparison by Drug', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    return fig
