"""Feature selection utilities for improving model accuracy."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb


def select_top_features_mi(X, y, n_features=100, random_state=42):
    """Select top features using mutual information scoring."""
    n_features = min(n_features, X.shape[1])
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    selector = SelectKBest(score_func=lambda X, y: mi_scores, k=n_features)
    X_selected = selector.fit_transform(X, y)
    return X_selected, mi_scores


def filter_correlated_features(X, threshold=0.95):
    """Remove highly correlated features to reduce multicollinearity."""
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.abs(corr_matrix)
    features_to_remove = set()
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if corr_matrix[i, j] > threshold:
                features_to_remove.add(j)
    remaining_indices = [i for i in range(X.shape[1]) if i not in features_to_remove]
    X_filtered = X[:, remaining_indices]
    return X_filtered, remaining_indices


def recursive_feature_elimination(X, y, n_features=100, random_state=42):
    """Recursive feature elimination using XGBoost feature importance."""
    n_features = min(n_features, X.shape[1])
    remaining_indices = list(range(X.shape[1]))
    X_current = X.copy()
    
    while X_current.shape[1] > n_features:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric='auc',
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_current, y, verbose=False)
        importance = model.feature_importances_
        least_important_idx = np.argmin(importance)
        remaining_indices.pop(least_important_idx)
        X_current = np.delete(X_current, least_important_idx, axis=1)
    
    return X_current, remaining_indices


def apply_pca(X, n_components=None, variance_explained=0.95):
    """Apply Principal Component Analysis for dimensionality reduction."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if n_components is None:
        pca = PCA(n_components=variance_explained)
    else:
        pca = PCA(n_components=n_components)
    
    X_transformed = pca.fit_transform(X_scaled)
    return X_transformed, pca


def select_features_for_accuracy(X, y, method='mutual_info', n_features=None, random_state=42):
    """Intelligently select features to improve model accuracy."""
    if n_features is None:
        n_features = max(50, X.shape[0] // 10)
        n_features = min(n_features, X.shape[1])
    
    metadata = {
        'original_n_features': X.shape[1],
        'target_n_features': n_features,
        'method': method,
    }
    
    # Step 1: Filter correlated features
    X_filtered, remaining_indices = filter_correlated_features(X, threshold=0.95)
    metadata['after_correlation_filter'] = X_filtered.shape[1]
    
    # Step 2: Apply selected feature selection method
    if method == 'mutual_info':
        X_selected, mi_scores = select_top_features_mi(
            X_filtered, y, n_features=n_features, random_state=random_state
        )
        metadata['feature_scores'] = mi_scores
    elif method == 'rfe':
        X_selected, selected_feature_indices = recursive_feature_elimination(
            X_filtered, y, n_features=n_features, random_state=random_state
        )
        metadata['selected_indices'] = selected_feature_indices
    elif method == 'pca':
        X_selected, pca = apply_pca(X_filtered, n_components=n_features)
        metadata['pca_components'] = pca.n_components_
        metadata['explained_variance_ratio'] = float(pca.explained_variance_ratio_.sum())
    elif method == 'combined':
        # Correlation filter → mutual information → RFE refinement
        n_mi = min(max(n_features * 2, n_features), X_filtered.shape[1])
        X_mi, mi_scores = select_top_features_mi(
            X_filtered, y, n_features=n_mi, random_state=random_state
        )
        metadata['feature_scores'] = mi_scores
        if X_mi.shape[1] > n_features:
            X_selected, selected_indices = recursive_feature_elimination(
                X_mi, y, n_features=n_features, random_state=random_state
            )
            metadata['selected_indices'] = selected_indices
        else:
            X_selected = X_mi
    else:
        raise ValueError("Unknown feature selection method: {}".format(method))
    
    metadata['final_n_features'] = X_selected.shape[1]
    
    return X_selected, metadata
