"""
Model training and prediction for HIV drug resistance.

This module provides functions for:
- Training logistic regression and XGBoost classifiers
- Per-drug model training
- Cross-validation evaluation
- PyTorch-based attention-weighted pooling model
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score


class AttentionWeightedClassifier(nn.Module):
    """
    Classifier with learned attention-weighted pooling.
    
    Implements the architecture described in the manuscript:
    1. Input: Per-residue ESM-2 embeddings
    2. Attention: Two-layer neural network (Linear -> Tanh -> Linear -> Softmax)
    3. Pooling: Weighted average of embeddings
    4. Classifier: Linear layer (equivalent to Logistic Regression)
    """
    def __init__(self, input_dim: int = 1280, attention_hidden_dim: int = 256):
        super().__init__()
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        # Classification head
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x, mask=None, rare_mutation_weights=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Boolean mask of shape (batch_size, seq_len), 0 for padding
            rare_mutation_weights: Optional tensor of shape (batch_size, seq_len)
                with per-residue rarity weights. When provided, attention weights
                are multiplied by these values and renormalized. When None,
                behavior is identical to the original implementation.
            
        Returns:
            logits: Classification logits (batch_size, 1)
            attention_weights: Normalized attention weights (batch_size, seq_len)
        """
        # Compute raw attention scores: (batch, seq_len, 1)
        scores = self.attention(x)
        
        # Squeeze to (batch, seq_len)
        scores = scores.squeeze(-1)
        
        # Mask padding positions (set to -infinity before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Normalize weights
        weights = F.softmax(scores, dim=1)
        
        # Integrate rare mutation weighting (if provided)
        if rare_mutation_weights is not None:
            weights = weights * rare_mutation_weights
            # Re-normalize so weights sum to 1 per sample
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Weighted pooling: (batch, 1, seq_len) @ (batch, seq_len, dim) -> (batch, 1, dim)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, weights

    def extract_pooled_vectors(
        self,
        x,
        mask=None,
        rare_mutation_weights=None,
    ) -> torch.Tensor:
        """Return the attention-pooled embedding vector(s) without classification."""
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        if rare_mutation_weights is not None:
            weights = weights * rare_mutation_weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled


class MultiHeadAttentionPoolingClassifier(nn.Module):
    """Multi-head attention pooling followed by a projection and classifier."""
    def __init__(
        self,
        input_dim: int = 1280,
        attention_hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1),
            )
            for _ in range(n_heads)
        ])
        self.attention_dropout = nn.Dropout(dropout)
        self.project = nn.Linear(n_heads * input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None, rare_mutation_weights=None):
        """Return logits and attention weights for each head."""
        raw_scores = torch.stack(
            [head(x).squeeze(-1) for head in self.attention_heads], dim=-1
        )
        if mask is not None:
            raw_scores = raw_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        weights = F.softmax(raw_scores, dim=1)
        if rare_mutation_weights is not None:
            weights = weights * rare_mutation_weights.unsqueeze(-1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        weights = self.attention_dropout(weights)
        pooled = torch.einsum('bsh,bsd->bhd', weights, x)
        pooled = pooled.view(pooled.size(0), -1)
        pooled = self.project(pooled)
        logits = self.classifier(pooled)
        return logits, weights

    def extract_pooled_vectors(
        self,
        x,
        mask=None,
        rare_mutation_weights=None,
    ) -> torch.Tensor:
        raw_scores = torch.stack(
            [head(x).squeeze(-1) for head in self.attention_heads], dim=-1
        )
        if mask is not None:
            raw_scores = raw_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        weights = F.softmax(raw_scores, dim=1)
        if rare_mutation_weights is not None:
            weights = weights * rare_mutation_weights.unsqueeze(-1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = torch.einsum('bsh,bsd->bhd', weights, x)
        pooled = pooled.view(pooled.size(0), -1)
        pooled = self.project(pooled)
        return pooled


class EmbeddingDataset(Dataset):
    """Dataset for variable-length embeddings with optional rare mutation weights."""
    def __init__(self, embeddings_list, labels, rare_mutation_weights_list=None):
        self.embeddings = embeddings_list
        self.labels = torch.FloatTensor(labels)
        self.rare_weights = rare_mutation_weights_list
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        emb = torch.FloatTensor(self.embeddings[idx])
        label = self.labels[idx]
        if self.rare_weights is not None:
            rw = torch.FloatTensor(self.rare_weights[idx])
            return emb, label, rw
        return emb, label, None


def collate_embeddings(batch):
    """Collate function to pad variable-length embedding sequences.
    
    Handles both legacy (emb, label) and extended (emb, label, rare_weights)
    tuple formats from EmbeddingDataset.
    """
    # Unpack — supports both 2-element and 3-element tuples
    if len(batch[0]) == 3:
        embeddings, labels, rare_weights_items = zip(*batch)
        has_rare_weights = rare_weights_items[0] is not None
    else:
        embeddings, labels = zip(*batch)
        has_rare_weights = False
        rare_weights_items = None
    
    # Get lengths
    lengths = torch.LongTensor([len(e) for e in embeddings])
    max_len = max(lengths)
    
    # Pad sequences
    input_dim = embeddings[0].shape[1]
    padded_embeddings = torch.zeros(len(embeddings), max_len, input_dim)
    mask = torch.zeros(len(embeddings), max_len)
    
    for i, emb in enumerate(embeddings):
        end = lengths[i]
        padded_embeddings[i, :end, :] = emb
        mask[i, :end] = 1
    
    # Pad rare mutation weights if present
    if has_rare_weights:
        padded_rare_weights = torch.zeros(len(embeddings), max_len)
        for i, rw in enumerate(rare_weights_items):
            end = lengths[i]
            padded_rare_weights[i, :end] = rw[:end]
        return padded_embeddings, torch.FloatTensor(labels), mask, padded_rare_weights
        
    return padded_embeddings, torch.FloatTensor(labels), mask


def train_attention_model(
    embeddings_list: List[np.ndarray],
    labels: np.ndarray,
    val_embeddings_list: Optional[List[np.ndarray]] = None,
    val_labels: Optional[np.ndarray] = None,
    rare_mutation_weights_list: Optional[List[np.ndarray]] = None,
    val_rare_mutation_weights_list: Optional[List[np.ndarray]] = None,
    input_dim: int = 1280,
    attention_dim: int = 256,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-4,
    early_stopping_patience: int = 5,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> AttentionWeightedClassifier:
    """
    Train the attention-weighted classifier end-to-end.
    
    Args:
        embeddings_list: List of (seq_len, embed_dim) arrays
        labels: Binary labels
        val_embeddings_list: Validation embeddings
        val_labels: Validation labels
        rare_mutation_weights_list: Optional list of per-residue rarity
            weight vectors for training sequences. When provided, attention
            weights are modulated by these values during training.
        val_rare_mutation_weights_list: Same as above for validation set.
        input_dim: Embedding dimension
        attention_dim: Hidden dimension for attention net
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        early_stopping_patience: Stop if validation AUC does not improve for this many epochs
        device: Torch device
        verbose: Print progress
        
    Returns:
        Trained AttentionWeightedClassifier model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Prepare data
    train_dataset = EmbeddingDataset(
        embeddings_list, labels,
        rare_mutation_weights_list=rare_mutation_weights_list
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_embeddings
    )
    
    use_rare_weights = rare_mutation_weights_list is not None
    
    # Init model
    model = AttentionWeightedClassifier(input_dim, attention_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop with early stopping on validation AUC
    best_val_auc = -1.0
    best_state = None
    patience_counter = 0
    has_validation = val_embeddings_list is not None and val_labels is not None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_data in train_loader:
            # Unpack — 3 elements (legacy) or 4 elements (with rare weights)
            if use_rare_weights and len(batch_data) == 4:
                batch_emb, batch_y, batch_mask, batch_rw = batch_data
                batch_rw = batch_rw.to(device)
            else:
                batch_emb, batch_y, batch_mask = batch_data[:3]
                batch_rw = None
            
            batch_emb = batch_emb.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_emb, batch_mask, rare_mutation_weights=batch_rw)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        if val_embeddings_list is not None and val_labels is not None:
            model.eval()
            val_probs = []
            
            # Process validation in batches
            val_dataset = EmbeddingDataset(
                val_embeddings_list, val_labels,
                rare_mutation_weights_list=val_rare_mutation_weights_list
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_embeddings
            )
            
            val_use_rw = val_rare_mutation_weights_list is not None
            
            with torch.no_grad():
                for val_batch in val_loader:
                    if val_use_rw and len(val_batch) == 4:
                        v_emb, _, v_mask, v_rw = val_batch
                        v_rw = v_rw.to(device)
                    else:
                        v_emb, _, v_mask = val_batch[:3]
                        v_rw = None
                    v_emb = v_emb.to(device)
                    v_mask = v_mask.to(device)
                    logits, _ = model(v_emb, v_mask, rare_mutation_weights=v_rw)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_probs.extend(probs)
            
            val_auc = roc_auc_score(val_labels, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")

    if has_validation and best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_multihead_attention_model(
    embeddings_list: List[np.ndarray],
    labels: np.ndarray,
    val_embeddings_list: Optional[List[np.ndarray]] = None,
    val_labels: Optional[np.ndarray] = None,
    rare_mutation_weights_list: Optional[List[np.ndarray]] = None,
    val_rare_mutation_weights_list: Optional[List[np.ndarray]] = None,
    input_dim: int = 1280,
    attention_dim: int = 256,
    n_heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-4,
    early_stopping_patience: int = 5,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> MultiHeadAttentionPoolingClassifier:
    """Train a multi-head attention pooling classifier."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = EmbeddingDataset(
        embeddings_list, labels,
        rare_mutation_weights_list=rare_mutation_weights_list,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_embeddings,
    )
    use_rare_weights = rare_mutation_weights_list is not None

    model = MultiHeadAttentionPoolingClassifier(
        input_dim=input_dim,
        attention_hidden_dim=attention_dim,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0
    has_validation = val_embeddings_list is not None and val_labels is not None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_data in train_loader:
            if use_rare_weights and len(batch_data) == 4:
                batch_emb, batch_y, batch_mask, batch_rw = batch_data
                batch_rw = batch_rw.to(device)
            else:
                batch_emb, batch_y, batch_mask = batch_data[:3]
                batch_rw = None

            batch_emb = batch_emb.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_emb, batch_mask, rare_mutation_weights=batch_rw)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if has_validation:
            model.eval()
            val_dataset = EmbeddingDataset(
                val_embeddings_list, val_labels,
                rare_mutation_weights_list=val_rare_mutation_weights_list,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_embeddings,
            )
            val_use_rw = val_rare_mutation_weights_list is not None
            val_probs = []
            with torch.no_grad():
                for val_batch in val_loader:
                    if val_use_rw and len(val_batch) == 4:
                        v_emb, _, v_mask, v_rw = val_batch
                        v_rw = v_rw.to(device)
                    else:
                        v_emb, _, v_mask = val_batch[:3]
                        v_rw = None
                    v_emb = v_emb.to(device)
                    v_mask = v_mask.to(device)
                    logits, _ = model(v_emb, v_mask, rare_mutation_weights=v_rw)
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())

            val_auc = roc_auc_score(val_labels, np.array(val_probs).flatten())
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")

    if has_validation and best_state is not None:
        model.load_state_dict(best_state)

    return model


def get_default_xgb_params(class_weight: float = 1.0) -> Dict:
    """
    Get default XGBoost parameters for resistance prediction.

    Args:
        class_weight: Weight for positive class (resistant)

    Returns:
        Dictionary of XGBoost parameters
    """
    return {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': class_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train logistic regression classifier with standardization.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization strength
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_scaled, y_train)

    return model, scaler


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 50,
    verbose: int = 0
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with optional early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (for early stopping)
        y_val: Validation labels
        params: XGBoost parameters (uses defaults if None)
        early_stopping_rounds: Early stopping patience
        verbose: Verbosity level

    Returns:
        Trained XGBClassifier
    """
    if params is None:
        # Compute class weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = n_neg / n_pos if n_pos > 0 else 1.0
        params = get_default_xgb_params(class_weight)

    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
    else:
        model.fit(X_train, y_train, verbose=verbose)

    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 10,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = 'rbf',
    random_state: int = 42
) -> Tuple[SVC, StandardScaler]:
    """
    Train SVM classifier with standardization.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter
        kernel: Kernel type
        random_state: Random seed

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = SVC(
        C=C,
        kernel=kernel,
        probability=True,
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_scaled, y_train)

    return model, scaler


def per_drug_training(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    model_type: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Train models for each drug and evaluate with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features) or list of embeddings
        phenotypes: DataFrame with drug resistance labels
        drugs: List of drug names to train models for
        model_type: 'logistic', 'xgboost', 'rf', 'svm', or 'attention'
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with results per drug
    """
    results = {}

    for drug in drugs:
        # Get binary labels (prefer explicit binarized columns)
        class2_col = f"{drug}_class2"
        class3_col = f"{drug}_class3"
        fc_col = f"{drug}_FC"

        if class2_col in phenotypes.columns:
            y = phenotypes[class2_col].values
        elif class3_col in phenotypes.columns:
            # Prefer class3 if class2 isn't available (may encode resistance differently)
            y = phenotypes[class3_col].values
        elif fc_col in phenotypes.columns:
            # Binarize fold-change values using the standard threshold (>=2.5)
            fc_vals = phenotypes[fc_col].values
            y = np.full_like(fc_vals, np.nan, dtype=float)
            valid_fc = ~np.isnan(fc_vals)
            y[valid_fc] = (fc_vals[valid_fc] >= 2.5).astype(float)
        elif drug in phenotypes.columns:
            # Fall back to the raw drug column if present (assume already binary)
            y = phenotypes[drug].values
        else:
            reason = "no label column found"
            print(f"  Skipping {drug}: {reason}")
            results[drug] = {
                'auc': np.nan,
                'n_samples': 0,
                'n_resistant': 0,
                'n_susceptible': 0,
                'skipped': True,
                'reason': reason
            }
            continue

        # Filter valid samples
        valid_mask = ~np.isnan(y)
        
        if isinstance(X, list):
            X_valid = [X[i] for i in range(len(X)) if valid_mask[i]]
        else:
            X_valid = X[valid_mask]
            
        y_valid = y[valid_mask].astype(int)

        n_resistant = int(y_valid.sum())
        n_susceptible = int(len(y_valid) - n_resistant)

        if len(np.unique(y_valid)) < 2:
            reason = "single class only"
            print(f"  Skipping {drug}: {reason}")
            results[drug] = {
                'auc': np.nan,
                'n_samples': int(len(y_valid)),
                'n_resistant': n_resistant,
                'n_susceptible': n_susceptible,
                'skipped': True,
                'reason': reason
            }
            continue

        # Ensure we have enough samples per class for stratified CV
        unique, counts = np.unique(y_valid, return_counts=True)
        min_class_count = counts.min()

        if min_class_count < 2:
            reason = f"not enough samples in at least one class (min={min_class_count})"
            print(f"  Skipping {drug}: {reason}")
            results[drug] = {
                'auc': np.nan,
                'n_samples': int(len(y_valid)),
                'n_resistant': n_resistant,
                'n_susceptible': n_susceptible,
                'skipped': True,
                'reason': reason
            }
            continue

        effective_splits = min(n_splits, int(min_class_count))
        if effective_splits < 2:
            reason = "effective CV folds < 2"
            print(f"  Skipping {drug}: {reason}")
            results[drug] = {
                'auc': np.nan,
                'n_samples': int(len(y_valid)),
                'n_resistant': n_resistant,
                'n_susceptible': n_susceptible,
                'skipped': True,
                'reason': reason
            }
            continue

        # Cross-validation predictions
        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

        # Placeholder for predictions
        y_pred = np.zeros(len(y_valid))

        if model_type == 'attention':
            # Custom CV loop for attention model
            for train_idx, val_idx in cv.split(np.zeros(len(y_valid)), y_valid):
                # Handle list indexing
                X_train_fold = [X_valid[i] for i in train_idx]
                X_val_fold = [X_valid[i] for i in val_idx]
                y_train_fold = y_valid[train_idx]
                y_val_fold = y_valid[val_idx]
                
                # Train
                model = train_attention_model(
                    X_train_fold, y_train_fold,
                    val_embeddings_list=X_val_fold,
                    val_labels=y_val_fold,
                    epochs=20,
                    verbose=False
                )
                
                # Predict
                model.eval()
                val_dataset = EmbeddingDataset(X_val_fold, y_val_fold)
                val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_embeddings)
                
                fold_preds = []
                with torch.no_grad():
                    device = next(model.parameters()).device
                    for v_emb, _, v_mask in val_loader:
                        v_emb = v_emb.to(device)
                        v_mask = v_mask.to(device)
                        logits, _ = model(v_emb, v_mask)
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        fold_preds.extend(probs)
                        
                y_pred[val_idx] = fold_preds
                
        elif model_type == 'logistic':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
            y_pred = cross_val_predict(model, X_scaled, y_valid, cv=cv, method='predict_proba')[:, 1]
        elif model_type == 'xgboost':
            n_pos = y_valid.sum()
            n_neg = len(y_valid) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight, random_state=random_state,
                eval_metric='auc', n_jobs=-1
            )
            y_pred = cross_val_predict(model, X_valid, y_valid, cv=cv, method='predict_proba')[:, 1]
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=random_state, n_jobs=-1)
            y_pred = cross_val_predict(model, X_valid, y_valid, cv=cv, method='predict_proba')[:, 1]
        elif model_type == 'svm':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            model = SVC(probability=True, class_weight='balanced', random_state=random_state)
            y_pred = cross_val_predict(model, X_scaled, y_valid, cv=cv, method='predict_proba')[:, 1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute AUC
        auc = roc_auc_score(y_valid, y_pred)

        results[drug] = {
            'auc': auc,
            'n_samples': len(y_valid),
            'n_resistant': int(n_resistant),
            'n_susceptible': int(n_susceptible),
            'y_true': y_valid,
            'y_pred': y_pred
        }

        print(f"  {drug}: AUC = {auc:.4f} (n={len(y_valid)}, R={n_resistant}, S={n_susceptible})")

    return results


def aggregate_drug_results(results: Dict) -> pd.DataFrame:
    """
    Aggregate per-drug results into a summary DataFrame.

    Args:
        results: Dictionary from per_drug_training()

    Returns:
        DataFrame with summary statistics
    """
    summary = []

    for drug, res in results.items():
        # Use .get with defaults to be robust against malformed result entries
        auc = res.get('auc', np.nan)
        n_samples = res.get('n_samples', 0)
        n_resistant = res.get('n_resistant', 0)
        n_susceptible = res.get('n_susceptible', 0)

        prevalence = np.nan
        try:
            if n_samples and n_samples > 0:
                prevalence = n_resistant / n_samples
        except Exception:
            prevalence = np.nan

        summary.append({
            'drug': drug,
            'auc': auc,
            'n_samples': n_samples,
            'n_resistant': n_resistant,
            'n_susceptible': n_susceptible,
            'prevalence': prevalence
        })

    df = pd.DataFrame(summary, columns=['drug', 'auc', 'n_samples', 'n_resistant', 'n_susceptible', 'prevalence'])

    if df.empty:
        # Return an empty DataFrame with expected columns rather than failing
        return df

    # Compute aggregated "MEAN" row safely
    valid_auc = df['auc'].dropna()
    mean_auc = valid_auc.mean() if not valid_auc.empty else np.nan
    total_n_samples = int(df['n_samples'].sum()) if 'n_samples' in df.columns else 0
    total_n_resistant = int(df['n_resistant'].sum()) if 'n_resistant' in df.columns else 0
    total_n_susceptible = int(df['n_susceptible'].sum()) if 'n_susceptible' in df.columns else 0
    mean_prevalence = (total_n_resistant / total_n_samples) if total_n_samples > 0 else np.nan

    mean_row = pd.DataFrame([{
        'drug': 'MEAN',
        'auc': mean_auc,
        'n_samples': total_n_samples,
        'n_resistant': total_n_resistant,
        'n_susceptible': total_n_susceptible,
        'prevalence': mean_prevalence
    }])

    df = pd.concat([df, mean_row], ignore_index=True)

    return df


def compare_models(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    model_types: List[str] = ['logistic', 'xgboost', 'rf'],
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple model types across all drugs.

    Args:
        X: Feature matrix
        phenotypes: DataFrame with resistance labels
        drugs: List of drugs to evaluate
        model_types: List of model types to compare
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        DataFrame with comparison results
    """
    comparison = []

    for model_type in model_types:
        print(f"\n{model_type.upper()}:")
        results = per_drug_training(
            X, phenotypes, drugs,
            model_type=model_type,
            n_splits=n_splits,
            random_state=random_state
        )

        for drug, res in results.items():
            comparison.append({
                'model': model_type,
                'drug': drug,
                'auc': res['auc'],
                'n_samples': res['n_samples']
            })

    return pd.DataFrame(comparison)
