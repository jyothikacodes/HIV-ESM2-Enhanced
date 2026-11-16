"""
Biologically Informed Multi-Task Drug Resistance Prediction.

Experiment B:
Shared trunk representation with drug-specific classification heads.
Exploits biological cross-resistance across drugs within the same class
(e.g., NRTI cross-resistance via K65R or M184V; PI active site cross-resistance).

Includes:
- MaskedMultiTaskBCELoss: strictly ignores unmeasured phenotypes (NaN labels)
  so missing assays do not bias or penalize the shared trunk.
- Uncertainty-weighted multi-task loss (Kendall & Gal).
- Scikit-learn compatible MultiTaskDrugPredictor.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
    _BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    _HAS_TORCH = False
    _BaseModule = object

from sklearn.linear_model import LogisticRegression


if _HAS_TORCH:
    class MaskedMultiTaskBCELoss(nn.Module):
        """
        Binary Cross Entropy loss with strict NaN masking for unmeasured drugs.
        
        If sample i was not phenotyped for drug d (mask[i, d] == 0),
        it contributes exactly 0 to the gradient.
        """
        def __init__(self, reduction: str = "mean"):
            super().__init__()
            self.reduction = reduction

        def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Args:
                logits: (B, D) predicted unnormalized logits.
                targets: (B, D) binary labels (0 or 1).
                mask: (B, D) binary mask (1 if measured, 0 if missing/untested).
                      If None, mask is inferred as (~torch.isnan(targets)).
            """
            if mask is None:
                mask = ~torch.isnan(targets)
                # Replace NaNs in targets with 0 to prevent NaN gradients in untouched positions
                targets = torch.where(mask, targets, torch.zeros_like(targets))
            else:
                mask = mask.bool()
                targets = torch.where(mask, targets, torch.zeros_like(targets))
                
            # Element-wise BCE with logits
            bce_per_element = F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction="none"
            )
            
            # Mask out unmeasured drug-sample pairs
            masked_loss = bce_per_element * mask.float()
            
            total_valid_entries = mask.float().sum()
            if total_valid_entries == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
                
            if self.reduction == "mean":
                return masked_loss.sum() / (total_valid_entries + 1e-8)
            elif self.reduction == "sum":
                return masked_loss.sum()
            else:
                return masked_loss


    class MultiTaskDrugTrunkNet(nn.Module):
        """
        Neural network with shared representation trunk and task-specific prediction heads.
        """
        def __init__(
            self,
            input_dim: int,
            num_drugs: int,
            trunk_hidden_dims: List[int] = [512, 256],
            dropout_rate: float = 0.3,
            use_uncertainty_weighting: bool = False
        ):
            super().__init__()
            self.num_drugs = num_drugs
            self.use_uncertainty = use_uncertainty_weighting
            
            # Shared trunk
            layers = []
            curr_dim = input_dim
            for h_dim in trunk_hidden_dims:
                layers.extend([
                    nn.Linear(curr_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                curr_dim = h_dim
            self.trunk = nn.Sequential(*layers)
            
            # Per-drug classification heads
            self.heads = nn.ModuleList([
                nn.Linear(curr_dim, 1) for _ in range(num_drugs)
            ])
            
            # Optional Kendall & Gal homoscedastic uncertainty log-variances
            if self.use_uncertainty:
                self.log_vars = nn.Parameter(torch.zeros(num_drugs))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Returns:
                logits: (B, num_drugs) predicted logits for each drug.
            """
            h = self.trunk(x)
            head_outputs = [head(h) for head in self.heads]
            logits = torch.cat(head_outputs, dim=-1)
            return logits

        def compute_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            mask: torch.Tensor,
            loss_fn: MaskedMultiTaskBCELoss
        ) -> torch.Tensor:
            """
            Compute loss with optional uncertainty weighting.
            """
            if not self.use_uncertainty:
                return loss_fn(logits, targets, mask)
                
            # Uncertainty-weighted loss per drug:
            total_loss = 0.0
            valid_tasks = 0
            for d in range(self.num_drugs):
                task_mask = mask[:, d:d+1]
                if task_mask.sum() == 0:
                    continue
                task_loss = loss_fn(logits[:, d:d+1], targets[:, d:d+1], task_mask)
                precision = torch.exp(-self.log_vars[d])
                total_loss = total_loss + 0.5 * precision * task_loss + 0.5 * self.log_vars[d]
                valid_tasks += 1
                
            return total_loss / max(valid_tasks, 1)


class MultiTaskDrugPredictor:
    """
    High-level predictor interface for multi-task drug resistance prediction.
    Falls back gracefully to independent regularized models when PyTorch is not installed.
    """
    def __init__(
        self,
        drug_names: List[str],
        trunk_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 40,
        random_state: int = 42
    ):
        self.drug_names = drug_names
        self.num_drugs = len(drug_names)
        self.trunk_hidden_dims = trunk_hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.model: Optional[Union[MultiTaskDrugTrunkNet, Dict[str, LogisticRegression]]] = None

    def fit(
        self,
        X: np.ndarray,
        Y_matrix: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> "MultiTaskDrugPredictor":
        """
        Fit multi-task model on cohort.
        
        Args:
            X: Feature matrix of shape (N, feature_dim).
            Y_matrix: Resistance labels of shape (N, num_drugs), containing {0, 1, np.nan}.
        """
        if Y_matrix.shape[1] != self.num_drugs:
            raise ValueError(
                f"Y_matrix has {Y_matrix.shape[1]} columns, expected {self.num_drugs} ({self.drug_names})."
            )

        if not _HAS_TORCH:
            # Fallback to independent Logistic Regression models with NaN masking
            self.model = {}
            for d_idx, drug in enumerate(self.drug_names):
                y_col = Y_matrix[:, d_idx]
                valid = ~np.isnan(y_col)
                if valid.sum() > 10 and len(np.unique(y_col[valid])) > 1:
                    clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=self.random_state)
                    clf.fit(X[valid], y_col[valid])
                    self.model[drug] = clf
            return self

        # PyTorch multi-task implementation
        torch.manual_seed(self.random_state)
        input_dim = X.shape[1]
        self.net = MultiTaskDrugTrunkNet(
            input_dim=input_dim,
            num_drugs=self.num_drugs,
            trunk_hidden_dims=self.trunk_hidden_dims,
            dropout_rate=self.dropout
        )
        
        # Prepare tensors and masks
        mask = (~np.isnan(Y_matrix)).astype(np.float32)
        y_clean = np.nan_to_num(Y_matrix, nan=0.0).astype(np.float32)
        
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y_clean),
            torch.FloatTensor(mask)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        loss_fn = MaskedMultiTaskBCELoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        
        self.net.train()
        for epoch in range(self.max_epochs):
            for batch_x, batch_y, batch_mask in loader:
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = loss_fn(logits, batch_y, batch_mask)
                loss.backward()
                optimizer.step()
                
        self.net.eval()
        self.model = self.net
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of resistance for all drugs.
        
        Returns:
            probs: (N, num_drugs) array of predicted probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
            
        N = len(X)
        probs = np.full((N, self.num_drugs), 0.5, dtype=np.float32)
        
        if not _HAS_TORCH or isinstance(self.model, dict):
            for d_idx, drug in enumerate(self.drug_names):
                if drug in self.model:
                    probs[:, d_idx] = self.model[drug].predict_proba(X)[:, 1]
            return probs
            
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X)
            logits = self.net(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        return probs
