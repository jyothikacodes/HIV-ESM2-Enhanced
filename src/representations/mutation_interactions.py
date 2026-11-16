"""
Epistatic Mutation Interaction Modeling for HIV-1 Drug Resistance.

Biologically grounded modeling of pairwise mutation interactions:
1. Fitness compensation (e.g. Primary protease DRMs D30N/V82A/L90M + accessory M46I/I54V/A71V).
2. Resensitization / antagonism (e.g. M184V reversing AZT/TDF resistance caused by TAMs).
3. Mutually exclusive evolutionary pathways (e.g. K65R vs L74V).

Features:
- Rigorous training-fold-only co-occurrence filtering (avoids O(M^2) combinatorial explosion).
- Min-cooccurrence thresholding and optional statistical association ranking.
- Supports representation modes: 'mutation_only', 'interaction_only', and 'combined'.
- L1/ElasticNet regularized models for interpretable sparse interaction weights.
"""

from typing import Dict, List, Literal, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .mutation_features import FoldSpecificMutationEncoder, STANDARD_AMINO_ACIDS


InteractionMode = Literal["mutation_only", "interaction_only", "combined"]


class PairwiseMutationInteractionEncoder:
    """
    Encoder that generates filtered pairwise epistatic interaction features.
    
    Guarantees:
    - Pairwise interactions are derived and filtered strictly on training fold sequences.
    - Rare combinations below min_cooccurrence are pruned to prevent p >> n overfitting.
    - Validation and test sequences are transformed using the fixed training vocabulary.
    """
    
    def __init__(
        self,
        reference_seq: str,
        min_cooccurrence: int = 5,
        max_interactions: Optional[int] = 200,
        rare_mutation_threshold: float = 0.01
    ):
        """
        Args:
            reference_seq: Consensus wildtype reference sequence.
            min_cooccurrence: Minimum number of training isolates carrying both mutations.
            max_interactions: Maximum number of interaction pairs to retain.
            rare_mutation_threshold: Frequency below which a mutation is rare.
        """
        self.reference_seq = reference_seq
        self.seq_len = len(reference_seq)
        self.min_cooccurrence = min_cooccurrence
        self.max_interactions = max_interactions
        self.mutation_encoder = FoldSpecificMutationEncoder(
            reference_seq=reference_seq,
            rare_threshold=rare_mutation_threshold
        )
        
        self.is_fitted = False
        self.interaction_pairs: List[Tuple[Tuple[int, str], Tuple[int, str]]] = []
        self.pair_to_idx: Dict[Tuple[Tuple[int, str], Tuple[int, str]], int] = {}
        self.pair_cooccurrences: Dict[Tuple[Tuple[int, str], Tuple[int, str]], int] = {}

    def fit(self, training_sequences: List[str], y: Optional[np.ndarray] = None) -> "PairwiseMutationInteractionEncoder":
        """
        Fit encoder on training sequences strictly within training fold.
        """
        self.mutation_encoder.fit(training_sequences)
        known_muts = self.mutation_encoder.known_mutations_list
        mut_to_idx = self.mutation_encoder.mutation_to_idx
        
        n_samples = len(training_sequences)
        # Binary mutation matrix on train
        X_single = self.mutation_encoder.transform_binary_mutations(training_sequences)
        
        # Count pairwise co-occurrences
        # Dot product: C = X^T X gives co-occurrence counts
        # (N, M)^T @ (N, M) -> (M, M)
        cooccur_matrix = (X_single.T @ X_single).astype(int)
        
        candidates = []
        M = len(known_muts)
        for i in range(M):
            for j in range(i + 1, M):
                count = cooccur_matrix[i, j]
                if count >= self.min_cooccurrence:
                    mut_a = known_muts[i]
                    mut_b = known_muts[j]
                    candidates.append(((mut_a, mut_b), count))
                    
        # If max_interactions is set and candidates exceed it, rank by count
        # or correlation with phenotype if y is provided
        if y is not None and len(candidates) > (self.max_interactions or 0) > 0:
            valid_y = ~np.isnan(y)
            ranked_candidates = []
            for (mut_a, mut_b), count in candidates:
                idx_a = mut_to_idx[mut_a]
                idx_b = mut_to_idx[mut_b]
                pair_feat = X_single[valid_y, idx_a] * X_single[valid_y, idx_b]
                # Pearson correlation with phenotype on train
                corr, _ = stats.pointbiserialr(pair_feat, y[valid_y])
                ranked_candidates.append(((mut_a, mut_b), count, abs(corr) if not np.isnan(corr) else 0.0))
            ranked_candidates.sort(key=lambda x: x[2], reverse=True)
            candidates = [(x[0], x[1]) for x in ranked_candidates[:self.max_interactions]]
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
            if self.max_interactions is not None:
                candidates = candidates[:self.max_interactions]
                
        self.interaction_pairs = [c[0] for c in candidates]
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.interaction_pairs)}
        self.pair_cooccurrences = {c[0]: c[1] for c in candidates}
        self.is_fitted = True
        return self

    def transform(
        self,
        sequences: List[str],
        mode: InteractionMode = "combined"
    ) -> np.ndarray:
        """
        Transform sequences into mutation / interaction / combined feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before calling transform.")
            
        X_single = self.mutation_encoder.transform_binary_mutations(sequences)
        if mode == "mutation_only":
            return X_single
            
        N = len(sequences)
        P = len(self.interaction_pairs)
        X_pairs = np.zeros((N, P), dtype=np.float32)
        
        mut_to_idx = self.mutation_encoder.mutation_to_idx
        for p_idx, (mut_a, mut_b) in enumerate(self.interaction_pairs):
            if mut_a in mut_to_idx and mut_b in mut_to_idx:
                idx_a = mut_to_idx[mut_a]
                idx_b = mut_to_idx[mut_b]
                X_pairs[:, p_idx] = X_single[:, idx_a] * X_single[:, idx_b]
                
        if mode == "interaction_only":
            return X_pairs
        elif mode == "combined":
            return np.hstack([X_single, X_pairs])
        else:
            raise ValueError(f"Unknown mode: {mode}")


class EpistaticInteractionClassifier:
    """
    Classifier equipped with sparse regularized learning to analyze epistatic interactions.
    """
    def __init__(
        self,
        reference_seq: str,
        mode: InteractionMode = "combined",
        penalty: str = "l1",
        C: float = 1.0,
        min_cooccurrence: int = 5,
        max_interactions: int = 150,
        random_state: int = 42
    ):
        self.reference_seq = reference_seq
        self.mode = mode
        self.penalty = penalty
        self.C = C
        self.min_cooccurrence = min_cooccurrence
        self.max_interactions = max_interactions
        self.random_state = random_state
        
        self.encoder = PairwiseMutationInteractionEncoder(
            reference_seq=reference_seq,
            min_cooccurrence=min_cooccurrence,
            max_interactions=max_interactions
        )
        self.scaler = StandardScaler(with_mean=False)
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver="liblinear" if penalty == "l1" else "lbfgs",
            class_weight="balanced",
            max_iter=500,
            random_state=random_state
        )

    def fit(self, training_sequences: List[str], y: np.ndarray) -> "EpistaticInteractionClassifier":
        valid = ~np.isnan(y)
        seqs_clean = [training_sequences[i] for i in range(len(training_sequences)) if valid[i]]
        y_clean = y[valid].astype(int)
        
        self.encoder.fit(seqs_clean, y_clean)
        X = self.encoder.transform(seqs_clean, mode=self.mode)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_clean)
        return self

    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        X = self.encoder.transform(sequences, mode=self.mode)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def extract_top_interactions(self, top_k: int = 10) -> pd.DataFrame:
        """
        Extract the strongest positive (synergistic) and negative (antagonistic) interaction pairs.
        """
        if self.mode == "mutation_only" or len(self.encoder.interaction_pairs) == 0:
            return pd.DataFrame()
            
        n_single = len(self.encoder.mutation_encoder.known_mutations_list)
        coefs = self.model.coef_[0]
        
        records = []
        if self.mode == "combined":
            pair_coefs = coefs[n_single:]
        else:
            pair_coefs = coefs
            
        for idx, (mut_a, mut_b) in enumerate(self.encoder.interaction_pairs):
            if idx < len(pair_coefs):
                w = pair_coefs[idx]
                records.append({
                    "mut_a": f"{mut_a[1]}{mut_a[0]+1}",
                    "mut_b": f"{mut_b[1]}{mut_b[0]+1}",
                    "pair": f"{mut_a[1]}{mut_a[0]+1} + {mut_b[1]}{mut_b[0]+1}",
                    "weight": float(w),
                    "abs_weight": abs(float(w)),
                    "cooccurrence": self.encoder.pair_cooccurrences.get((mut_a, mut_b), 0),
                    # Positive weight in regularized logistic model indicates positive statistical
                    # association conditional on main effects; must not be conflated with verified
                    # biochemical synergy due to potential subtype/lineage confounding.
                    "interaction_type": "positive_association" if w > 0 else "negative_association"
                })
                
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
        return df.head(top_k)


def compare_main_effects_vs_interactions(
    sequences: List[str],
    y: np.ndarray,
    reference_seq: str,
    n_splits: int = 5,
    min_cooccurrence: int = 5,
    max_interactions: int = 50,
    random_state: int = 42
) -> Dict[str, Union[float, Dict, pd.DataFrame]]:
    """
    Rigorously compare Main-Effects-Only model vs Main-Effects-Plus-Interactions model.
    
    Statistical Formulation:
        Model 0 (Main Effects Only):
            logit(P(Y=1)) = beta_0 + sum_j beta_j * X_j
        Model 1 (Main Effects + Pairwise Interactions):
            logit(P(Y=1)) = beta_0 + sum_j beta_j * X_j + sum_{j < k} gamma_{jk} * (X_j * X_k)
            
    Evaluation Protocol:
        - Out-of-fold cross-validation on identical CV folds.
        - Evaluates incremental predictive value on test folds (AUC, Log-Loss/Cross-Entropy).
        - Tests for statistical interaction significance (paired Wilcoxon test on fold log-losses).
        - Measures interaction coefficient stability (mean, std, sign consistency across folds).
        - Prevents confusing positive statistical regression weights with biochemical synergy.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, log_loss
    
    valid = ~np.isnan(y)
    seqs_clean = [sequences[i] for i in range(len(sequences)) if valid[i]]
    y_clean = y[valid].astype(int)
    N = len(seqs_clean)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_p_main = np.zeros(N)
    oof_p_comb = np.zeros(N)
    
    fold_log_loss_main = []
    fold_log_loss_comb = []
    fold_auc_main = []
    fold_auc_comb = []
    
    pair_weights_by_fold: Dict[str, List[float]] = {}
    
    for tr_idx, te_idx in skf.split(seqs_clean, y_clean):
        tr_seqs = [seqs_clean[i] for i in tr_idx]
        te_seqs = [seqs_clean[i] for i in te_idx]
        y_tr = y_clean[tr_idx]
        y_te = y_clean[te_idx]
        
        # Fit main effects only
        clf_main = EpistaticInteractionClassifier(
            reference_seq=reference_seq,
            mode="mutation_only",
            min_cooccurrence=min_cooccurrence,
            random_state=random_state
        ).fit(tr_seqs, y_tr)
        
        p_te_main = clf_main.predict_proba(te_seqs)[:, 1]
        oof_p_main[te_idx] = p_te_main
        
        # Fit combined model (main + interactions)
        clf_comb = EpistaticInteractionClassifier(
            reference_seq=reference_seq,
            mode="combined",
            min_cooccurrence=min_cooccurrence,
            max_interactions=max_interactions,
            random_state=random_state
        ).fit(tr_seqs, y_tr)
        
        p_te_comb = clf_comb.predict_proba(te_seqs)[:, 1]
        oof_p_comb[te_idx] = p_te_comb
        
        # Fold metrics
        eps = 1e-15
        p_main_clip = np.clip(p_te_main, eps, 1 - eps)
        p_comb_clip = np.clip(p_te_comb, eps, 1 - eps)
        
        fold_log_loss_main.append(log_loss(y_te, p_main_clip))
        fold_log_loss_comb.append(log_loss(y_te, p_comb_clip))
        fold_auc_main.append(roc_auc_score(y_te, p_te_main))
        fold_auc_comb.append(roc_auc_score(y_te, p_te_comb))
        
        # Record interaction weights
        top_pairs = clf_comb.extract_top_interactions(top_k=max_interactions)
        for _, r in top_pairs.iterrows():
            pair_weights_by_fold.setdefault(r["pair"], []).append(r["weight"])
            
    # Aggregate OOF metrics
    oof_auc_main = float(roc_auc_score(y_clean, oof_p_main))
    oof_auc_comb = float(roc_auc_score(y_clean, oof_p_comb))
    oof_loss_main = float(log_loss(y_clean, np.clip(oof_p_main, 1e-15, 1 - 1e-15)))
    oof_loss_comb = float(log_loss(y_clean, np.clip(oof_p_comb, 1e-15, 1 - 1e-15)))
    
    # Statistical significance of incremental improvement
    # Paired Wilcoxon signed-rank test on fold log-losses (lower is better)
    diff_loss = np.array(fold_log_loss_main) - np.array(fold_log_loss_comb)
    try:
        w_stat, p_value = stats.wilcoxon(diff_loss, alternative="greater")
    except Exception:
        p_value = 1.0
        
    # Analyze interaction stability across folds
    stability_records = []
    for pair, weights in pair_weights_by_fold.items():
        n_folds_present = len(weights)
        mean_w = float(np.mean(weights))
        std_w = float(np.std(weights)) if len(weights) > 1 else 0.0
        # Sign consistency: fraction of folds sharing the majority sign
        pos_frac = sum(1 for w in weights if w > 0) / n_folds_present
        stability_records.append({
            "pair": pair,
            "folds_detected": n_folds_present,
            "detection_rate": n_folds_present / n_splits,
            "mean_weight": round(mean_w, 4),
            "std_weight": round(std_w, 4),
            "sign_consistency": round(max(pos_frac, 1.0 - pos_frac), 2),
            "stable": (n_folds_present >= (n_splits // 2 + 1)) and (max(pos_frac, 1.0 - pos_frac) >= 0.8)
        })
        
    stability_df = pd.DataFrame(stability_records)
    if not stability_df.empty:
        stability_df = stability_df.sort_values(["detection_rate", "sign_consistency"], ascending=False).reset_index(drop=True)
        
    return {
        "oof_auc_main_effects": round(oof_auc_main, 4),
        "oof_auc_combined": round(oof_auc_comb, 4),
        "auc_difference": round(oof_auc_comb - oof_auc_main, 4),
        "oof_log_loss_main_effects": round(oof_loss_main, 4),
        "oof_log_loss_combined": round(oof_loss_comb, 4),
        "log_loss_reduction": round(oof_loss_main - oof_loss_comb, 4),
        "incremental_improvement_p_value": round(float(p_value), 4),
        "interaction_significantly_better": bool(p_value < 0.05 and (oof_loss_comb < oof_loss_main)),
        "stability_analysis": stability_df
    }

