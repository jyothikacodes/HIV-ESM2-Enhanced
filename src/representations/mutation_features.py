"""
Fold-specific mutation encoding and clinical cutoff definitions for HIV-1 drug resistance.

Ensures zero data leakage between cross-validation folds:
- Reference amino acids are never counted as mutations.
- Mutation frequencies and rare mutation weights are fitted exclusively on training folds.
- Drug-specific clinical resistance cutoffs aligned with Stanford HIVDB and IAS-USA guidelines.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


# Canonical 20 amino acids
STANDARD_AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")

# Stanford HIVDB Consensus Reference Sequences
HIV_PROTEASE_REFERENCE = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYD"
    "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)  # 99 aa

HIV_RT_REFERENCE = (
    "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"
    "FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL"
    "DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIY"
    "QYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWTV"
)  # 240 aa

# Drug Class Definitions
DRUG_CLASSES: Dict[str, str] = {
    # Protease Inhibitors (PI) - 99 aa
    "ATV": "PI", "DRV": "PI", "FPV": "PI", "IDV": "PI",
    "LPV": "PI", "NFV": "PI", "SQV": "PI", "TPV": "PI",
    # Nucleoside Reverse Transcriptase Inhibitors (NRTI) - 240 aa
    "3TC": "NRTI", "ABC": "NRTI", "AZT": "NRTI", "D4T": "NRTI",
    "DDI": "NRTI", "TDF": "NRTI",
    # Non-Nucleoside Reverse Transcriptase Inhibitors (NNRTI) - 240 aa
    "EFV": "NNRTI", "ETR": "NNRTI", "NVP": "NNRTI", "RPV": "NNRTI",
}

# Stanford HIVDB / IAS-USA Phenotypic Fold-Change Cutoffs
# Literature-grounded clinical cutoffs for resistance classification
# (Replaces indiscriminate single threshold across all drugs)
CLINICAL_CUTOFFS: Dict[str, Dict[str, float]] = {
    # PIs: cutoffs reflect boosted PI susceptibility breakpoints
    "ATV": {"susceptible_max": 2.0, "resistant_min": 3.0},
    "DRV": {"susceptible_max": 3.0, "resistant_min": 10.0},
    "FPV": {"susceptible_max": 2.5, "resistant_min": 3.5},
    "IDV": {"susceptible_max": 2.5, "resistant_min": 3.5},
    "LPV": {"susceptible_max": 3.0, "resistant_min": 9.0},   # LPV/r has high genetic barrier
    "NFV": {"susceptible_max": 2.5, "resistant_min": 3.5},
    "SQV": {"susceptible_max": 2.5, "resistant_min": 3.5},
    "TPV": {"susceptible_max": 2.0, "resistant_min": 3.0},
    # NRTIs: TDF cutoff is 1.4-1.5x because small fold changes confer high failure rates
    "3TC": {"susceptible_max": 2.5, "resistant_min": 3.0},   # High-level resistance via M184V is >>10x
    "ABC": {"susceptible_max": 1.5, "resistant_min": 2.0},   # Clinical resistance begins at 2x
    "AZT": {"susceptible_max": 2.5, "resistant_min": 3.0},
    "D4T": {"susceptible_max": 1.5, "resistant_min": 2.0},
    "DDI": {"susceptible_max": 1.5, "resistant_min": 2.0},
    "TDF": {"susceptible_max": 1.4, "resistant_min": 1.5},   # Critical fix for TDF class starvation
    # NNRTIs:
    "EFV": {"susceptible_max": 2.5, "resistant_min": 3.0},
    "ETR": {"susceptible_max": 2.0, "resistant_min": 2.9},
    "NVP": {"susceptible_max": 2.5, "resistant_min": 3.0},
    "RPV": {"susceptible_max": 1.5, "resistant_min": 2.0},
}


def get_clinical_cutoff(drug: str, binary: bool = True) -> Union[float, Dict[str, float]]:
    """
    Get the verified clinical cutoff for a specific antiretroviral drug.
    
    Args:
        drug: Drug code (e.g. 'TDF', 'LPV', '3TC')
        binary: If True, returns single float threshold (resistant_min).
                If False, returns dict with susceptible_max and resistant_min.
    """
    drug_upper = drug.upper()
    if drug_upper not in CLINICAL_CUTOFFS:
        # Fallback default
        return 3.0 if binary else {"susceptible_max": 2.0, "resistant_min": 3.0}
    
    cutoffs = CLINICAL_CUTOFFS[drug_upper]
    return cutoffs["resistant_min"] if binary else cutoffs


def binarize_resistance(
    values: Union[np.ndarray, pd.Series, List[float]],
    drug: str
) -> np.ndarray:
    """
    Binarize continuous fold-change resistance values using drug-specific clinical cutoffs.
    NaN values are strictly preserved as NaN.
    
    0 = Susceptible, 1 = Resistant.
    """
    arr = np.array(values, dtype=float)
    threshold = get_clinical_cutoff(drug, binary=True)
    binary = np.full(arr.shape, np.nan, dtype=float)
    valid_mask = ~np.isnan(arr)
    binary[valid_mask] = (arr[valid_mask] >= threshold).astype(float)
    return binary


def categorize_resistance_3class(
    values: Union[np.ndarray, pd.Series, List[float]],
    drug: str
) -> np.ndarray:
    """
    Categorize continuous fold-change values into 3 classes:
    0 = Susceptible
    1 = Intermediate
    2 = Resistant
    NaN values are preserved as NaN.
    """
    arr = np.array(values, dtype=float)
    cutoffs = get_clinical_cutoff(drug, binary=False)
    cat = np.full(arr.shape, np.nan, dtype=float)
    valid_mask = ~np.isnan(arr)
    
    sus_mask = valid_mask & (arr < cutoffs["susceptible_max"])
    res_mask = valid_mask & (arr >= cutoffs["resistant_min"])
    int_mask = valid_mask & (~sus_mask) & (~res_mask)
    
    cat[sus_mask] = 0.0
    cat[int_mask] = 1.0
    cat[res_mask] = 2.0
    return cat


class FoldSpecificMutationEncoder:
    """
    Mutation encoder fitted strictly on training fold data.
    
    Guarantees:
    1. Reference amino acids are NEVER treated as mutations (prevents false rare mutations).
    2. Position-specific mutation frequencies are computed exclusively from training data.
    3. Rare mutation weights reflect true mutation rarity in the training cohort.
    4. Validation and test folds do not leak into frequency distributions.
    """
    
    def __init__(
        self,
        reference_seq: str,
        rare_threshold: float = 0.01,
        laplace_alpha: float = 1.0
    ):
        """
        Args:
            reference_seq: Consensus wildtype amino acid sequence (e.g. Protease 99 aa).
            rare_threshold: Frequency threshold below which a mutation is classified as rare (default 1%).
            laplace_alpha: Additive smoothing parameter for unseen mutations in validation/test.
        """
        self.reference_seq = reference_seq
        self.seq_len = len(reference_seq)
        self.rare_threshold = rare_threshold
        self.laplace_alpha = laplace_alpha
        
        # Fitted attributes (strictly populated in fit())
        self.is_fitted = False
        self.n_train_samples = 0
        self.mutation_counts: Dict[Tuple[int, str], int] = {}
        self.mutation_frequencies: Dict[Tuple[int, str], float] = {}
        self.rare_mutation_weights: Dict[Tuple[int, str], float] = {}
        self.known_mutations_list: List[Tuple[int, str]] = []
        self.mutation_to_idx: Dict[Tuple[int, str], int] = {}
        
    def fit(self, training_sequences: List[str]) -> "FoldSpecificMutationEncoder":
        """
        Fit encoder on training sequences.
        
        Args:
            training_sequences: List of amino acid sequences in the training fold.
        """
        self.n_train_samples = len(training_sequences)
        if self.n_train_samples == 0:
            raise ValueError("Cannot fit FoldSpecificMutationEncoder on empty training sequences.")
            
        self.mutation_counts = {}
        observed_mutations: Set[Tuple[int, str]] = set()
        
        for seq in training_sequences:
            limit = min(len(seq), self.seq_len)
            for i in range(limit):
                aa = seq[i]
                ref_aa = self.reference_seq[i]
                
                # CRITICAL AUDIT FIX:
                # 1. Must be a valid standard amino acid
                # 2. Must NOT match reference sequence (wildtype is NOT a mutation)
                # 3. Must not be gap/ambiguous character
                if aa in STANDARD_AMINO_ACIDS and aa != ref_aa:
                    key = (i, aa)
                    self.mutation_counts[key] = self.mutation_counts.get(key, 0) + 1
                    observed_mutations.add(key)
                    
        # Sort mutations deterministically by position, then amino acid
        self.known_mutations_list = sorted(list(observed_mutations), key=lambda x: (x[0], x[1]))
        self.mutation_to_idx = {mut: idx for idx, mut in enumerate(self.known_mutations_list)}
        
        # Compute frequencies and inverse-frequency rare weights
        self.mutation_frequencies = {}
        self.rare_mutation_weights = {}
        
        for key, count in self.mutation_counts.items():
            freq = count / self.n_train_samples
            self.mutation_frequencies[key] = freq
            
            # Smooth inverse frequency weight: w = log(1 + 1 / (freq + eps))
            # Rare mutations receive higher weight; common mutations receive lower weight
            weight = np.log1p(1.0 / (freq + 1e-4))
            self.rare_mutation_weights[key] = float(weight)
            
        self.is_fitted = True
        return self

    def transform_binary_mutations(self, sequences: List[str]) -> np.ndarray:
        """
        Transform sequences into a binary mutation matrix (N, num_known_mutations)
        based strictly on mutations observed during training.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before calling transform.")
            
        n_samples = len(sequences)
        n_features = len(self.known_mutations_list)
        matrix = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for s_idx, seq in enumerate(sequences):
            limit = min(len(seq), self.seq_len)
            for i in range(limit):
                aa = seq[i]
                if aa in STANDARD_AMINO_ACIDS and aa != self.reference_seq[i]:
                    key = (i, aa)
                    if key in self.mutation_to_idx:
                        matrix[s_idx, self.mutation_to_idx[key]] = 1.0
                        
        return matrix

    def compute_positional_rare_weights(
        self,
        sequences: List[str],
        max_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-residue rare mutation weights and rare mutation masks.
        
        Returns:
            weights: Array of shape (N, L), default 1.0 for wildtype/no mutation.
            rare_mask: Array of shape (N, L), 1.0 where a rare mutation exists, 0.0 otherwise.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before calling transform.")
            
        L = max_len if max_len is not None else self.seq_len
        N = len(sequences)
        weights = np.ones((N, L), dtype=np.float32)
        rare_mask = np.zeros((N, L), dtype=np.float32)
        
        # Default weight for a novel/unseen rare mutation in test data (smoothed)
        unseen_rare_weight = float(np.log1p(1.0 / (1.0 / (self.n_train_samples + 1) + 1e-4)))
        
        for s_idx, seq in enumerate(sequences):
            limit = min(len(seq), L, self.seq_len)
            for i in range(limit):
                aa = seq[i]
                ref_aa = self.reference_seq[i]
                
                # Check for genuine mutation
                if aa in STANDARD_AMINO_ACIDS and aa != ref_aa:
                    key = (i, aa)
                    if key in self.rare_mutation_weights:
                        w = self.rare_mutation_weights[key]
                        freq = self.mutation_frequencies[key]
                    else:
                        # Mutation was not seen in training set: definitely rare
                        w = unseen_rare_weight
                        freq = 0.0
                        
                    weights[s_idx, i] = w
                    if freq < self.rare_threshold:
                        rare_mask[s_idx, i] = 1.0
                else:
                    # Wildtype or gap: neutral weight 1.0, rare_mask 0.0
                    weights[s_idx, i] = 1.0
                    rare_mask[s_idx, i] = 0.0
                    
        return weights, rare_mask
