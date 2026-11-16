"""
Dual Sequence Preprocessing Configurations for HIV-1 Drug Resistance:

1. BenchmarkPreprocessingPipeline:
   - Reproduces Hayden Farquhar's benchmark configuration.
   - Enforces strict canonical lengths (PR=99, RT=240).
   - Excludes sequences with >10% missing positions (>9 for PR, >24 for RT).
   - Replaces mixtures using first-listed amino acid or training-fold-only consensus.
   - Logs all excluded sequences with exact failure criteria.

2. EnhancedPreprocessingPipeline:
   - Resilient preprocessing that preserves clinical isolates with indels/mixtures.
   - Alignment-aware coordinate mapping with per-residue validity masks.
   - Separates canonical functional RT (codons 1-240) from unsequenced connection domain.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from collections import Counter
import pandas as pd
import numpy as np

from ..data_processing import (
    HXB2_PROTEASE, HXB2_RT,
    get_position_columns
)

CANONICAL_LENGTHS = {
    "PR": 99,
    "RT": 240
}

MISSING_THRESHOLD_PCT = 0.10  # 10% maximum missing positions


class BenchmarkPreprocessingPipeline:
    """
    Strict Farquhar-compatible benchmark preprocessing pipeline.
    """
    def __init__(self, gene: str, exclude_stop_codons: bool = True, max_missing_pct: float = 0.10):
        self.gene = gene.upper()
        if self.gene not in CANONICAL_LENGTHS:
            raise ValueError(f"Unknown gene: {gene}. Must be PR or RT.")
            
        self.canonical_len = CANONICAL_LENGTHS[self.gene]
        self.ref_seq = HXB2_PROTEASE if self.gene == "PR" else HXB2_RT[:self.canonical_len]
        self.exclude_stop_codons = exclude_stop_codons
        self.max_missing_pct = max_missing_pct
        self.max_missing_allowed = int(self.canonical_len * max_missing_pct)
        
        self.exclusion_log: List[Dict[str, Union[str, int, float]]] = []

    def process_dataset(self, df: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
        """
        Process raw HIVDB dataframe, returning valid reconstructed sequences,
        valid SeqIDs, and the filtered dataframe.
        """
        pos_cols = [f"P{i}" for i in range(1, self.canonical_len + 1)]
        for col in pos_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required position column {col} in dataset.")
                
        valid_indices = []
        valid_sequences = []
        valid_seq_ids = []
        
        for idx, row in df.iterrows():
            seq_id = str(row.get("SeqID", f"idx_{idx}"))
            
            # 1. Check missing unsequenced positions ('.')
            n_missing = sum(1 for c in pos_cols if str(row[c]).strip() == ".")
            if n_missing > self.max_missing_allowed:
                self.exclusion_log.append({
                    "SeqID": seq_id,
                    "gene": self.gene,
                    "reason": f"excessive_missing_positions ({n_missing} > {self.max_missing_allowed})",
                    "missing_count": n_missing,
                    "missing_pct": round(n_missing / self.canonical_len * 100, 2)
                })
                continue
                
            # 2. Check premature stop codons ('*')
            has_stop = any("*" in str(row[c]) for c in pos_cols)
            if has_stop and self.exclude_stop_codons:
                self.exclusion_log.append({
                    "SeqID": seq_id,
                    "gene": self.gene,
                    "reason": "premature_stop_codon",
                    "missing_count": n_missing,
                    "missing_pct": round(n_missing / self.canonical_len * 100, 2)
                })
                continue
                
            # 3. Check deletions ('~') or insertions ('#') in core alignment
            has_indel = any("~" in str(row[c]) or "#" in str(row[c]) for c in pos_cols)
            if has_indel:
                self.exclusion_log.append({
                    "SeqID": seq_id,
                    "gene": self.gene,
                    "reason": "indel_disrupts_canonical_length",
                    "missing_count": n_missing,
                    "missing_pct": round(n_missing / self.canonical_len * 100, 2)
                })
                continue
                
            # Reconstruct sequence using first-listed amino acid convention
            seq_chars = []
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip() if pd.notna(row[col]) else "."
                if val in ["-", "."]:
                    seq_chars.append(self.ref_seq[i])
                elif val[0].isalpha():
                    seq_chars.append(val[0].upper())
                else:
                    seq_chars.append(self.ref_seq[i])
                    
            seq_str = "".join(seq_chars)
            if len(seq_str) != self.canonical_len:
                self.exclusion_log.append({
                    "SeqID": seq_id,
                    "gene": self.gene,
                    "reason": f"length_mismatch ({len(seq_str)} != {self.canonical_len})",
                    "missing_count": n_missing,
                    "missing_pct": round(n_missing / self.canonical_len * 100, 2)
                })
                continue
                
            valid_indices.append(idx)
            valid_sequences.append(seq_str)
            valid_seq_ids.append(seq_id)
            
        filtered_df = df.iloc[valid_indices].copy().reset_index(drop=True)
        return valid_sequences, valid_seq_ids, filtered_df

    def get_exclusion_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.exclusion_log)


class EnhancedPreprocessingPipeline:
    """
    Alternative resilient preprocessing pipeline that retains indels and mixtures
    using per-residue coordinate and validity masks.
    """
    def __init__(self, gene: str):
        self.gene = gene.upper()
        self.canonical_len = CANONICAL_LENGTHS[self.gene]
        self.ref_seq = HXB2_PROTEASE if self.gene == "PR" else HXB2_RT[:self.canonical_len]

    def process_dataset(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str], pd.DataFrame]:
        pos_cols = [f"P{i}" for i in range(1, self.canonical_len + 1)]
        sequences = []
        validity_masks = []
        seq_ids = []
        
        for idx, row in df.iterrows():
            seq_id = str(row.get("SeqID", f"idx_{idx}"))
            seq_chars = []
            mask = np.ones(self.canonical_len, dtype=np.float32)
            
            for i, col in enumerate(pos_cols):
                val = str(row[col]).strip() if pd.notna(row[col]) else "."
                if val in ["-"]:
                    seq_chars.append(self.ref_seq[i])
                elif val in [".", "~", "#", "*"]:
                    seq_chars.append(self.ref_seq[i])
                    mask[i] = 0.0  # Mask out unsequenced, deleted, or premature stop codons
                elif val[0].isalpha():
                    seq_chars.append(val[0].upper())
                else:
                    seq_chars.append(self.ref_seq[i])
                    mask[i] = 0.0
                    
            sequences.append("".join(seq_chars))
            validity_masks.append(mask)
            seq_ids.append(seq_id)
            
        return sequences, np.array(validity_masks), seq_ids, df.copy()
