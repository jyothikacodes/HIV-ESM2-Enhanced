"""
Data processing utilities for Stanford HIVDB data.

This module provides functions for:
- Parsing HIVDB sequence and phenotype data
- Extracting resistance labels for binary classification
- Creating stratified train/test splits
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Drug lists by class
PI_DRUGS = ['ATV', 'DRV', 'FPV', 'IDV', 'LPV', 'NFV', 'SQV', 'TPV']
NRTI_DRUGS = ['ABC', 'AZT', 'D4T', 'DDI', '3TC', 'TDF']
NNRTI_DRUGS = ['EFV', 'ETR', 'NVP', 'RPV']

ALL_DRUGS = PI_DRUGS + NRTI_DRUGS + NNRTI_DRUGS


def get_drug_list(drug_class: Optional[str] = None) -> List[str]:
    """
    Get list of drugs by class.

    Args:
        drug_class: 'PI', 'NRTI', 'NNRTI', or None for all drugs

    Returns:
        List of drug abbreviations
    """
    if drug_class is None:
        return ALL_DRUGS
    elif drug_class == 'PI':
        return PI_DRUGS
    elif drug_class == 'NRTI':
        return NRTI_DRUGS
    elif drug_class == 'NNRTI':
        return NNRTI_DRUGS
    else:
        raise ValueError(f"Unknown drug class: {drug_class}")


def get_drug_class(drug: str) -> str:
    """
    Get the class for a given drug.

    Args:
        drug: Drug abbreviation

    Returns:
        Drug class ('PI', 'NRTI', or 'NNRTI')
    """
    if drug in PI_DRUGS:
        return 'PI'
    elif drug in NRTI_DRUGS:
        return 'NRTI'
    elif drug in NNRTI_DRUGS:
        return 'NNRTI'
    else:
        raise ValueError(f"Unknown drug: {drug}")


def load_fasta(filepath: Path) -> Tuple[List[str], List[str]]:
    """
    Load sequences from FASTA file.

    Args:
        filepath: Path to FASTA file

    Returns:
        Tuple of (sequences, sequence_ids)
    """
    try:
        from Bio import SeqIO
    except ImportError as exc:
        raise ImportError(
            "Biopython is required to load FASTA files. "
            "Install it with `pip install biopython` or add it to your environment."
        ) from exc

    sequences = []
    seq_ids = []

    with open(filepath, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_ids.append(record.id)
            sequences.append(str(record.seq))

    return sequences, seq_ids


def save_fasta(
    sequences: List[str],
    seq_ids: List[str],
    filepath: Path
) -> None:
    """
    Save sequences to FASTA file.

    Args:
        sequences: List of amino acid sequences
        seq_ids: List of sequence identifiers
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for seq_id, seq in zip(seq_ids, sequences):
            f.write(f">{seq_id}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")


def parse_hivdb_sequences(
    filepath: Path,
    min_length: int = 50,
    max_length: int = 1000
) -> pd.DataFrame:
    """
    Parse Stanford HIVDB sequence data.

    Args:
        filepath: Path to HIVDB data file (TSV or FASTA)
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include

    Returns:
        DataFrame with sequence data
    """
    if str(filepath).endswith('.fasta') or str(filepath).endswith('.fa'):
        sequences, seq_ids = load_fasta(filepath)
        df = pd.DataFrame({
            'seq_id': seq_ids,
            'sequence': sequences
        })
    else:
        # Assume TSV format
        df = pd.read_csv(filepath, sep='\t')

    # Filter by length
    if 'sequence' in df.columns:
        df['seq_length'] = df['sequence'].str.len()
        df = df[(df['seq_length'] >= min_length) & (df['seq_length'] <= max_length)]

    return df


from .feature_engineering import HIV_PROTEASE_REFERENCE, HIV_RT_REFERENCE


def reconstruct_sequences_from_positions(
    df: pd.DataFrame,
    reference: str,
    position_prefix: str = 'P'
) -> List[str]:
    """
    Reconstruct amino acid sequences from position columns in HIVDB genopheno files.

    Args:
        df: DataFrame with position columns (e.g. P1, P2, ...)
        reference: Reference amino acid sequence
        position_prefix: Prefix for position columns

    Returns:
        List of reconstructed sequences
    """
    pos_cols = sorted(
        [c for c in df.columns if c.startswith(position_prefix)
         and c[len(position_prefix):].isdigit()],
        key=lambda c: int(c[len(position_prefix):])
    )

    ref_len = min(len(reference), len(pos_cols))
    sequences = []

    for _, row in df.iterrows():
        seq = list(reference[:ref_len])
        for i, col in enumerate(pos_cols[:ref_len]):
            aa = row[col]
            if isinstance(aa, str) and aa != '-' and len(aa) == 1 and aa.isalpha():
                seq[i] = aa
        sequences.append(''.join(seq))

    return sequences


def extract_resistance_labels(
    phenotypes: pd.DataFrame,
    drug: str,
    resistance_col: str = 'class2'
) -> np.ndarray:
    """
    Extract binary resistance labels for a drug.

    The HIVDB uses fold-change (FC) values and class labels:
    - class2: 0 = susceptible, 1 = resistant (binary)
    - class3: 0 = susceptible, 1 = intermediate, 2 = resistant

    Args:
        phenotypes: DataFrame with phenotype data
        drug: Drug abbreviation
        resistance_col: Column suffix to use ('class2', 'class3', or 'FC')

    Returns:
        Binary labels array (0 = susceptible, 1 = resistant)
    """
    col_name = f"{drug}_{resistance_col}"

    if col_name not in phenotypes.columns:
        raise ValueError(f"Column {col_name} not found in phenotypes")

    labels = phenotypes[col_name].values.copy()

    # Handle class3 by binarizing (intermediate -> resistant)
    if resistance_col == 'class3':
        labels = (labels >= 1).astype(int)
    elif resistance_col == 'class2':
        labels = labels.astype(int)
    elif resistance_col == 'FC':
        # Convert fold-change to binary using standard thresholds
        labels = (labels >= 2.5).astype(int)

    return labels


def create_stratified_split(
    sequences: List[str],
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Create stratified train/test split.

    Args:
        sequences: List of sequences
        labels: Binary resistance labels
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        Tuple of (train_seqs, test_seqs, train_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split

    # Filter out samples with missing labels
    valid_mask = ~np.isnan(labels)
    valid_seqs = [s for s, v in zip(sequences, valid_mask) if v]
    valid_labels = labels[valid_mask]

    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        valid_seqs, valid_labels,
        test_size=test_size,
        stratify=valid_labels,
        random_state=random_state
    )

    return train_seqs, test_seqs, train_labels, test_labels


def load_unified_data(data_dir: Path) -> Dict:
    """
    Load unified data for all drug classes.

    Expected structure:
        data_dir/
            PI_sequences.fasta
            PI_phenotypes.csv
            NRTI_sequences.fasta
            NRTI_phenotypes.csv
            NNRTI_sequences.fasta
            NNRTI_phenotypes.csv

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with data for each drug class
    """
    unified_data = {}

    for drug_class in ['PI', 'NRTI', 'NNRTI']:
        fasta_path = data_dir / f'{drug_class}_sequences.fasta'
        pheno_path = data_dir / f'{drug_class}_phenotypes.csv'

        if fasta_path.exists() and pheno_path.exists():
            sequences, seq_ids = load_fasta(fasta_path)
            phenotypes = pd.read_csv(pheno_path)

            # Get drug columns
            exclude_cols = {'Unnamed: 0', 'seq_id', 'index', 'SeqID', 'IsolateID', 'Subtype'}
            drug_columns = [c for c in phenotypes.columns
                          if c not in exclude_cols and not c.startswith('Unnamed')]

            unified_data[drug_class] = {
                'sequences': sequences,
                'seq_ids': seq_ids,
                'phenotypes': phenotypes,
                'drugs': drug_columns
            }

    return unified_data


def get_dataset_statistics(unified_data: Dict) -> pd.DataFrame:
    """
    Compute dataset statistics.

    Args:
        unified_data: Dictionary from load_unified_data()

    Returns:
        DataFrame with statistics by drug class
    """
    stats = []

    for drug_class, data in unified_data.items():
        n_seqs = len(data['sequences'])
        seq_lengths = [len(s) for s in data['sequences']]

        stats.append({
            'drug_class': drug_class,
            'n_sequences': n_seqs,
            'n_drugs': len(data['drugs']),
            'mean_length': np.mean(seq_lengths),
            'min_length': np.min(seq_lengths),
            'max_length': np.max(seq_lengths)
        })

    return pd.DataFrame(stats)
