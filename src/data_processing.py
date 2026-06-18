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


# ---------------------------------------------------------------------------
# HIVDB genotype-phenotype format
# ---------------------------------------------------------------------------
# The Stanford HIVDB genotype-phenotype datasets do NOT contain an amino-acid
# sequence column. Each isolate's sequence is stored *differentially* across
# position columns named P1, P2, ... Pn (one per residue of the target
# protein), where:
#   '-' / '.' / blank  -> residue matches the HXB2 consensus (wild-type)
#   single letter      -> amino-acid substitution at that position
#   several letters    -> a mixture (we take the first listed residue)
#   '~'                -> deletion (position skipped)
#   '#'                -> insertion (position skipped)
# The full protein is reconstructed by overlaying these columns onto the HXB2
# reference. CompMutList is a human-readable summary of the same mutations.

# HXB2 reference sequences (UniProt/HXB2 numbering) used for reconstruction.
HXB2_PROTEASE = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
    "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)  # 99 aa

HXB2_RT = (
    "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"
    "FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL"
    "DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVI"
    "YQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT"
    "VQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGTKALTEVIPLTEEAE"
    "LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGA"
    "HTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTP"
    "PLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQ"
    "AIYLALQDSGLEVNIVTDSQYALGIIQAQPDQSESELVNQIIEQLIKKEKVYLAWVPAHK"
    "GIGGNEQVDKLVSAGIRKVLFLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASC"
)  # >= 318 aa (covers all RT position columns in the NRTI/NNRTI datasets)

# Fold-change resistance cutoffs used in this study (Stanford HIVDB convention).
FC_RESISTANT = 3.0   # class2: fold-change >= 3.0 -> resistant
FC_HIGH = 10.0       # class3: <3 susceptible, 3-10 intermediate, >=10 resistant

_GAP_CHARS = {'-', '.', ''}
_DELETION_CHARS = {'~'}
_INSERTION_CHARS = {'#'}


def get_position_columns(df: pd.DataFrame) -> List[str]:
    """Return HIVDB position columns (P1, P2, ... Pn) sorted by position number."""
    cols = [c for c in df.columns if c.startswith('P') and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def reconstruct_sequence(
    row: pd.Series,
    position_cols: List[str],
    reference: str
) -> str:
    """
    Reconstruct one full-length protein sequence from HIVDB position columns.

    Args:
        row: A DataFrame row containing the position columns.
        position_cols: Ordered list of position column names (P1..Pn).
        reference: HXB2 reference sequence for the relevant protein.

    Returns:
        The reconstructed amino-acid sequence (gaps filled from the reference).
    """
    residues = []
    for i, col in enumerate(position_cols):
        aa = row[col]
        if pd.isna(aa):
            residues.append(reference[i] if i < len(reference) else 'X')
            continue
        aa = str(aa).strip()
        if aa in _GAP_CHARS:
            residues.append(reference[i] if i < len(reference) else 'X')
        elif aa in _DELETION_CHARS or aa in _INSERTION_CHARS:
            continue  # deletion / insertion: drop the position
        elif aa[0].isalpha():
            residues.append(aa[0].upper())  # substitution or mixture (first residue)
        else:
            residues.append(reference[i] if i < len(reference) else 'X')
    return ''.join(residues)


def reconstruct_sequences(
    df: pd.DataFrame,
    gene: str
) -> Tuple[List[str], List[str]]:
    """
    Reconstruct full-length sequences for every isolate in a HIVDB dataset.

    Args:
        df: Parsed HIVDB dataset (with P1..Pn position columns).
        gene: 'PR' for protease (PI dataset) or 'RT' for reverse
            transcriptase (NRTI/NNRTI datasets).

    Returns:
        Tuple of (sequences, position_cols).
    """
    position_cols = get_position_columns(df)
    if not position_cols:
        raise ValueError(
            "No HIVDB position columns (P1, P2, ... Pn) found. The genotype-"
            "phenotype download stores sequences as position columns, not as a "
            "single amino-acid column."
        )

    reference = HXB2_PROTEASE if gene == 'PR' else HXB2_RT
    sequences = [
        reconstruct_sequence(row, position_cols, reference)
        for _, row in df.iterrows()
    ]
    return sequences, position_cols


def classify_fold_change(fold_change, threshold: float = FC_RESISTANT) -> float:
    """Binary resistance label from a fold-change value (1 = resistant)."""
    try:
        fc = float(fold_change)
    except (ValueError, TypeError):
        return np.nan
    return 1.0 if fc >= threshold else 0.0


def classify_fold_change_3class(fold_change) -> float:
    """Three-class resistance label (0 susceptible, 1 intermediate, 2 resistant)."""
    try:
        fc = float(fold_change)
    except (ValueError, TypeError):
        return np.nan
    if fc < FC_RESISTANT:
        return 0.0
    elif fc < FC_HIGH:
        return 1.0
    return 2.0


def build_phenotypes(df: pd.DataFrame, drug_class: str) -> pd.DataFrame:
    """
    Build phenotype labels from a HIVDB dataset's bare drug fold-change columns.

    The download stores fold-change under bare drug abbreviations (e.g. 'ATV',
    '3TC'). This derives the '{drug}_FC', '{drug}_class2' and '{drug}_class3'
    columns the rest of the pipeline expects.

    Args:
        df: Parsed HIVDB dataset.
        drug_class: 'PI', 'NRTI', or 'NNRTI'.

    Returns:
        DataFrame (aligned to df.index) of fold-change and class columns.
    """
    phenotypes = pd.DataFrame(index=df.index)
    for drug in get_drug_list(drug_class):
        if drug not in df.columns:
            continue
        fc = pd.to_numeric(df[drug], errors='coerce')
        phenotypes[f'{drug}_FC'] = fc
        phenotypes[f'{drug}_class2'] = fc.apply(classify_fold_change)
        phenotypes[f'{drug}_class3'] = fc.apply(classify_fold_change_3class)
    return phenotypes


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
        # Convert fold-change to binary using the study cutoff (FC >= 3.0)
        labels = (labels >= FC_RESISTANT).astype(int)

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
