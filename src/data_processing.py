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


# Reference sequences for HIV proteins
HIV_PROTEASE_REFERENCE = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
    "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)

HIV_RT_REFERENCE = (
    "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"
    "FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL"
    "DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVI"
    "YQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT"
)


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
        sequences = []
        seq_ids = []

        with open(filepath, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq_ids.append(record.id)
                sequences.append(str(record.seq))

        return sequences, seq_ids
    except ImportError:
        # Fallback to simple FASTA parser if Biopython is not installed.
        sequences = []
        seq_ids = []
        current_id = None
        current_seq = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_id is not None:
                        seq_ids.append(current_id)
                        sequences.append(''.join(current_seq))
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

        if current_id is not None:
            seq_ids.append(current_id)
            sequences.append(''.join(current_seq))

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


def _ensure_processed_data(
    processed_dir: Path,
    raw_data_dir: Path,
    force: bool = False
) -> None:
    """
    Ensure processed FASTA and phenotype files exist in the processed directory.

    If any required processed file is missing, this helper will attempt to
    build the processed artifacts from the raw HIVDB source data.
    """
    required_files = []
    for drug_class in ['PI', 'NRTI', 'NNRTI']:
        required_files.extend([
            processed_dir / f'{drug_class}_sequences.fasta',
            processed_dir / f'{drug_class}_phenotypes.csv'
        ])

    if force or any(not p.exists() for p in required_files):
        build_processed_hivdb_data(raw_data_dir, processed_dir, force=force)


def load_unified_data(
    data_dir: Path,
    raw_data_dir: Optional[Path] = None,
    force_rebuild: bool = False
) -> Dict:
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
        data_dir: Path to processed data directory
        raw_data_dir: Optional path to raw HIVDB files if processed data must be built
        force_rebuild: If True, rebuild processed files from raw data even if they exist

    Returns:
        Dictionary with data for each drug class
    """
    if raw_data_dir is None:
        raw_data_dir = data_dir.parent if data_dir.name == 'processed' else data_dir

    # Ensure processed artifacts are available when raw data exists.
    _ensure_processed_data(data_dir, raw_data_dir, force=force_rebuild)

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


def _get_raw_dataset_path(data_dir: Path, drug_class: str) -> Path:
    """
    Resolve the raw HIVDB dataset path for a drug class.

    Supports common naming conventions used by Stanford HIVDB exports.
    """
    candidates = [
        data_dir / f"{drug_class}_DataSet.txt",
        data_dir / f"{drug_class}_DataSet.tsv",
        data_dir / f"{drug_class}_dataset.txt",
        data_dir / f"{drug_class}_dataset.tsv",
        data_dir / f"{drug_class}_genopheno.csv",
        data_dir / f"{drug_class}_genopheno.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Raw dataset for {drug_class} not found in {data_dir}. "
        f"Expected one of: {[str(p.name) for p in candidates]}"
    )


def parse_hivdb_genopheno_file(filepath: Path, drug_class: str) -> pd.DataFrame:
    """
    Load and parse a Stanford HIVDB genotype-phenotype file.

    Args:
        filepath: Path to raw dataset file
        drug_class: 'PI', 'NRTI', or 'NNRTI'

    Returns:
        DataFrame with reconstructed sequences and phenotype columns.
    """
    # Support both tab-delimited and comma-delimited exports
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath, low_memory=False)
    else:
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
        except pd.errors.ParserError:
            df = pd.read_csv(filepath, sep=',', low_memory=False)

    seq_col = None
    # Prefer existing sequence column
    for candidate in ['sequence', 'seq', 'Sequence', 'SEQ', 'AAseq']:
        if candidate in df.columns:
            seq_col = candidate
            break

    if seq_col is None:
        # Reconstruct from position columns (e.g. P1, P2, ...)
        ref = HIV_PROTEASE_REFERENCE if drug_class == 'PI' else HIV_RT_REFERENCE
        df['sequence'] = reconstruct_sequences_from_positions(df, ref, position_prefix='P')
    else:
        df['sequence'] = df[seq_col].astype(str)

    # Ensure an explicit sequence ID column
    if 'SeqID' in df.columns:
        df['seq_id'] = df['SeqID'].astype(str)
    elif 'seq_id' in df.columns:
        df['seq_id'] = df['seq_id'].astype(str)
    elif 'IsolateID' in df.columns:
        df['seq_id'] = df['IsolateID'].astype(str)
    else:
        df['seq_id'] = df.index.astype(str)

    # Keep the raw phenotype columns and sequence metadata
    phenotypes = df.copy()
    return phenotypes


def build_processed_hivdb_data(
    raw_data_dir: Path,
    processed_dir: Path,
    force: bool = False
) -> None:
    """
    Build processed FASTA and phenotype CSV files from raw HIVDB files.

    Args:
        raw_data_dir: Directory containing raw HIVDB files
        processed_dir: Output directory for processed artifacts
        force: If True, overwrite existing processed files
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    drug_classes = ['PI', 'NRTI', 'NNRTI']
    drug_lists = {
        'PI': PI_DRUGS,
        'NRTI': NRTI_DRUGS,
        'NNRTI': NNRTI_DRUGS,
    }

    for drug_class in drug_classes:
        try:
            raw_path = _get_raw_dataset_path(raw_data_dir, drug_class)
        except FileNotFoundError:
            # Skip classes for which raw data is not available.
            continue

        phenotypes = parse_hivdb_genopheno_file(raw_path, drug_class)
        drug_cols = [col for col in phenotypes.columns if col in drug_lists[drug_class]]

        # Save sequences FASTA
        fasta_path = processed_dir / f"{drug_class}_sequences.fasta"
        pheno_path = processed_dir / f"{drug_class}_phenotypes.csv"

        if fasta_path.exists() and pheno_path.exists() and not force:
            continue

        sequences = phenotypes['sequence'].astype(str).tolist()
        seq_ids = phenotypes['seq_id'].astype(str).tolist()
        save_fasta(sequences, seq_ids, fasta_path)

        # Save phenotype table (including sequence IDs and drug columns)
        output_cols = ['seq_id'] + [c for c in phenotypes.columns if c != 'sequence']
        phenotypes[output_cols].to_csv(pheno_path, index=False)


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
