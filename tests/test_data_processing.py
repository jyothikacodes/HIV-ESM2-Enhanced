import tempfile
from pathlib import Path
import pandas as pd

from src.data_processing import build_processed_hivdb_data, parse_hivdb_genopheno_file, load_unified_data


def test_parse_hivdb_genopheno_file_reconstructs_sequence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_path = tmpdir / 'PI_DataSet.txt'
        df = pd.DataFrame({
            'SeqID': ['S1', 'S2'],
            'P1': ['M', 'A'],
            'P2': ['T', 'T'],
            'ATV_FC': [1.0, 10.0],
            'ATV_class2': [0, 1],
            'ATV_class3': [0, 2],
        })
        df.to_csv(raw_path, sep='\t', index=False)

        parsed = parse_hivdb_genopheno_file(raw_path, 'PI')
        assert 'sequence' in parsed.columns
        assert parsed.loc[0, 'seq_id'] == 'S1'
        assert parsed.loc[1, 'seq_id'] == 'S2'
        assert parsed.loc[0, 'sequence'].startswith('MT')
        assert parsed.loc[1, 'sequence'].startswith('AT')


def test_build_processed_hivdb_data_creates_fasta_and_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_dir = tmpdir / 'raw'
        processed_dir = tmpdir / 'processed'
        raw_dir.mkdir()
        processed_dir.mkdir()

        raw_path = raw_dir / 'PI_DataSet.txt'
        df = pd.DataFrame({
            'SeqID': ['S1', 'S2'],
            'P1': ['M', 'A'],
            'P2': ['T', 'T'],
            'ATV_FC': [1.0, 10.0],
            'ATV_class2': [0, 1],
            'ATV_class3': [0, 2],
        })
        df.to_csv(raw_path, sep='\t', index=False)

        build_processed_hivdb_data(raw_dir, processed_dir)

        fasta_path = processed_dir / 'PI_sequences.fasta'
        csv_path = processed_dir / 'PI_phenotypes.csv'

        assert fasta_path.exists()
        assert csv_path.exists()

        fasta_text = fasta_path.read_text()
        assert '>S1' in fasta_text
        assert '>S2' in fasta_text
        assert 'MT' in fasta_text
        assert 'AT' in fasta_text

        phenotypes = pd.read_csv(csv_path)
        assert 'seq_id' in phenotypes.columns
        assert 'ATV_FC' in phenotypes.columns


def test_load_unified_data_reads_processed_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        processed_dir = tmpdir / 'processed'
        processed_dir.mkdir()

        fasta_path = processed_dir / 'PI_sequences.fasta'
        fasta_path.write_text('>S1\nMT\n> S2\nAT\n')
        pheno_path = processed_dir / 'PI_phenotypes.csv'
        pd.DataFrame({
            'seq_id': ['S1', 'S2'],
            'ATV_FC': [1.0, 10.0]
        }).to_csv(pheno_path, index=False)

        unified = load_unified_data(processed_dir)
        assert 'PI' in unified
        assert unified['PI']['sequences'][0] == 'MT'
        assert unified['PI']['seq_ids'][0] == 'S1'
        assert 'ATV_FC' in unified['PI']['phenotypes'].columns
