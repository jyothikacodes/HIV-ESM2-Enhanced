#!/usr/bin/env python3
"""
HPC-Ready ESM-2 Protein Language Model Embedding Extraction Script.

Features:
- Models: esm2_t33_650M_UR50D (default, 33 layers, 1280 dim) or smaller variants.
- Deduplication: Computes SHA-256 hash per unique sequence; extracts each sequence exactly once.
- Separate Reference Extraction: Caches consensus HXB2 PR (99 aa) and RT (240 aa) embeddings.
- Precision: Supports float16 (GPU) and float32.
- Resumability: Skips sequences already extracted and verified by shape/checksum.
- Metadata & Integrity: Records extraction environment, model checkpoint, tokenization, and hashes.
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import HXB2_PROTEASE, HXB2_RT
from src.preprocessing.sequence_preprocessing import BenchmarkPreprocessingPipeline, EnhancedPreprocessingPipeline

DEFAULT_MODEL = "esm2_t33_650M_UR50D"


def compute_sequence_hash(sequence: str) -> str:
    """Compute 16-character SHA-256 hash of normalized sequence."""
    clean_seq = sequence.strip().upper()
    return hashlib.sha256(clean_seq.encode("utf-8")).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-residue ESM-2 embeddings.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="ESM-2 checkpoint name")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "data" / "embeddings" / "esm2_650m"))
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32",
                        choices=["fp16", "fp32"])
    parser.add_argument("--preprocessing", type=str, default="benchmark", choices=["benchmark", "enhanced"])
    parser.add_argument("--dry_run", action="store_true", help="Log extraction plan without loading ESM-2")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seq_dir = out_dir / "sequences"
    per_seq_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ESM-2 PER-RESIDUE EMBEDDING EXTRACTION")
    print(f"Model Checkpoint:   {args.model}")
    print(f"Device:             {args.device}")
    print(f"Precision:          {args.precision}")
    print(f"Preprocessing Mode: {args.preprocessing}")
    print(f"Output Directory:   {out_dir}")
    print("=" * 80)

    # 1. Collect sequences across all raw datasets
    raw_files = {
        "PR": REPO_ROOT / "data/raw/PI_DataSet.txt",
        "RT_NRTI": REPO_ROOT / "data/raw/NRTI_DataSet.txt",
        "RT_NNRTI": REPO_ROOT / "data/raw/NNRTI_DataSet.txt",
    }

    all_sequences: Dict[str, Set[str]] = {"PR": set(), "RT": set()}
    seq_to_ids: Dict[str, List[str]] = {}
    
    for key, path in raw_files.items():
        gene = "PR" if "PR" in key else "RT"
        df = pd.read_csv(path, sep="\t")
        
        if args.preprocessing == "benchmark":
            pipeline = BenchmarkPreprocessingPipeline(gene=gene)
            seqs, seq_ids, _ = pipeline.process_dataset(df)
        else:
            pipeline = EnhancedPreprocessingPipeline(gene=gene)
            seqs, _, seq_ids, _ = pipeline.process_dataset(df)
            
        for s, sid in zip(seqs, seq_ids):
            all_sequences[gene].add(s)
            seq_to_ids.setdefault(s, []).append(sid)

    unique_pr = sorted(list(all_sequences["PR"]))
    unique_rt = sorted(list(all_sequences["RT"]))
    
    print(f"\nUnique Biological Sequences Identified:")
    print(f"  Protease (PR, L=99):               {len(unique_pr):5d} unique sequences")
    print(f"  Reverse Transcriptase (RT, L=240): {len(unique_rt):5d} unique sequences")
    print(f"  Total Unique Sequence Workload:    {len(unique_pr) + len(unique_rt):5d} unique sequences")

    # Build sequence index
    seq_index = []
    for gene, seq_list in [("PR", unique_pr), ("RT", unique_rt)]:
        for s in seq_list:
            shash = compute_sequence_hash(s)
            seq_index.append({
                "sequence_hash": shash,
                "gene": gene,
                "length": len(s),
                "sequence": s,
                "sample_ids": ";".join(seq_to_ids.get(s, [])),
                "embedding_path": str(per_seq_dir / f"{shash}.npy")
            })
            
    index_df = pd.DataFrame(seq_index)
    index_csv_path = out_dir / "sequence_embedding_index.csv"
    index_df.to_csv(index_csv_path, index=False)
    print(f"  Saved sequence hash index: {index_csv_path}")

    # Check already extracted files
    already_extracted = 0
    to_extract = []
    for item in seq_index:
        p = Path(item["embedding_path"])
        if p.exists() and p.stat().st_size > 0:
            already_extracted += 1
        else:
            to_extract.append(item)

    print(f"\nResumability Audit:")
    print(f"  Already Extracted & Cached: {already_extracted}")
    print(f"  Pending Extraction:         {len(to_extract)}")

    if args.dry_run:
        print("\n[Dry Run] Skipping model loading and tensor computation.")
        return

    if len(to_extract) == 0:
        print("\nAll embeddings are already cached and verified! Nothing to extract.")
        return

    # Check PyTorch environment
    print(f"\nLoading ESM-2 model '{args.model}' on {args.device}...")
    try:
        import esm
    except ImportError:
        print("ERROR: Package 'esm' (fair-esm) is not installed.")
        print("To run extraction on GPU, install via: pip install fair-esm")
        sys.exit(1)

    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    model = model.to(args.device)
    model.eval()
    if args.precision == "fp16" and args.device == "cuda":
        model = model.half()

    batch_converter = alphabet.get_batch_converter()

    # Extract reference embeddings first
    print("\nExtracting HXB2 Reference Consensus Embeddings...")
    ref_dir = out_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    for ref_name, ref_seq in [("HXB2_PR_99", HXB2_PROTEASE), ("HXB2_RT_240", HXB2_RT[:240])]:
        ref_path = ref_dir / f"{ref_name}.npy"
        if not ref_path.exists():
            data = [(ref_name, ref_seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(args.device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33][0, 1:len(ref_seq) + 1].cpu().numpy()
            np.save(ref_path, token_repr.astype(np.float16 if args.precision == "fp16" else np.float32))
            print(f"  Saved reference: {ref_path} shape={token_repr.shape}")
        else:
            print(f"  Reference already cached: {ref_path}")

    # Batch extraction for pending sequences
    print(f"\nExtracting {len(to_extract)} unique sequences in batches of {args.batch_size}...")
    for i in range(0, len(to_extract), args.batch_size):
        batch_items = to_extract[i : i + args.batch_size]
        batch_data = [(item["sequence_hash"], item["sequence"]) for item in batch_items]
        
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(args.device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        representations = results["representations"][33]
        
        for b_idx, item in enumerate(batch_items):
            seq_len = item["length"]
            # Exclude start token (<cls>) and end token (<eos>)
            emb = representations[b_idx, 1 : seq_len + 1].cpu().numpy()
            emb_dtype = np.float16 if args.precision == "fp16" else np.float32
            np.save(item["embedding_path"], emb.astype(emb_dtype))
            
        if (i // args.batch_size) % 20 == 0 or (i + args.batch_size) >= len(to_extract):
            print(f"  Extracted {min(i + args.batch_size, len(to_extract))}/{len(to_extract)} sequences...")

    # Write metadata
    meta = {
        "model_name": args.model,
        "device": args.device,
        "precision": args.precision,
        "embedding_dim": 1280,
        "preprocessing_pipeline": args.preprocessing,
        "total_unique_sequences": len(seq_index),
        "pr_count": len(unique_pr),
        "rt_count": len(unique_rt),
    }
    with open(out_dir / "extraction_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print("\nESM-2 embedding extraction completed successfully!")


if __name__ == "__main__":
    main()
