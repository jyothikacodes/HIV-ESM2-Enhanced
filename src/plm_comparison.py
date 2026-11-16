"""
Protein language model comparison for HIV drug resistance prediction.

This module extends the ESM-2 pipeline to support additional PLMs:
- ESM C (Cambrian) 600M: Meta's recommended ESM-2 successor for embeddings
- ESM-1v: Zero-shot variant-effect scoring for mutation impact prediction

To assess robustness to PLM backbone choice, this module provides a
unified comparison pipeline that extracts embeddings from each model
and trains identical classifiers for fair evaluation.

Installation:
    ESM C:  pip install esm  (EvolutionaryScale package, >=3.0)
    ESM-1v: pip install fair-esm  (Facebook Research package)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None
from tqdm import tqdm


# ── ESM C (Cambrian) ────────────────────────────────────────────────────────

def load_esmc_model(
    model_name: str = "esmc_600m",
    device: Optional[torch.device] = None
) -> Tuple:
    """
    Load ESM Cambrian model for embedding extraction.

    ESM C is EvolutionaryScale's recommended successor to ESM-2 for
    representation/embedding tasks. Unlike ESM-3 (which is generative),
    ESM C is designed specifically for producing high-quality protein embeddings.

    Args:
        model_name: Model variant. Options:
            - "esmc_600m": 600M params, 1152-dim embeddings (recommended)
            - "esmc_300m": 300M params, 960-dim embeddings
        device: torch device (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer, device, embed_dim)
    """
    from esm.models.esmc import ESMC
    from esm.tokenization import EsmSequenceTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    if model_name == "esmc_600m":
        model = ESMC.from_pretrained("esmc_600m", device=device)
        embed_dim = 1152
    elif model_name == "esmc_300m":
        model = ESMC.from_pretrained("esmc_300m", device=device)
        embed_dim = 960
    else:
        raise ValueError(f"Unknown ESM C model: {model_name}. Use 'esmc_600m' or 'esmc_300m'.")

    model.eval()
    tokenizer = EsmSequenceTokenizer()

    return model, tokenizer, device, embed_dim


def extract_esmc_embeddings(
    sequences: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
    pooling: str = 'mean'
) -> np.ndarray:
    """
    Extract embeddings from ESM C model.

    Args:
        sequences: List of amino acid sequences
        model: ESM C model
        tokenizer: ESM sequence tokenizer
        device: torch device
        batch_size: Batch size for inference
        pooling: Pooling method ('mean', 'max', 'mean_max')

    Returns:
        Array of shape (n_sequences, embed_dim) or (n_sequences, embed_dim*2) for mean_max
    """
    all_pooled = []

    for i in tqdm(range(0, len(sequences), batch_size), desc=f"ESM-C {pooling} embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize
        tokens = tokenizer(batch_seqs, padding=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)

        with torch.no_grad():
            output = model(input_ids)
            # ESM C returns embeddings directly
            token_embeddings = output.embeddings  # (batch, seq_len, embed_dim)

        for j, seq in enumerate(batch_seqs):
            # Remove special tokens (BOS at position 0, EOS at end)
            seq_len = len(seq)
            emb = token_embeddings[j, 1:seq_len + 1, :].cpu().numpy()

            if pooling == 'mean':
                pooled = np.mean(emb, axis=0)
            elif pooling == 'max':
                pooled = np.max(emb, axis=0)
            elif pooling == 'mean_max':
                pooled = np.concatenate([np.mean(emb, axis=0), np.max(emb, axis=0)])
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_pooled.append(pooled)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(all_pooled)


def extract_esmc_per_residue_embeddings(
    sequences: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
) -> List[np.ndarray]:
    """
    Extract per-residue embeddings from ESM C (for attention-weighted classifier).

    Args:
        sequences: List of amino acid sequences
        model: ESM C model
        tokenizer: ESM sequence tokenizer
        device: torch device
        batch_size: Batch size

    Returns:
        List of (seq_len, embed_dim) numpy arrays
    """
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-C per-residue"):
        batch_seqs = sequences[i:i + batch_size]

        tokens = tokenizer(batch_seqs, padding=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)

        with torch.no_grad():
            output = model(input_ids)
            token_embeddings = output.embeddings

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = token_embeddings[j, 1:seq_len + 1, :].cpu().numpy()
            all_embeddings.append(emb)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_embeddings


# ── ESM-1v (variant effect scoring) ─────────────────────────────────────────

def load_esm1v_model(
    model_index: int = 1,
    device: Optional[torch.device] = None
) -> Tuple:
    """
    Load ESM-1v model for variant effect scoring.

    ESM-1v uses masked marginal scoring to predict the effect of mutations
    on protein function in a zero-shot manner. This is complementary to
    the embedding-based approach.

    There are 5 ESM-1v models (ensemble); model_index selects which one.

    Args:
        model_index: Which model in the ensemble (1-5)
        device: torch device

    Returns:
        Tuple of (model, alphabet, batch_converter, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # Load via torch.hub (avoids namespace conflict with EvolutionaryScale esm package)
    model_name = f"esm1v_t33_650M_UR90S_{model_index}"
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    # Note: for full ensemble, load models 1-5 and average scores

    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    return model, alphabet, batch_converter, device


def compute_variant_effect_scores(
    sequences: List[str],
    reference: str,
    model,
    alphabet,
    device: torch.device,
) -> np.ndarray:
    """
    Compute zero-shot variant effect scores using ESM-1v masked marginal scoring.

    For each sequence, computes the log-likelihood ratio of the mutant vs
    wild-type amino acid at each mutated position, summed across all mutations.

    Args:
        sequences: List of (potentially mutant) amino acid sequences
        reference: Wild-type reference sequence
        model: ESM-1v model
        alphabet: ESM alphabet
        device: torch device

    Returns:
        Array of shape (n_sequences,) with aggregate variant effect scores.
        More negative = more deleterious mutations.
    """
    batch_converter = alphabet.get_batch_converter()
    scores = []

    for seq in tqdm(sequences, desc="ESM-1v variant scoring"):
        ref_len = min(len(seq), len(reference))

        # Find mutated positions
        mutations = []
        for i in range(ref_len):
            if seq[i] != reference[i]:
                mutations.append((i, reference[i], seq[i]))

        if not mutations:
            scores.append(0.0)
            continue

        # Compute masked marginal log-likelihood for each mutation
        total_score = 0.0
        for pos, wt_aa, mut_aa in mutations:
            # Create masked sequence
            masked_seq = list(seq[:ref_len])
            masked_seq[pos] = '<mask>'
            masked_str = ''.join(masked_seq).replace('<mask>', alphabet.get_tok(alphabet.mask_idx) if hasattr(alphabet, 'mask_idx') else '<mask>')

            # For ESM-1v, we mask the position and get log-probs
            data = [("seq", seq[:ref_len])]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            # Mask the target position (+1 for BOS token)
            masked_tokens = tokens.clone()
            masked_tokens[0, pos + 1] = alphabet.mask_idx

            with torch.no_grad():
                logits = model(masked_tokens)['logits']

            # Get log-probabilities at the masked position
            log_probs = torch.log_softmax(logits[0, pos + 1], dim=-1)

            # Score = log P(mutant) - log P(wildtype)
            wt_idx = alphabet.get_idx(wt_aa)
            mut_idx = alphabet.get_idx(mut_aa)

            score = (log_probs[mut_idx] - log_probs[wt_idx]).item()
            total_score += score

        scores.append(total_score)

    return np.array(scores)


def extract_esm1v_embeddings(
    sequences: List[str],
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    batch_size: int = 8,
    pooling: str = 'mean',
    repr_layer: int = 33
) -> np.ndarray:
    """
    Extract embeddings from ESM-1v (same architecture as ESM-2 650M).

    ESM-1v produces 1280-dim embeddings identical in structure to ESM-2.
    Can be used as an alternative embedding backbone.

    Args:
        sequences: List of amino acid sequences
        model: ESM-1v model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: torch device
        batch_size: Batch size
        pooling: 'mean', 'max', or 'mean_max'
        repr_layer: Representation layer (33 for ESM-1v 650M)

    Returns:
        Array of shape (n_sequences, embed_dim)
    """
    # ESM-1v uses the same API as ESM-2, so we can reuse the extraction logic
    from .feature_engineering import batch_extract_pooled_embeddings

    return batch_extract_pooled_embeddings(
        sequences, model, alphabet, batch_converter, device,
        pooling_method=pooling, batch_size=batch_size, repr_layer=repr_layer
    )


# ── Multi-PLM comparison pipeline ───────────────────────────────────────────

def run_plm_comparison(
    sequences: Dict[str, List[str]],
    phenotypes: Dict[str, pd.DataFrame],
    drug_lists: Dict[str, List[str]],
    embeddings_cache_dir: Optional[str] = None,
    models_to_run: List[str] = ['esm2', 'esmc', 'esm1v'],
    classifier: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Run the full multi-PLM comparison across all drug classes.

    Extracts embeddings from each PLM backbone, trains the same classifier,
    and reports per-drug AUC for direct comparison.

    Args:
        sequences: Dict of drug_class -> list of AA sequences
        phenotypes: Dict of drug_class -> phenotype DataFrame
        drug_lists: Dict of drug_class -> list of drug names
        embeddings_cache_dir: Directory to cache embeddings (saves time on re-runs)
        models_to_run: Which PLM backbones to evaluate
        classifier: Classifier type ('logistic', 'xgboost')
        n_splits: CV folds
        random_state: Random seed
        device: Torch device

    Returns:
        DataFrame with columns [plm, drug_class, drug, auc, n_samples, embed_dim]
    """
    from pathlib import Path
    from .models import per_drug_training

    cache_dir = Path(embeddings_cache_dir) if embeddings_cache_dir else None
    all_results = []

    for plm_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Extracting embeddings: {plm_name.upper()}")
        print(f"{'='*60}")

        for drug_class in ['PI', 'NRTI', 'NNRTI']:
            if drug_class not in sequences:
                continue

            seqs = sequences[drug_class]
            pheno = phenotypes[drug_class]
            drugs = drug_lists[drug_class]

            # Check cache
            cache_file = None
            if cache_dir:
                cache_file = cache_dir / f"{plm_name}_{drug_class}_mean.npy"
                if cache_file.exists():
                    print(f"  Loading cached {plm_name} embeddings for {drug_class}")
                    X = np.load(cache_file)

                    print(f"\n  {plm_name.upper()} + {classifier} on {drug_class}:")
                    results = per_drug_training(
                        X, pheno, drugs,
                        model_type=classifier,
                        n_splits=n_splits,
                        random_state=random_state
                    )

                    for drug, res in results.items():
                        all_results.append({
                            'plm': plm_name,
                            'drug_class': drug_class,
                            'drug': drug,
                            'auc': res['auc'],
                            'n_samples': res['n_samples'],
                            'embed_dim': X.shape[1],
                        })
                    continue

            # Extract embeddings
            if plm_name == 'esm2':
                from .feature_engineering import (
                    load_esm2_model, batch_extract_pooled_embeddings
                )
                model, alphabet, batch_converter, dev = load_esm2_model(device=device)
                X = batch_extract_pooled_embeddings(
                    seqs, model, alphabet, batch_converter, dev,
                    pooling_method='mean'
                )
                embed_dim = 1280

            elif plm_name == 'esmc':
                model, tokenizer, dev, embed_dim = load_esmc_model(device=device)
                X = extract_esmc_embeddings(
                    seqs, model, tokenizer, dev, pooling='mean'
                )

            elif plm_name == 'esm1v':
                model, alphabet, batch_converter, dev = load_esm1v_model(device=device)
                X = extract_esm1v_embeddings(
                    seqs, model, alphabet, batch_converter, dev, pooling='mean'
                )
                embed_dim = 1280

            else:
                raise ValueError(f"Unknown PLM: {plm_name}")

            # Cache embeddings
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, X)
                print(f"  Cached embeddings to {cache_file}")

            # Train and evaluate
            print(f"\n  {plm_name.upper()} + {classifier} on {drug_class}:")
            results = per_drug_training(
                X, pheno, drugs,
                model_type=classifier,
                n_splits=n_splits,
                random_state=random_state
            )

            for drug, res in results.items():
                all_results.append({
                    'plm': plm_name,
                    'drug_class': drug_class,
                    'drug': drug,
                    'auc': res['auc'],
                    'n_samples': res['n_samples'],
                    'embed_dim': X.shape[1],
                })

            # Free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(all_results)


def format_plm_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format PLM comparison results into a publication-ready table.

    Args:
        results_df: DataFrame from run_plm_comparison()

    Returns:
        Pivot table with drugs as rows and PLMs as columns, showing AUC values
    """
    # Pivot: rows = drug, columns = plm
    pivot = results_df.pivot_table(
        index='drug', columns='plm', values='auc', aggfunc='first'
    )

    # Add mean row
    mean_row = pivot.mean()
    mean_row.name = 'MEAN'
    pivot = pd.concat([pivot, mean_row.to_frame().T])

    # Add drug class column
    from .data_processing import get_drug_class
    drug_classes = {}
    for drug in pivot.index:
        if drug == 'MEAN':
            drug_classes[drug] = ''
        else:
            try:
                drug_classes[drug] = get_drug_class(drug)
            except ValueError:
                drug_classes[drug] = '?'

    pivot.insert(0, 'class', pd.Series(drug_classes))

    # Sort by drug class then drug name
    class_order = {'PI': 0, 'NRTI': 1, 'NNRTI': 2, '': 3, '?': 4}
    pivot['_sort'] = pivot['class'].map(class_order)
    pivot = pivot.sort_values(['_sort', pivot.index.name]).drop(columns='_sort')

    return pivot
