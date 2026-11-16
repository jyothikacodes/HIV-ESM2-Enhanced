"""
Feature engineering for HIV drug resistance prediction.

This module provides functions for:
- Loading ESM-2 protein language model
- Extracting embeddings from sequences
- Attention-weighted and mean pooling
- Binary mutation encoding for baseline models
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None
from tqdm import tqdm


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


def load_esm2_model(
    model_name: str = "esm2_t33_650M_UR50D",
    device: Optional[torch.device] = None
) -> Tuple:
    """
    Load ESM-2 model for embedding extraction.

    Args:
        model_name: ESM-2 model variant
        device: torch device (auto-detected if None)

    Returns:
        Tuple of (model, alphabet, batch_converter, device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model via torch.hub (avoids namespace conflict with EvolutionaryScale esm package)
    model, alphabet = torch.hub.load(
        "facebookresearch/esm:main", model_name
    )

    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    return model, alphabet, batch_converter, device


def extract_embeddings(
    sequences: List[str],
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    batch_size: int = 8,
    repr_layer: int = 33
) -> np.ndarray:
    """
    Extract ESM-2 embeddings for sequences.

    Args:
        sequences: List of amino acid sequences
        model: ESM-2 model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: torch device
        batch_size: Batch size for inference
        repr_layer: Which layer to extract representations from

    Returns:
        Array of shape (n_sequences, seq_len, embed_dim)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])

        token_embeddings = results['representations'][repr_layer]

        # Remove BOS/EOS tokens
        for j, seq in enumerate(batch_seqs):
            seq_emb = token_embeddings[j, 1:len(seq)+1, :].cpu().numpy()
            all_embeddings.append(seq_emb)

        torch.cuda.empty_cache()

    return all_embeddings


def extract_attention_weights(
    sequence: str,
    model,
    alphabet,
    device: torch.device,
    layer: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract attention weights from ESM-2 for a single sequence.

    Args:
        sequence: Amino acid sequence
        model: ESM-2 model
        alphabet: ESM alphabet
        device: torch device
        layer: Which layer's attention to extract (-1 = last)

    Returns:
        Tuple of (attention_matrix, position_attention)
        - attention_matrix: (heads, seq_len, seq_len)
        - position_attention: (seq_len,) averaged attention per position
    """
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[], return_contacts=False, need_head_weights=True)

    # Get attention weights: (layers, heads, seq_len+2, seq_len+2)
    attentions = results['attentions']

    # Select layer and remove BOS/EOS tokens
    layer_attn = attentions[0, layer, :, 1:-1, 1:-1]  # (heads, seq_len, seq_len)

    # Average across heads
    head_avg = layer_attn.mean(dim=0)  # (seq_len, seq_len)

    # Position-wise attention received (column sum)
    position_attention = head_avg.sum(dim=0).cpu().numpy()

    return layer_attn.cpu().numpy(), position_attention


def attention_weighted_pooling(
    embeddings: np.ndarray,
    attention_weights: np.ndarray
) -> np.ndarray:
    """
    Pool sequence embeddings using attention weights.

    Args:
        embeddings: (seq_len, embed_dim) token embeddings
        attention_weights: (seq_len,) attention weights per position

    Returns:
        (embed_dim,) pooled embedding
    """
    # Normalize attention weights
    weights = attention_weights / attention_weights.sum()

    # Weighted average
    pooled = np.average(embeddings, axis=0, weights=weights)

    return pooled


def mean_pooling(embeddings: np.ndarray) -> np.ndarray:
    """
    Pool sequence embeddings using mean pooling.

    Args:
        embeddings: (seq_len, embed_dim) token embeddings

    Returns:
        (embed_dim,) pooled embedding
    """
    return np.mean(embeddings, axis=0)


def max_pooling(embeddings: np.ndarray) -> np.ndarray:
    """
    Pool sequence embeddings using max pooling.

    Args:
        embeddings: (seq_len, embed_dim) token embeddings

    Returns:
        (embed_dim,) pooled embedding
    """
    return np.max(embeddings, axis=0)


def mean_max_pooling(embeddings: np.ndarray) -> np.ndarray:
    """
    Pool sequence embeddings using concatenated mean + max pooling.

    Args:
        embeddings: (seq_len, embed_dim) token embeddings

    Returns:
        (embed_dim * 2,) pooled embedding
    """
    return np.concatenate([mean_pooling(embeddings), max_pooling(embeddings)])


def create_binary_mutation_encoding(
    sequences: List[str],
    reference: str,
    positions: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create binary mutation encoding for baseline model.

    Each position is encoded as 1 if mutated from reference, 0 otherwise.

    Args:
        sequences: List of sequences
        reference: Reference sequence
        positions: Specific positions to encode (1-indexed), or None for all

    Returns:
        Binary encoding array of shape (n_sequences, n_positions)
    """
    if positions is None:
        positions = list(range(1, len(reference) + 1))

    encodings = []

    for seq in sequences:
        encoding = []
        for pos in positions:
            idx = pos - 1  # Convert to 0-indexed
            if idx < len(seq) and idx < len(reference):
                is_mutated = int(seq[idx] != reference[idx])
            else:
                is_mutated = 0
            encoding.append(is_mutated)
        encodings.append(encoding)

    return np.array(encodings)


def create_amino_acid_encoding(
    sequences: List[str],
    reference: str,
    positions: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create one-hot amino acid encoding.

    Args:
        sequences: List of sequences
        reference: Reference sequence
        positions: Specific positions to encode (1-indexed)

    Returns:
        One-hot encoding array of shape (n_sequences, n_positions * 20)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    if positions is None:
        positions = list(range(1, len(reference) + 1))

    encodings = []

    for seq in sequences:
        encoding = []
        for pos in positions:
            idx = pos - 1
            one_hot = [0] * 20
            if idx < len(seq):
                aa = seq[idx]
                if aa in aa_to_idx:
                    one_hot[aa_to_idx[aa]] = 1
            encoding.extend(one_hot)
        encodings.append(encoding)

    return np.array(encodings)


def batch_extract_pooled_embeddings(
    sequences: List[str],
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    pooling_method: str = 'mean',
    batch_size: int = 8,
    repr_layer: int = 33
) -> np.ndarray:
    """
    Extract and pool embeddings for all sequences.

    Args:
        sequences: List of amino acid sequences
        model: ESM-2 model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: torch device
        pooling_method: 'mean', 'max', or 'mean_max'
        batch_size: Batch size for inference
        repr_layer: Representation layer

    Returns:
        Array of shape (n_sequences, embed_dim) or (n_sequences, embed_dim*2) for mean_max
    """
    pooling_fn = {
        'mean': mean_pooling,
        'max': max_pooling,
        'mean_max': mean_max_pooling
    }[pooling_method]

    all_pooled = []

    for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting {pooling_method} embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])

        token_embeddings = results['representations'][repr_layer]

        for j, seq in enumerate(batch_seqs):
            seq_emb = token_embeddings[j, 1:len(seq)+1, :].cpu().numpy()
            pooled = pooling_fn(seq_emb)
            all_pooled.append(pooled)

        torch.cuda.empty_cache()

    return np.array(all_pooled)


def batch_extract_per_residue_embeddings(
    sequences: List[str],
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    batch_size: int = 8,
    repr_layer: int = 33
) -> List[np.ndarray]:
    """
    Extract per-residue embeddings (unpooled) for all sequences.
    
    Required for AttentionWeightedClassifier.
    
    Args:
        sequences: List of sequences
        model: ESM-2 model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: Torch device
        batch_size: Inference batch size
        repr_layer: Layer to extract
        
    Returns:
        List of (seq_len, embed_dim) numpy arrays
    """
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting per-residue embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])

        token_embeddings = results['representations'][repr_layer]

        # Remove BOS/EOS tokens
        for j, seq in enumerate(batch_seqs):
            seq_emb = token_embeddings[j, 1:len(seq)+1, :].cpu().numpy()
            all_embeddings.append(seq_emb)

        torch.cuda.empty_cache()

    return all_embeddings
