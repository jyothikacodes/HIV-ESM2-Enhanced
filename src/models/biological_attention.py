"""
Biologically Informed / Prior-Guided Attention Mechanism.

Experiment D:
Injects independent clinical domain knowledge (IAS-USA / Stanford HIVDB DRM consensus)
into the attention scoring function:
    a_i = softmax( v^T tanh( W * E_i + U * M_i + V * R_{i, d} ) )

Hypothesis:
Unconstrained attention over 99 or 240 positions overfits clade-specific
polymorphisms rather than true causal drug resistance mutations.
Biologically guided attention anchors attention heads to canonical resistance hotspots
while allowing contextual PLM embeddings to modulate the exact resistance magnitude.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False
    _BaseModule = object


# Consensus Drug Resistance Mutation (DRM) Positions from IAS-USA / Stanford HIVDB
# 0-indexed positions (1-indexed minus 1)
CANONICAL_DRM_POSITIONS: Dict[str, Set[int]] = {
    # Protease (PIs) - 0-indexed:
    # 24->23, 30->29, 32->31, 46->45, 47->46, 48->47, 50->49, 54->53, 76->75, 82->81, 84->83, 88->87, 90->89
    "PI": {23, 29, 31, 45, 46, 47, 49, 53, 75, 81, 83, 87, 89},
    # Reverse Transcriptase (NRTIs) - 0-indexed:
    # 41->40, 65->64, 67->66, 70->69, 74->73, 75->74, 115->114, 151->150, 184->183, 210->209, 215->214, 219->218
    "NRTI": {40, 64, 66, 69, 73, 74, 114, 150, 183, 209, 214, 218},
    # Reverse Transcriptase (NNRTIs) - 0-indexed:
    # 100->99, 101->100, 103->102, 106->105, 138->137, 179->178, 181->180, 188->187, 190->189, 225->224, 230->229
    "NNRTI": {99, 100, 102, 105, 137, 178, 180, 187, 189, 224, 229},
}

# Drug-specific primary resistance positions (0-indexed)
DRUG_SPECIFIC_PRIMARY_DRMS: Dict[str, Set[int]] = {
    "NFV": {29, 87, 89},
    "ATV": {49, 83},
    "DRV": {31, 46, 49, 53, 83},
    "LPV": {46, 49, 53, 81},
    "3TC": {183},        # M184V/I is decisive
    "TDF": {64},         # K65R is decisive
    "ABC": {64, 73, 114, 183},
    "AZT": {40, 66, 69, 209, 214, 218}, # TAMs
    "EFV": {102, 178, 180, 187, 189},   # K103N, Y181C, G190A
    "NVP": {100, 102, 180, 187, 189},
    "ETR": {99, 100, 178, 180},
    "RPV": {100, 137, 180},
}


def get_prior_drm_indicator(
    drug: str,
    seq_len: int,
    drug_class: Optional[str] = None
) -> np.ndarray:
    """
    Generate binary prior indicator vector R of shape (seq_len, 1).
    1.0 at canonical DRM positions, 0.0 elsewhere.
    """
    indicator = np.zeros((seq_len, 1), dtype=np.float32)
    drug_upper = drug.upper()
    
    # 1. Add class-wide DRMs
    if drug_class is not None and drug_class in CANONICAL_DRM_POSITIONS:
        for pos in CANONICAL_DRM_POSITIONS[drug_class]:
            if pos < seq_len:
                indicator[pos, 0] = 1.0
                
    # 2. Add drug-specific primary DRMs
    if drug_upper in DRUG_SPECIFIC_PRIMARY_DRMS:
        for pos in DRUG_SPECIFIC_PRIMARY_DRMS[drug_upper]:
            if pos < seq_len:
                indicator[pos, 0] = 1.5  # Higher prior weight for primary DRMs
                
    return indicator


if _HAS_TORCH:
    class BiologicallyGuidedAttention(nn.Module):
        """
        PyTorch attention module that combines ESM-2 residue embeddings
        with mutation indicators and canonical DRM position priors.
        """
        def __init__(
            self,
            embed_dim: int = 1280,
            hidden_dim: int = 128,
            use_mutation_indicator: bool = True,
            use_prior_drm: bool = True
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            self.use_mut = use_mutation_indicator
            self.use_prior = use_prior_drm
            
            self.w_embed = nn.Linear(embed_dim, hidden_dim)
            self.w_mut = nn.Linear(1, hidden_dim, bias=False) if use_mutation_indicator else None
            self.w_prior = nn.Linear(1, hidden_dim, bias=False) if use_prior_drm else None
            
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(
            self,
            x: torch.Tensor,
            mut_indicator: Optional[torch.Tensor] = None,
            prior_drm: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (B, L, embed_dim)
                mut_indicator: (B, L, 1) or None
                prior_drm: (L, 1) or (B, L, 1) or None
                mask: (B, L) padding mask (1 for valid residue, 0 for pad)
                
            Returns:
                pooled: (B, embed_dim)
                attn_weights: (B, L)
            """
            # Linear projection of embeddings
            h = self.w_embed(x)
            
            # Incorporate mutation indicator
            if self.use_mut and mut_indicator is not None:
                h = h + self.w_mut(mut_indicator)
                
            # Incorporate clinical DRM prior
            if self.use_prior and prior_drm is not None:
                if prior_drm.dim() == 2:
                    prior_drm = prior_drm.unsqueeze(0)  # (1, L, 1)
                h = h + self.w_prior(prior_drm)
                
            scores = self.v(torch.tanh(h)).squeeze(-1)  # (B, L)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(scores, dim=-1)  # (B, L)
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, embed_dim)
            
            return pooled, attn_weights


    class BiologicallyGuidedClassifier(nn.Module):
        """
        Complete model combining biologically guided attention pooling with linear classifier.
        """
        def __init__(
            self,
            input_dim: int = 1280,
            hidden_dim: int = 128,
            dropout: float = 0.2
        ):
            super().__init__()
            self.attention = BiologicallyGuidedAttention(embed_dim=input_dim, hidden_dim=hidden_dim)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, 1)
            )

        def forward(
            self,
            x: torch.Tensor,
            mut_indicator: Optional[torch.Tensor] = None,
            prior_drm: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            pooled, attn_weights = self.attention(x, mut_indicator, prior_drm, mask)
            logits = self.classifier(pooled)
            return logits, attn_weights
