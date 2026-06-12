import time
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from feature_engineering import load_esm2_model, extract_embeddings

print("Starting ESM-2 load test...")
t0 = time.time()
try:
    model, alphabet, batch_converter, device = load_esm2_model("esm2_t33_650M_UR50D")
    print(f"ESM-2 loaded in {time.time() - t0:.2f} seconds on device: {device}")
    
    # Test on a few sequences
    seqs = ["PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"] * 5
    t1 = time.time()
    embs = extract_embeddings(seqs, model, alphabet, batch_converter, device, batch_size=2)
    print(f"Extracted embeddings for {len(seqs)} sequences in {time.time() - t1:.2f} seconds")
    print(f"Embedding shape: {embs[0].shape}")
except Exception as e:
    print(f"Error during test: {e}")
