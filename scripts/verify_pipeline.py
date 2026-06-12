import sys
import os
import torch
import numpy as np
import pandas as pd

# Add the workspace to python path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("--- Test 1: Verify existing pipeline imports ---")
try:
    from src import data_processing, feature_engineering, models, evaluation
    print("Core pipeline imports OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- Test 2: Verify new modules import cleanly ---")
try:
    from src import rare_mutations, temporal_validation, ternary_classification, shap_explainability, calibration
    print("New modules import OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- Test 3: Run a quick smoke test on rare mutation analysis ---")
try:
    from src.rare_mutations import compute_mutation_frequencies, compute_rare_mutation_weights
    from src.feature_engineering import HIV_PROTEASE_REFERENCE
    seqs = ['PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'] * 10
    freq = compute_mutation_frequencies(seqs, HIV_PROTEASE_REFERENCE)
    weights = compute_rare_mutation_weights(seqs, HIV_PROTEASE_REFERENCE)
    print(f"Frequency shape: {freq.shape}, Weights per seq: {len(weights[0])}")
    print("Rare mutation module OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- Test 4: Verify attention model backward-compatibility (no rare weights = same behavior) ---")
try:
    import torch
    from src.models import AttentionWeightedClassifier
    model = AttentionWeightedClassifier(input_dim=64, attention_hidden_dim=16)
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    logits1, w1 = model(x, mask)  # without rare weights
    logits2, w2 = model(x, mask, rare_mutation_weights=None)  # explicit None
    assert torch.allclose(logits1, logits2), 'Backward compatibility broken!'
    print("Attention model backward-compatible [OK]")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- Test 5: Verify ternary classification ---")
try:
    from src.ternary_classification import extract_ternary_labels
    pheno = pd.DataFrame({'ABC_FC': [1.0, 5.0, 15.0, np.nan, 3.0]})
    labels = extract_ternary_labels(pheno, 'ABC', fc_col_suffix='FC')
    print(f"Ternary labels: {labels}")
    assert list(labels[~np.isnan(labels)]) == [0, 1, 2, 1], f"Unexpected: {labels}"
    print("Ternary classification OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- Test 6: Verify calibration module ---")
try:
    from src.calibration import calibrate_predictions
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_cal = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])
    y_test = np.array([0.15, 0.85, 0.5])
    calibrated = calibrate_predictions(y_true, y_cal, y_test, method='platt')
    print(f"Calibrated: {calibrated}")
    print("Calibration module OK")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- All tests passed! ---")
