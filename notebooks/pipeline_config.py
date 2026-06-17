"""
Shared configuration for the HIV-ESM-2 notebook chain.

Edit ENABLE_IMPROVED_PIPELINE to control the full proposal pipeline across
notebooks 02–07. When True, prerequisite sections (rare mutations, per-residue
embeddings, attention training, ternary/calibration evaluation, SHAP, and the
improved pipeline capstone in notebook 07) are enabled automatically.
"""

# Master toggle — set False for lightweight / legacy notebook runs
ENABLE_IMPROVED_PIPELINE = True

# ── Prerequisite sections (notebooks 02–06) ─────────────────────────────────
RUN_RARE_MUTATION_PREP = ENABLE_IMPROVED_PIPELINE
RUN_PER_RESIDUE_EXTRACTION = ENABLE_IMPROVED_PIPELINE
RUN_ATTENTION_TRAINING = ENABLE_IMPROVED_PIPELINE
RUN_TERNARY_CLASSIFICATION = ENABLE_IMPROVED_PIPELINE
RUN_CALIBRATION_COMPARISON = ENABLE_IMPROVED_PIPELINE
RUN_EXTENDED_TEMPORAL_VALIDATION = ENABLE_IMPROVED_PIPELINE
RUN_SHAP_RESIDUE_MAPPING = ENABLE_IMPROVED_PIPELINE
# Dual SHAP / IG aggregator — enabled when running the full pipeline
RUN_EXPLAINABILITY_EXTENSIONS = True
RUN_STATISTICAL_VALIDATION = ENABLE_IMPROVED_PIPELINE

# ── Improved pipeline capstone (notebook 07) ────────────────────────────────
RUN_IMPROVED_PIPELINE = ENABLE_IMPROVED_PIPELINE

# ── ACCURACY IMPROVEMENTS: Use FULL COHORT and enhanced feature selection ───
# Use full cohort when per-residue embeddings exist; else subsample
IMPROVED_USE_FULL_COHORT = True  # Use complete dataset for maximum accuracy
IMPROVED_SUBSET_SIZE = None  # None = full cohort (if IMPROVED_USE_FULL_COHORT=True)
IMPROVED_SEED = 42
IMPROVED_OUTER_SPLITS = 5
IMPROVED_INNER_SPLITS = 3
IMPROVED_N_REPEATS = 2
IMPROVED_OPTUNA_TRIALS = 50  # Increased for better hyperparameter tuning
IMPROVED_USE_OPTUNA = True

# ── Feature Selection & Ensemble Weighting (Accuracy Improvements) ───────────
IMPROVED_ENABLE_FEATURE_SELECTION = True  # Enable MI + PCA feature selection
IMPROVED_FEATURE_SELECTION_METHOD = 'combined'  # 'mutual_info', 'pca', or 'combined'
IMPROVED_N_SELECTED_FEATURES = None  # Auto-select optimal number (typically 100-150)
IMPROVED_ENABLE_ADAPTIVE_ENSEMBLE = True  # Use AUC-based model weighting
IMPROVED_ENABLE_DISTRIBUTION_MATCHING = True  # Calibration-aware weighting

# Option B / statistical validation (notebook 07)
STATISTICAL_VALIDATION_SUBSET_SIZE = IMPROVED_SUBSET_SIZE
STATISTICAL_VALIDATION_SEED = IMPROVED_SEED

# Per-residue extraction (notebook 03)
BATCH_SIZE_PER_RESIDUE = 4

# Calibration methods when improved pipeline is enabled
CALIBRATION_METHODS = ['platt', 'isotonic', 'temperature', 'auto']
