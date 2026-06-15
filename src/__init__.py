"""
HIV-ESM-2: HIV Drug Resistance Prediction with ESM-2 Protein Language Model

This package provides tools for:
- Parsing Stanford HIVDB sequence and resistance data
- Extracting ESM-2 protein language model embeddings
- Training classifiers for drug resistance prediction
- Interpreting model predictions via attention analysis
"""

__version__ = "1.0.0"
__author__ = "Hayden Farquhar"
__email__ = "hayden.farquhar@icloud.com"

from . import data_processing
from . import feature_engineering
from . import models
from . import evaluation
from . import visualization
from . import interpretability
from . import rare_mutations
from . import temporal_validation
from . import ternary_classification
from . import shap_explainability
from . import calibration
from . import dual_shap_explainability
from . import counterfactual_mutation_analysis
from . import scientific_validation

