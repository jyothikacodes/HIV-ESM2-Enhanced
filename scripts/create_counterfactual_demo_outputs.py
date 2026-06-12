"""
Demonstration of Counterfactual Mutation Analysis Module outputs.

This shows the expected structure and outputs of the counterfactual_mutation_analysis module.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Create sample outputs for demonstration
data = {
    'counterfactual_mutation_effects.csv': pd.DataFrame({
        'drug': ['ATV', 'ATV', 'ATV', 'DRV', 'DRV'],
        'drug_class': ['PI', 'PI', 'PI', 'PI', 'PI'],
        'mutation': ['M36I', 'L90M', 'I84V', 'L90M', 'M36I'],
        'position': [36, 90, 84, 90, 36],
        'delta_probability': [0.0543, 0.1234, -0.0312, 0.0876, 0.0421],
        'original_probability': [0.65, 0.72, 0.58, 0.78, 0.68],
        'counterfactual_probability': [0.7043, 0.8434, 0.5488, 0.8676, 0.7221],
        'causal_direction': ['increases_resistance', 'increases_resistance', 'decreases_resistance', 
                            'increases_resistance', 'increases_resistance']
    }),
    
    'mutation_causal_ranking.csv': pd.DataFrame({
        'drug': ['ATV', 'ATV', 'ATV', 'ATV', 'DRV'],
        'drug_class': ['PI', 'PI', 'PI', 'PI', 'PI'],
        'mutation': ['L90M', 'M36I', 'I84V', 'L33F', 'L90M'],
        'position': [90, 36, 84, 33, 90],
        'mean_causal_effect': [0.1234, 0.0543, 0.0312, 0.0198, 0.0876],
        'rank': [1, 2, 3, 4, 1]
    }),
    
    'drug_level_causal_summary.csv': pd.DataFrame({
        'drug': ['ATV', 'DRV', 'FPV', 'IDV', '3TC'],
        'drug_class': ['PI', 'PI', 'PI', 'PI', 'NRTI'],
        'mean_causal_sensitivity': [0.0632, 0.0754, 0.0521, 0.0418, 0.0856],
        'std_causal_sensitivity': [0.0234, 0.0312, 0.0198, 0.0156, 0.0234],
        'median_causal_sensitivity': [0.0589, 0.0698, 0.0478, 0.0382, 0.0812],
        'n_mutations_analyzed': [12, 15, 10, 8, 18]
    })
}

results_dir = Path(__file__).resolve().parent.parent / 'results'
results_dir.mkdir(exist_ok=True)

for filename, df in data.items():
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Created {filepath} with {len(df)} rows")
    print(f"Columns: {list(df.columns)}\n")

print("Sample outputs created for demonstration.")
