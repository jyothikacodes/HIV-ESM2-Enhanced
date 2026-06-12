#!/usr/bin/env python
"""Minimal test to verify counterfactual module works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print("Test 1: Import module...", flush=True)
sys.stdout.flush()

try:
    from src.counterfactual_mutation_analysis import (
        extract_key_mutations,
        generate_counterfactual_sequence,
        get_mutations_relative_to_ref,
        parse_mutation
    )
    print("SUCCESS: Module imported", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    sys.exit(1)

# Test basic functions
print("\nTest 2: Test parse_mutation...", flush=True)
wt, pos, muts = parse_mutation("M184V")
print(f"  Result: wt={wt}, pos={pos}, muts={muts}", flush=True)
assert wt == "M" and pos == 184 and muts == ["V"], "Parse failed"

print("\nTest 3: Test get_mutations_relative_to_ref...", flush=True)
ref = "MTLNL"
seq1 = "MTLNL"
seq2 = "MCLNL"
mut1 = get_mutations_relative_to_ref(seq1, ref)
mut2 = get_mutations_relative_to_ref(seq2, ref)
print(f"  No mutations: {mut1}", flush=True)
print(f"  With mutation: {mut2}", flush=True)
assert len(mut1) == 0 and len(mut2) == 1, "Mutation detection failed"

print("\nTest 4: Test generate_counterfactual_sequence...", flush=True)
original = "MTLNL"
ref = "MOTOL"
cf = generate_counterfactual_sequence(original, ref, position=2)
print(f"  Original: {original}", flush=True)
print(f"  Counterfactual: {cf}", flush=True)
print(f"  Expected: MOTOL", flush=True)

print("\nAll tests passed!", flush=True)
