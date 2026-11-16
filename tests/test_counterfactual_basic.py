"""Unit tests for counterfactual mutation analysis."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counterfactual_mutation_analysis import (
    extract_key_mutations,
    generate_counterfactual_sequence,
    get_mutations_relative_to_ref,
    parse_mutation,
)


def test_parse_mutation():
    wt, pos, muts = parse_mutation("M184V")
    assert wt == "M"
    assert pos == 184
    assert muts == ["V"]


def test_get_mutations_relative_to_ref():
    ref = "MTLNL"
    seq1 = "MTLNL"
    seq2 = "MCLNL"
    mut1 = get_mutations_relative_to_ref(seq1, ref)
    mut2 = get_mutations_relative_to_ref(seq2, ref)
    assert len(mut1) == 0
    assert len(mut2) == 1
    assert mut2[0] == "T2C"


def test_generate_counterfactual_sequence():
    original = "MTLNL"
    ref = "MOTOL"
    cf = generate_counterfactual_sequence(original, ref, position=2)
    assert cf == "MOTNL" or cf[1] == ref[1]
