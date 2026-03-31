#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_k_independence import paper1_h08_term3_k_independence


def test_matching_implies_mode_independent_ratio() -> None:
    out = paper1_h08_term3_k_independence()
    assert out["matching"].lhs == sp.Symbol("S[lm]")
    assert sp.sstr(out["ratio_condition"].rhs) == "C_k**2/r_k"

