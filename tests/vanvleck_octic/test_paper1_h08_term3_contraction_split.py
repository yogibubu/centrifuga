#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_contraction_split import paper1_h08_term3_contraction_split


def test_contraction_split_is_additive() -> None:
    out = paper1_h08_term3_contraction_split()
    lhs = out["split"].lhs
    rhs = out["split"].rhs
    Gk = sp.Symbol("G_k[R_lm]")
    Qk = sp.Symbol("Q_k[k3,R_n]")
    assert sp.simplify(rhs.subs({Gk: 1, Qk: 2}) - 3) == 0
    assert lhs == sp.Symbol("S_k[lm]")
