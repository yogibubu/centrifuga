#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_twomode_matching import paper1_h08_term3_twomode_matching


def test_twomode_matching_ratio_condition() -> None:
    out = paper1_h08_term3_twomode_matching()
    C1, C2 = sp.symbols("C1 C2", nonzero=True)
    assert sp.sstr(out["ratio_condition"].lhs) == "r'_1/r'_2"
    assert sp.simplify(out["ratio_condition"].rhs - C1**2 / C2**2) == 0
