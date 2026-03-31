#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_k4_linear_component import paper1_h08_k4_linear_component


def test_h08_k4_linear_component_formula() -> None:
    out = paper1_h08_k4_linear_component()
    k4 = sp.Symbol("k'_{klmn}")
    Ckyy = sp.Symbol("C_k^{yy}")
    Clyy = sp.Symbol("C_l^{yy}")
    Cmyy = sp.Symbol("C_m^{yy}")
    Cnyy = sp.Symbol("C_n^{yy}")
    assert out["formula"] == sp.Rational(1, 24) * k4 * Ckyy * Clyy * Cmyy * Cnyy

