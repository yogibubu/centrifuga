#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_final_orth_coefficients import paper1_octic_final_orth_coefficients


def test_final_orth_coefficients_have_all_15_entries() -> None:
    out = paper1_octic_final_orth_coefficients()
    assert len(out["orth_basis"]) == 15
    assert len(out["total_orth_coeffs"]) == 15


def test_linear_constant_is_c080_component() -> None:
    out = paper1_octic_final_orth_coefficients()
    assert out["linear_constant_L"] == out["total_orth_coeffs"][(0, 8, 0)]


def test_final_linear_constant_collapses_to_h08k4_at_principal_symbol_level() -> None:
    out = paper1_octic_final_orth_coefficients()
    L = out["linear_constant_L"]
    assert L == sp.Symbol("h08k4_080")
