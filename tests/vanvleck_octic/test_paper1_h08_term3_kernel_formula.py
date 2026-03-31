#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_term3_kernel_formula import paper1_h08_term3_kernel_formula


def test_term3_kernel_formula_has_expected_sign() -> None:
    summary = paper1_h08_term3_kernel_formula()
    omega_k = sp.Symbol("omega_k", nonzero=True)
    r_k = sp.Symbol("r_k")
    C_l, C_m = sp.symbols("C_l C_m")
    r_lm = sp.Symbol("r_lm")
    assert summary["term3_kernel"] == r_k * C_l * C_m * r_lm / omega_k
