#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_rprimekl_true_linear_reduction import paper1_h08_rprimekl_true_linear_reduction


def test_rprimekl_true_linear_reduction_has_expected_geometric_piece() -> None:
    summary = paper1_h08_rprimekl_true_linear_reduction()
    omega_k, omega_l, B = sp.symbols("omega_k omega_l B", nonzero=True)
    Ck, Cl = sp.symbols("C_k C_l")
    assert summary["r_kl_geometric"] == sp.Rational(3, 4) * omega_k * omega_l * Ck * Cl / B
    assert "k'_{klm}" in str(summary["r_kl_total"])
