#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_octic_true_linear_collapse import paper1_octic_true_linear_collapse


def test_true_linear_collapse_has_compact_expected_form() -> None:
    summary = paper1_octic_true_linear_collapse()
    text = str(summary["collapsed_true_linear_X0"])
    assert "C_k^{yz}" not in text
    assert "C_k^{zz}" not in text
    assert "tau_yyzz" not in text
    assert "Wzzz" not in text
    assert "K_term3" in text
