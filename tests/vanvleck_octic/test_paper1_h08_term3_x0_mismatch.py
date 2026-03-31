#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_term3_x0_mismatch import paper1_h08_term3_x0_mismatch


def test_term3_quotient_differs_from_linear_x0_image() -> None:
    X = sp.Symbol("X")
    summary = paper1_h08_term3_x0_mismatch()
    assert summary["x0_k0_polynomial"] == X**4 + 4 * X**3 - 14 * X**2 + 12 * X
    assert summary["difference"] != 0
