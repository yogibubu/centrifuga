#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_term3_h02_factor import paper1_h08_term3_h02_factor
from scripts.paper1_h08_term3_linear_hierarchy_split import paper1_h08_term3_linear_hierarchy_split


def test_term3_hierarchy_split_reconstructs_quotient() -> None:
    summary = paper1_h08_term3_linear_hierarchy_split()
    quotient = paper1_h08_term3_h02_factor()["quotient_after_dividing_by_X"]
    assert sp.expand(summary["reconstructed"] - quotient) == 0
    assert summary["hierarchy"]["octic_X0"] == sp.Rational(-63, 256)
