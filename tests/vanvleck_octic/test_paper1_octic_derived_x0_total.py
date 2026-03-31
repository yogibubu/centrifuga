#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_octic_derived_x0_total import paper1_octic_derived_x0_total


def test_derived_x0_total_includes_visible_term3_and_term5() -> None:
    summary = paper1_octic_derived_x0_total()
    total = summary["derived_X0_total_so_far"]
    assert "r_k" in str(total)
    assert "r_lm" in str(total)
    assert summary["visible_term1_X0"] == 0
    assert summary["visible_term4_X0"] == 0
