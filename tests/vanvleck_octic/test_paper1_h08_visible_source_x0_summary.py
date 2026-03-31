#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_term3_octic_projected_piece import paper1_h08_term3_octic_projected_piece
from scripts.paper1_h08_visible_source_x0_summary import paper1_h08_visible_source_x0_summary


def test_visible_source_summary_includes_exact_term3_coefficient() -> None:
    summary = paper1_h08_visible_source_x0_summary()
    assert summary["visible_term3_normalized_octic_X0"] == sp.Rational(-63, 256)
    assert summary["visible_term1_naive_X0"] == 0
    assert summary["visible_term4_true_linear_X0"] == 0
