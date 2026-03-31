#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_term3_octic_projected_piece import paper1_h08_term3_octic_projected_piece


def test_term3_octic_projected_piece_is_minus_63_over_256() -> None:
    summary = paper1_h08_term3_octic_projected_piece()
    assert summary["hierarchy_components"]["octic_X0_normalized"] == sp.Rational(-63, 256)
