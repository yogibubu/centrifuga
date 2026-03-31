#!/usr/bin/env python3
"""Summary of rigorously isolated X0 pieces from the visible one-line H08 source."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_k4_octic_projected_piece import paper1_h08_k4_octic_projected_piece
from scripts.paper1_h08_term3_octic_projected_piece import paper1_h08_term3_octic_projected_piece
from scripts.paper1_h08_term5_pure_octic import paper1_h08_term5_pure_octic


def paper1_h08_visible_source_x0_summary() -> dict[str, object]:
    k4 = paper1_h08_k4_octic_projected_piece()["hierarchy_components"]["octic_X0"]
    term3_data = paper1_h08_term3_octic_projected_piece()
    term3 = term3_data["hierarchy_components"]["octic_X0_physical"]
    term5 = paper1_h08_term5_pure_octic()["x0_coefficient"]
    return {
        "visible_H08_k4_octic_X0": k4,
        "visible_term3_reduced_octic_X0": term3,
        "visible_term3_normalized_octic_X0": term3_data["hierarchy_components"]["octic_X0_normalized"],
        "visible_term3_kernel": term3_data["term3_kernel"],
        "visible_term5_pure_octic_X0": term5,
        "visible_term1_naive_X0": sp.Integer(0),
        "visible_term4_true_linear_X0": sp.Integer(0),
        "statement": (
            "The visible one-line H08 source decomposes, at the rigorously isolated "
            "linear-octic level, into the quartic-source X0 piece, the reduced term-3 "
            "X0 piece, the pure term-5 X0 piece, and vanishing term-1/term-4 contributions."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_visible_source_x0_summary())
