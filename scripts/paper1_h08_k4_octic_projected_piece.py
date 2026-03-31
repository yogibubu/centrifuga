#!/usr/bin/env python3
"""Exact CeDiTT4 hierarchy projection of the visible quartic H08 source block."""

from __future__ import annotations

import sympy as sp

from scripts.octic_linear_hierarchy_projector import octic_linear_hierarchy_projector
from scripts.paper1_h08_k4_linear_projection import paper1_h08_k4_linear_projection


def paper1_h08_k4_octic_projected_piece() -> dict[str, object]:
    poly = paper1_h08_k4_linear_projection()["linear_polynomial"]
    coeff_vector = sp.Matrix([sp.expand(poly[f"X{i}"]) for i in range(5)])
    projector = octic_linear_hierarchy_projector()["projector"]
    comps = sp.simplify(projector * coeff_vector)
    return {
        "linear_polynomial": poly,
        "hierarchy_components": {
            "octic_X0": sp.simplify(comps[0]),
            "sextic_X3": sp.simplify(comps[1]),
            "quartic_X2": sp.simplify(comps[2]),
            "quadratic_X1": sp.simplify(comps[3]),
            "constant_X0": sp.simplify(comps[4]),
        },
        "statement": (
            "Projecting the visible quartic H08 source onto the exact linear "
            "CeDiTT4 hierarchy isolates its octic X0 piece together with any "
            "lower-order scalar renormalizations."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_k4_octic_projected_piece())
