#!/usr/bin/env python3
"""Exact octic piece of the visible H08 term-3 source after hierarchy projection."""

from __future__ import annotations

import sympy as sp

from scripts.octic_linear_hierarchy_projector import octic_linear_hierarchy_projector
from scripts.paper1_h08_term3_h02_factor import paper1_h08_term3_h02_factor
from scripts.paper1_h08_term3_kernel_formula import paper1_h08_term3_kernel_formula


def paper1_h08_term3_octic_projected_piece() -> dict[str, object]:
    X = sp.Symbol("X")
    quotient = paper1_h08_term3_h02_factor()["quotient_after_dividing_by_X"]
    kernel = paper1_h08_term3_kernel_formula()["term3_kernel"]
    coeff_vector = sp.Matrix([sp.expand(quotient).coeff(X, i) for i in range(5)])
    projector = octic_linear_hierarchy_projector()["projector"]
    comps = sp.simplify(projector * coeff_vector)
    x0_coeff = sp.simplify(comps[0])
    return {
        "quotient": quotient,
        "term3_kernel": kernel,
        "hierarchy_components": {
            "octic_X0_normalized": x0_coeff,
            "octic_X0_physical": sp.simplify(kernel * x0_coeff),
            "sextic_X3_normalized": sp.simplify(comps[1]),
            "quartic_X2_normalized": sp.simplify(comps[2]),
            "quadratic_X1_normalized": sp.simplify(comps[3]),
            "constant_X0_normalized": sp.simplify(comps[4]),
        },
        "statement": (
            "Projecting the divided visible H08 term-3 source onto the exact linear "
            "CeDiTT4 scalar hierarchy isolates a normalized octic X0 coefficient -63/256. "
            "The physical contribution is this coefficient multiplied by the exact "
            "true-linear source kernel r_k C_l C_m r_lm / omega_k."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_octic_projected_piece())
