#!/usr/bin/env python3
"""Exact mismatch between the visible H08 term-3 quotient and the linear X0 image."""

from __future__ import annotations

import sympy as sp

from scripts.octic_linear_k0_quantum_projection import project_x_axis_linear_polynomial
from scripts.paper1_h08_term3_h02_factor import paper1_h08_term3_h02_factor


def paper1_h08_term3_x0_mismatch() -> dict[str, object]:
    X = sp.Symbol("X")
    x0_k0 = project_x_axis_linear_polynomial(
        {
            (0, 8, 0): sp.Integer(1),
            (0, 6, 2): sp.Integer(4),
            (0, 4, 4): sp.Integer(6),
            (0, 2, 6): sp.Integer(4),
            (0, 0, 8): sp.Integer(1),
        }
    )
    x0_poly = sp.expand(sum(x0_k0[f"X{i}"] * X**i for i in range(5)))
    quotient = paper1_h08_term3_h02_factor()["quotient_after_dividing_by_X"]
    difference = sp.expand(quotient - x0_poly)
    return {
        "x0_k0_polynomial": x0_poly,
        "term3_quotient": quotient,
        "difference": difference,
        "statement": (
            "After factoring out the explicit H02 ~ X from the visible one-line "
            "term 3, the remaining quartic polynomial is not the K=0 image of "
            "the linear octic scalar X0 = (J_perp^2)^4. The final Watson-Aliev "
            "effective reduction therefore cannot be a simple division by H02."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_x0_mismatch())
