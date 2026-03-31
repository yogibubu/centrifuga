#!/usr/bin/env python3
"""Exact true-linear X-polynomial status of the visible H08 subprincipal terms."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term1_k0_vanishing import paper1_h08_term1_k0_vanishing
from scripts.paper1_h08_term3_onemode_source_probe import paper1_h08_term3_onemode_source_probe
from scripts.paper1_h08_term4_true_linear_vanishing import paper1_h08_term4_true_linear_vanishing


def paper1_h08_subprincipal_x_polynomial() -> dict[str, object]:
    term1 = paper1_h08_term1_k0_vanishing()
    term3 = paper1_h08_term3_onemode_source_probe()
    term4 = paper1_h08_term4_true_linear_vanishing()
    zero = sp.Integer(0)
    return {
        "term1_k0_values": term1["k0_diagonal_values"],
        "term1_X_polynomial": {"X0": zero, "X1": zero, "X2": zero, "X3": zero, "X4": zero},
        "term3_X_polynomial": {
            "X0": sp.expand(term3["X0"]),
            "X1": sp.expand(term3["X1"]),
            "X2": sp.expand(term3["X2"]),
            "X3": sp.expand(term3["X3"]),
            "X4": sp.expand(term3["X4"]),
            "X5": sp.expand(term3["X5"]),
        },
        "term4_commutator": sp.expand(term4["commutator"]),
        "term4_X_polynomial": {"X0": zero, "X1": zero, "X2": zero, "X3": zero, "X4": zero},
        "statement": (
            "In the true linear limit, the visible H08 subprincipal terms are "
            "best discussed through their exact K=0 polynomial in X=J(J+1). "
            "Term 1 vanishes identically, term 4 vanishes because [R_k,R_l]=0, "
            "and term 3 survives as a genuine source polynomial containing both "
            "X^5 and X^4 pieces before the final effective reduction."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_subprincipal_x_polynomial())
