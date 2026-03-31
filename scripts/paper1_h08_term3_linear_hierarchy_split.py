#!/usr/bin/env python3
"""Exact linear-scalar hierarchy split of the visible H08 term-3 quotient."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term3_effective_split import paper1_h08_term3_effective_split


def paper1_h08_term3_linear_hierarchy_split() -> dict[str, object]:
    X = sp.Symbol("X")
    split = paper1_h08_term3_effective_split()["split"]
    hierarchy = {
        "octic_X0": split["a_x0"],
        "sextic_X3": split["b_x3"],
        "quartic_X2": split["c_x2"],
        "quadratic_X1": split["d_x1"],
        "constant_X0": split["e_x0scalar"],
    }
    reconstructed = sp.expand(
        hierarchy["octic_X0"] * (X**4 + 4 * X**3 - 14 * X**2 + 12 * X)
        + hierarchy["sextic_X3"] * X**3
        + hierarchy["quartic_X2"] * X**2
        + hierarchy["quadratic_X1"] * X
        + hierarchy["constant_X0"]
    )
    return {
        "hierarchy": hierarchy,
        "reconstructed": reconstructed,
        "statement": (
            "After removing the explicit H02 factor, the visible one-line H08 term 3 "
            "splits exactly into one octic X0 contribution plus lower-order scalar "
            "renormalizations at sextic, quartic, quadratic, and constant levels."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_linear_hierarchy_split())
