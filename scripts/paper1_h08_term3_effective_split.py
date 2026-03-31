#!/usr/bin/env python3
"""Exact split of the visible H08 term-3 quotient into octic and lower-order parts."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term3_h02_factor import paper1_h08_term3_h02_factor
from scripts.paper1_h08_term3_x0_mismatch import paper1_h08_term3_x0_mismatch


def paper1_h08_term3_effective_split() -> dict[str, object]:
    X = sp.Symbol("X")
    quotient = paper1_h08_term3_h02_factor()["quotient_after_dividing_by_X"]
    x0_poly = paper1_h08_term3_x0_mismatch()["x0_k0_polynomial"]
    split = {
        "a_x0": sp.Rational(-63, 256),
        "b_x3": sp.Rational(17, 16),
        "c_x2": sp.Rational(-459, 128),
        "d_x1": sp.Rational(-199, 64),
        "e_x0scalar": sp.Rational(31, 2),
    }
    reconstructed = sp.expand(
        split["a_x0"] * x0_poly
        + split["b_x3"] * X**3
        + split["c_x2"] * X**2
        + split["d_x1"] * X
        + split["e_x0scalar"]
    )
    return {
        "quotient": quotient,
        "linear_x0_image": x0_poly,
        "split": split,
        "reconstructed": reconstructed,
        "difference": sp.expand(quotient - reconstructed),
        "statement": (
            "After removing the explicit H02 ~ X factor, the surviving visible "
            "one-line term 3 decomposes exactly into an octic piece "
            "(-63/256) X0 plus lower-order scalar renormalizations in X^3, X^2, X, and 1."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_effective_split())
