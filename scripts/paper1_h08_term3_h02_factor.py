#!/usr/bin/env python3
"""Exact H02/X factorization of the visible H08 term-3 source polynomial."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term3_universal_polynomial import paper1_h08_term3_universal_polynomial


def paper1_h08_term3_h02_factor() -> dict[str, object]:
    X = sp.Symbol("X")
    U = paper1_h08_term3_universal_polynomial()["universal_polynomial"]
    quotient = sp.expand(sp.factor(U / X))
    return {
        "universal_polynomial": U,
        "quotient_after_dividing_by_X": quotient,
        "remainder": sp.rem(sp.Poly(U, X), sp.Poly(X, X)),
        "statement": (
            "The universal K=0 source polynomial of the visible one-line term "
            "3 is exactly divisible by X=J(J+1). In the linear rotor this means "
            "that the source carries an explicit H02 factor. Any final octic "
            "effective contribution must therefore arise after removing one H02 "
            "factor from this source-level decatic object."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_h02_factor())
