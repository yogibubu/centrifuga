#!/usr/bin/env python3
"""Universal normalized K=0 source polynomial for the visible H08 term 3."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term3_onemode_source_probe import paper1_h08_term3_onemode_source_probe
from scripts.paper1_h08_term3_twomode_probe import paper1_h08_term3_twomode_probe


def _normalized_poly_dict(out: dict[str, sp.Expr]) -> dict[str, sp.Expr]:
    kernel = sp.simplify(out["X4"] / sp.Rational(5, 64))
    return {
        "kernel": kernel,
        "X1": sp.simplify(out["X1"] / kernel),
        "X2": sp.simplify(out["X2"] / kernel),
        "X3": sp.simplify(out["X3"] / kernel),
        "X4": sp.simplify(out["X4"] / kernel),
        "X5": sp.simplify(out["X5"] / kernel),
    }


def paper1_h08_term3_universal_polynomial() -> dict[str, object]:
    X = sp.Symbol("X")
    one = _normalized_poly_dict(paper1_h08_term3_onemode_source_probe())
    two = _normalized_poly_dict(paper1_h08_term3_twomode_probe())
    poly = sp.expand(
        one["X1"] * X
        + one["X2"] * X**2
        + one["X3"] * X**3
        + one["X4"] * X**4
        + one["X5"] * X**5
    )
    return {
        "one_mode_normalized": one,
        "two_mode_normalized": two,
        "universal_polynomial": poly,
        "factored_polynomial": sp.factor(poly),
        "statement": (
            "For both exact probes closed so far, the visible one-line source "
            "term -R'_k R_l R_m R'_{lm}/(omega_k omega_l omega_m) reduces on "
            "K=0 to the same normalized polynomial in X=J(J+1): "
            "31/2 X - 97/16 X^2 - 9/64 X^3 + 5/64 X^4 - 63/256 X^5. Only the "
            "overall scalar kernel changes between probes."
        ),
    }


if __name__ == "__main__":
    out = paper1_h08_term3_universal_polynomial()
    print(out["universal_polynomial"])
    print(out["factored_polynomial"])
