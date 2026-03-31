#!/usr/bin/env python3
"""Exact true-linear collapse of the currently derived X0 expression."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_derived_x0_no_rkl import paper1_octic_derived_x0_no_rkl


def paper1_octic_true_linear_collapse() -> dict[str, object]:
    expr = paper1_octic_derived_x0_no_rkl()["primitive_total_without_rkl"]

    B, C = sp.symbols("B C")
    S111, S113, S131 = sp.symbols("S111 S113 S131")
    tau_yyyy, tau_yyzz, tau_zzzz = sp.symbols("tau_yyyy tau_yyzz tau_zzzz")
    Wyyy, Wyyz, Wzzy, Wzzz = sp.symbols("Wyyy Wyyz Wzzy Wzzz")

    linear_subs = {
        C: B,
        S131: S113,
        tau_zzzz: tau_yyyy,
        tau_yyzz: sp.Rational(2, 3) * tau_yyyy,
        Wzzz: Wyyy,
        Wzzy: Wyyz,
    }

    transverse_subs = {}
    for sym in expr.free_symbols:
        text = str(sym)
        if "^{zz}" in text:
            transverse_subs[sym] = sp.Symbol(text.replace("^{zz}", "^{yy}"))
        elif "^{yz}" in text:
            transverse_subs[sym] = sp.Integer(0)

    collapsed = sp.expand(expr.subs(linear_subs).subs(transverse_subs))

    return {
        "collapsed_true_linear_X0": collapsed,
        "statement": (
            "After imposing the exact true-linear symmetry constraints on the "
            "derived primitive X0 expression, the current CeDiTT4 program collapses "
            "to a compact linear formula containing the visible quartic block, "
            "the S111^2 tau term, the normalized visible term-3 contribution times "
            "its unresolved scalar kernel, and the "
            "explicit contracted square from the printed one-index/two-index prime sector."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_true_linear_collapse())
