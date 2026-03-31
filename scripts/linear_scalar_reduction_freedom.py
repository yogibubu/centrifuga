#!/usr/bin/env python3
"""Exact gauge-freedom audit of the one-dimensional linear scalar reduction."""

from __future__ import annotations

import sympy as sp


def linear_scalar_reduction_freedom() -> dict[str, sp.Expr]:
    B, D = sp.symbols("B D", nonzero=True)
    X = sp.symbols("X")
    a, b, c = sp.symbols("a b c")

    x = X + a * X**2 + b * X**3 + c * X**4
    expr = sp.expand(B * x - D * x**2)

    quartic_coeff = sp.expand(expr.coeff(X, 2))
    sextic_coeff = sp.expand(expr.coeff(X, 3))
    octic_coeff = sp.expand(expr.coeff(X, 4))

    a_from_quartic_invariance = sp.solve(sp.Eq(quartic_coeff, -D), a)[0]
    sextic_after_quartic_fix = sp.simplify(sextic_coeff.subs(a, a_from_quartic_invariance))
    octic_after_quartic_fix = sp.simplify(octic_coeff.subs(a, a_from_quartic_invariance))

    return {
        "quartic_coeff": quartic_coeff,
        "sextic_coeff": sextic_coeff,
        "octic_coeff": octic_coeff,
        "a_from_quartic_invariance": sp.simplify(a_from_quartic_invariance),
        "sextic_after_quartic_fix": sextic_after_quartic_fix,
        "octic_after_quartic_fix": octic_after_quartic_fix,
        "statement": sp.Symbol(
            "Quartic invariance fixes a = 0, but leaves the cubic reduction parameter b free. "
            "Therefore the hidden shared sextic counterterm is not fixed by the one-dimensional "
            "scalar branch alone."
        ),
    }


def main() -> None:
    out = linear_scalar_reduction_freedom()
    for key, value in out.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
