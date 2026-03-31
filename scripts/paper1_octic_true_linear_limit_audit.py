#!/usr/bin/env python3
"""Audit the true y/z-symmetric linear limit of the octic X^4 coefficient."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_k4_linear_projection import paper1_h08_k4_linear_projection
from scripts.paper1_octic_final_linear_x4 import paper1_octic_final_linear_x4


def true_linear_limit_substitutions() -> dict[sp.Symbol, sp.Expr]:
    B, C = sp.symbols("B C")
    S111 = sp.Symbol("S111")
    subs: dict[sp.Symbol, sp.Expr] = {
        C: B,
        sp.Symbol("q004"): sp.Symbol("q040"),
        sp.Symbol("q022"): 2 * sp.Symbol("q040"),
        sp.Symbol("q202"): sp.Symbol("q220"),
        sp.Symbol("w006"): sp.Symbol("w060"),
        sp.Symbol("w024"): sp.Symbol("w042"),
        sp.Symbol("s113"): sp.Symbol("s131"),
        sp.Symbol("C_k^{zz}"): sp.Symbol("C_k^{yy}"),
        sp.Symbol("C_l^{zz}"): sp.Symbol("C_l^{yy}"),
        sp.Symbol("C_m^{zz}"): sp.Symbol("C_m^{yy}"),
        sp.Symbol("C_n^{zz}"): sp.Symbol("C_n^{yy}"),
        sp.Symbol("C_k^{yz}"): 0,
        sp.Symbol("C_l^{yz}"): 0,
        sp.Symbol("C_m^{yz}"): 0,
        sp.Symbol("C_n^{yz}"): 0,
    }
    return subs


def audit_true_linear_limit() -> dict[str, sp.Expr]:
    subs = true_linear_limit_substitutions()
    total = paper1_octic_final_linear_x4()["X4_total"]
    h08 = paper1_h08_k4_linear_projection()["X4"]
    quartic_expected = (
        sp.Rational(1, 24)
        * sp.Symbol("k'_{klmn}")
        * sp.Symbol("C_k^{yy}")
        * sp.Symbol("C_l^{yy}")
        * sp.Symbol("C_m^{yy}")
        * sp.Symbol("C_n^{yy}")
    )
    return {
        "X4_total_linear": sp.expand(total.subs(subs)),
        "X4_h08_linear": sp.expand(h08.subs(subs)),
        "quartic_expected": quartic_expected,
        "gap_total_minus_quartic": sp.expand(total.subs(subs) - quartic_expected),
        "gap_h08_minus_quartic": sp.expand(h08.subs(subs) - quartic_expected),
    }


def main() -> None:
    out = audit_true_linear_limit()
    for k, v in out.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()
