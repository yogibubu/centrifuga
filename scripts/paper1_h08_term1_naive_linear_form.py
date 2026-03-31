#!/usr/bin/env python3
"""Naive true-linear geometric form of the one-line H08 term RRR R_klm."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term1_naive_linear_form() -> dict[str, sp.Expr]:
    B, wk, wl, wm = sp.symbols("B omega_k omega_l omega_m", nonzero=True)
    Ck, Cl, Cm, X = sp.symbols("C_k C_l C_m X")

    Rprod = -wk * wl * wm * Ck * Cl * Cm * X**3
    Rklm = -sp.Rational(1, 2) * wk * wl * wm * Ck * Cl * Cm * sp.Symbol("(J_y^3+J_z^3)") / B**2
    term1 = sp.expand(-Rprod * Rklm / (wk * wl * wm))

    return {
        "R_product": Rprod,
        "R_klm_linear": Rklm,
        "term1_naive_linear": term1,
        "statement": (
            "If one inserts the true-linear geometric forms of R_k and R_klm into "
            "the one-line term -R_k R_l R_m R_klm/(omega_k omega_l omega_m), the "
            "naive commuting result is proportional to X^3 (J_y^3 + J_z^3), not to "
            "a pure scalar X^4."
        ),
    }


def main() -> None:
    out = paper1_h08_term1_naive_linear_form()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
