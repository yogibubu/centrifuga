#!/usr/bin/env python3
"""Linear commuting-sector ansatz for the sub-principal one-line H08 terms.

This is not the principal-symbol grading. It is the minimal K=0 effective
grading consistent with the known linear scalar formula for L.
"""

from __future__ import annotations

import sympy as sp


def paper1_h08_subprincipal_linear_ansatz() -> dict[str, sp.Expr]:
    wk, wl, wm, wn = sp.symbols("omega_k omega_l omega_m omega_n", nonzero=True)
    Ck, Cl, Cm, Cn = sp.symbols("C_k C_l C_m C_n")
    uk = sp.Symbol("u_k")
    rlm = sp.Symbol("r_lm")
    tklm = sp.Symbol("t_klm")
    k4 = sp.Symbol("k'_{klmn}")
    X = sp.Symbol("X")

    Rk = -wk * Ck * X
    Rl = -wl * Cl * X
    Rm = -wm * Cm * X
    Rn = -wn * Cn * X
    Rpk = uk * X**2
    # Minimal K=0 effective degree required for term3 to contribute to X^4.
    Rplm = rlm
    # Minimal K=0 effective degree required for term1 to contribute to X^4.
    Rklm = tklm * X

    term1 = sp.expand(-Rk * Rl * Rm * Rklm / (wk * wl * wm))
    term2 = sp.expand(Rk * Rl * Rm * Rn * k4 / (24 * wk * wl * wm * wn))
    term3 = sp.expand(-Rpk * Rl * Rm * Rplm / (wk * wl * wm))
    term5 = sp.expand(-(Rpk**2) / (2 * wk))

    coeff_x4 = {
        "term1_x4": sp.expand(sp.Poly(term1, X).coeff_monomial(X**4)),
        "term2_x4": sp.expand(sp.Poly(term2, X).coeff_monomial(X**4)),
        "term3_x4": sp.expand(sp.Poly(term3, X).coeff_monomial(X**4)),
        "term5_x4": sp.expand(sp.Poly(term5, X).coeff_monomial(X**4)),
    }
    coeff_x4["total_x4"] = sp.expand(sum(coeff_x4.values()))
    return coeff_x4


def main() -> None:
    out = paper1_h08_subprincipal_linear_ansatz()
    for k, v in out.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()
