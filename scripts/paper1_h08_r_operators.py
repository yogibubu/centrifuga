#!/usr/bin/env python3
"""Exact visible R-operator definitions from the notation screenshot.

The one-index Coriolis-like object in the notation panel is R-tilde_k, not R'_k.
The explicit one-index prime definition is recorded separately in
paper1_h08_tablev_rprimek.py from the dedicated screenshot.
"""

from __future__ import annotations

import sympy as sp


def paper1_r_operator_definitions() -> dict[str, sp.Eq]:
    wk, wl, wm = sp.symbols("omega_k omega_l omega_m", positive=True)
    Ba, Bg, Bd = sp.symbols("B_a B_g B_d", positive=True)

    Rtk = sp.Symbol("R̃_k", commutative=False)
    Rtl = sp.Symbol("R̃_l", commutative=False)
    Rk = sp.Symbol("R_k", commutative=False)
    Rkl = sp.Symbol("R_kl", commutative=False)
    Rlk = sp.Symbol("R_lk", commutative=False)
    Rklm = sp.Symbol("R_klm", commutative=False)
    Rmlk = sp.Symbol("R_mlk", commutative=False)

    Ja = sp.Symbol("J_a", commutative=False)
    Jab = sp.Symbol("J_a J_b", commutative=False)
    Jabg = sp.Symbol("J_a J_d J_g", commutative=False)

    Ckab = sp.Symbol("C_k^{ab}")
    Clab = sp.Symbol("C_l^{ab}")
    Cmag = sp.Symbol("C_m^{ag}")
    Cmdb = sp.Symbol("C_m^{db}")
    zeta_kla = sp.Symbol("zeta_{kl}^a")
    zeta_lma = sp.Symbol("zeta_{lm}^a")

    return {
        "R_tilde_k": sp.Eq(
            Rtk,
            -2 * sp.sqrt(wl / wk) * sp.Symbol("Σ_a B_a zeta_{kl}^a J_a", commutative=False),
        ),
        "R_tilde_k_to_R_tilde_l": sp.Eq(Rtk, -sp.sqrt(wl / wk) * Rtl),
        "R_klm": sp.Eq(
            Rklm,
            -sp.Rational(1, 4)
            * wk
            * wl
            * wm
            * sp.Symbol(
                "Σ_abgd ((C_k^{ag} C_m^{db} + C_m^{ag} C_k^{db}) C_l^{gb} / (B_g B_d)) J_a J_d J_g",
                commutative=False,
            ),
        ),
        "R_mlk": sp.Eq(Rmlk, Rklm),
        "R_k": sp.Eq(
            Rk,
            -wk * sp.Symbol("Σ_ab C_k^{ab} J_a J_b", commutative=False),
        ),
        "R_kl": sp.Eq(
            Rkl,
            sp.Rational(3, 8)
            * wk
            * wl
            * sp.Symbol(
                "Σ_abg ((C_k^{ag} C_l^{bg} + C_l^{ag} C_k^{bg})/B_g) J_a J_b",
                commutative=False,
            ),
        ),
        "R_lk": sp.Eq(Rlk, Rkl),
        "k'_klm": sp.Eq(
            sp.Symbol("k'_klm"),
            sp.Symbol("(∂^3 V / ∂q_k ∂q_l ∂q_m)_e / hc"),
        ),
        "k'_{klmn}": sp.Eq(
            sp.Symbol("k'_{klmn}"),
            sp.Symbol("(∂^4 V / ∂q_k ∂q_l ∂q_m ∂q_n)_e / hc"),
        ),
    }


def main() -> None:
    for key, value in paper1_r_operator_definitions().items():
        print(key, ":", value)


if __name__ == "__main__":
    main()
