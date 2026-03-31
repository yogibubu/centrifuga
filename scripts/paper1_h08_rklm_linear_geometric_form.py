#!/usr/bin/env python3
"""True linear-limit geometric form of the printed three-index operator R_klm."""

from __future__ import annotations

import sympy as sp


def paper1_h08_rklm_linear_geometric_form() -> dict[str, sp.Expr]:
    B, wk, wl, wm = sp.symbols("B omega_k omega_l omega_m", nonzero=True)
    Ck, Cl, Cm = sp.symbols("C_k C_l C_m")

    # In the true linear a-axis limit:
    # C_n^{yy} = C_n^{zz} = C_n,  C_n^{yz} = 0,  B_y = B_z = B.
    # The printed R_klm definition then keeps only the yyy and zzz index chains.
    coeff = -sp.Rational(1, 2) * wk * wl * wm * Ck * Cl * Cm / B**2
    Rklm_linear = sp.expand(coeff * (sp.Symbol("J_y^3") + sp.Symbol("J_z^3")))

    return {
        "coefficient": coeff,
        "R_klm_linear_geom": Rklm_linear,
        "statement": (
            "Under the true linear y/z symmetry, the geometric part of the printed "
            "three-index operator R_klm reduces to -(1/2) omega_k omega_l omega_m "
            "C_k C_l C_m / B^2 times (J_y^3 + J_z^3)."
        ),
    }


def main() -> None:
    out = paper1_h08_rklm_linear_geometric_form()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
