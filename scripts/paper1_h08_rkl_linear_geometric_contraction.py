#!/usr/bin/env python3
"""Linear-limit geometric contraction induced by the printed R_lm definition."""

from __future__ import annotations

import sympy as sp


def paper1_h08_rkl_linear_geometric_contraction() -> dict[str, sp.Expr]:
    B, wl, wm, X = sp.symbols("B omega_l omega_m X", nonzero=True)
    Cl, Cm = sp.symbols("C_l C_m")

    # Printed geometric definition:
    # R_lm = (3/8) wl wm sum_abg ((C_l^{ag} C_m^{bg} + C_m^{ag} C_l^{bg})/B_g) J_a J_b
    #
    # Linear a-axis limit with transverse cylindrical symmetry:
    # C_n^{yy} = C_n^{zz} = C_n,  C_n^{yz} = 0,  B_y = B_z = B
    # so only yy and zz survive.
    Rlm_geom = sp.expand(
        sp.Rational(3, 8)
        * wl
        * wm
        * (
            2 * Cl * Cm / B * sp.Symbol("J_y^2")
            + 2 * Cl * Cm / B * sp.Symbol("J_z^2")
        )
    )

    # K=0 scalar reduction:
    # J_y^2 + J_z^2 -> X
    rlm_scalar = sp.expand(sp.Rational(3, 4) * wl * wm * Cl * Cm / B)

    Gk_geom = sp.Symbol("Σ_lm C_l C_m r_lm^(geom)")
    Gk_geom_closed = sp.Symbol("(3/4B) Σ_lm omega_l omega_m C_l^2 C_m^2")

    return {
        "Rlm_geom_linear": Rlm_geom,
        "rlm_scalar_geom": rlm_scalar,
        "Gk_geom_symbol": Gk_geom,
        "Gk_geom_closed": Gk_geom_closed,
        "statement": (
            "Using the printed geometric definition of R_lm and imposing the true "
            "linear y/z symmetry, the K=0 scalar reduction gives r_lm^(geom) = "
            "(3/4B) omega_l omega_m C_l C_m. Therefore the full geometric lm "
            "contraction is proportional to (3/4B) sum_lm omega_l omega_m C_l^2 C_m^2."
        ),
    }


def main() -> None:
    out = paper1_h08_rkl_linear_geometric_contraction()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
