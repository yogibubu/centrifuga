#!/usr/bin/env python3
"""Exact identification of the visible H08 term-3 source with the transverse decatic scalar."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term3_universal_polynomial import paper1_h08_term3_universal_polynomial


def paper1_h08_term3_transverse_decatic_identity() -> dict[str, object]:
    X = sp.Symbol("X")
    decatic_k0 = sp.expand(
        -sp.Rational(31, 2) * X
        + sp.Rational(97, 16) * X**2
        + sp.Rational(9, 64) * X**3
        - sp.Rational(5, 64) * X**4
        + sp.Rational(63, 256) * X**5
    )
    universal = paper1_h08_term3_universal_polynomial()["universal_polynomial"]
    return {
        "k0_of_Jperp2_pow5": decatic_k0,
        "visible_term3_universal_poly": universal,
        "difference": sp.expand(universal + decatic_k0),
        "statement": (
            "The normalized K=0 source polynomial of the visible one-line term "
            "-R'_k R_l R_m R'_{lm}/(omega_k omega_l omega_m) is exactly the "
            "negative of the K=0 diagonal polynomial of (J_perp^2)^5. The "
            "visible source therefore carries a pure transverse decatic scalar, "
            "not yet the final octic scalar."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_transverse_decatic_identity())
