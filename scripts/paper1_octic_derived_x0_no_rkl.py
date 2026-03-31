#!/usr/bin/env python3
"""Exact X0 total-so-far with r_k and r_kl removed in favor of primitive contractions."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_derived_x0_no_rk import paper1_octic_derived_x0_no_rk


def paper1_octic_derived_x0_no_rkl() -> dict[str, object]:
    omega_k, omega_l, B = sp.symbols("omega_k omega_l B", nonzero=True)
    Ck, Cl = sp.symbols("C_k C_l")
    contraction = sp.Symbol("Σ_l C_l ((3/(4B)) omega_k omega_l C_k C_l + (1/2) Σ_m k'_{klm} C_m)")

    total = paper1_octic_derived_x0_no_rk()["primitive_total_without_rk"]
    no_rkl = sp.expand(total.subs(sp.Symbol("Σ_l C_l r_kl"), contraction))

    return {
        "rk_contraction_reduced": sp.Eq(sp.Symbol("Σ_l C_l r_kl"), contraction),
        "primitive_total_without_rkl": no_rkl,
        "statement": (
            "Using the exact true-linear reduction of R'_{kl}, the current derived X0 "
            "total can be rewritten without r_{kl} in favor of the geometric contraction "
            "(3/4B) omega_k omega_l C_k C_l and the explicit cubic scalar contraction "
            "(1/2) Σ_m k'_{klm} C_m."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_derived_x0_no_rkl())
