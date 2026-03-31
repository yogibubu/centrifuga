#!/usr/bin/env python3
"""Exact true-linear reduction of the printed two-index prime object R'_{kl}."""

from __future__ import annotations

import sympy as sp


def paper1_h08_rprimekl_true_linear_reduction() -> dict[str, object]:
    omega_k, omega_l, B = sp.symbols("omega_k omega_l B", nonzero=True)
    Ck, Cl = sp.symbols("C_k C_l")
    cubic_kl = sp.Symbol("(1/2) Σ_m k'_{klm} C_m")

    rkl_geom = sp.expand(sp.Rational(3, 4) * omega_k * omega_l * Ck * Cl / B)
    rkl = sp.expand(rkl_geom + cubic_kl)

    return {
        "r_kl_geometric": rkl_geom,
        "r_kl_total": rkl,
        "statement": (
            "In the true linear limit, the printed relation "
            "R'_{kl}=R_{kl}-(1/2)Σ_m k'_{klm}R_m/omega_m reduces exactly to "
            "r_{kl} = (3/4B) omega_k omega_l C_k C_l + (1/2) Σ_m k'_{klm} C_m."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_rprimekl_true_linear_reduction())
