#!/usr/bin/env python3
"""Exact true-linear kernel formula for the visible one-line H08 term 3."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_kernel_formula() -> dict[str, object]:
    omega_k = sp.Symbol("omega_k", nonzero=True)
    r_k = sp.Symbol("r_k")
    C_l, C_m = sp.symbols("C_l C_m")
    r_lm = sp.Symbol("r_lm")

    scalar_prefactor = sp.expand(-r_k * C_l * C_m * r_lm / omega_k)
    kernel = sp.expand(-scalar_prefactor)

    return {
        "source_scalar_prefactor": scalar_prefactor,
        "term3_kernel": kernel,
        "normalized_octic_piece": sp.expand(-sp.Rational(63, 256) * kernel),
        "statement": (
            "In the true linear limit the visible one-line term "
            "-R'_k R_l R_m R'_{lm}/(omega_k omega_l omega_m) reduces to a scalar "
            "prefactor times the universal K=0 polynomial of X_perp^5. Since that "
            "polynomial is minus the normalized universal source polynomial, the "
            "physical kernel entering the octic X0 piece is K_term3 = r_k C_l C_m r_lm / omega_k."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_kernel_formula())
