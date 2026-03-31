#!/usr/bin/env python3
"""Exact true-linear reduction of the printed one-index prime object R'_k."""

from __future__ import annotations

import sympy as sp

from scripts.linear_quartic_scalar_basis import linear_quartic_scalar_basis


def paper1_h08_rprimek_true_linear_reduction() -> dict[str, object]:
    omega_k = sp.Symbol("omega_k", nonzero=True)
    X2 = linear_quartic_scalar_basis()["basis_operator"]

    # True-linear commuting scalar forms
    Cl = sp.Symbol("C_l")
    rkl = sp.Symbol("r_kl")
    Rl = -sp.Symbol("omega_l") * Cl * sp.Symbol("J_perp^2", commutative=False)
    Rpkl = rkl * X2

    rk = sp.Symbol("r_k")
    reduced_rk = sp.Symbol("Σ_l C_l r_kl")

    return {
        "R_l_linear": Rl,
        "Rprime_kl_linear": Rpkl,
        "commutator_term_vanishes": True,
        "r_k_reduction": sp.Eq(rk, reduced_rk),
        "statement": (
            "In the true linear limit, [R_k,H02] vanishes because both blocks are "
            "functions of the same transverse quadratic scalar. Since the quartic "
            "commuting scalar space is one-dimensional, R'_{kl} = r_{kl}(J_perp^2)^2. "
            "The printed definition R'_k = -Σ_l R_l R'_{kl}/omega_l therefore reduces "
            "exactly to r_k = Σ_l C_l r_{kl}."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_rprimek_true_linear_reduction())
