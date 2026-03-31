#!/usr/bin/env python3
"""Exact classification of the visible (R'_k)^2 term in the linear CeDiTT4 hierarchy."""

from __future__ import annotations

import sympy as sp

from scripts.linear_quartic_scalar_basis import linear_quartic_scalar_basis


def paper1_h08_term5_pure_octic() -> dict[str, object]:
    rho, omega = sp.symbols("rho_k omega_k")
    Jy, Jz = sp.symbols("Jy Jz")
    quartic = linear_quartic_scalar_basis()["basis_operator"]
    rp = sp.expand(rho * quartic)
    term5 = sp.expand(-rp**2 / (2 * omega))
    x0 = sp.expand((Jy**2 + Jz**2) ** 4)
    coeff = sp.simplify(-rho**2 / (2 * omega))
    return {
        "Rp_linear_form": rp,
        "term5_linear": term5,
        "x0_operator": x0,
        "x0_coefficient": coeff,
        "difference": sp.expand(term5 - coeff * x0),
        "statement": (
            "Because the true-linear quartic scalar space is one-dimensional, "
            "R'_k must be proportional to (J_perp^2)^2. Therefore the visible "
            "one-line term -(R'_k)^2/(2 omega_k) is a pure octic X0 contribution "
            "with no lower-order contamination."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term5_pure_octic())
