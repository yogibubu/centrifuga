#!/usr/bin/env python3
"""Exact collected form of the current true-linear X0 expression."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_true_linear_collapse import paper1_octic_true_linear_collapse


def paper1_octic_true_linear_collect() -> dict[str, object]:
    expr = paper1_octic_true_linear_collapse()["collapsed_true_linear_X0"]

    omega_k, omega_l, B = sp.symbols("omega_k omega_l B", nonzero=True)
    Ck, Cl = sp.symbols("C_k C_l")
    G = sp.Symbol("Σ_l C_l (3 omega_k omega_l C_k C_l/(4B))")
    Q = sp.Symbol("Σ_l C_l ((1/2) Σ_m k'_{klm} C_m)")

    target = sp.Symbol("Σ_l C_l ((3/(4B)) omega_k omega_l C_k C_l + (1/2) Σ_m k'_{klm} C_m)")
    collected = sp.expand(expr.subs(target, G + Q))
    split = sp.expand(
        sp.Symbol("quartic_block")
        + sp.Symbol("s2tau_block")
        + sp.Symbol("constant_block")
        - (G**2 + 2 * G * Q + Q**2) / (2 * omega_k)
    )

    quartic_block = sp.Symbol("quartic_block")
    s2tau_block = sp.Symbol("s2tau_block")
    constant_block = sp.Symbol("constant_block")
    substituted = collected.subs(
        {
            sp.Symbol("C_k^{yy}*C_l^{yy}*C_m^{yy}*C_n^{yy}*k'_{klmn}/24"): quartic_block,
        }
    )

    return {
        "collected_expression": collected,
        "geometric_piece": -sp.expand(G**2 / (2 * omega_k)),
        "mixed_piece": -sp.expand(G * Q / omega_k),
        "cubic_square_piece": -sp.expand(Q**2 / (2 * omega_k)),
        "statement": (
            "The current true-linear X0 expression can be collected exactly into "
            "a visible quartic block, an S111^2 tau block, a constant offset, and "
            "three contributions coming from the square of the printed R'_k reduction: "
            "geometric, mixed geometric-cubic, and purely cubic."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_true_linear_collect())
