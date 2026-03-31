#!/usr/bin/env python3
"""Minimal matching identities between one-line H08 subprincipal terms and linear L."""

from __future__ import annotations

import sympy as sp


def paper1_h08_linear_matching() -> dict[str, sp.Expr]:
    omega = sp.Symbol("omega_k", nonzero=True)
    u = sp.Symbol("u_k")
    r = sp.Symbol("r_k")
    C = sp.Symbol("C_k")
    B = sp.Symbol("B")
    DJ = sp.Symbol("D_J")

    term5 = -u**2 / (2 * omega)
    target_rsq = -r**2 / (2 * omega)
    u_match_gap = sp.expand(term5.subs({u: r}) - target_rsq)

    geom = 8 * B * DJ * C**2 / omega
    offset = -8 * DJ**3 / B**2

    return {
        "term5": term5,
        "target_rsq": target_rsq,
        "u_eq_r_gap": u_match_gap,
        "remaining_geometric_piece": geom + offset,
        "statement": (
            "Setting u_k = r_k makes the one-line (R'_k)^2 contribution "
            "coincide exactly with the -r_k^2/(2 omega_k) term of the known "
            "linear formula for L. The remaining geometric + offset part must "
            "therefore be carried by the RRRR_klm and R'RRR'_{lm} one-line terms."
        ),
    }


def main() -> None:
    out = paper1_h08_linear_matching()
    for k, v in out.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()
