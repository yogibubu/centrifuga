#!/usr/bin/env python3
"""Effective matching condition for the full one-line term R'_k R_l R_m R'_{lm}."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_effective_matching() -> dict[str, sp.Expr]:
    wk, B, DJ = sp.symbols("omega_k B D_J", nonzero=True)
    rk = sp.Symbol("r_k")
    Ck = sp.Symbol("C_k")
    Slm = sp.Symbol("S_k[lm]")  # effective lm contraction after linear reduction

    term3_eff = -rk * Slm / wk
    target_geom = 8 * B * DJ * Ck**2 / wk
    condition = sp.Eq(Slm, sp.simplify(-8 * B * DJ * Ck**2 / rk))

    rho_diag = sp.Symbol("rho_k")
    diagonal_ansatz = sp.Eq(rho_diag, sp.simplify(-8 * B * DJ / rk))

    return {
        "term3_eff": term3_eff,
        "target_geom": target_geom,
        "matching_condition": condition,
        "diagonal_only_ansatz": diagonal_ansatz,
        "statement": (
            "If u_k = r_k is already fixed, then the full lm-reduced one-line "
            "term R'_k R_l R_m R'_{lm} must satisfy S_k[lm] = -8 B D_J C_k^2/r_k "
            "after linear reduction. Under a minimal diagonal-only contraction "
            "ansatz this becomes rho_k = -8 B D_J / r_k."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_effective_matching()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
