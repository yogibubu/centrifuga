#!/usr/bin/env python3
"""Structural k-independence of the full lm contraction in the one-line H08 term3."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_k_independence() -> dict[str, sp.Expr]:
    wk = sp.Symbol("omega_k", nonzero=True)
    rk = sp.Symbol("r_k")
    Ck = sp.Symbol("C_k")
    B = sp.Symbol("B")
    DJ = sp.Symbol("D_J")

    S = sp.Symbol("S[lm]")
    term3_eff = -rk * S / wk
    target = 8 * B * DJ * Ck**2 / wk
    matching = sp.Eq(S, sp.simplify(-8 * B * DJ * Ck**2 / rk))
    ratio_condition = sp.Eq(sp.Symbol("const"), sp.simplify(Ck**2 / rk))

    return {
        "term3_eff": term3_eff,
        "target": target,
        "matching": matching,
        "ratio_condition": ratio_condition,
        "statement": (
            "Because the printed one-line term R'_k R_l R_m R'_{lm} contracts only "
            "over l,m inside R'_{lm}, the reduced scalar contraction is independent "
            "of k. Therefore exact matching of 8 B D_J C_k^2/omega_k requires the "
            "ratio C_k^2/r_k to be mode-independent."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_k_independence()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
