#!/usr/bin/env python3
"""Exact split of the full lm contraction entering the one-line H08 term3."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_contraction_split() -> dict[str, sp.Expr]:
    Ck = sp.Symbol("C_k")
    Slm = sp.Symbol("S_k[lm]")

    geom = sp.Symbol("G_k[R_lm]")
    cubic = sp.Symbol("Q_k[k3,R_n]")

    # From the printed relation:
    # R'_{lm} = R_{lm} - 1/2 sum_n k'_{lmn} R_n / omega_n
    split = sp.Eq(Slm, geom + cubic)

    # Since R_n = - omega_n C_n X, the subtracted cubic part contributes
    # +1/2 sum_{lmn} C_l C_m k'_{lmn} C_n after K=0 scalar reduction.
    cubic_scalar = sp.Symbol("1/2 Σ_lmn C_l C_m C_n k'_{lmn}")
    cubic_eq = sp.Eq(cubic, cubic_scalar)

    return {
        "split": split,
        "cubic_piece": cubic_eq,
        "statement": (
            "The full contraction S_k[lm] entering the one-line term R'_k R_l R_m R'_{lm} "
            "splits exactly into a piece carried by R_lm and a cubic-subtraction piece "
            "coming from -(1/2) sum_n k'_{lmn} R_n/omega_n. After K=0 reduction the latter "
            "is proportional to 1/2 sum_lmn C_l C_m C_n k'_{lmn}."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_contraction_split()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
