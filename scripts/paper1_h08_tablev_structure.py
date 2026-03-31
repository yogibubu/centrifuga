#!/usr/bin/env python3
"""Transcription of the bare H08 block from Table V of Paper 1.

This module separates:

1. the exact visible leading H08 formula, which is fully legible in the
   high-resolution crop of Table V;
2. the remaining additive channel structure of the bare octic block;
3. the visible object dependencies before the Eq. (99) block-diagonal
   completion.
"""

from __future__ import annotations

import sympy as sp


def paper1_h08_tablev_leading_terms() -> dict[str, sp.Expr]:
    """Return the exact visible bare H08 formula from the first Table V line."""
    wk, wl, wm, wn = sp.symbols("omega_k omega_l omega_m omega_n", positive=True)
    Rk, Rl, Rm, Rn = sp.symbols("R_k R_l R_m R_n", commutative=False)
    Rklm = sp.Symbol("R_klm", commutative=False)
    Rplm = sp.Symbol("R'_lm", commutative=False)
    Rpk = sp.Symbol("R'_k", commutative=False)
    k4 = sp.Symbol("k'_klmn")
    comm_kl = sp.Symbol("[R_k,R_l]", commutative=False)

    term1 = -Rk * Rl * Rm * Rklm / (wk * wl * wm)
    term2 = Rk * Rl * Rm * Rn * k4 / (24 * wk * wl * wm * wn)
    term3 = -Rpk * Rl * Rm * Rplm / (wk * wl * wm)
    term4 = -sp.I * Rpk * Rl * comm_kl / (wk**2 * wl)
    term5 = -(Rpk**2) / (2 * wk)
    return {
        "term1_rrrr": term1,
        "term2_rrrrk4": term2,
        "term3_rprime_rr_rprime": term3,
        "term4_rprime_r_comm": term4,
        "term5_rprime_sq": term5,
        "visible_h08_formula": sp.simplify(term1 + term2 + term3 + term4 + term5),
    }


def paper1_h08_tablev_exact_visible_statement() -> str:
    return (
        "The first H08 line of Table V is fully legible and contains exactly five "
        "terms: RRRR/omega^3, RRRR*k4/omega^4, R'RRR'/omega^3, "
        "i R'R[R,R]/(omega^2*omega), and (R')^2/(2 omega)."
    )


def paper1_h08_tablev_channel_structure() -> dict[str, tuple[str, ...]]:
    """Return the additive channel structure visibly present in Table V for H08."""
    return {
        "pure_cubic_cubic_scalar": (
            "k3*k3 over nonresonant energy denominators",
            "k3*k3 with doubly summed nonresonant denominators",
            "k3*k3 starred resonant-exclusion sums",
        ),
        "r_blocks": (
            "R*R*R*R over omega-products",
            "R*R*R*R*k4 over omega-products",
            "R*R*[R,R] over omega-products",
            "(R')^2 over omega",
        ),
        "x_f_blocks": (
            "R*(X*R - X*R) / omega",
            "(X/omega + X/omega)[R,R]",
            "X*X*F terms",
        ),
        "uv_block": (
            "-8 * sum_m { U_km U_lm (2 omega_m + omega_k + omega_l) + V_km V_lm (2 omega_m - omega_k - omega_l) }",
        ),
        "ladder_blocks": (
            "quartic ladder products times k4",
            "quartic ladder products times R and X",
            "quartic ladder products times B*zeta*R kernels",
        ),
    }


def paper1_h08_tablev_visible_dependencies() -> dict[str, tuple[str, ...]]:
    """Return the visible object dependencies of the bare H08 block."""
    return {
        "depends_on": ("k3", "k4", "R", "X", "F", "U", "V", "B", "zeta", "omega"),
        "not_visibly_needed": ("mu3",),
    }


def paper1_h08_tablev_minimal_statement() -> str:
    return (
        "The bare H08 entry of Table V is a sum of k3^2, R-block, X/F-block, "
        "U/V-block, and ladder-type sectors before the Eq. (99) block-diagonal completion."
    )


def main() -> None:
    print(paper1_h08_tablev_exact_visible_statement())
    for key, value in paper1_h08_tablev_channel_structure().items():
        print(key, ":", value)
    print("visible_h08_formula:", paper1_h08_tablev_leading_terms()["visible_h08_formula"])


if __name__ == "__main__":
    main()
