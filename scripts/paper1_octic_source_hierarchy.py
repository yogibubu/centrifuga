#!/usr/bin/env python3
"""Exact source hierarchy for the non-linear octic block from Paper 1.

This file records only what is explicitly printed in Watson--Aliev:

1. the transformed octic block Eq. (99),
2. the fact that the bare H08 depends linearly on quartic and quadratically on
   cubic potential constants,
3. the exact list of lower-order blocks entering the block-diagonal completion.
"""

from __future__ import annotations

import sympy as sp


def comm(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return sp.Symbol(f"[{sp.sstr(a)},{sp.sstr(b)}]", commutative=False)


def paper1_tilde_h08_structure() -> dict[str, sp.Expr]:
    """Return Eq. (99) as a symbolic noncommutative expression."""
    H02 = sp.Symbol("H02", commutative=False)
    H04 = sp.Symbol("H04", commutative=False)
    H06 = sp.Symbol("H06", commutative=False)
    H08 = sp.Symbol("H08", commutative=False)
    S03 = sp.Symbol("S03", commutative=False)
    S05 = sp.Symbol("S05", commutative=False)
    S07 = sp.Symbol("S07", commutative=False)

    term0 = H08
    term1 = -sp.I / 6 * comm(S03, comm(S03, comm(S03, H02)))
    term2 = -sp.Rational(1, 2) * comm(S03, comm(S03, H04))
    term3 = sp.I * comm(S03, H06)
    term4 = sp.I * comm(S05, H04)
    term5 = -comm(S05, comm(S03, H02))
    term6 = sp.I * comm(S07, H02)
    expr = term0 + term1 + term2 + term3 + term4 + term5 + term6
    return {
        "tilde_H08": expr,
        "H08": H08,
        "S03": S03,
        "S05": S05,
        "S07": S07,
        "H02": H02,
        "H04": H04,
        "H06": H06,
        "terms": (term0, term1, term2, term3, term4, term5, term6),
    }


def paper1_octic_dependency_statement() -> dict[str, str]:
    """Return the exact printed dependency statement for the bare octic block."""
    return {
        "paper1_statement": (
            "In Table V we give for the first time a general expression for H08, "
            "which is seen to depend linearly on the quartic potential and "
            "quadratically on the cubic potential constants."
        ),
        "minimal_force_field_content": "Phi3 and Phi4",
    }


def paper1_octic_source_blocks() -> dict[str, tuple[str, ...]]:
    """Return the exact source blocks entering Eq. (99)."""
    return {
        "bare_block": ("H08",),
        "block_diagonal_completion": (
            "[S03,[S03,[S03,H02]]]",
            "[S03,[S03,H04]]",
            "[S03,H06]",
            "[S05,H04]",
            "[S05,[S03,H02]]",
            "[S07,H02]",
        ),
        "lower_order_inputs": ("H02", "H04", "H06", "S03", "S05", "S07"),
    }


def main() -> None:
    out = paper1_tilde_h08_structure()
    print("tilde_H08 =", out["tilde_H08"])
    print(paper1_octic_dependency_statement()["paper1_statement"])


if __name__ == "__main__":
    main()
