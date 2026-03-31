#!/usr/bin/env python3
"""Exact X0 total-so-far with r_k eliminated in favor of C_l r_kl."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_derived_x0_primitives import paper1_octic_derived_x0_primitives


def paper1_octic_derived_x0_no_rk() -> dict[str, object]:
    r_k = sp.Symbol("r_k")
    reduced = sp.Symbol("Σ_l C_l r_kl")
    total = paper1_octic_derived_x0_primitives()["primitive_total_so_far"]
    no_rk = sp.expand(total.subs(r_k, reduced))
    return {
        "primitive_total_so_far": total,
        "rk_reduction": sp.Eq(r_k, reduced),
        "primitive_total_without_rk": no_rk,
        "statement": (
            "Using the exact true-linear reduction of the printed one-index prime block, "
            "the derived X0 total can be rewritten without r_k in favor of the contraction "
            "Σ_l C_l r_kl."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_derived_x0_no_rk())
