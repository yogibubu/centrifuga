#!/usr/bin/env python3
"""Explicit CeDiTT4-style X0 decomposition of the direct octic channels."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_axial_decomposition import paper1_octic_axial_decomposition


def paper1_octic_x0_decomposition() -> dict[str, object]:
    out = paper1_octic_axial_decomposition()
    x0_terms = {name: sp.expand(block["axial"]["X0"]) for name, block in out["termwise"].items()}
    total = sp.expand(sum(x0_terms.values()))
    return {
        "termwise_X0": x0_terms,
        "total_X0": total,
        "statement": (
            "In the CeDiTT4 axial formulation the linear optical constant is "
            "introduced only at the end, as L = X0. The direct octic channels "
            "therefore enter the linear limit through their X0 projections."
        ),
    }


if __name__ == "__main__":
    out = paper1_octic_x0_decomposition()
    for key, val in out["termwise_X0"].items():
        print(key, ":", val)
    print("total X0 =", out["total_X0"])
