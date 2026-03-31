#!/usr/bin/env python3
"""Extract BCH generators S^(n) and summarize them in Watson-compatible form."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict
from typing import Dict

import sympy as sp

import vanvleck_bch_engine as vbe
from tools.vanvleck import derive_watson_generic_vanvleck as dvgv


def _group_generator(expr: Dict[vbe.Key, sp.Expr]) -> dict[str, object]:
    by_vj: dict[tuple[int, int], int] = defaultdict(int)
    by_origin: dict[vbe.Origin, int] = defaultdict(int)
    for (vword, jword, origin), coeff in expr.items():
        if coeff == 0:
            continue
        by_vj[(len(vword), len(jword))] += 1
        by_origin[origin] += 1
    return {
        "term_count": len(expr),
        "by_vj_degree": dict(sorted(by_vj.items())),
        "by_origin": dict(sorted(by_origin.items())),
    }


def extract_reference_generators(
    *,
    n_modes: int = 1,
    diag_rot_only: bool = True,
    max_order: int = 4,
    max_j: int = 8,
    max_v: int = 8,
) -> dict[str, object]:
    out = dvgv.build_generic_reference_problem(
        n_modes=n_modes,
        potential_orders=(3, 4, 5, 6),
        rotder_orders=(1, 2, 3),
        order_assignment={
            1: ("H12", "H30"),
            2: ("H22", "H40"),
            3: ("H32", "H50"),
            4: ("H60",),
        },
        max_order=max_order,
        max_j=max_j,
        max_v=max_v,
        diag_rot_only=diag_rot_only,
    )
    s_series = out["s_series"]
    summary = {}
    watson_bridge = {1: "S03", 2: "S05", 3: "S07"}
    for order in sorted(s_series):
        summary[f"S^({order})"] = {
            "watson_label": watson_bridge.get(order),
            "summary": _group_generator(s_series[order]),
            "expr": s_series[order],
        }
    return {
        "generators": summary,
        "notation_bridge": {
            "S^(1)": "S03",
            "S^(2)": "S05",
            "S^(3)": "S07",
        },
    }


if __name__ == "__main__":
    report = extract_reference_generators()
    for name, payload in report["generators"].items():
        print(name, payload["watson_label"], payload["summary"])
