#!/usr/bin/env python3
"""Detailed mapping between BCH generators and Watson-labelled source sectors."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict

from tools.vanvleck import derive_watson_generic_vanvleck as dvgv
import vanvleck_bch_engine as vbe


def map_generators_to_watson(
    *,
    n_modes: int = 1,
    max_order: int = 4,
    max_j: int = 8,
    max_v: int = 8,
    diag_rot_only: bool = True,
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
    watson_bridge = {1: "S03", 2: "S05", 3: "S07"}
    report = {}
    for order, expr in sorted(out["s_series"].items()):
        by_origin_vj: dict[tuple[vbe.Origin, int, int], int] = defaultdict(int)
        for (vword, jword, origin), coeff in expr.items():
            if coeff == 0:
                continue
            by_origin_vj[(origin, len(vword), len(jword))] += 1
        report[f"S^({order})"] = {
            "watson_label": watson_bridge.get(order),
            "origin_vj_table": dict(sorted(by_origin_vj.items())),
            "term_count": len(expr),
        }
    return {"mapping": report}


if __name__ == "__main__":
    print(map_generators_to_watson())
