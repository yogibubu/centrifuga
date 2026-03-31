#!/usr/bin/env python3
"""Extract Watson-style octic channels from the BCH generators."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sympy as sp

from tools.vanvleck import derive_watson_generic_vanvleck as dvgv
from tools.vanvleck import derive_watson_octic_vanvleck as dovv
import vanvleck_bch_engine as vbe
from vanvleck_harmonic_rotor import build_h02


def _nested_comm(
    left: dict[vbe.Key, sp.Expr],
    right: dict[vbe.Key, sp.Expr],
    *,
    omega,
    hbar,
    max_j: int,
    max_v: int,
) -> dict[vbe.Key, sp.Expr]:
    return vbe.comm_expr(left, right, max_j=max_j, max_v=max_v)


def extract_reference_octic_channels(
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
    s = out["s_series"]
    b = out["blocks"]
    omega = out["omega"]
    hbar = out["hbar"]
    h02 = build_h02()

    channels = {}
    if 1 in s:
        channels["[S^(1),[S^(1),[S^(1),H02]]]"] = _nested_comm(
            s[1],
            _nested_comm(
                s[1],
                _nested_comm(s[1], h02, omega=omega, hbar=hbar, max_j=max_j, max_v=max_v),
                omega=omega,
                hbar=hbar,
                max_j=max_j,
                max_v=max_v,
            ),
            omega=omega,
            hbar=hbar,
            max_j=max_j,
            max_v=max_v,
        )
    if 1 in s and "H40" in b:
        channels["[S^(1),[S^(1),H04]]"] = _nested_comm(
            s[1],
            _nested_comm(s[1], b["H40"], omega=omega, hbar=hbar, max_j=max_j, max_v=max_v),
            omega=omega,
            hbar=hbar,
            max_j=max_j,
            max_v=max_v,
        )
    if 1 in s and "H60" in b:
        channels["[S^(1),H06]"] = _nested_comm(
            s[1], b["H60"], omega=omega, hbar=hbar, max_j=max_j, max_v=max_v
        )
    if 2 in s and "H40" in b:
        channels["[S^(2),H04]"] = _nested_comm(
            s[2], b["H40"], omega=omega, hbar=hbar, max_j=max_j, max_v=max_v
        )

    summaries = {}
    for name, expr in channels.items():
        if not expr:
            summaries[name] = {"term_count": 0, "octic_rot_ground_count": 0, "octic_commuting": sp.Integer(0)}
            continue
        octic = dovv.extract_rot_ground_degree(expr, 8)
        summaries[name] = {
            "term_count": len(expr),
            "octic_rot_ground_count": len(octic),
            "octic_commuting": dovv.commuting_polynomial_degree(octic),
        }
    return {"channels": channels, "summaries": summaries}


if __name__ == "__main__":
    out = extract_reference_octic_channels()
    for name, payload in out["summaries"].items():
        print(name, payload)
