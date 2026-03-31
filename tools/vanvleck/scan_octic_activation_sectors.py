#!/usr/bin/env python3
"""Scan exact BCH sectors that activate the octic rotational-ground block."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import dataclass

from tools.vanvleck import derive_watson_octic_from_loworder_vanvleck as dlo


@dataclass(frozen=True)
class OcticSectorScanCase:
    name: str
    n_modes: int
    diag_rot_only: bool
    max_order: int = 4
    max_j: int = 8
    max_v: int = 8


def default_scan_cases() -> tuple[OcticSectorScanCase, ...]:
    return (
        OcticSectorScanCase("1mode_diag", n_modes=1, diag_rot_only=True),
        OcticSectorScanCase("1mode_fullrot", n_modes=1, diag_rot_only=False),
        OcticSectorScanCase("2mode_diag", n_modes=2, diag_rot_only=True),
        OcticSectorScanCase("2mode_fullrot", n_modes=2, diag_rot_only=False),
    )


def run_octic_activation_scan() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    first_active: str | None = None
    for case in default_scan_cases():
        out = dlo.derive_loworder_octic_problem(
            n_modes=case.n_modes,
            diag_rot_only=case.diag_rot_only,
            max_order=case.max_order,
            max_j=case.max_j,
            max_v=case.max_v,
        )
        octic_total = out["analysis"]["octic_total"]
        row = {
            "name": case.name,
            "n_modes": case.n_modes,
            "diag_rot_only": case.diag_rot_only,
            "quartic_count": len(out["analysis"]["quartic_total"]),
            "sextic_count": len(out["analysis"]["sextic_total"]),
            "octic_count": len(octic_total),
            "octic_active": bool(octic_total),
        }
        if first_active is None and row["octic_active"]:
            first_active = case.name
        rows.append(row)
    return {
        "cases": rows,
        "first_active_case": first_active,
    }


if __name__ == "__main__":
    print(run_octic_activation_scan())
