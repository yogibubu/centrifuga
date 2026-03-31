#!/usr/bin/env python3
"""Targeted symbolic scans for the first post-low-order octic sectors.

This script avoids the old monolithic search. It evaluates only the sectors
that remain plausible after the low-order scan:
  - H32_only
  - H50_only
  - H32_plus_H50
  - H60_only
  - highorder_full

Each run reports ordered/commuting J^8 blocks by perturbative order.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from dataclasses import dataclass

import sympy as sp

from tools.vanvleck import derive_watson_octic_vanvleck as dovv
import vanvleck_bch_engine as vbe
import vanvleck_hamiltonian_builders as vhb
import vanvleck_perturbative_input as vpi


@dataclass(frozen=True)
class SectorSpec:
    name: str
    order_to_labels: dict[int, tuple[str, ...]]


def _collapsed_two_mode_blocks() -> dict[str, object]:
    generic = vhb.build_generic_blocks(
        n_modes=2,
        potential_orders=(3, 4, 5, 6),
        rotder_orders=(1, 2, 3),
        diag_rot_only=True,
    )
    labels = ("H12", "H30", "H22", "H40", "H32", "H50", "H60")
    collapsed = {}
    for label in labels:
        symbol = sp.Symbol(label, real=True)
        collapsed[label] = vpi.collapse_block_coefficients(
            generic["blocks"][label], class_symbol=symbol
        )
    return {
        "omega": generic["omega"],
        "hbar": generic["hbar"],
        "blocks": collapsed,
    }


def _sector_specs() -> dict[str, SectorSpec]:
    return {
        "H32_only": SectorSpec("H32_only", {3: ("H32",)}),
        "H50_only": SectorSpec("H50_only", {3: ("H50",)}),
        "H32_plus_H50": SectorSpec("H32_plus_H50", {3: ("H32", "H50")}),
        "H60_only": SectorSpec("H60_only", {4: ("H60",)}),
        "highorder_full": SectorSpec(
            "highorder_full",
            {
                1: ("H12", "H30"),
                2: ("H22", "H40"),
                3: ("H32", "H50"),
                4: ("H60",),
            },
        ),
    }


def _series_for_spec(blocks: dict[str, dict], spec: SectorSpec) -> vbe.Series:
    return vpi.compose_input_series(
        {
            order: tuple(blocks[label] for label in labels)
            for order, labels in spec.order_to_labels.items()
        }
    )


def analyze_sector(
    *,
    sector_name: str,
    max_order: int = 4,
    max_j: int = 8,
    max_v: int = 8,
) -> dict[str, object]:
    specs = _sector_specs()
    spec = specs[sector_name]
    setup = _collapsed_two_mode_blocks()
    blocks = setup["blocks"]
    h_input = _series_for_spec(blocks, spec)
    k_full, s_series = vbe.build_effective_to_order(
        h_input,
        omega=setup["omega"],
        hbar=setup["hbar"],
        max_order=max_order,
        max_j=max_j,
        max_v=max_v,
    )
    octic_by_order = {
        order: dovv.extract_rot_ground_degree(k_full[order], 8)
        for order in range(1, max_order + 1)
    }
    commuting_by_order = {
        order: sp.expand(dovv.commuting_polynomial_degree(block))
        for order, block in octic_by_order.items()
    }
    return {
        "sector": sector_name,
        "generator_orders": tuple(sorted(s_series)),
        "ordered_counts": {order: len(block) for order, block in octic_by_order.items()},
        "commuting_by_order": commuting_by_order,
        "first_ordered_order": next(
            (order for order in range(1, max_order + 1) if octic_by_order[order]), None
        ),
        "first_commuting_order": next(
            (
                order
                for order in range(1, max_order + 1)
                if commuting_by_order[order] != 0
            ),
            None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", choices=tuple(_sector_specs()), required=True)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--max-j", type=int, default=8)
    parser.add_argument("--max-v", type=int, default=8)
    args = parser.parse_args()
    print(
        analyze_sector(
            sector_name=args.sector,
            max_order=args.max_order,
            max_j=args.max_j,
            max_v=args.max_v,
        )
    )


if __name__ == "__main__":
    main()
