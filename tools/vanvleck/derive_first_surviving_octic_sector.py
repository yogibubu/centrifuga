#!/usr/bin/env python3
"""Locate the first low-order BCH sector with a surviving commuting J^8 block.

The search is exact and symbolic. To keep the algebra tractable without any
numerical specialization, each input block is coefficient-collapsed to a single
class symbol. This preserves BCH connectivity and exact cancellations while
avoiding irrelevant tensor-index clutter.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _collapsed_two_mode_diag_blocks() -> dict[str, object]:
    generic = vhb.build_generic_blocks(
        n_modes=2,
        potential_orders=(3, 4),
        rotder_orders=(1, 2),
        diag_rot_only=True,
    )
    raw_blocks = generic["blocks"]
    labels = ("H12", "H30", "H22", "H40")
    collapsed = {}
    for label in labels:
        symbol = sp.Symbol(label, real=True)
        collapsed[label] = vpi.collapse_block_coefficients(raw_blocks[label], class_symbol=symbol)
    return {
        "omega": generic["omega"],
        "hbar": generic["hbar"],
        "blocks": collapsed,
    }


def _sector_specs() -> tuple[SectorSpec, ...]:
    return (
        SectorSpec("H12_only", {1: ("H12",)}),
        SectorSpec("H30_only", {1: ("H30",)}),
        SectorSpec("H12_plus_H30", {1: ("H12", "H30")}),
        SectorSpec("H22_only", {2: ("H22",)}),
        SectorSpec("H40_only", {2: ("H40",)}),
        SectorSpec("H22_plus_H40", {2: ("H22", "H40")}),
        SectorSpec("loworder_full", {1: ("H12", "H30"), 2: ("H22", "H40")}),
    )


def _series_for_spec(blocks: dict[str, dict], spec: SectorSpec) -> vbe.Series:
    return vpi.compose_input_series(
        {
            order: tuple(blocks[label] for label in labels)
            for order, labels in spec.order_to_labels.items()
        }
    )


def _analyze_sector(blocks: dict[str, dict], omega, hbar, spec: SectorSpec) -> dict[str, object]:
    h_input = _series_for_spec(blocks, spec)
    k_full, _s_series = vbe.build_effective_to_order(
        h_input,
        omega=omega,
        hbar=hbar,
        max_order=4,
        max_j=8,
        max_v=8,
    )
    octic_by_order = {
        order: dovv.extract_rot_ground_degree(k_full[order], 8)
        for order in range(1, 5)
    }
    commuting_by_order = {
        order: sp.expand(dovv.commuting_polynomial_degree(block))
        for order, block in octic_by_order.items()
    }
    first_ordered = next((order for order in range(1, 5) if octic_by_order[order]), None)
    first_commuting = next((order for order in range(1, 5) if commuting_by_order[order] != 0), None)
    return {
        "name": spec.name,
        "first_ordered_order": first_ordered,
        "first_commuting_order": first_commuting,
        "ordered_counts": {order: len(block) for order, block in octic_by_order.items()},
        "commuting_by_order": commuting_by_order,
        "ordered_by_order": octic_by_order,
    }


def derive_first_surviving_octic_sector() -> dict[str, object]:
    setup = _collapsed_two_mode_diag_blocks()
    blocks = setup["blocks"]
    omega = setup["omega"]
    hbar = setup["hbar"]

    analyses = tuple(_analyze_sector(blocks, omega, hbar, spec) for spec in _sector_specs())
    first_surviving = next((entry for entry in analyses if entry["first_commuting_order"] is not None), None)
    return {
        "analyses": analyses,
        "first_surviving": first_surviving,
    }


if __name__ == "__main__":
    print(derive_first_surviving_octic_sector())
