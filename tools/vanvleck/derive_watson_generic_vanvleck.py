#!/usr/bin/env python3
"""Generic Van Vleck driver built on the new BCH engine."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Iterable

import sympy as sp

from tools.vanvleck import derive_watson_octic_vanvleck as dovv
import vanvleck_bch_engine as vbe
import vanvleck_hamiltonian_builders as vhb
import vanvleck_perturbative_input as vpi


def default_generator_keep(vword: vbe.VibWord, jword: vbe.JWord) -> bool:
    return len(jword) <= 2 and len(vword) <= 4


def build_generic_reference_problem(
    *,
    n_modes: int,
    potential_orders: Iterable[int],
    rotder_orders: Iterable[int],
    order_assignment: dict[int, tuple[str, ...]],
    max_order: int,
    max_j: int,
    max_v: int,
    diag_rot_only: bool = False,
    rot_pairs: set[tuple[int, int]] | None = None,
    keep_origin_fn=None,
    combine_origin_fn=None,
    generator_keep_fn=default_generator_keep,
) -> dict[str, object]:
    generic = vhb.build_generic_blocks(
        n_modes=n_modes,
        potential_orders=potential_orders,
        rotder_orders=rotder_orders,
        diag_rot_only=diag_rot_only,
        rot_pairs=rot_pairs,
    )
    blocks = generic["blocks"]
    h_input = vpi.compose_input_series(
        {
            order: tuple(blocks[label] for label in labels)
            for order, labels in order_assignment.items()
        }
    )
    k_full, s_series = vbe.build_effective_to_order(
        h_input,
        omega=generic["omega"],
        hbar=generic["hbar"],
        max_order=max_order,
        max_j=max_j,
        max_v=max_v,
        keep_origin_fn=keep_origin_fn,
        combine_origin_fn=combine_origin_fn,
        generator_keep_fn=generator_keep_fn,
    )
    analysis = dovv.analyze_effective_series(k_full, max_order=max_order)
    return {
        "omega": generic["omega"],
        "hbar": generic["hbar"],
        "blocks": blocks,
        "h_input": h_input,
        "k_full": k_full,
        "s_series": s_series,
        "analysis": analysis,
    }


__all__ = ["default_generator_keep", "build_generic_reference_problem"]
