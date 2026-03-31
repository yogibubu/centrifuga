#!/usr/bin/env python3
"""Exact octic rotational-ground block from the low-order rovibrational Hamiltonian.

This is the first fully internal use of the new generic Van Vleck stack:
starting from H12/H22/H30/H40 only, it constructs K_eff and extracts the
quartic, sextic, and octic rotational-ground blocks directly from the BCH
reduction.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vanvleck_bch_engine as vbe
import vanvleck_hamiltonian_builders as vhb
import vanvleck_perturbative_input as vpi
from tools.vanvleck import derive_watson_octic_vanvleck as dovv


def derive_loworder_octic_problem(
    *,
    n_modes: int,
    max_order: int = 4,
    max_j: int = 8,
    max_v: int = 8,
    diag_rot_only: bool = False,
    rot_pairs: set[tuple[int, int]] | None = None,
    keep_origin_fn=None,
    combine_origin_fn=None,
    generator_keep_fn=None,
) -> dict[str, object]:
    generic = vhb.build_generic_blocks(
        n_modes=n_modes,
        potential_orders=(3, 4),
        rotder_orders=(1, 2),
        diag_rot_only=diag_rot_only,
        rot_pairs=rot_pairs,
    )
    blocks = generic["blocks"]
    h_input = vpi.compose_input_series(
        {
            1: (blocks["H12"], blocks["H30"]),
            2: (blocks["H22"], blocks["H40"]),
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


__all__ = ["derive_loworder_octic_problem"]
