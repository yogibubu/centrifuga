#!/usr/bin/env python3
"""First exact BCH sector that activates the octic rotational-ground block."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vanvleck import derive_watson_octic_vanvleck as dovv
import vanvleck_bch_engine as vbe
import vanvleck_hamiltonian_builders as vhb
import vanvleck_perturbative_input as vpi


def derive_first_active_octic_sector() -> dict[str, object]:
    generic = vhb.build_generic_blocks(
        n_modes=2,
        potential_orders=(),
        rotder_orders=(1,),
        diag_rot_only=True,
    )
    h12 = generic["blocks"]["H12"]
    h_input = vpi.compose_input_series({1: (h12,)})
    k_full, s_series = vbe.build_effective_to_order(
        h_input,
        omega=generic["omega"],
        hbar=generic["hbar"],
        max_order=4,
        max_j=8,
        max_v=8,
    )
    octic_by_order = {
        order: dovv.extract_rot_ground_degree(k_full[order], 8)
        for order in range(1, 5)
    }
    octic_commuting_order4 = dovv.commuting_polynomial_degree(octic_by_order[4])
    return {
        "sector": "H12_only_two_mode_diagonal",
        "h_input": h_input,
        "k_full": k_full,
        "s_series": s_series,
        "octic_by_order": octic_by_order,
        "octic_commuting_order4": octic_commuting_order4,
        "first_active_order": next((order for order in range(1, 5) if octic_by_order[order]), None),
        "octic_count_order4": len(octic_by_order[4]),
        "quartic_count_order2": len(dovv.extract_rot_ground_degree(k_full[2], 4)),
    }


if __name__ == "__main__":
    print(derive_first_active_octic_sector())
