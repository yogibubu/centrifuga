#!/usr/bin/env python3
"""Manual BCH breakdown of the H30H30 iii/iij sectors.

This script uses the explicit recursion implemented in
``scripts/h30h30_sector_symbolic_probe.py`` to compare the three restricted
inputs

- ``iii``
- ``iij``
- ``iii+iij``

and extracts the interference piece

- ``(iii+iij) - iii - iij``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv
from scripts.h30h30_sector_symbolic_probe import MON_KEYS, _filter_v3, _manual_effective_h30h30


def _manual_sector_exprs(allowed: set[str]):
    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_J = 4
    dv.PRUNE_MAX_V = 4
    _h, hrv1, hrv2, v3, v4, omega, hbar, class_syms = dv.build_hprime_collapsed(
        n_modes=2,
        seed=7,
        diag_rot_only=False,
        rot_pairs={(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)},
        symbolic_omega=False,
    )
    v3f = _filter_v3(v3, allowed)
    h1 = dv.build_targeted_input("H30,H30", hrv1, hrv2, v3f, v4)[1]
    k_full, _s = _manual_effective_h30h30(h1, omega=omega, hbar=hbar, max_order=4)
    quartic = dv.extract_quartic_rot_ground(k_full[4])
    tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
    return dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {}), class_syms, hbar


def main() -> None:
    iii, class_syms, hbar = _manual_sector_exprs({"iii"})
    iij, _, _ = _manual_sector_exprs({"iij"})
    both, _, _ = _manual_sector_exprs({"iii", "iij"})
    A, _B, C, _D = class_syms
    scale = A**2 * C**2 * hbar

    print("Manual H30H30 iii/iij breakdown (n_modes=2, seed=7, rot=diag-plus-ab)")
    for key in MON_KEYS:
        iii_n = sp.simplify(iii[key] / scale)
        iij_n = sp.simplify(iij[key] / scale)
        both_n = sp.simplify(both[key] / scale)
        inter_n = sp.simplify((both[key] - iii[key] - iij[key]) / scale)
        print(f"\n{key}:")
        print(f"  iii          = {iii_n}")
        print(f"  iij          = {iij_n}")
        print(f"  iii+iij      = {both_n}")
        print(f"  interference = {inter_n}")


if __name__ == "__main__":
    main()
