#!/usr/bin/env python3
"""Audit the BCH-order content of the manual H30H30 iij pivot."""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv
from scripts.h30h30_sector_symbolic_probe import _filter_v3, _manual_effective_h30h30


def main() -> None:
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
    v3f = _filter_v3(v3, {"iij"})
    h1 = dv.build_targeted_input("H30,H30", hrv1, hrv2, v3f, v4)[1]
    h_series = {0: {}, 1: h1, 2: {}, 3: {}, 4: {}}
    final, s_series = _manual_effective_h30h30(h1, omega, hbar, max_order=4)

    term = {n: h_series.get(n, {}) for n in range(5)}
    has_h0 = True
    A, _B, C, _D = class_syms
    scale = A**2 * C**2 * hbar

    print("Manual H30H30 iij BCH-order audit")
    for order in range(1, 5):
        term = dv.ad_series(s_series, term, has_h0, omega, hbar, 4)
        has_h0 = False
        scaled = {m: dv.scale_expr(term.get(m, {}), sp.Rational(1, order)) for m in range(5)}
        expr4 = dv.prune_expr(dv.normal_order_expr(scaled[4]))
        quartic = dv.extract_quartic_rot_ground(expr4)
        tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
        piece = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
        if not piece:
            print(f"order {order}: empty")
            continue
        print(
            f"order {order}: "
            f"tau_xxxx={sp.simplify(piece['tau_xxxx']/scale)}, "
            f"tau_xxyy={sp.simplify(piece['tau_xxyy']/scale)}"
        )

    quartic = dv.extract_quartic_rot_ground(final[4])
    tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
    piece = dv.decompose_tau_by_symbol_class(tau)["H30,H30"]
    print(
        "final: "
        f"tau_xxxx={sp.simplify(piece['tau_xxxx']/scale)}, "
        f"tau_xxyy={sp.simplify(piece['tau_xxyy']/scale)}"
    )


if __name__ == "__main__":
    main()
