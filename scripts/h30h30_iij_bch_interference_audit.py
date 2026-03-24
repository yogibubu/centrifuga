#!/usr/bin/env python3
"""Audit the S1/S2 interference structure of the manual H30H30 iij pivot."""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv
from scripts.h30h30_sector_symbolic_probe import _filter_v3


def _tau_pair(expr, scale):
    exprn = dv.prune_expr(dv.normal_order_expr(expr))
    quartic = dv.extract_quartic_rot_ground(exprn)
    tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
    piece = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
    if not piece:
        return None
    return sp.simplify(piece["tau_xxxx"] / scale), sp.simplify(piece["tau_xxyy"] / scale)


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
    partial1 = dv.bch_transform(h_series, {}, omega, hbar, max_order=1)
    _, off1 = dv.split_diag_offdiag(partial1[1], omega, hbar)
    s1 = dv.solve_s_order(off1, omega, hbar)
    partial2 = dv.bch_transform(h_series, {1: s1}, omega, hbar, max_order=2)
    _, off2 = dv.split_diag_offdiag(partial2[2], omega, hbar)
    s2 = dv.solve_s_order(off2, omega, hbar)
    A, _B, C, _D = class_syms
    scale = A**2 * C**2 * hbar

    for label, sdict in (("S2 only", {2: s2}), ("S1+S2", {1: s1, 2: s2})):
        term = {n: h_series.get(n, {}) for n in range(5)}
        has_h0 = True
        print(label)
        for power in range(1, 4):
            term = dv.ad_series(sdict, term, has_h0, omega, hbar, 4)
            has_h0 = False
            contrib = dv.scale_expr(term[4], sp.Rational(1, power))
            print(f"  power {power}: {_tau_pair(contrib, scale)}")


if __name__ == "__main__":
    main()
