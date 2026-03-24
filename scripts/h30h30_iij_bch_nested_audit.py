#!/usr/bin/env python3
"""Audit simple nested-commutator candidates for the H30H30 iij prefactor."""

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

    print("Manual H30H30 iij nested-commutator audit")
    print("order-2 target =", (-sp.Rational(624163585, 449474199552), -sp.Rational(929643954971, 288947699712000)))
    print("1/4 [S2,[S1,H1]] =", _tau_pair(dv.scale_expr(dv.comm_expr(s2, dv.comm_expr(s1, h1)), sp.Rational(1, 4)), scale))
    print("1/2 [S2,[S1,H1]] =", _tau_pair(dv.scale_expr(dv.comm_expr(s2, dv.comm_expr(s1, h1)), sp.Rational(1, 2)), scale))
    h0_branch = dv.comm_with_h0(s1, omega, hbar)
    print("order-3 target =", (sp.Rational(624163585, 674211299328), sp.Rational(929643954971, 433421549568000)))
    print(
        "1/6 [S2,[S1,[S1,H0]]] =",
        _tau_pair(dv.scale_expr(dv.comm_expr(s2, dv.comm_expr(s1, h0_branch)), sp.Rational(1, 6)), scale),
    )

    for name, expr in {
        "121": dv.comm_expr(s1, dv.comm_expr(s2, dv.comm_expr(s1, h1))),
        "211": dv.comm_expr(s2, dv.comm_expr(s1, dv.comm_expr(s1, h1))),
        "122": dv.comm_expr(s1, dv.comm_expr(s2, dv.comm_expr(s2, h1))),
        "212": dv.comm_expr(s2, dv.comm_expr(s1, dv.comm_expr(s2, h1))),
        "221": dv.comm_expr(s2, dv.comm_expr(s2, dv.comm_expr(s1, h1))),
    }.items():
        print(f"{name} =", _tau_pair(expr, scale))


if __name__ == "__main__":
    main()
