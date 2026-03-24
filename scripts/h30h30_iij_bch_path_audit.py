#!/usr/bin/env python3
"""Track explicit BCH generator paths for the manual H30H30 iij pivot."""

from __future__ import annotations

from collections import defaultdict
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
    ops = {"1": (1, s1), "2": (2, s2)}
    A, _B, C, _D = class_syms
    scale = A**2 * C**2 * hbar

    term_by_path = defaultdict(dict)
    for s_label, (s_ord, s_expr) in ops.items():
        expr_h1 = dv.comm_expr(s_expr, h1)
        if expr_h1:
            term_by_path[(s_label, "H1")][1 + s_ord] = expr_h1
        expr_h0 = dv.comm_with_h0(s_expr, omega, hbar)
        if expr_h0:
            term_by_path[(s_label, "H0")][s_ord] = expr_h0

    for _ in range(2):
        new = defaultdict(dict)
        for path, orders in term_by_path.items():
            for order, expr in orders.items():
                for s_label, (s_ord, s_expr) in ops.items():
                    comm = dv.comm_expr(s_expr, expr)
                    tgt = order + s_ord
                    if not comm or tgt > 4:
                        continue
                    cur = new[path + (s_label,)].get(tgt, {})
                    new[path + (s_label,)][tgt] = dv.add_expr(cur, comm)
        term_by_path = new

    print("Manual H30H30 iij BCH path audit")
    print("Nonzero path contributions to the 1/6 ad^3 sector:")
    found = False
    for path, orders in sorted(term_by_path.items()):
        expr = orders.get(4, {})
        if not expr:
            continue
        val = _tau_pair(dv.scale_expr(expr, sp.Rational(1, 6)), scale)
        if val:
            found = True
            print(path, val)
    if not found:
        print("  none at the single-path level")


if __name__ == "__main__":
    main()
