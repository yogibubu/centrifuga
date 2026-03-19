#!/usr/bin/env python3
"""Probe iii-only H30H30 samples under diagonal vs full rotational couplings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


MON_KEYS = ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")


def _v3_diag_only(v3):
    out = {}
    for (vword, jword, origin), coeff in v3.items():
        if len({mode for _, mode in vword}) == 1:
            out[(vword, jword, origin)] = coeff
    return out


def _collapsed_diag_exprs(
    n_modes: int,
    seed: int,
    *,
    diag_rot_only: bool,
) -> dict[str, sp.Expr]:
    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_V = 4
    dv.PRUNE_MAX_J = 4
    _h, hrv1, hrv2, v3, v4, omega, hbar, class_syms = dv.build_hprime_collapsed(
        n_modes=n_modes,
        seed=seed,
        diag_rot_only=diag_rot_only,
        symbolic_omega=False,
    )
    h_input = dv.build_targeted_input("H30,H30", hrv1, hrv2, _v3_diag_only(v3), v4)
    k_full, _s_series = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
    quartic = dv.extract_quartic_rot_ground(k_full[4])
    tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
    exprs = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
    A, _B, C, _D = class_syms
    return {
        key: sp.simplify(exprs.get(key, sp.Integer(0)) / (A**2 * C**2 * hbar))
        for key in MON_KEYS
    }


def _collapsed_diag_vector(
    n_modes: int,
    seed: int,
    *,
    diag_rot_only: bool,
) -> np.ndarray:
    exprs = _collapsed_diag_exprs(n_modes, seed, diag_rot_only=diag_rot_only)
    return np.array([float(sp.N(exprs[key])) for key in MON_KEYS], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", nargs="+", type=int, default=(1, 2))
    ap.add_argument("--seeds", nargs="+", type=int, default=(7, 11, 13))
    args = ap.parse_args()

    rows_diag = []
    rows_full = []
    labels = []
    for n_modes in args.n_modes:
        for seed in args.seeds:
            labels.append((n_modes, seed))
            rows_diag.append(_collapsed_diag_vector(n_modes, seed, diag_rot_only=True))
            rows_full.append(_collapsed_diag_vector(n_modes, seed, diag_rot_only=False))

    diag_mat = np.array(rows_diag, dtype=float)
    full_mat = np.array(rows_full, dtype=float)
    print("iii-only H30H30 sample probe")
    print(f"cases = {labels}")
    print(f"sample-span rank (diag-only rotational) = {np.linalg.matrix_rank(diag_mat)}")
    print(f"sample-span rank (full rotational)      = {np.linalg.matrix_rank(full_mat)}")
    print("NOTE: these ranks reflect the sampled collapsed benchmark vectors, not the intrinsic scaffold rank.")
    print()
    print("diag-only rotational vectors")
    for label, row in zip(labels, diag_mat):
        print(label, row)
    print()
    print("full rotational vectors")
    for label, row in zip(labels, full_mat):
        print(label, row)


if __name__ == "__main__":
    main()
