#!/usr/bin/env python3
"""Probe targeted H30H30 symbolic sectors built from filtered cubic blocks."""

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


def _classify_vword(vword) -> str:
    modes = tuple(mode for _op, mode in vword)
    uniq = sorted((modes.count(i) for i in set(modes)))
    if uniq == [3]:
        return "iii"
    if uniq == [1, 2]:
        return "iij"
    if uniq == [1, 1, 1]:
        return "ijk"
    raise ValueError(f"Unexpected multiplicity pattern for {vword}")


def _filter_v3(v3: dict, allowed: set[str]) -> dict:
    out = {}
    for key, coeff in v3.items():
        vword, _jword, _origin = key
        if _classify_vword(vword) in allowed:
            out[key] = coeff
    return dv._clean(out)


def _manual_effective_h30h30(h1: dict, omega, hbar, max_order: int = 4):
    """Build the H30,H30 effective series by explicit recursive S construction.

    This is algebraically equivalent to ``build_effective_to_order`` for the
    present targeted input ``{1: H1, 2: 0, 3: 0, 4: 0}``, but exposes the
    sector-by-sector recursion directly:

    ``S1 -> K2 -> S2 -> K3 -> S3 -> K4``.
    """
    h_series = {0: {}, 1: h1}
    for order in range(2, max_order + 1):
        h_series[order] = {}

    partial1 = dv.bch_transform(h_series, {}, omega, hbar, max_order=1)
    _, off1 = dv.split_diag_offdiag(partial1[1], omega, hbar)
    s_series = {1: dv.solve_s_order(off1, omega, hbar)}

    if max_order >= 2:
        partial2 = dv.bch_transform(h_series, s_series, omega, hbar, max_order=2)
        _, off2 = dv.split_diag_offdiag(partial2[2], omega, hbar)
        s_series[2] = dv.solve_s_order(off2, omega, hbar)
    if max_order >= 3:
        partial3 = dv.bch_transform(h_series, s_series, omega, hbar, max_order=3)
        _, off3 = dv.split_diag_offdiag(partial3[3], omega, hbar)
        s_series[3] = dv.solve_s_order(off3, omega, hbar)
    if max_order >= 4:
        partial4 = dv.bch_transform(h_series, s_series, omega, hbar, max_order=4)
        return partial4, s_series
    return dv.bch_transform(h_series, s_series, omega, hbar, max_order=max_order), s_series


def _collapsed_sector_vector(
    n_modes: int,
    seed: int,
    *,
    allowed_classes: set[str],
    diag_rot_only: bool = False,
    rot_pairs=None,
    manual: bool = False,
) -> np.ndarray:
    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_V = 4
    dv.PRUNE_MAX_J = 4
    _h, hrv1, hrv2, v3, v4, omega, hbar, class_syms = dv.build_hprime_collapsed(
        n_modes=n_modes,
        seed=seed,
        diag_rot_only=diag_rot_only,
        rot_pairs=rot_pairs,
        symbolic_omega=False,
    )
    v3f = _filter_v3(v3, allowed_classes)
    h_input = dv.build_targeted_input("H30,H30", hrv1, hrv2, v3f, v4)
    if manual:
        k_full, _ = _manual_effective_h30h30(h_input[1], omega=omega, hbar=hbar, max_order=4)
    else:
        k_full, _ = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
    quartic = dv.extract_quartic_rot_ground(k_full[4])
    tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
    exprs = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
    A, _B, C, _D = class_syms
    return np.array([float(sp.N(sp.simplify(exprs.get(key, 0) / (A**2 * C**2 * hbar)))) for key in MON_KEYS], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", nargs="+", type=int, default=(2,))
    ap.add_argument("--seeds", nargs="+", type=int, default=(7, 11, 13))
    ap.add_argument(
        "--rot-mode",
        choices=("diag", "diag-plus-ab", "full"),
        default="diag-plus-ab",
        help="Restrict the rotational couplings to keep the probe tractable.",
    )
    ap.add_argument(
        "--manual",
        action="store_true",
        help="Use explicit S1->K2->S2->K3->S3->K4 recursion instead of build_effective_to_order.",
    )
    args = ap.parse_args()

    rot_pairs = None
    if args.rot_mode == "diag":
        rot_pairs = {(0, 0), (1, 1), (2, 2)}
    elif args.rot_mode == "diag-plus-ab":
        rot_pairs = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}

    sectors = {
        "iii_only": {"iii"},
        "iii_iij": {"iii", "iij"},
        "iij_only": {"iij"},
        "iii_iij_minus_iii": None,
    }
    for n_modes in args.n_modes:
        print(f"\nn_modes={n_modes}")
        sector_rows = {name: [] for name in sectors}
        for seed in args.seeds:
            iii = _collapsed_sector_vector(
                n_modes,
                seed,
                allowed_classes={"iii"},
                diag_rot_only=False,
                rot_pairs=rot_pairs,
                manual=args.manual,
            )
            iii_iij = _collapsed_sector_vector(
                n_modes,
                seed,
                allowed_classes={"iii", "iij"},
                diag_rot_only=False,
                rot_pairs=rot_pairs,
                manual=args.manual,
            )
            iij = _collapsed_sector_vector(
                n_modes,
                seed,
                allowed_classes={"iij"},
                diag_rot_only=False,
                rot_pairs=rot_pairs,
                manual=args.manual,
            )
            rows = {
                "iii_only": iii,
                "iii_iij": iii_iij,
                "iij_only": iij,
                "iii_iij_minus_iii": iii_iij - iii,
            }
            print(f"  seed={seed}")
            for name, row in rows.items():
                sector_rows[name].append(row)
                print(f"    {name:>18}: {row}")
        print("  sample-span ranks:")
        for name, rows in sector_rows.items():
            mat = np.array(rows, dtype=float)
            print(f"    {name:>18}: {np.linalg.matrix_rank(mat)}")


if __name__ == "__main__":
    main()
