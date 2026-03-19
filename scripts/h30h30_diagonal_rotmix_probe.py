#!/usr/bin/env python3
"""Probe how off-diagonal rotational couplings deform the iii-only H30H30 core."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


UNIVERSAL_DIAG = {
    "tau_xxxx": -sp.Rational(1, 1728),
    "tau_yyyy": -sp.Rational(1, 1728),
    "tau_zzzz": -sp.Rational(1, 1728),
    "tau_xxyy": -sp.Rational(1, 864),
    "tau_xxzz": -sp.Rational(1, 864),
    "tau_yyzz": -sp.Rational(1, 864),
}


def _v3_diag_only(v3):
    out = {}
    for (vword, jword, origin), coeff in v3.items():
        if len({mode for _, mode in vword}) == 1:
            out[(vword, jword, origin)] = coeff
    return out


def _collapsed_tau_vector(n_modes: int, seed: int, rot_pairs=None):
    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_V = 4
    dv.PRUNE_MAX_J = 4
    _h, hrv1, hrv2, v3, v4, omega, hbar, class_syms = dv.build_hprime_collapsed(
        n_modes=n_modes,
        seed=seed,
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        symbolic_omega=False,
    )
    h_input = dv.build_targeted_input("H30,H30", hrv1, hrv2, _v3_diag_only(v3), v4)
    k_full, _s_series = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
    quartic = dv.extract_quartic_rot_ground(k_full[4])
    tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
    exprs = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
    A, _B, C, _D = class_syms
    return {
        key: sp.simplify(exprs[key] / (A**2 * C**2 * hbar))
        for key in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    diag_pairs = {(0, 0), (1, 1), (2, 2)}
    one_offdiag = diag_pairs | {(0, 1), (1, 0)}

    print(f"n_modes={args.n_modes} seed={args.seed}")
    print("universal pure-diagonal ratios:")
    for key, val in UNIVERSAL_DIAG.items():
        print(f"  {key}: {val}")
    print()

    full = _collapsed_tau_vector(args.n_modes, args.seed, rot_pairs=one_offdiag)
    print("with one off-diagonal rotational block (ab + ba):")
    for key, val in full.items():
        print(f"  {key}: {val}   [relative to pure core: {sp.simplify(val / UNIVERSAL_DIAG[key])}]")


if __name__ == "__main__":
    main()
