#!/usr/bin/env python3
"""Probe the pure diagonal iii-only symbolic core of H30H30."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


def _build_diag_only_collapsed(n_modes: int, seed: int):
    rng = random.Random(seed)
    hbar = sp.symbols("hbar", positive=True)
    omega = tuple(sp.Rational(i + 2, 1) for i in range(n_modes))
    A, C = sp.symbols("A C", real=True)

    def rr() -> sp.Rational:
        return sp.Rational(rng.randint(1, 9), rng.randint(1, 5))

    mu1 = sp.MutableDenseNDimArray([A * rr() for _ in range(3 * 3 * n_modes)], (3, 3, n_modes))
    phi3 = sp.MutableDenseNDimArray([sp.Integer(0) for _ in range(n_modes * n_modes * n_modes)], (n_modes, n_modes, n_modes))
    for i in range(n_modes):
        phi3[i, i, i] = C * rr()

    hrv1 = {}
    for a in range(3):
        for k in range(n_modes):
            c = sp.Rational(1, 2) * mu1[a, a, k] * sp.sqrt(hbar / (2 * omega[k]))
            jword = (dv.JOPS[a], dv.JOPS[a])
            hrv1 = dv.add_expr(
                hrv1,
                dv.one_term(c, (("a", k),), jword, ("H12",)),
                dv.one_term(c, (("ad", k),), jword, ("H12",)),
            )

    v3 = {}
    for i in range(n_modes):
        c = sp.Rational(1, 6) * phi3[i, i, i] * (sp.sqrt(hbar / (2 * omega[i])) ** 3)
        for op_i in ("a", "ad"):
            for op_j in ("a", "ad"):
                for op_k in ("a", "ad"):
                    v3 = dv.add_expr(v3, dv.one_term(c, ((op_i, i), (op_j, i), (op_k, i)), (), ("H30",)))
    return hrv1, v3, omega, hbar, mu1, phi3


def _diag_scaffold_sum(mu1, phi3, omega, component: str) -> sp.Expr:
    if component == "tau_xxxx":
        a, b = 0, 0
    elif component == "tau_yyyy":
        a, b = 1, 1
    elif component == "tau_zzzz":
        a, b = 2, 2
    elif component == "tau_xxyy":
        a, b = 0, 1
    elif component == "tau_xxzz":
        a, b = 0, 2
    elif component == "tau_yyzz":
        a, b = 1, 2
    else:
        raise ValueError(component)

    total = sp.Integer(0)
    for i in range(mu1.shape[2]):
        total += phi3[i, i, i] ** 2 * mu1[a, a, i] * mu1[b, b, i] / omega[i] ** 7
    return sp.simplify(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", nargs="+", type=int, default=(1, 2))
    ap.add_argument("--seeds", nargs="+", type=int, default=(7, 11, 13))
    args = ap.parse_args()

    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_V = 4
    dv.PRUNE_MAX_J = 4

    for n_modes in args.n_modes:
        for seed in args.seeds:
            hrv1, v3, omega, hbar, mu1, phi3 = _build_diag_only_collapsed(n_modes, seed)
            h_input = dv.build_targeted_input("H30,H30", hrv1, {}, v3, {})
            k_full, _s_series = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
            quartic = dv.extract_quartic_rot_ground(k_full[4])
            tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
            exprs = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
            print(f"n_modes={n_modes} seed={seed}")
            for component in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
                ratio = sp.simplify(exprs[component] / (_diag_scaffold_sum(mu1, phi3, omega, component) * hbar))
                print(f"  {component}: {ratio}")
            print()


if __name__ == "__main__":
    main()
