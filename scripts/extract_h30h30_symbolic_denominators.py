#!/usr/bin/env python3
"""Extract symbolic denominator families for the H30,H30 quartic channel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv
from h30h30_resonance import denominator_signature, extract_denominator_families


def build_tau_h30h30(
    n_modes: int,
    *,
    diag_rot_only: bool,
    collapsed: bool,
    seed: int,
    symbolic_omega: bool,
    channel_aware: bool = False,
    max_vib_word: int = 4,
    max_j_word: int = 4,
):
    old_channel_aware = dv.CHANNEL_AWARE
    old_prune_max_v = dv.PRUNE_MAX_V
    old_prune_max_j = dv.PRUNE_MAX_J
    dv.CHANNEL_AWARE = channel_aware
    dv.PRUNE_MAX_V = max_vib_word
    dv.PRUNE_MAX_J = max_j_word
    try:
        if collapsed:
            _hprime, hrv1, hrv2, v3, v4, omega, hbar, _class_syms = dv.build_hprime_collapsed(
                n_modes=n_modes,
                seed=seed,
                diag_rot_only=diag_rot_only,
                symbolic_omega=symbolic_omega,
            )
        else:
            _hprime, hrv1, hrv2, v3, v4, omega, hbar = dv.build_hprime(
                n_modes=n_modes,
                diag_rot_only=diag_rot_only,
            )
        h_input = dv.build_targeted_input("H30,H30", hrv1, hrv2, v3, v4)
        k_series, _s_series = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        quartic = dv.extract_quartic_rot_ground(k_series[4])
        proj = dv.commuting_projection(quartic)
        tau = dv.tau_constants_from_poly(proj)
        return dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
    finally:
        dv.CHANNEL_AWARE = old_channel_aware
        dv.PRUNE_MAX_V = old_prune_max_v
        dv.PRUNE_MAX_J = old_prune_max_j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", type=int, default=1, help="Symbolic mode count")
    ap.add_argument("--diag-rot-only", action="store_true", help="Restrict to diagonal rotational couplings")
    ap.add_argument("--collapsed", action="store_true", help="Use collapsed couplings for faster but less informative probing")
    ap.add_argument("--seed", type=int, default=7, help="Collapsed-coupling seed")
    ap.add_argument("--symbolic-omega", action="store_true", help="Keep omega_i symbolic in collapsed mode")
    ap.add_argument("--channel-aware", action="store_true")
    ap.add_argument("--max-vib-word", type=int, default=4)
    ap.add_argument("--max-j-word", type=int, default=4)
    args = ap.parse_args()

    exprs = build_tau_h30h30(
        args.n_modes,
        diag_rot_only=args.diag_rot_only,
        collapsed=args.collapsed,
        seed=args.seed,
        symbolic_omega=args.symbolic_omega,
        channel_aware=args.channel_aware,
        max_vib_word=args.max_vib_word,
        max_j_word=args.max_j_word,
    )
    if not exprs:
        print("No H30,H30 symbolic expressions produced.")
        return

    print("H30,H30 symbolic denominator families")
    print(
        f"n_modes={args.n_modes}, diag_rot_only={args.diag_rot_only}, "
        f"collapsed={args.collapsed}, symbolic_omega={args.symbolic_omega}, "
        f"channel_aware={args.channel_aware}, max_vib={args.max_vib_word}, max_j={args.max_j_word}"
    )
    print()
    families = extract_denominator_families(exprs)
    for sig, names in families.items():
        print(f"DEN = {sig}")
        print("components =", ", ".join(names))
        print()

    print("Component detail")
    for name, expr in exprs.items():
        print(f"{name} = {sp.simplify(expr)}")
        print(f"DEN  = {denominator_signature(expr)}")
        print()


if __name__ == "__main__":
    main()
