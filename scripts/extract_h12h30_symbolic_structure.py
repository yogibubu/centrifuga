#!/usr/bin/env python3
"""Extract symbolic H12H30 structure from the BCH engine.

This is a diagnostic helper used to recover the lost mixed-channel formula from
the symbolic Van Vleck reference. It mirrors the H30H30 extraction utility but
targets the ``H12,H30`` symbolic class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


def build_h12h30_tau(
    *,
    n_modes: int,
    collapsed_couplings: bool,
    diag_rot_only: bool,
    symbolic_omega: bool,
    seed: int,
    channel_aware: bool = False,
    max_vib_word: int = 4,
    max_j_word: int = 4,
    rot_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, sp.Expr]:
    old_channel_aware = dv.CHANNEL_AWARE
    old_prune_max_v = dv.PRUNE_MAX_V
    old_prune_max_j = dv.PRUNE_MAX_J
    dv.CHANNEL_AWARE = channel_aware
    dv.PRUNE_MAX_V = max_vib_word
    dv.PRUNE_MAX_J = max_j_word

    if collapsed_couplings:
        _, hrv1, hrv2, v3, v4, omega, hbar, _ = dv.build_hprime_collapsed(
            n_modes=n_modes,
            seed=seed,
            diag_rot_only=diag_rot_only,
            symbolic_omega=symbolic_omega,
            rot_pairs=rot_pairs,
        )
    else:
        _, hrv1, hrv2, v3, v4, omega, hbar = dv.build_hprime(
            n_modes=n_modes,
            diag_rot_only=diag_rot_only,
            rot_pairs=rot_pairs,
        )
    try:
        h_input = dv.build_targeted_input("H12,H30", hrv1, hrv2, v3, v4)
        k_series, _ = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        quartic_order4 = dv.extract_quartic_rot_ground(k_series[4])
        proj = dv.commuting_projection(quartic_order4)
        tau = dv.tau_constants_from_poly(proj)
        return dv.decompose_tau_by_symbol_class(tau).get("H12,H30", {})
    finally:
        dv.CHANNEL_AWARE = old_channel_aware
        dv.PRUNE_MAX_V = old_prune_max_v
        dv.PRUNE_MAX_J = old_prune_max_j


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract symbolic H12H30 structure from the BCH engine.")
    ap.add_argument("--n-modes", type=int, default=1)
    ap.add_argument("--collapsed-couplings", action="store_true")
    ap.add_argument("--diag-rot-only", action="store_true")
    ap.add_argument("--symbolic-omega", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--channel-aware", action="store_true")
    ap.add_argument("--max-vib-word", type=int, default=4)
    ap.add_argument("--max-j-word", type=int, default=4)
    ap.add_argument(
        "--rot-pairs",
        type=str,
        default="",
        help="Comma-separated rotational index pairs among xx,xy,xz,yx,yy,yz,zx,zy,zz.",
    )
    args = ap.parse_args()

    rot_pairs = None
    if args.rot_pairs.strip():
        code = {"x": 0, "y": 1, "z": 2}
        rot_pairs = set()
        for tok in args.rot_pairs.split(","):
            t = tok.strip().lower()
            if len(t) != 2 or t[0] not in code or t[1] not in code:
                raise ValueError(f"Invalid rot pair token: {tok}")
            rot_pairs.add((code[t[0]], code[t[1]]))

    block = build_h12h30_tau(
        n_modes=args.n_modes,
        collapsed_couplings=args.collapsed_couplings,
        diag_rot_only=args.diag_rot_only,
        symbolic_omega=args.symbolic_omega,
        seed=args.seed,
        channel_aware=args.channel_aware,
        max_vib_word=args.max_vib_word,
        max_j_word=args.max_j_word,
        rot_pairs=rot_pairs,
    )
    for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
        print(f"{name} = {sp.simplify(block.get(name, sp.Integer(0)))}")


if __name__ == "__main__":
    main()
