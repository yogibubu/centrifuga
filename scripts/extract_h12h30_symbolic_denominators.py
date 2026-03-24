#!/usr/bin/env python3
"""Extract symbolic denominator families for the H12,H30 quartic channel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h30h30_resonance import denominator_signature, extract_denominator_families
from scripts.extract_h12h30_symbolic_structure import build_h12h30_tau


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

    exprs = build_h12h30_tau(
        n_modes=args.n_modes,
        collapsed_couplings=args.collapsed,
        diag_rot_only=args.diag_rot_only,
        symbolic_omega=args.symbolic_omega,
        seed=args.seed,
        channel_aware=args.channel_aware,
        max_vib_word=args.max_vib_word,
        max_j_word=args.max_j_word,
        rot_pairs=rot_pairs,
    )
    if not exprs:
        print("No H12,H30 symbolic expressions produced.")
        return

    print("H12,H30 symbolic denominator families")
    print(
        f"n_modes={args.n_modes}, diag_rot_only={args.diag_rot_only}, "
        f"collapsed={args.collapsed}, symbolic_omega={args.symbolic_omega}"
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
