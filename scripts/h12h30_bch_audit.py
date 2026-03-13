#!/usr/bin/env python3
"""Light BCH audit for the tri-index H12H30 block.

This script answers one narrow question: in a tri-index-only V3 setup, where do
the H12,H30 terms survive inside K^(4) before the final quartic rotational
projection?
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


def _filter_v3_three_index(v3: dict) -> dict:
    out = {}
    for key, coeff in v3.items():
        vword, _jword, _origin = key
        modes = tuple(mode for _op, mode in vword)
        if len(modes) == 3 and len(set(modes)) == 3:
            out[key] = coeff
    return dv._clean(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collapsed", action="store_true", help="Use collapsed couplings.")
    ap.add_argument("--max-vib-word", type=int, default=3)
    ap.add_argument("--max-j-word", type=int, default=2)
    args = ap.parse_args()

    old_channel_aware = dv.CHANNEL_AWARE
    old_prune_max_v = dv.PRUNE_MAX_V
    old_prune_max_j = dv.PRUNE_MAX_J
    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_V = args.max_vib_word
    dv.PRUNE_MAX_J = args.max_j_word
    try:
        if args.collapsed:
            _h, hrv1, hrv2, v3, _v4, omega, hbar, _ = dv.build_hprime_collapsed(
                n_modes=3,
                diag_rot_only=True,
                symbolic_omega=True,
                seed=7,
                rot_pairs={(0, 0)},
            )
        else:
            _h, hrv1, hrv2, v3, _v4, omega, hbar = dv.build_hprime(
                n_modes=3,
                diag_rot_only=True,
                rot_pairs={(0, 0)},
            )

        tri_v3 = _filter_v3_three_index(v3)
        h_input = dv.build_targeted_input("H12,H30", hrv1, hrv2, tri_v3, {})
        k_series, _ = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        normal_k4 = dv.normal_order_expr(k_series[4])
        h1230 = {k: c for k, c in normal_k4.items() if dv.classify_term_by_symbols(c) == "H12,H30"}

        print("H12H30 BCH audit")
        print(
            f"collapsed={args.collapsed} max_vib={args.max_vib_word} max_j={args.max_j_word} "
            f"v3_total={len(v3)} tri_v3={len(tri_v3)}"
        )
        print(f"K^4 total terms = {len(k_series[4])}")
        print(f"K^4 H12,H30 terms after normal ordering = {len(h1230)}")

        buckets = Counter()
        samples: dict[tuple[int, int, tuple], list[tuple[tuple, sp.Expr]]] = defaultdict(list)
        for key, coeff in h1230.items():
            vword, jword, origin = key
            bucket = (len(vword), len(jword), dv.normalize_origin(origin))
            buckets[bucket] += 1
            if len(samples[bucket]) < 5:
                samples[bucket].append((key, sp.simplify(coeff)))

        print("\nBuckets (vib_len, rot_len, origin)")
        for bucket, n in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
            print(f"  {bucket}: {n}")
            for key, coeff in samples[bucket]:
                print(f"    key   = {key}")
                print(f"    coeff = {coeff}")

        subsets = {
            "v0_anyJ": sum(1 for key in h1230 if len(key[0]) == 0),
            "anyv_J4": sum(1 for key in h1230 if len(key[1]) == 4),
            "v0_J2": sum(1 for key in h1230 if len(key[0]) == 0 and len(key[1]) == 2),
            "v2_J2": sum(1 for key in h1230 if len(key[0]) == 2 and len(key[1]) == 2),
            "v0_J4": sum(1 for key in h1230 if len(key[0]) == 0 and len(key[1]) == 4),
        }
        print("\nKey subsets")
        for name, count in subsets.items():
            print(f"  {name} = {count}")

        quartic = dv.extract_quartic_rot_ground(k_series[4])
        print(f"\nquartic_rot_ground terms = {len(quartic)}")
    finally:
        dv.CHANNEL_AWARE = old_channel_aware
        dv.PRUNE_MAX_V = old_prune_max_v
        dv.PRUNE_MAX_J = old_prune_max_j


if __name__ == "__main__":
    main()
