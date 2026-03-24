#!/usr/bin/env python3
"""Non-collapsed symbolic probe for the three-index H12H30 block."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


def _filter_v3_three_index(v3: dict, *, one_permutation_only: bool) -> dict:
    out = {}
    seen_mode_sets: set[tuple[int, ...]] = set()
    for key, coeff in v3.items():
        vword, _jword, _origin = key
        modes = tuple(mode for _op, mode in vword)
        if len(modes) != 3 or len(set(modes)) != 3:
            continue
        mode_set = tuple(sorted(modes))
        if one_permutation_only and mode_set in seen_mode_sets:
            continue
        out[key] = coeff
        seen_mode_sets.add(mode_set)
    return dv._clean(out)


def _probe(*, max_vib_word: int, max_j_word: int, timeout_s: int, one_permutation_only: bool) -> dict[str, sp.Expr]:
    def _timeout_handler(signum, frame):
        raise TimeoutError("non-collapsed symbolic probe timed out")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    old_channel_aware = dv.CHANNEL_AWARE
    old_prune_max_v = dv.PRUNE_MAX_V
    old_prune_max_j = dv.PRUNE_MAX_J
    signal.alarm(timeout_s)
    try:
        dv.CHANNEL_AWARE = True
        dv.PRUNE_MAX_V = max_vib_word
        dv.PRUNE_MAX_J = max_j_word
        _hprime, hrv1, hrv2, v3, _v4, omega, hbar = dv.build_hprime(
            n_modes=3,
            diag_rot_only=True,
            rot_pairs={(0, 0)},
        )
        tri_v3 = _filter_v3_three_index(v3, one_permutation_only=one_permutation_only)
        h_input = dv.build_targeted_input("H12,H30", hrv1, hrv2, tri_v3, {})
        k_series, _ = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        quartic_order4 = dv.extract_quartic_rot_ground(k_series[4])
        proj = dv.commuting_projection(quartic_order4)
        tau = dv.tau_constants_from_poly(proj)
        return dv.decompose_tau_by_symbol_class(tau).get("H12,H30", {})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        dv.CHANNEL_AWARE = old_channel_aware
        dv.PRUNE_MAX_V = old_prune_max_v
        dv.PRUNE_MAX_J = old_prune_max_j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--configs", nargs="*", default=("3:2", "4:2", "3:3", "4:3"), help="Sequence max_vib:max_j to try.")
    ap.add_argument("--one-permutation-only", action="store_true", help="Keep only one representative v3 term per unordered three-mode set.")
    args = ap.parse_args()

    print("H12H30 S3 non-collapsed probe")
    print(f"timeout={args.timeout}s one_permutation_only={args.one_permutation_only}")
    for item in args.configs:
        max_vib_word, max_j_word = [int(x) for x in item.split(":", 1)]
        print(f"\ntrying max_vib={max_vib_word} max_j={max_j_word}")
        try:
            block = _probe(
                max_vib_word=max_vib_word,
                max_j_word=max_j_word,
                timeout_s=args.timeout,
                one_permutation_only=args.one_permutation_only,
            )
        except TimeoutError:
            print("  timed out")
            continue
        if not block:
            print("  no nonzero H12,H30 block recovered")
            continue
        print("  recovered components:")
        for name, expr in block.items():
            print(f"  {name} = {sp.simplify(expr)}")
            print(f"    den = {sp.factor(sp.denom(sp.together(expr)))}")
        break
    else:
        print("\nNo nonzero non-collapsed three-index block recovered.")


if __name__ == "__main__":
    main()
