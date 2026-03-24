#!/usr/bin/env python3
"""Progressively probe the symbolic three-index block of H12H30."""

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


def _filter_v3_three_index(v3: dict) -> dict:
    out = {}
    for key, coeff in v3.items():
        vword, _jword, _origin = key
        modes = tuple(mode for _op, mode in vword)
        if len(modes) == 3 and len(set(modes)) == 3:
            out[key] = coeff
    return dv._clean(out)


def _run_probe(*, n_modes: int, max_vib_word: int, max_j_word: int, timeout_s: int):
    def _timeout_handler(signum, frame):
        raise TimeoutError("symbolic probe timed out")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    old_channel_aware = dv.CHANNEL_AWARE
    old_prune_max_v = dv.PRUNE_MAX_V
    old_prune_max_j = dv.PRUNE_MAX_J
    try:
        dv.CHANNEL_AWARE = True
        dv.PRUNE_MAX_V = max_vib_word
        dv.PRUNE_MAX_J = max_j_word
        _hprime, hrv1, hrv2, v3, _v4, omega, hbar, _ = dv.build_hprime_collapsed(
            n_modes=n_modes,
            diag_rot_only=True,
            symbolic_omega=True,
            seed=7,
            rot_pairs={(0, 0)},
        )
        tri_v3 = _filter_v3_three_index(v3)
        h_input = dv.build_targeted_input("H12,H30", hrv1, hrv2, tri_v3, {})
        k_series, _ = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        quartic_order4 = dv.extract_quartic_rot_ground(k_series[4])
        proj = dv.commuting_projection(quartic_order4)
        tau = dv.tau_constants_from_poly(proj)
        exprs = dv.decompose_tau_by_symbol_class(tau).get("H12,H30", {})
        out = {}
        for name, expr in exprs.items():
            if expr != 0:
                out[name] = sp.simplify(expr)
        return out
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        dv.CHANNEL_AWARE = old_channel_aware
        dv.PRUNE_MAX_V = old_prune_max_v
        dv.PRUNE_MAX_J = old_prune_max_j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument(
        "--configs",
        nargs="*",
        default=("3:2", "4:2", "3:3", "4:3", "4:4"),
        help="Sequence max_vib:max_j to try.",
    )
    args = ap.parse_args()

    print("H12H30 S3 symbolic probe")
    print(f"n_modes={args.n_modes} timeout={args.timeout}s")
    for item in args.configs:
        max_vib_word, max_j_word = [int(x) for x in item.split(":", 1)]
        print(f"\ntrying max_vib={max_vib_word} max_j={max_j_word}")
        try:
            tri_block = _run_probe(
                n_modes=args.n_modes,
                max_vib_word=max_vib_word,
                max_j_word=max_j_word,
                timeout_s=args.timeout,
            )
        except TimeoutError:
            print("  timed out")
            continue

        if not tri_block:
            print("  no three-index block recovered")
            continue

        print("  recovered components:")
        for name, expr in tri_block.items():
            print(f"  {name} = {expr}")
            print(f"    den = {sp.factor(sp.denom(sp.together(expr)))}")
        break
    else:
        print("\nNo nonzero three-index symbolic block recovered.")


if __name__ == "__main__":
    main()
