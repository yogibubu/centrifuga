#!/usr/bin/env python3
"""Probe the H12H30 three-index block with a single explicit triad family."""

from __future__ import annotations

import argparse
import itertools
import signal
import sys
from pathlib import Path

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv


def _phi3_index(i: int, j: int, k: int, *, n_modes: int) -> int:
    return i * n_modes * n_modes + j * n_modes + k


def _filter_v3_single_triad(
    v3: dict,
    *,
    triad: tuple[int, int, int],
    n_modes: int,
) -> dict:
    wanted = {
        f"phi3_{_phi3_index(i, j, k, n_modes=n_modes)}"
        for i, j, k in set(itertools.permutations(triad))
    }
    out = {}
    for key, coeff in v3.items():
        names = {sym.name for sym in coeff.free_symbols if sym.name.startswith("phi3_")}
        if names and names.issubset(wanted):
            out[key] = coeff
    return dv._clean(out)


def _probe(
    *,
    n_modes: int,
    triad: tuple[int, int, int],
    max_vib_word: int,
    max_j_word: int,
    timeout_s: int,
) -> dict[str, sp.Expr]:
    def _timeout_handler(signum, frame):
        raise TimeoutError("single-triad probe timed out")

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
            n_modes=n_modes,
            diag_rot_only=True,
            rot_pairs={(0, 0)},
        )
        tri_v3 = _filter_v3_single_triad(v3, triad=triad, n_modes=n_modes)
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
    ap.add_argument("--n-modes", type=int, default=3)
    ap.add_argument("--triad", nargs=3, type=int, default=(0, 1, 2))
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--configs", nargs="*", default=("3:2", "4:2", "3:3", "4:3"), help="Sequence max_vib:max_j to try.")
    args = ap.parse_args()

    triad = tuple(args.triad)
    print("H12H30 S3 single-triad probe")
    print(f"n_modes={args.n_modes} triad={triad} timeout={args.timeout}s")
    for item in args.configs:
        max_vib_word, max_j_word = [int(x) for x in item.split(":", 1)]
        print(f"\ntrying max_vib={max_vib_word} max_j={max_j_word}")
        try:
            block = _probe(
                n_modes=args.n_modes,
                triad=triad,
                max_vib_word=max_vib_word,
                max_j_word=max_j_word,
                timeout_s=args.timeout,
            )
        except TimeoutError:
            print("  timed out")
            continue
        if not block:
            print("  no nonzero H12,H30 block recovered")
            continue
        print("  recovered components:")
        for name, expr in block.items():
            expr = sp.simplify(expr)
            print(f"  {name} = {expr}")
            print(f"    den = {sp.factor(sp.denom(sp.together(expr)))}")
        break
    else:
        print("\nNo nonzero single-triad three-index block recovered.")


if __name__ == "__main__":
    main()
