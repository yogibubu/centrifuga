#!/usr/bin/env python3
"""Sweep pruning/rotation settings for the minimal H30H30 sector probe."""

from __future__ import annotations

import argparse
import math
import random
import signal
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import derive_watson_quartic_vanvleck as dv
import sympy as sp


def _timeout_handler(signum, frame):
    raise TimeoutError("probe timed out")


def _build_minimal_input(*, include_iij: bool, rot_mode: str):
    rng = random.Random(7)
    hbar = sp.symbols("hbar", positive=True)
    omega = (sp.Integer(2), sp.Integer(3))
    A, C = sp.symbols("A C", real=True)

    def rr():
        return sp.Float(rng.uniform(0.5, 2.0))

    allowed_pairs = {(0, 0), (1, 1), (2, 2)}
    if rot_mode == "diag-plus-ab":
        allowed_pairs |= {(0, 1), (1, 0)}
    elif rot_mode == "full":
        allowed_pairs = {(a, b) for a in range(3) for b in range(3)}

    mu1 = {}
    for a, b in allowed_pairs:
        for k in range(2):
            val = rr()
            c = sp.Rational(1, 2) * A * val * sp.sqrt(hbar / (2 * omega[k]))
            jword = (dv.JOPS[a], dv.JOPS[b])
            mu1 = dv.add_expr(
                mu1,
                dv.one_term(c, (("a", k),), jword, ("H12",)),
                dv.one_term(c, (("ad", k),), jword, ("H12",)),
            )

    v3 = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                counts = sorted((tuple((i, j, k)).count(x) for x in set((i, j, k))))
                family = "iii" if counts == [3] else ("iij" if counts == [1, 2] else "ijk")
                if family == "ijk" or (family == "iij" and not include_iij):
                    continue
                c = (sp.Rational(1, 6) * C * rr() * sp.sqrt(hbar / (2 * omega[i])) * sp.sqrt(hbar / (2 * omega[j])) * sp.sqrt(hbar / (2 * omega[k])))
                for op_i in ("a", "ad"):
                    for op_j in ("a", "ad"):
                        for op_k in ("a", "ad"):
                            v3 = dv.add_expr(v3, dv.one_term(c, ((op_i, i), (op_j, j), (op_k, k)), (), ("H30",)))
    return mu1, v3, omega, hbar


def _run_once(*, include_iij: bool, rot_mode: str, max_v: int, max_j: int, timeout_s: int) -> bool:
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    old_channel_aware = dv.CHANNEL_AWARE
    old_prune_v = dv.PRUNE_MAX_V
    old_prune_j = dv.PRUNE_MAX_J
    signal.alarm(timeout_s)
    try:
        dv.CHANNEL_AWARE = True
        dv.PRUNE_MAX_V = max_v
        dv.PRUNE_MAX_J = max_j
        hrv1, v3, omega, hbar = _build_minimal_input(include_iij=include_iij, rot_mode=rot_mode)
        h_input = dv.build_targeted_input("H30,H30", hrv1, {}, v3, {})
        k_series, _ = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        quartic = dv.extract_quartic_rot_ground(k_series[4])
        tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
        exprs = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
        return any(expr != 0 for expr in exprs.values())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        dv.CHANNEL_AWARE = old_channel_aware
        dv.PRUNE_MAX_V = old_prune_v
        dv.PRUNE_MAX_J = old_prune_j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    print("H30H30 sector timeout sweep")
    for include_iij in (False, True):
        label = "iii+iij" if include_iij else "iii-only"
        for rot_mode in ("diag", "diag-plus-ab", "full"):
            print(f"\nsector={label} rot_mode={rot_mode}")
            for max_v in (2, 3, 4):
                for max_j in (2, 3, 4):
                    try:
                        ok = _run_once(
                            include_iij=include_iij,
                            rot_mode=rot_mode,
                            max_v=max_v,
                            max_j=max_j,
                            timeout_s=args.timeout,
                        )
                        print(f"  max_v={max_v} max_j={max_j} -> {'ok' if ok else 'zero'}")
                    except TimeoutError:
                        print(f"  max_v={max_v} max_j={max_j} -> timeout")


if __name__ == "__main__":
    main()
