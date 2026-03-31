#!/usr/bin/env python3
"""Combined exact-linear report: scalar pure-rotational branch plus pairwise octic branch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.our_linear_octic_gaussian_report import build_our_linear_octic_pure_report  # noqa: E402
from scripts.paper1_octic_linear_degenerate_bridge import paper1_octic_linear_degenerate_bridge  # noqa: E402


def build_combined_report(*, species: str, gaussian_dir: str | Path) -> str:
    gdir = Path(gaussian_dir)
    scalar = build_our_linear_octic_pure_report(species=species, gaussian_dir=gdir)
    deg = paper1_octic_linear_degenerate_bridge(species=species, gaussian_dir=gdir)
    lines = [
        f"=== {species} ===",
        "exact-linear octic observables from the current CeDiTT4 program:",
        f"  L_parallel_pure = {float(scalar.total):+.12e} cm^-1",
        f"  L_Aliev         = {float(scalar.aliev_total):+.12e} cm^-1",
        f"  Delta_scalar    = {float(scalar.difference_vs_aliev):+.12e} cm^-1",
        "  perpendicular-degenerate octic observables:",
    ]
    for pair in deg["pairwise_qH_values_hz"]:
        lines.append(
            f"    {pair['pair_label']} modes={list(pair['modes'])}: q_H = {float(pair['q_H_hz']):+.12e} Hz"
        )
    lines.append(
        "  interpretation: the exact linear octic sector splits into one scalar pure-rotational observable and one pairwise degenerate observable per perpendicular doublet."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hcn", "hccd", "ocs"])
    args = ap.parse_args()
    for species in args.species:
        print(build_combined_report(species=species, gaussian_dir=args.gaussian_dir))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
