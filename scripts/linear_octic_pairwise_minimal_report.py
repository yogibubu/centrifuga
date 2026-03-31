#!/usr/bin/env python3
"""Minimal exact report for the linear degenerate octic branch from our equations."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.linear_octic_pairwise_from_ours import build_linear_octic_pairwise_from_ours  # noqa: E402


def format_linear_octic_pairwise_minimal_report(*, species: str, gaussian_dir: Path) -> str:
    report = build_linear_octic_pairwise_from_ours(species=species, gaussian_dir=gaussian_dir)
    lines = [
        f"=== {report.species} ===",
        f"tau_perp = {report.tau_perp_cm:+.12e} cm^-1",
    ]
    for pair in report.pairs:
        lines.extend(
            [
                f"pair = {list(pair['pair'])}",
                f"Sigma_t = {pair['sigma_t']:+.12e}",
                f"q_H^(ours) = {pair['q_H_hz_from_ours']:+.12e} Hz",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hcn"])
    args = ap.parse_args()

    for species in args.species:
        print(format_linear_octic_pairwise_minimal_report(species=species, gaussian_dir=Path(args.gaussian_dir)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
