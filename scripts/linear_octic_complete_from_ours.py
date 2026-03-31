#!/usr/bin/env python3
"""Complete linear octic wrapper from our equations only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.linear_octic_pairwise_from_ours import (  # noqa: E402
    LinearOcticPairwiseFromOurs,
    build_linear_octic_pairwise_from_ours,
)
from scripts.linear_octic_pure_from_ours import (  # noqa: E402
    LinearOcticPureFromOursReport,
    build_linear_octic_pure_from_ours,
)


@dataclass(frozen=True)
class LinearOcticCompleteFromOurs:
    species: str
    pure: LinearOcticPureFromOursReport
    degenerate: LinearOcticPairwiseFromOurs


def build_linear_octic_complete_from_ours(*, species: str, gaussian_dir: Path) -> LinearOcticCompleteFromOurs:
    pure = build_linear_octic_pure_from_ours(species=species, gaussian_dir=gaussian_dir)
    degenerate = build_linear_octic_pairwise_from_ours(species=species, gaussian_dir=gaussian_dir)
    return LinearOcticCompleteFromOurs(
        species=species,
        pure=pure,
        degenerate=degenerate,
    )


def format_linear_octic_complete_report(*, species: str, gaussian_dir: Path) -> str:
    report = build_linear_octic_complete_from_ours(species=species, gaussian_dir=gaussian_dir)
    lines = [
        f"=== {report.species} ===",
        "pure rotational scalar branch:",
        f"parallel modes (0-based) = {list(report.pure.parallel_mode_indices_0based)}",
        f"B_perp = {report.pure.B_perp_cm:+.12e} cm^-1",
        f"L4_parallel = {report.pure.quartic_block_cm:+.12e} cm^-1",
        f"L3_parallel = {report.pure.term3_block_cm:+.12e} cm^-1",
        f"L5_parallel = {report.pure.term5_block_cm:+.12e} cm^-1",
        f"L_parallel^(ours) = {report.pure.total_cm:+.12e} cm^-1",
        "degenerate pairwise branch:",
        f"tau_perp = {report.degenerate.tau_perp_cm:+.12e} cm^-1",
    ]
    for pair in report.degenerate.pairs:
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
        print(format_linear_octic_complete_report(species=species, gaussian_dir=Path(args.gaussian_dir)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
