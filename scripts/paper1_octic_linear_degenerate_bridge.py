#!/usr/bin/env python3
"""Bridge the derived degenerate linear octic block to the observable pairwise X_l branch."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.linear_s111_builder import build_linear_s111_builder_report  # noqa: E402


def paper1_octic_linear_degenerate_bridge(*, species: str, gaussian_dir: str | Path) -> dict[str, object]:
    report = build_linear_s111_builder_report(species=species, gaussian_dir=Path(gaussian_dir))
    return {
        "species": species,
        "derived_degenerate_block_symbol": "L_perp^deg = (18/35) S_perp^2 tau_perp",
        "observable_exact_linear_replacement": "(J^2)^2 X_l pairwise branch",
        "pairwise_qH_values_hz": tuple(
            {
                "pair_label": pair["pair_label"],
                "modes": pair["modes"],
                "q_H_hz": pair["q_H_hz"],
            }
            for pair in report.pairs
        ),
        "statement": (
            "In the exact linear regime the derived degenerate octic block does not "
            "feed the scalar pure-rotational constant L. Its observable replacement "
            "is the pairwise degenerate carrier X_l, and the octic coefficient of "
            "that branch is q_H for each perpendicular doublet."
        ),
    }


if __name__ == "__main__":
    print(
        paper1_octic_linear_degenerate_bridge(
            species="c2h2",
            gaussian_dir="/Users/vincenzobarone/centrifugal/gaussian",
        )
    )
