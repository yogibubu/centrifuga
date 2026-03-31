#!/usr/bin/env python3
"""Exact linear-limit builder for the degenerate perpendicular branch.

This does not fabricate a unique scalar ``S111`` in the exact linear regime.
Instead it returns the observable pairwise l-type carrier that survives after
the CeDiTT4 projection onto the degenerate perpendicular subspace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs  # noqa: E402
from distortion_workflow import linear_ltype_terms  # noqa: E402


@dataclass(frozen=True)
class LinearS111BuilderReport:
    species: str
    exact_linear_identifiability: str
    observable_replacement: str
    pairs: tuple[dict[str, object], ...]


def _species_paths(gaussian_dir: Path, species: str) -> tuple[Path, Path]:
    lower = species.lower()
    candidates = [
        (gaussian_dir / f"{lower}.fchk", gaussian_dir / f"{lower}.log"),
        (ROOT / f"{lower}.fchk", ROOT / f"{lower}.log"),
        (ROOT / f"{species}.fchk", ROOT / f"{species}.log"),
    ]
    for fchk, log in candidates:
        if fchk.exists() and log.exists():
            return fchk, log
    raise FileNotFoundError(f"Missing Gaussian inputs for {species}.")


def build_linear_s111_builder_report(*, species: str, gaussian_dir: Path) -> LinearS111BuilderReport:
    fchk, log = _species_paths(gaussian_dir, species)
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=str(fchk))
    ltype = linear_ltype_terms(model, gaussian_log_path=str(log))
    if ltype is None:
        raise ValueError(f"{species} is not classified as a linear molecule with degenerate pairs.")

    pairs: list[dict[str, object]] = []
    for pair in ltype["pairs"]:
        effective_pair = pair["effective_linear_model_hz"]["constants_hz"]
        final_conv = pair["effective_linear_model_final_hz"]["constants_hz"]
        pairs.append(
            {
                "pair_label": pair.get("pair_label", tuple(pair["modes"])),
                "modes": tuple(pair["modes"]),
                "freq_cm": float(pair["freq_cm"]),
                "carrier": "X_l",
                "q_e_hz": float(final_conv["q_e"]),
                "q_J_hz": float(final_conv["q_J"]),
                "q_H_hz": float(effective_pair["q_H_pair"]),
            }
        )

    return LinearS111BuilderReport(
        species=species,
        exact_linear_identifiability="no_unique_scalar_S111",
        observable_replacement="pairwise_degenerate_X_l_branch",
        pairs=tuple(pairs),
    )


def _format_report(report: LinearS111BuilderReport) -> str:
    lines = [
        f"=== {report.species} ===",
        "exact linear S111 builder:",
        f"  identifiability = {report.exact_linear_identifiability}",
        f"  observable replacement = {report.observable_replacement}",
    ]
    for pair in report.pairs:
        lines.extend(
            [
                f"  pair {pair['pair_label']} modes={list(pair['modes'])} freq={pair['freq_cm']:.6f} cm^-1",
                f"    carrier = {pair['carrier']}",
                f"    q_e = {pair['q_e_hz']:+.12e} Hz",
                f"    q_J = {pair['q_J_hz']:+.12e} Hz",
                f"    q_H = {pair['q_H_hz']:+.12e} Hz",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hccd", "hcn", "ocs"])
    args = ap.parse_args()
    for species in args.species:
        report = build_linear_s111_builder_report(species=species, gaussian_dir=Path(args.gaussian_dir))
        print(_format_report(report))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
