#!/usr/bin/env python3
"""Reconstruct the linear octic offset from the quartic CeDiTT4 scalar D only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs  # noqa: E402
from compare_gaussian_sextic import CMINV_TO_MHZ  # noqa: E402
from distortion_workflow import compute_order2_quartic  # noqa: E402


@dataclass(frozen=True)
class LinearOcticOffsetFromOursD:
    species: str
    D_cm: float
    B_perp_cm: float
    offset_cm: float


def _species_fchk(gaussian_dir: Path, species: str) -> Path:
    lower = species.lower()
    candidates = [
        gaussian_dir / f"{lower}.fchk",
        ROOT / f"{lower}.fchk",
        ROOT / f"{species}.fchk",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing FCHK for {species}.")


def build_linear_octic_offset_from_ours_D(*, species: str, gaussian_dir: Path) -> LinearOcticOffsetFromOursD:
    fchk = _species_fchk(gaussian_dir, species)
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=str(fchk))
    quartic = compute_order2_quartic(model)
    D_cm = float(quartic["special_quartic_projection"]["quartic_mhz"]["D"] / CMINV_TO_MHZ)
    B_perp_cm = float((model.abc_mhz[1] / CMINV_TO_MHZ + model.abc_mhz[2] / CMINV_TO_MHZ) / 2.0)
    offset_cm = float(-8.0 * D_cm**3 / B_perp_cm**2)
    return LinearOcticOffsetFromOursD(
        species=species,
        D_cm=D_cm,
        B_perp_cm=B_perp_cm,
        offset_cm=offset_cm,
    )


def format_linear_octic_offset_from_ours_D(*, species: str, gaussian_dir: Path) -> str:
    out = build_linear_octic_offset_from_ours_D(species=species, gaussian_dir=gaussian_dir)
    return "\n".join(
        [
            f"=== {out.species} ===",
            f"D^(ours) = {out.D_cm:+.12e} cm^-1",
            f"B_perp = {out.B_perp_cm:+.12e} cm^-1",
            f"offset = -8 D^3 / B^2 = {out.offset_cm:+.12e} cm^-1",
        ]
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hcn"])
    args = ap.parse_args()
    for species in args.species:
        print(format_linear_octic_offset_from_ours_D(species=species, gaussian_dir=Path(args.gaussian_dir)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
