#!/usr/bin/env python3
"""Audit the precise mismatch channels between our linear octic branch and the repo Aliev branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import build_explicit_aliev_L_model, build_explicit_aliev_families, make_linear_aliev_explicit_inputs_from_mapping  # noqa: E402
from scripts.build_linear_aliev_payload_from_gaussian import build_payload  # noqa: E402
from scripts.linear_octic_complete_from_ours import build_linear_octic_complete_from_ours  # noqa: E402


@dataclass(frozen=True)
class LinearOcticAlievAudit:
    species: str
    same_Cn: bool
    quartic_total_cm: float
    geom_total_cm: float
    r_square_total_cm: float
    aliev_shared_offset_cm: float
    aliev_total_cm: float
    ours_pure_total_cm: float


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


def build_linear_octic_aliev_audit(*, species: str, gaussian_dir: Path) -> LinearOcticAlievAudit:
    fchk, log = _species_paths(gaussian_dir, species)
    ours = build_linear_octic_complete_from_ours(species=species, gaussian_dir=gaussian_dir)
    payload = build_payload(
        fchk_path=str(fchk),
        log_path=str(log),
        cn_source="didq_linear_v_iscr",
        zeta_reduction="pair_offdiag",
        pair_seed_source="gaussian_qe_source",
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced_input")
    fam = build_explicit_aliev_families(inputs)
    model = build_explicit_aliev_L_model(inputs)

    same_Cn = tuple(float(x) for x in ours.pure.C_parallel_cm) == tuple(float(fam.C_n[i]) for i in range(len(fam.C_n)))

    B = float(inputs.B)
    D_J = float(inputs.D_J)
    geom_total = sum(8.0 * B * D_J * float(fam.C_n[n]) ** 2 / float(inputs.omega_parallel[n]) for n in range(len(inputs.omega_parallel)))
    r_square_total = sum(-(float(fam.r_n[n]) ** 2) / (2.0 * float(inputs.omega_parallel[n])) for n in range(len(inputs.omega_parallel)))

    # The quartic branch in the repo model is the remainder once the explicit
    # geometric, r^2, and shared-offset pieces are removed.
    quartic_total = float(model.value) - float(model.shared_offset) - geom_total - r_square_total

    return LinearOcticAlievAudit(
        species=species,
        same_Cn=same_Cn,
        quartic_total_cm=float(quartic_total),
        geom_total_cm=float(geom_total),
        r_square_total_cm=float(r_square_total),
        aliev_shared_offset_cm=float(model.shared_offset),
        aliev_total_cm=float(model.value),
        ours_pure_total_cm=float(ours.pure.total_cm),
    )


def format_linear_octic_aliev_audit(*, species: str, gaussian_dir: Path) -> str:
    out = build_linear_octic_aliev_audit(species=species, gaussian_dir=gaussian_dir)
    return "\n".join(
        [
            f"=== {out.species} ===",
            f"same_Cn = {out.same_Cn}",
            f"ours_pure_total = {out.ours_pure_total_cm:+.12e} cm^-1",
            f"aliev_total = {out.aliev_total_cm:+.12e} cm^-1",
            f"aliev_shared_offset = {out.aliev_shared_offset_cm:+.12e} cm^-1",
            f"aliev_quartic_total = {out.quartic_total_cm:+.12e} cm^-1",
            f"aliev_geom_total = {out.geom_total_cm:+.12e} cm^-1",
            f"aliev_rsquare_total = {out.r_square_total_cm:+.12e} cm^-1",
        ]
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hcn"])
    args = ap.parse_args()
    for species in args.species:
        print(format_linear_octic_aliev_audit(species=species, gaussian_dir=Path(args.gaussian_dir)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
