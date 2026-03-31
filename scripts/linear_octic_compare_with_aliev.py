#!/usr/bin/env python3
"""Direct comparison between our linear octic builders and the repo Aliev branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import build_explicit_aliev_L_model, make_linear_aliev_explicit_inputs_from_mapping  # noqa: E402
from scripts.build_linear_aliev_payload_from_gaussian import build_payload  # noqa: E402
from scripts.linear_octic_complete_from_ours import build_linear_octic_complete_from_ours  # noqa: E402
from scripts.linear_s111_builder import build_linear_s111_builder_report  # noqa: E402


@dataclass(frozen=True)
class LinearOcticCompareWithAliev:
    species: str
    L_parallel_ours_cm: float
    L_aliev_cm: float
    delta_L_cm: float
    pairwise_rows: tuple[dict[str, object], ...]


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


def build_linear_octic_compare_with_aliev(*, species: str, gaussian_dir: Path) -> LinearOcticCompareWithAliev:
    ours = build_linear_octic_complete_from_ours(species=species, gaussian_dir=gaussian_dir)
    fchk, log = _species_paths(gaussian_dir, species)
    payload = build_payload(
        fchk_path=str(fchk),
        log_path=str(log),
        cn_source="didq_linear_v_iscr",
        zeta_reduction="pair_offdiag",
        pair_seed_source="gaussian_qe_source",
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced_input")
    l_aliev = float(build_explicit_aliev_L_model(inputs).value)
    pairwise_aliev = build_linear_s111_builder_report(species=species, gaussian_dir=gaussian_dir)

    rows = []
    for ours_pair, aliev_pair in zip(ours.degenerate.pairs, pairwise_aliev.pairs, strict=True):
        rows.append(
            {
                "pair": ours_pair["pair"],
                "q_H_ours_hz": float(ours_pair["q_H_hz_from_ours"]),
                "q_H_aliev_hz": float(aliev_pair["q_H_hz"]),
                "delta_q_H_hz": float(ours_pair["q_H_hz_from_ours"] - aliev_pair["q_H_hz"]),
            }
        )

    return LinearOcticCompareWithAliev(
        species=species,
        L_parallel_ours_cm=float(ours.pure.total_cm),
        L_aliev_cm=l_aliev,
        delta_L_cm=float(ours.pure.total_cm - l_aliev),
        pairwise_rows=tuple(rows),
    )


def format_linear_octic_compare_with_aliev(*, species: str, gaussian_dir: Path) -> str:
    out = build_linear_octic_compare_with_aliev(species=species, gaussian_dir=gaussian_dir)
    lines = [
        f"=== {out.species} ===",
        f"L_parallel^(ours) = {out.L_parallel_ours_cm:+.12e} cm^-1",
        f"L_Aliev = {out.L_aliev_cm:+.12e} cm^-1",
        f"Delta_L = {out.delta_L_cm:+.12e} cm^-1",
        "pairwise q_H comparison:",
    ]
    for row in out.pairwise_rows:
        lines.extend(
            [
                f"pair = {list(row['pair'])}",
                f"q_H^(ours) = {row['q_H_ours_hz']:+.12e} Hz",
                f"q_H^(Aliev-branch) = {row['q_H_aliev_hz']:+.12e} Hz",
                f"Delta_q_H = {row['delta_q_H_hz']:+.12e} Hz",
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
        print(format_linear_octic_compare_with_aliev(species=species, gaussian_dir=Path(args.gaussian_dir)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
