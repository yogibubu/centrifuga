#!/usr/bin/env python3
"""Autonomous pure linear octic builder from the current CeDiTT4 equations only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs  # noqa: E402
from compare_gaussian_sextic import CMINV_TO_MHZ, _classify_rotor_limit, _degenerate_mode_metadata  # noqa: E402
from gaussian_force_constant_units import (  # noqa: E402
    gaussian_bxx_to_aliev,
    raw_cubic_to_reduced_cm,
    raw_quartic_to_reduced_cm,
    reduced_cubic_to_aliev_k3,
    reduced_quartic_to_aliev_k4,
)
from gaussian_vpt_parser import parse_gaussian_anharmonic_force_data  # noqa: E402


@dataclass(frozen=True)
class LinearOcticPureFromOursReport:
    species: str
    parallel_mode_indices_0based: tuple[int, ...]
    perpendicular_pairs_0based: tuple[tuple[int, int], ...]
    B_perp_cm: float
    C_parallel_cm: tuple[float, ...]
    quartic_block_cm: float
    term3_block_cm: float
    term5_block_cm: float
    total_cm: float


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


def _parallel_mode_indices(n_modes: int, pair_meta: list[dict[str, object]]) -> list[int]:
    deg = {int(x) for meta in pair_meta for x in meta["pair"]}
    return [idx for idx in range(n_modes) if idx not in deg]


def _perpendicular_rotational_constant_cm(model, rotor_limit: dict[str, object]) -> float:
    abc = list(map(float, model.abc_mhz))
    axis_map = {"a": 0, "b": 1, "c": 2}
    vals = [abc[axis_map[str(ax)]] / CMINV_TO_MHZ for ax in rotor_limit.get("degenerate_axes", ())]
    if not vals:
        raise ValueError("Linear rotor does not expose perpendicular axes.")
    return float(sum(vals) / len(vals))


def _build_bxx_parallel_from_didq(model, parallel_indices: list[int], rotor_limit: dict[str, object]) -> list[float]:
    from compare_gaussian_sextic import DIDQ_AU_PER_AMU_SQRT_ANG, FACTG  # noqa: E402
    import numpy as np

    deg_axes = tuple(str(x) for x in rotor_limit.get("degenerate_axes", ()))
    if not deg_axes:
        raise ValueError("Linear-rotor limit does not expose degenerate axes for dI/dQ bootstrap.")
    try:
        perp_axis = tuple(model.xyz_to_abc).index(deg_axes[0])
    except ValueError as exc:
        raise ValueError("Could not map the perpendicular linear axis in xyz_to_abc.") from exc
    moments_xyz = np.asarray(model.moments_amu_a2, dtype=float).reshape(3)
    pmom = float(moments_xyz[perp_axis])
    if abs(pmom) <= 1.0e-30:
        raise ValueError("Perpendicular moment of inertia is too small for dI/dQ bootstrap.")
    didq_au_sqrt_ang = np.asarray(model.dI_au, dtype=float) / DIDQ_AU_PER_AMU_SQRT_ANG
    freq = np.asarray(model.vib_freq_cm, dtype=float).reshape(-1)
    out: list[float] = []
    for idx in parallel_indices:
        wi = float(abs(freq[idx]))
        if wi <= 1.0e-30:
            out.append(0.0)
            continue
        x = float((FACTG * wi) ** 1.5)
        den = 2.0 * x * pmom * pmom
        num = float(didq_au_sqrt_ang[perp_axis, perp_axis, idx])
        out.append(0.0 if abs(den) <= 1.0e-30 else num / den)
    return out


def _quartic_term_supported(*indices: int) -> bool:
    return len(set(indices)) < 4


def _reduced_quartic_key(*indices: int) -> tuple[int, int, int] | None:
    if not _quartic_term_supported(*indices):
        return None
    counts: dict[int, int] = {}
    for idx in indices:
        counts[idx] = counts.get(idx, 0) + 1
    repeated = max(counts, key=lambda idx: (counts[idx], -idx))
    if counts[repeated] < 2:
        return None
    tail = list(indices)
    tail.remove(repeated)
    tail.remove(repeated)
    if len(tail) == 1:
        tail = [tail[0], tail[0]]
    elif len(tail) == 0:
        tail = [repeated, repeated]
    tail_a, tail_b = sorted(tail)
    return repeated, tail_a, tail_b


def _build_k4_reduced_local(phi4_cm, parallel_indices: list[int]) -> list[list[list[float]]]:
    size = len(parallel_indices)
    out = [[[0.0 for _ in range(size)] for _ in range(size)] for _ in range(size)]
    for ii, i in enumerate(parallel_indices):
        for jj, j in enumerate(parallel_indices):
            for kk, k in enumerate(parallel_indices):
                a, b = sorted((j, k))
                out[ii][jj][kk] = float(phi4_cm[i, i, a, b])
    return out


def _quartic_value_from_reduced(tensor3: list[list[list[float]]], *indices: int) -> float:
    key = _reduced_quartic_key(*indices)
    if key is None:
        return 0.0
    i, j, k = key
    return float(tensor3[i][j][k])


def build_linear_octic_pure_from_ours(*, species: str, gaussian_dir: Path) -> LinearOcticPureFromOursReport:
    import numpy as np

    fchk, log = _species_paths(gaussian_dir, species)
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=str(fchk))
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    if rotor_limit.get("kind") != "linear":
        raise ValueError("Pure linear octic builder currently supports only linear molecules.")

    pair_meta = _degenerate_mode_metadata(model, rotor_limit)
    parallel = _parallel_mode_indices(int(len(model.vib_freq_cm)), pair_meta)
    omega_parallel = [float(abs(model.vib_freq_cm[i])) for i in parallel]
    B_perp_cm = _perpendicular_rotational_constant_cm(model, rotor_limit)

    anh = parse_gaussian_anharmonic_force_data(str(log))
    phi3_reduced_cm = raw_cubic_to_reduced_cm(np.asarray(anh.phi3_raw_au, dtype=float), np.asarray(model.vib_freq_cm, dtype=float))
    phi4_reduced_cm = raw_quartic_to_reduced_cm(np.asarray(anh.phi4_raw_au, dtype=float), np.asarray(model.vib_freq_cm, dtype=float))
    k3 = reduced_cubic_to_aliev_k3(phi3_reduced_cm, np.asarray(model.vib_freq_cm, dtype=float))
    k4 = reduced_quartic_to_aliev_k4(phi4_reduced_cm, np.asarray(model.vib_freq_cm, dtype=float))
    k4_reduced = _build_k4_reduced_local(k4, parallel)

    bxx_gaussian = _build_bxx_parallel_from_didq(model, parallel, rotor_limit)
    bxx_aliev = gaussian_bxx_to_aliev(np.asarray(bxx_gaussian, dtype=float), np.asarray(omega_parallel, dtype=float))
    c_parallel = [float(-B_perp_cm * bxx_aliev[n] / omega_parallel[n]) for n in range(len(parallel))]

    n_parallel = len(parallel)
    quartic_block = 0.0
    for n in range(n_parallel):
        for np_ in range(n_parallel):
            for npp in range(n_parallel):
                for nppp in range(n_parallel):
                    q = _quartic_value_from_reduced(k4_reduced, n, np_, npp, nppp)
                    quartic_block += (q * c_parallel[n] * c_parallel[np_] * c_parallel[npp] * c_parallel[nppp]) / 24.0

    r_nn = [[0.0 for _ in range(n_parallel)] for _ in range(n_parallel)]
    for n, i in enumerate(parallel):
        for np_, j in enumerate(parallel):
            cubic_sum = 0.5 * sum(float(k3[i, j, parallel[npp]]) * c_parallel[npp] for npp in range(n_parallel))
            geom = (3.0 / (4.0 * B_perp_cm)) * omega_parallel[n] * omega_parallel[np_] * c_parallel[n] * c_parallel[np_]
            r_nn[n][np_] = geom + cubic_sum

    r_n = [sum(c_parallel[np_] * r_nn[n][np_] for np_ in range(n_parallel)) for n in range(n_parallel)]
    term5_block = -0.5 * sum((r_n[n] ** 2) / omega_parallel[n] for n in range(n_parallel))
    inner = sum(c_parallel[n] * c_parallel[np_] * r_nn[n][np_] for n in range(n_parallel) for np_ in range(n_parallel))
    term3_block = -(63.0 / 256.0) * sum(r_n[n] / omega_parallel[n] for n in range(n_parallel)) * inner
    total = quartic_block + term3_block + term5_block

    return LinearOcticPureFromOursReport(
        species=species,
        parallel_mode_indices_0based=tuple(parallel),
        perpendicular_pairs_0based=tuple(tuple(int(x) for x in meta["pair"]) for meta in pair_meta),
        B_perp_cm=B_perp_cm,
        C_parallel_cm=tuple(float(x) for x in c_parallel),
        quartic_block_cm=float(quartic_block),
        term3_block_cm=float(term3_block),
        term5_block_cm=float(term5_block),
        total_cm=float(total),
    )


def _format_report(report: LinearOcticPureFromOursReport) -> str:
    return "\n".join(
        [
            f"=== {report.species} ===",
            f"parallel modes (0-based): {list(report.parallel_mode_indices_0based)}",
            f"perpendicular pairs (0-based): {[list(x) for x in report.perpendicular_pairs_0based]}",
            f"B_perp = {report.B_perp_cm:+.12e} cm^-1",
            f"C_parallel = {[float(x) for x in report.C_parallel_cm]}",
            f"L4_parallel = {report.quartic_block_cm:+.12e} cm^-1",
            f"L3_parallel = {report.term3_block_cm:+.12e} cm^-1",
            f"L5_parallel = {report.term5_block_cm:+.12e} cm^-1",
            f"L_parallel^(ours) = {report.total_cm:+.12e} cm^-1",
        ]
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hcn"])
    args = ap.parse_args()
    for species in args.species:
        report = build_linear_octic_pure_from_ours(species=species, gaussian_dir=Path(args.gaussian_dir))
        print(_format_report(report))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
