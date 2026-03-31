#!/usr/bin/env python3
"""Exact linear degenerate octic builder from the current CeDiTT4 primitives only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs  # noqa: E402
from compare_gaussian_sextic import (  # noqa: E402
    _c1_from_mu1,
    _classify_rotor_limit,
    _degenerate_mode_metadata,
    _quartic_tau_from_model,
    _special_alpha_context,
)
from rovib_distortion import coriolis_zeta_tensor  # noqa: E402

CMINV_TO_HZ = 2.99792458e10


@dataclass(frozen=True)
class LinearOcticPairwiseFromOurs:
    species: str
    tau_perp_cm: float
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


def _perpendicular_rotational_constant_cm(model, rotor_limit: dict[str, object]) -> float:
    abc = np.asarray(model.abc_mhz, dtype=float)
    axis = {"a": 0, "b": 1, "c": 2}
    vals = [float(abc[axis[str(ax)]] / 29979.2458) for ax in rotor_limit.get("degenerate_axes", ())]
    if not vals:
        raise ValueError("Linear rotor does not expose perpendicular degenerate axes.")
    return float(sum(vals) / len(vals))


def _rotate_mode_pairs(vib_vecs_mw_pa: np.ndarray, pairs: list[tuple[int, int]], rots: list[np.ndarray]) -> np.ndarray:
    out = np.array(vib_vecs_mw_pa, dtype=float, copy=True)
    for (i, j), rot in zip(pairs, rots, strict=True):
        out[:, [i, j]] = out[:, [i, j]] @ rot
    return out


def _canonical_pair_blocks(model) -> dict[str, object]:
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    pair_meta = _degenerate_mode_metadata(model, rotor_limit)
    if not pair_meta:
        raise ValueError("No degenerate perpendicular pairs detected.")

    c1 = _c1_from_mu1(model)
    special_ctx = _special_alpha_context(model, rotor_limit, c1=c1)
    pair_rotation_map = {
        tuple(int(x) for x in meta["pair"]): np.asarray(meta["rotation_2x2"], dtype=float)
        for meta in special_ctx.get("canonical_pair_data", ())
    }
    pair_list = [tuple(int(x) for x in meta["pair"]) for meta in pair_meta]
    rots = [pair_rotation_map.get(pair, np.eye(2, dtype=float)) for pair in pair_list]

    vib_canonical = _rotate_mode_pairs(np.asarray(model.vib_vecs_mw_pa, dtype=float), pair_list, rots)
    canonical_coriolis_xyz = coriolis_zeta_tensor(vib_canonical)

    deg_modes = {int(x) for meta in pair_meta for x in meta["pair"]}
    parallel_indices = tuple(idx for idx in range(len(model.vib_freq_cm)) if idx not in deg_modes)
    axis_map = {label: idx for idx, label in enumerate(model.xyz_to_abc)}
    deg_lbls = tuple(rotor_limit.get("degenerate_axes", ()))
    if len(deg_lbls) != 2:
        raise ValueError("Linear pair builder requires two degenerate perpendicular axes.")
    ax0 = int(axis_map[str(deg_lbls[0])])
    ax1 = int(axis_map[str(deg_lbls[1])])

    return {
        "pair_meta": pair_meta,
        "pair_list": tuple(pair_list),
        "parallel_indices": parallel_indices,
        "degenerate_axis_indices": (ax0, ax1),
        "canonical_coriolis_xyz": canonical_coriolis_xyz,
    }


def _pairwise_sigma_t(model) -> tuple[dict[str, object], ...]:
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    pair_data = _canonical_pair_blocks(model)
    pair_meta = pair_data["pair_meta"]
    B_perp_cm = _perpendicular_rotational_constant_cm(model, rotor_limit)
    omega = np.abs(np.asarray(model.vib_freq_cm, dtype=float))
    ax0, ax1 = pair_data["degenerate_axis_indices"]
    canonical_coriolis_xyz = pair_data["canonical_coriolis_xyz"]
    parallel_indices = pair_data["parallel_indices"]
    pair_list = pair_data["pair_list"]

    out: list[dict[str, object]] = []
    for t_pos, pair in enumerate(pair_list):
        wt_cm = float(0.5 * (omega[pair[0]] + omega[pair[1]]))
        sigma_t = 0.0
        i, j = pair
        for n_pos, n in enumerate(parallel_indices):
            B_nt = np.array(
                [
                    [canonical_coriolis_xyz[ax0, n, i], canonical_coriolis_xyz[ax0, n, j]],
                    [canonical_coriolis_xyz[ax1, n, i], canonical_coriolis_xyz[ax1, n, j]],
                ],
                dtype=float,
            )
            for np_pos, np_ in enumerate(parallel_indices):
                B_npt = np.array(
                    [
                        [canonical_coriolis_xyz[ax0, np_, i], canonical_coriolis_xyz[ax0, np_, j]],
                        [canonical_coriolis_xyz[ax1, np_, i], canonical_coriolis_xyz[ax1, np_, j]],
                    ],
                    dtype=float,
                )
                G_nnp_t = float(np.sum(B_nt * B_npt))
                weight = B_perp_cm * np.sqrt((wt_cm * wt_cm) / (omega[n] * omega[np_]))
                sigma_t += float(weight * G_nnp_t)
        out.append(
            {
                "pair": tuple(int(x) for x in pair),
                "pair_label": str(tuple(int(x) for x in pair)),
                "freq_cm": wt_cm,
                "sigma_t": float(sigma_t),
            }
        )
    return tuple(out)


def build_linear_octic_pairwise_from_ours(*, species: str, gaussian_dir: Path) -> LinearOcticPairwiseFromOurs:
    fchk, _log = _species_paths(gaussian_dir, species)
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=str(fchk))
    tau, _tau_prime = _quartic_tau_from_model(model)
    tau_perp_cm = float(tau[1, 1, 1, 1])
    pairs = []
    for item in _pairwise_sigma_t(model):
        q_h_hz = float((18.0 / 35.0) * tau_perp_cm * item["sigma_t"] * CMINV_TO_HZ)
        pairs.append(
            {
                "pair": item["pair"],
                "pair_label": item["pair_label"],
                "freq_cm": item["freq_cm"],
                "sigma_t": item["sigma_t"],
                "q_H_hz_from_ours": q_h_hz,
            }
        )
    return LinearOcticPairwiseFromOurs(
        species=species,
        tau_perp_cm=tau_perp_cm,
        pairs=tuple(pairs),
    )


def _format_report(report: LinearOcticPairwiseFromOurs) -> str:
    lines = [
        f"=== {report.species} ===",
        f"tau_perp = {report.tau_perp_cm:+.12e} cm^-1",
        "pairwise octic branch from our equations:",
    ]
    for pair in report.pairs:
        lines.extend(
            [
                f"  {pair['pair_label']} modes={list(pair['pair'])} freq={pair['freq_cm']:.6f} cm^-1",
                f"    Sigma_t = {pair['sigma_t']:+.12e}",
                f"    q_H^(ours) = {pair['q_H_hz_from_ours']:+.12e} Hz",
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
        report = build_linear_octic_pairwise_from_ours(species=species, gaussian_dir=Path(args.gaussian_dir))
        print(_format_report(report))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
