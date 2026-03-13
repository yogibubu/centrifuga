#!/usr/bin/env python3
"""Compute per-channel quartic contributions from Gaussian inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp

from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
)
from rovib_distortion import (
    AMU_TO_AU_MASS,
    ANGSTROM_TO_BOHR,
    AU_FREQ_TO_CMINV,
    CMINV_TO_MHZ,
    harmonic_inertia_model_from_geometry_hessian,
)
from quartic_channels import (
    channel_h12h12,
    channel_h12h30,
    channel_h22,
    channel_h30h30,
    tau_to_watson_a,
)

WILSON_TAU_AU_TO_CMINV = 4.0 * AU_FREQ_TO_CMINV


@dataclass
class ChannelResult:
    species: str
    channel: str
    watson_khz: dict[str, float]


def sympy_array_from_numpy(array: np.ndarray) -> sp.MutableDenseNDimArray:
    return sp.MutableDenseNDimArray([[float(v) for v in row] for row in array])


def parse_model(fchk: Path, log: Path):
    harm = parse_gaussian_fchk_harmonic_data(fchk)
    anh = parse_gaussian_anharmonic_force_data(log)
    masses = harm.masses_amu
    coords_ang = harm.coords_bohr * (1.0 / ANGSTROM_TO_BOHR)
    hessian = harm.cartesian_force_constants
    model = harmonic_inertia_model_from_geometry_hessian(masses, coords_ang, hessian)
    phi3 = sp.MutableDenseNDimArray(anh.phi3_raw_au.tolist())
    return model, phi3


def channel_contributions(model, phi3):
    mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
    mu2 = sp.MutableDenseNDimArray(model.d2Inv_au.tolist())
    omega = tuple(float(x) for x in model.vib_freq_cm)
    hbar = sp.Float(1.0)
    taus = {
        "H12H12": channel_h12h12(mu1, omega),
        "H22": channel_h22(mu2, omega, hbar),
        "H12H30": channel_h12h30(mu1, mu2, phi3, omega, hbar, seed=None, exact_calibration=False),
        "H30H30": channel_h30h30(mu1, phi3, omega, hbar, seed=7, exact_calibration=False),
    }
    taus["total"] = {k: sum(tau[k] for tau in taus.values() if k in tau) for k in taus["H12H12"]}
    return taus


def to_watson_khz(tau: dict[str, sp.Expr]) -> dict[str, float]:
    tau_cm = {k: sp.simplify(WILSON_TAU_AU_TO_CMINV * v) for k, v in tau.items()}
    watson = tau_to_watson_a(tau_cm)
    return {k: float(v * CMINV_TO_MHZ * 1e3) for k, v in watson.items()}


def print_table(results: Iterable[ChannelResult]):
    headers = ("Species", "Channel", "ΔJ", "ΔJK", "ΔK", "d1", "d2")
    print(f"{' | '.join(headers)}")
    print("-" * 80)
    for res in results:
        vals = [f"{res.watson_khz[key]:.3e}" for key in ("DJ", "DJK", "DK", "d1", "d2")]
        print(f"{res.species} | {res.channel} | " + " | ".join(vals))


def main():
    ap = argparse.ArgumentParser(description="Compute channel-resolved quartic contributions.")
    ap.add_argument("--species", nargs=2, action="append", metavar=("NAME", "BASE"), help="Provide species name and base filename (without extension).", required=True)
    args = ap.parse_args()

    results: list[ChannelResult] = []
    for name, base in args.species:
        fchk = Path(base + ".fchk")
        log = Path(base + ".log")
        if not fchk.exists() or not log.exists():
            raise FileNotFoundError(f"Missing files for {name}: {fchk}, {log}")
        model, phi3 = parse_model(fchk, log)
        taus = channel_contributions(model, phi3)
        for channel, tau in taus.items():
            watson = to_watson_khz(tau)
            results.append(ChannelResult(name, channel, watson))

    print_table(results)


if __name__ == "__main__":
    main()
