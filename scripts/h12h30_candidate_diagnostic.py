#!/usr/bin/env python3
"""Diagnostic for minimal H12H30 scaffold candidates.

This script does not define the final production formula. It is used to test
whether simple one-mode inspired scaffolds have the correct order of magnitude
 before touching ``quartic_channels.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channel_contributions import to_watson_khz
from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
    parse_gaussian_quartic_benchmark,
)
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


def h12h30_one_mode_diagonal_trial(mu1, mu2, phi3_diag, omega):
    tau = {k: sp.Integer(0) for k in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")}
    for i in range(len(omega)):
        w = sp.Rational(1, 144) / (omega[i] ** 5)
        phi = sp.Float(float(phi3_diag[i, i, i]))
        xx1, yy1, zz1 = mu1[0, 0, i], mu1[1, 1, i], mu1[2, 2, i]
        xx2, yy2, zz2 = mu2[0, 0, i, i], mu2[1, 1, i, i], mu2[2, 2, i, i]
        xy1, yx1 = mu1[0, 1, i], mu1[1, 0, i]
        xz1, zx1 = mu1[0, 2, i], mu1[2, 0, i]
        yz1, zy1 = mu1[1, 2, i], mu1[2, 1, i]
        xy2, yx2 = mu2[0, 1, i, i], mu2[1, 0, i, i]
        xz2, zx2 = mu2[0, 2, i, i], mu2[2, 0, i, i]
        yz2, zy2 = mu2[1, 2, i, i], mu2[2, 1, i, i]
        tau["tau_xxxx"] += phi * xx1 * xx2 * w
        tau["tau_yyyy"] += phi * yy1 * yy2 * w
        tau["tau_zzzz"] += phi * zz1 * zz2 * w
        tau["tau_xxyy"] += phi * (xx1 * yy2 + xy1 * xy2 + xy1 * yx2 + yx1 * xy2 + yx1 * yx2 + yy1 * xx2) * w
        tau["tau_xxzz"] += phi * (xx1 * zz2 + xz1 * xz2 + xz1 * zx2 + zx1 * xz2 + zx1 * zx2 + zz1 * xx2) * w
        tau["tau_yyzz"] += phi * (yy1 * zz2 + yz1 * yz2 + yz1 * zy2 + zy1 * yz2 + zy1 * zy2 + zz1 * yy2) * w
    return tau


def main() -> None:
    for species in ("h2o", "h2s", "h2co", "h2cs"):
        fchk = parse_gaussian_fchk_harmonic_data(Path(f"{species}.fchk"))
        anh = parse_gaussian_anharmonic_force_data(Path(f"{species}.log"))
        quart = parse_gaussian_quartic_benchmark(Path(f"{species}.log"))
        model, meta = build_default_harmonic_model(
            fchk.masses_amu,
            fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
            fchk.cartesian_force_constants,
        )
        cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
        mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
        mu2 = sp.MutableDenseNDimArray(model.d2Inv_au.tolist())
        omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)

        trial_red = h12h30_one_mode_diagonal_trial(mu1, mu2, cubic.reduced_cm, omega)
        trial_raw = h12h30_one_mode_diagonal_trial(mu1, mu2, cubic.raw_au, omega)

        wat_red = to_watson_khz(
            trial_red,
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=quart.spectroscopic_axes,
            tau_cm_scale=1.0,
        )
        wat_raw = to_watson_khz(
            trial_raw,
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=quart.spectroscopic_axes,
        )
        print(species, meta)
        print("  reduced-cm trial:", {k: round(v, 6) for k, v in wat_red.items()})
        print("  raw-au trial    :", {k: round(v, 6) for k, v in wat_raw.items()})


if __name__ == "__main__":
    main()
