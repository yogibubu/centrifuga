#!/usr/bin/env python3
"""Dedicated H22 projection helpers.

This module isolates the validated CeDiTT3 path for the quartic H22 channel.
The raw channel tensor remains source-independent and lives in
``quartic_channels.channel_h22``; this module only handles the projection path
used to compare or report the channel in spectroscopic constants.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp

from compare_gaussian_sextic import _quartic_h22_tau_from_model
from derive_watson_quartic_vanvleck import (
    gaussian_asymmetric_a_from_t,
    gaussian_symmetric_from_t,
    gaussian_t_from_tauprime,
)
from gaussian_alignment import load_gaussian_aligned_model
from gaussian_vpt_parser import parse_gaussian_fchk_harmonic_data, parse_gaussian_quartic_benchmark
from quartic_channels import channel_h22
from rovib_distortion import ANGSTROM_TO_BOHR, CMINV_TO_MHZ, harmonic_inertia_model_from_geometry_hessian

AU_FREQ_TO_CMINV = 219474.6313705
WILSON_TAU_AU_TO_CMINV = 4.0 * AU_FREQ_TO_CMINV


def _sigma_values_quartic(abc_mhz) -> tuple[float, float]:
    a, b, c = [float(x) for x in abc_mhz]
    sigma = (2.0 * a - b - c) / (b - c)
    return sigma, 1.0 / sigma


def _gaussian_tauprime_from_compressed_tau(
    tau: dict[str, sp.Expr],
    spectroscopic_axes: dict[str, int] | None = None,
) -> dict[tuple[int, int], sp.Expr]:
    if spectroscopic_axes is None:
        order = (0, 1, 2)
    else:
        order = (
            int(spectroscopic_axes["a"]),
            int(spectroscopic_axes["b"]),
            int(spectroscopic_axes["c"]),
        )
    diag = {0: tau["tau_xxxx"], 1: tau["tau_yyyy"], 2: tau["tau_zzzz"]}
    off = {
        (0, 1): sp.simplify(tau["tau_xxyy"] / 2),
        (1, 0): sp.simplify(tau["tau_xxyy"] / 2),
        (0, 2): sp.simplify(tau["tau_xxzz"] / 2),
        (2, 0): sp.simplify(tau["tau_xxzz"] / 2),
        (1, 2): sp.simplify(tau["tau_yyzz"] / 2),
        (2, 1): sp.simplify(tau["tau_yyzz"] / 2),
    }
    out: dict[tuple[int, int], sp.Expr] = {}
    for p, i in enumerate(order):
        for q, j in enumerate(order):
            out[(p, q)] = diag[i] if i == j else off[(i, j)]
    return out


def _to_watson_khz(
    tau: dict[str, sp.Expr],
    *,
    abc_mhz,
    reduction: str = "S",
    spectroscopic_axes: dict[str, int] | None = None,
    tau_cm_scale: float = WILSON_TAU_AU_TO_CMINV,
) -> dict[str, float]:
    tau_cm = {k: sp.simplify(tau_cm_scale * v) for k, v in tau.items()}
    taup = _gaussian_tauprime_from_compressed_tau(tau_cm, spectroscopic_axes=spectroscopic_axes)
    tmat = gaussian_t_from_tauprime(taup)
    sigma, sigma1 = _sigma_values_quartic(abc_mhz)
    if reduction.strip().upper() == "A":
        watson = gaussian_asymmetric_a_from_t(tmat, sp.Float(sigma))
    else:
        watson = gaussian_symmetric_from_t(tmat, sp.Float(sigma1))
    return {k: float(v * CMINV_TO_MHZ * 1e3) for k, v in watson.items()}


def project_aligned_h22_benchmark(model, sigma: float) -> dict[str, float]:
    """Project H22 from the Gaussian-aligned Wilson tensor."""
    _tau, tau_prime = _quartic_h22_tau_from_model(model)
    t = tau_prime / 4.0
    t11, t22, t33 = t[0, 0], t[1, 1], t[2, 2]
    t12, t13, t23 = t[0, 1], t[0, 2], t[1, 2]
    t400 = (3.0 * t11 + 3.0 * t22 + 2.0 * t12) / 8.0
    t220 = t13 + t23 - 2.0 * t400
    t040 = t33 - t220 - t400
    t202 = (t11 - t22) / 4.0
    t022 = (t13 - t23) / 2.0 - t202
    t004 = (t11 + t22 - 2.0 * t12) / 16.0
    sigma1 = 1.0 / sigma
    return {
        "DJ": float((-t400 + 0.5 * t022 * sigma1) * CMINV_TO_MHZ * 1000.0),
        "DJK": float((-t220 - 3.0 * t022 * sigma1) * CMINV_TO_MHZ * 1000.0),
        "DK": float((-t040 + 2.5 * t022 * sigma1) * CMINV_TO_MHZ * 1000.0),
        "d1": float(t202 * CMINV_TO_MHZ * 1000.0),
        "d2": float((t004 + 0.25 * t022 * sigma1) * CMINV_TO_MHZ * 1000.0),
    }


def project_ir_h22_with_spectroscopic_axes(fchk_path: str | Path, quart) -> dict[str, float]:
    """Source-independent fallback used in the CeDiTT3 H22 workflow."""
    fchk = parse_gaussian_fchk_harmonic_data(fchk_path)
    model = harmonic_inertia_model_from_geometry_hessian(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
        representation="I",
    )
    mu2 = sp.MutableDenseNDimArray(model.d2Inv_au.tolist())
    omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)
    tau = channel_h22(mu2, omega, sp.Float(1.0))
    return _to_watson_khz(
        tau,
        abc_mhz=model.abc_mhz,
        reduction="S",
        spectroscopic_axes=quart.spectroscopic_axes,
    )


def project_h22_ceditt3_reference_path(fchk_path: str | Path, log_path: str | Path) -> dict[str, float]:
    """Return the validated CeDiTT3 H22 projection path.

    Policy:
    - use the Gaussian-aligned Wilson-tensor path when that alignment is
      reliable;
    - otherwise fall back to the source-independent Ir path with the benchmark
      spectroscopic axes from the Gaussian quartic section.
    """
    quart = parse_gaussian_quartic_benchmark(log_path)
    try:
        aligned = load_gaussian_aligned_model(str(fchk_path), str(log_path))
    except Exception:
        aligned = None
    if aligned is not None and aligned.is_reliable:
        return project_aligned_h22_benchmark(aligned.model, quart.sigma)
    return project_ir_h22_with_spectroscopic_axes(fchk_path, quart)
