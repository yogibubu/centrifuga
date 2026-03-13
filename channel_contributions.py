#!/usr/bin/env python3
"""Compute per-channel quartic contributions from Gaussian inputs.

The channel builders operate on the same internal frequency convention used by
the harmonic order-2 workflow, namely vibrational frequencies in atomic units.
The order-4 mixed and cubic-cubic channels still rely on placeholder formulas
in ``quartic_channels.py``; their output is therefore diagnostic only until
the full Paper 2 implementation is restored.
"""

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
    parse_gaussian_quartic_benchmark,
)
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from rovib_distortion import (
    AMU_TO_AU_MASS,
    ANGSTROM_TO_BOHR,
    AU_FREQ_TO_CMINV,
    CMINV_TO_MHZ,
)
from quartic_channels import (
    channel_h12h12,
    channel_h12h30,
    channel_h22,
    channel_h30h30,
)
from h30h30_resonance import enumerate_current_h30h30_terms, summarize_terms_by_metric
from derive_watson_quartic_vanvleck import (
    gaussian_asymmetric_a_from_t,
    gaussian_symmetric_from_t,
    gaussian_t_from_tauprime,
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
    anh = parse_gaussian_anharmonic_force_data(log)
    harm = parse_gaussian_fchk_harmonic_data(fchk)
    masses = harm.masses_amu
    coords_ang = harm.coords_bohr * (1.0 / ANGSTROM_TO_BOHR)
    hessian = harm.cartesian_force_constants
    model, convention = build_default_harmonic_model(masses, coords_ang, hessian)
    phi3_raw_au, cubic_check = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    phi3 = sp.MutableDenseNDimArray(phi3_raw_au.tolist())
    meta = {
        "harmonic_representation": convention["representation"],
        "harmonic_kappa": convention["kappa"],
        "cubic_mode_mapping": cubic_check.mapping,
        "cubic_mode_order_identity": cubic_check.is_identity,
        "cubic_max_abs_freq_delta_cm": cubic_check.max_abs_freq_delta_cm,
    }
    return model, phi3, meta


def channel_contributions(model, phi3):
    mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
    mu2 = sp.MutableDenseNDimArray(model.d2Inv_au.tolist())
    omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)
    hbar = sp.Float(1.0)
    taus = {
        "H12H12": channel_h12h12(mu1, omega),
        "H22": channel_h22(mu2, omega, hbar),
        "H12H30": channel_h12h30(mu1, mu2, phi3, omega, hbar, seed=None, exact_calibration=False),
        "H30H30": channel_h30h30(mu1, phi3, omega, hbar, seed=7, exact_calibration=False),
    }
    taus["total"] = {k: sum(tau[k] for tau in taus.values() if k in tau) for k in taus["H12H12"]}
    return taus


def _sigma_values_quartic(abc_mhz: Iterable[float]) -> tuple[float, float]:
    A, B, C = [float(x) for x in abc_mhz]
    sigma = (2.0 * A - B - C) / (B - C)
    sigma1 = 1.0 / sigma
    return sigma, sigma1


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

    diag = {
        0: tau["tau_xxxx"],
        1: tau["tau_yyyy"],
        2: tau["tau_zzzz"],
    }
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


def to_watson_khz(
    tau: dict[str, sp.Expr],
    *,
    abc_mhz: Iterable[float],
    reduction: str = "S",
    spectroscopic_axes: dict[str, int] | None = None,
    tau_cm_scale: float = WILSON_TAU_AU_TO_CMINV,
) -> dict[str, float]:
    tau_cm = {k: sp.simplify(tau_cm_scale * v) for k, v in tau.items()}
    taup = _gaussian_tauprime_from_compressed_tau(tau_cm, spectroscopic_axes=spectroscopic_axes)
    tmat = gaussian_t_from_tauprime(taup)
    sigma, sigma1 = _sigma_values_quartic(abc_mhz)
    red = reduction.strip().upper()
    if red == "A":
        watson = gaussian_asymmetric_a_from_t(tmat, sp.Float(sigma))
    else:
        watson = gaussian_symmetric_from_t(tmat, sp.Float(sigma1))
    return {k: float(v * CMINV_TO_MHZ * 1e3) for k, v in watson.items()}


def print_table(results: Iterable[ChannelResult]):
    headers = ("Species", "Channel", "ΔJ", "ΔJK", "ΔK", "d1", "d2")
    print(f"{' | '.join(headers)}")
    print("-" * 80)
    for res in results:
        vals = [f"{res.watson_khz[key]:.3e}" for key in ("DJ", "DJK", "DK", "d1", "d2")]
        print(f"{res.species} | {res.channel} | " + " | ".join(vals))


def print_convention_summary(species: str, meta: dict[str, object]) -> None:
    mapping = tuple(int(x) for x in meta["cubic_mode_mapping"])
    print(
        f"{species}: harmonic={meta['harmonic_representation']} "
        f"kappa={float(meta['harmonic_kappa']):+.6f} "
        f"cubic_order_identity={bool(meta['cubic_mode_order_identity'])} "
        f"cubic_mapping={mapping} "
        f"max_dnu={float(meta['cubic_max_abs_freq_delta_cm']):.3e} cm^-1"
    )


def print_h30h30_diagnostic(model, phi3, *, top_terms: int, wilson_factor_cm_per_au: float) -> None:
    terms = enumerate_current_h30h30_terms(
        model.dInv_au,
        np.asarray(phi3.tolist(), dtype=float),
        model.vib_freq_cm,
        wilson_factor_cm_per_au=wilson_factor_cm_per_au,
    )
    top = summarize_terms_by_metric(terms, top_n=top_terms)
    diag_only = [term for term in top if term.diagonal_modes]

    print("\nCurrent H30H30 resonance diagnostic")
    print(f"wilson_factor_cm_per_au = {wilson_factor_cm_per_au}")
    print("NOTE: only the currently coded positive-sum family omega_i + omega_j + 1 is included.")
    print("Top terms by Martin ratio:")
    for term in top:
        modes_1based = tuple(idx + 1 for idx in term.modes)
        print(
            f"  {term.component:>8}  modes={modes_1based!s:<12} "
            f"diag={str(term.diagonal_modes):<5} coupling_cm={term.coupling_cm: .6e} "
            f"denominator_cm={term.denominator_cm: .6e} Martin={term.martin_ratio: .6e}"
        )
    if diag_only:
        print("Diagonal modal terms within the printed set:")
        for term in diag_only:
            modes_1based = tuple(idx + 1 for idx in term.modes)
            print(
                f"  {term.component:>8}  modes={modes_1based!s:<12} "
                f"coupling_cm={term.coupling_cm: .6e} denominator_cm={term.denominator_cm: .6e} "
                f"Martin={term.martin_ratio: .6e}"
            )


def main():
    ap = argparse.ArgumentParser(description="Compute channel-resolved quartic contributions.")
    ap.add_argument("--species", nargs=2, action="append", metavar=("NAME", "BASE"), help="Provide species name and base filename (without extension).", required=True)
    ap.add_argument("--show-h30h30-diagnostic", action="store_true", help="Print the current H30H30 Martin/denominator diagnostic.")
    ap.add_argument("--h30h30-top-terms", type=int, default=10, help="How many H30H30 terms to print in the diagnostic.")
    ap.add_argument(
        "--h30h30-wilson-factor-cm-per-au",
        type=float,
        default=1.0,
        help="Temporary numerator conversion factor used in the current H30H30 Martin diagnostic.",
    )
    args = ap.parse_args()

    results: list[ChannelResult] = []
    for name, base in args.species:
        fchk = Path(base + ".fchk")
        log = Path(base + ".log")
        if not fchk.exists() or not log.exists():
            raise FileNotFoundError(f"Missing files for {name}: {fchk}, {log}")
        model, phi3, meta = parse_model(fchk, log)
        print_convention_summary(name, meta)
        quart = parse_gaussian_quartic_benchmark(log)
        axes = quart.spectroscopic_axes
        taus = channel_contributions(model, phi3)
        for channel, tau in taus.items():
            watson = to_watson_khz(
                tau,
                abc_mhz=model.abc_mhz,
                reduction="S",
                spectroscopic_axes=axes,
            )
            results.append(ChannelResult(name, channel, watson))
        if args.show_h30h30_diagnostic:
            print(f"\n=== {name} ===")
            print_h30h30_diagnostic(
                model,
                phi3,
                top_terms=args.h30h30_top_terms,
                wilson_factor_cm_per_au=args.h30h30_wilson_factor_cm_per_au,
            )

    print_table(results)


if __name__ == "__main__":
    main()
