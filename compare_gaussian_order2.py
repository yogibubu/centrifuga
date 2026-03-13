#!/usr/bin/env python3
"""Compare harmonic order-2 quartic tensors against Gaussian ``QCent`` data."""

from __future__ import annotations

import argparse
import math

import numpy as np

from gaussian_vpt_parser import (
    parse_gaussian_fchk_harmonic_data,
    parse_gaussian_harmonic_data,
    parse_gaussian_quartic_benchmark,
)
from rovib_distortion import (
    ANGSTROM_TO_BOHR,
    AU_FREQ_TO_CMINV,
    inverse_inertia_derivatives_from_normal_modes,
    mass_weight_hessian,
    normal_modes,
)


EH_TO_MHZ = 6.57968392061e9
CMINV_TO_MHZ = 29979.2458
BOHR_TO_ANG = 1.0 / ANGSTROM_TO_BOHR
_INV_FACTG = 6.62607015e-34 / (4.0 * math.pi**2 * 29979245800.0) / (1.66053906660e-27 * (1.0e-10) ** 2)
QCENT_CONST1 = _INV_FACTG**3


def _gaussian_like_didq(harm) -> np.ndarray:
    """Replicate Gaussian's ``dPMdQ`` formula in amu^1/2 * Angstrom units."""
    # QCent/dPMdQ is fed with the internal Eckart-frame Cartesian coordinates
    # and eigenvectors; reproducing the printed harmonic block directly gives a
    # much closer match than forcing an extra principal-axis rotation here.
    coords_pa = harm.coords_std_ang
    modes_pa = harm.normal_modes.reshape(harm.atomic_numbers.size, 3, harm.frequencies_cm.size)
    didq = np.zeros((6, harm.frequencies_cm.size), dtype=float)

    ijx = 0
    for ix in range(3):
        for jx in range(ix + 1):
            for mode in range(harm.frequencies_cm.size):
                acc = 0.0
                for atom in range(harm.atomic_numbers.size):
                    mass = harm.masses_amu[atom]
                    acc -= mass * coords_pa[atom, ix] * modes_pa[atom, jx, mode]
                    if ix == jx:
                        for kx in range(3):
                            acc += mass * coords_pa[atom, kx] * modes_pa[atom, kx, mode]
                didq[ijx, mode] = 2.0 * acc
            ijx += 1
    return didq


def _principal_moments_from_coords(coords_pa: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    i_tensor = np.zeros((3, 3), dtype=float)
    for mass, coord in zip(masses_amu, coords_pa):
        r2 = float(np.dot(coord, coord))
        i_tensor += mass * (r2 * np.eye(3) - np.outer(coord, coord))
    return np.diag(i_tensor)


def _tauprime_spectroscopic(tau4: dict[tuple[int, int, int, int], float], axes: dict[str, int] | None) -> dict[str, float]:
    if axes is None:
        axes = {"a": 0, "b": 1, "c": 2}

    def tau_prime(i: int, j: int) -> float:
        if i == j:
            return tau4[(i, i, i, i)]
        return tau4[(i, i, j, j)] + 2.0 * tau4[(i, j, i, j)]

    ia = axes["a"]
    ib = axes["b"]
    ic = axes["c"]
    return {
        "aaaa": tau_prime(ia, ia),
        "aabb": tau_prime(ia, ib),
        "aacc": tau_prime(ia, ic),
        "bbbb": tau_prime(ib, ib),
        "bbcc": tau_prime(ib, ic),
        "cccc": tau_prime(ic, ic),
    }


def h12h12_wilson_tauprime_mhz(log_path: str) -> dict[str, float]:
    harm = parse_gaussian_harmonic_data(log_path).reordered_to_anharmonic()
    quart = parse_gaussian_quartic_benchmark(log_path)
    coords_pa = harm.principal_axis_coordinates()
    vib_vecs_mw = harm.mass_weighted_modes()
    omega = np.abs(harm.frequencies_cm) / AU_FREQ_TO_CMINV
    _, d_inv, _ = inverse_inertia_derivatives_from_normal_modes(harm.masses_amu, coords_pa, vib_vecs_mw)

    tau4 = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    val = 0.0
                    for k in range(len(omega)):
                        val += -0.125 * d_inv[a, b, k] * d_inv[c, d, k] / (omega[k] ** 2)
                    tau4[(a, b, c, d)] = val

    taup = _tauprime_spectroscopic(tau4, quart.spectroscopic_axes)
    return {key: EH_TO_MHZ * val for key, val in taup.items()}


def qcent_tauprime_mhz(log_path: str) -> dict[str, float]:
    """Replicate Gaussian ``QCent`` using dI/dQ, PMom, and harmonic frequencies."""
    harm = parse_gaussian_harmonic_data(log_path).reordered_to_anharmonic()
    quart = parse_gaussian_quartic_benchmark(log_path)
    didq = _gaussian_like_didq(harm)
    pmom = _principal_moments_from_coords(harm.coords_std_ang, harm.masses_amu)
    freq = np.abs(harm.frequencies_cm)

    tau4 = {}
    idx_map = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (2, 0): 3,
        (2, 1): 4,
        (2, 2): 5,
    }
    for a in range(3):
        for b in range(3):
            ab = idx_map[(max(a, b), min(a, b))]
            for c in range(3):
                for d in range(3):
                    cd = idx_map[(max(c, d), min(c, d))]
                    val = 0.0
                    for k in range(freq.size):
                        denom = (freq[k] ** 2) * pmom[a] * pmom[b] * pmom[c] * pmom[d]
                        val += didq[ab, k] * didq[cd, k] / denom
                    tau4[(a, b, c, d)] = -0.5 * QCENT_CONST1 * val

    taup = _tauprime_spectroscopic(tau4, quart.spectroscopic_axes)
    return {key: CMINV_TO_MHZ * val for key, val in taup.items()}


def qcent_tauprime_from_logged_didq_mhz(log_path: str) -> dict[str, float]:
    """Replicate Gaussian ``QCent`` using the ``dIdQ`` block printed in the log."""
    harm = parse_gaussian_harmonic_data(log_path).reordered_to_anharmonic()
    quart = parse_gaussian_quartic_benchmark(log_path)
    if quart.didq_amu_sqrt_ang is None:
        raise ValueError("No Gaussian dIdQ block found in log.")

    didq = quart.didq_amu_sqrt_ang
    pmom = _principal_moments_from_coords(harm.coords_std_ang, harm.masses_amu)
    freq = np.abs(harm.frequencies_cm)

    tau4 = {}
    idx_map = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (2, 0): 3,
        (2, 1): 4,
        (2, 2): 5,
    }
    for a in range(3):
        for b in range(3):
            ab = idx_map[(max(a, b), min(a, b))]
            for c in range(3):
                for d in range(3):
                    cd = idx_map[(max(c, d), min(c, d))]
                    val = 0.0
                    for k in range(freq.size):
                        denom = (freq[k] ** 2) * pmom[a] * pmom[b] * pmom[c] * pmom[d]
                        val += didq[ab, k] * didq[cd, k] / denom
                    tau4[(a, b, c, d)] = -0.5 * QCENT_CONST1 * val

    taup = _tauprime_spectroscopic(tau4, quart.spectroscopic_axes)
    return {key: CMINV_TO_MHZ * val for key, val in taup.items()}


def qcent_tauprime_from_fchk_mhz(fchk_path: str, log_path: str) -> dict[str, float]:
    """Replicate Gaussian QCent from geometry + Cartesian Hessian in the fchk."""
    fchk = parse_gaussian_fchk_harmonic_data(fchk_path)
    quart = parse_gaussian_quartic_benchmark(log_path)

    f_mw = mass_weight_hessian(fchk.cartesian_force_constants, fchk.masses_amu)
    freq_cm, vecs_mw, _, _ = normal_modes(
        f_mw,
        fchk.atomic_numbers.size,
        masses_amu=fchk.masses_amu,
        coords_ang=fchk.coords_bohr * BOHR_TO_ANG,
        linear=False,
    )

    coords_ang = fchk.coords_bohr * 0.529177210903
    modes = vecs_mw.reshape(fchk.atomic_numbers.size, 3, -1)
    didq = np.zeros((6, freq_cm.size), dtype=float)
    ijx = 0
    for ix in range(3):
        for jx in range(ix + 1):
            for mode in range(freq_cm.size):
                acc = 0.0
                for atom in range(fchk.atomic_numbers.size):
                    # dPMdQ is called in Gaussian with MWeigh=.False.; the
                    # mass-weighted Hessian eigenvectors therefore contribute
                    # with a single sqrt(m) factor in Cartesian space.
                    mass = math.sqrt(fchk.masses_amu[atom])
                    acc -= mass * coords_ang[atom, ix] * modes[atom, jx, mode]
                    if ix == jx:
                        for kx in range(3):
                            acc += mass * coords_ang[atom, kx] * modes[atom, kx, mode]
                didq[ijx, mode] = 2.0 * acc
            ijx += 1

    pmom = _principal_moments_from_coords(coords_ang, fchk.masses_amu)
    tau4 = {}
    idx_map = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (2, 0): 3,
        (2, 1): 4,
        (2, 2): 5,
    }
    for a in range(3):
        for b in range(3):
            ab = idx_map[(max(a, b), min(a, b))]
            for c in range(3):
                for d in range(3):
                    cd = idx_map[(max(c, d), min(c, d))]
                    val = 0.0
                    for k in range(freq_cm.size):
                        denom = (abs(freq_cm[k]) ** 2) * pmom[a] * pmom[b] * pmom[c] * pmom[d]
                        val += didq[ab, k] * didq[cd, k] / denom
                    tau4[(a, b, c, d)] = -0.5 * QCENT_CONST1 * val

    # Empirically the Hessian-derived normal modes align with Gaussian's
    # spectroscopic ordering after the (a,b,c) -> (y,z,x) permutation.
    taup = _tauprime_spectroscopic(tau4, {"a": 1, "b": 2, "c": 0})
    return {key: CMINV_TO_MHZ * val for key, val in taup.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare current order-2 harmonic quartic tensor with Gaussian TauPrime.")
    ap.add_argument("log")
    ap.add_argument("--fchk", default="", help="Optional Gaussian .fchk for Hessian-derived normal modes.")
    args = ap.parse_args()

    benchmark = parse_gaussian_quartic_benchmark(args.log)
    model_inv = h12h12_wilson_tauprime_mhz(args.log)
    model_qcent = qcent_tauprime_mhz(args.log)
    model_logged = qcent_tauprime_from_logged_didq_mhz(args.log)
    model_fchk = qcent_tauprime_from_fchk_mhz(args.fchk, args.log) if args.fchk else None

    print("TauPrime comparison in MHz")
    header = f"{'component':<8} {'invI-model':>18} {'qcent-model':>18} {'log-dIdQ':>18}"
    if model_fchk is not None:
        header += f" {'fchk-qcent':>18}"
    header += f" {'gaussian':>18}"
    print(header)
    for key in ("aaaa", "aabb", "aacc", "bbbb", "bbcc", "cccc"):
        mi = model_inv[key]
        mq = model_qcent[key]
        ml = model_logged[key]
        g = benchmark.tau_prime_mhz[key]
        line = f"{key:<8} {mi:18.8f} {mq:18.8f} {ml:18.8f}"
        if model_fchk is not None:
            line += f" {model_fchk[key]:18.8f}"
        line += f" {g:18.8f}"
        print(line)


if __name__ == "__main__":
    main()
