from pathlib import Path

import sympy as sp

from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
)
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from quartic_channels import channel_h12h30, channel_h12h30_decomposed
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


def _load_case(name: str):
    fchk = parse_gaussian_fchk_harmonic_data(Path(f"{name}.fchk"))
    anh = parse_gaussian_anharmonic_force_data(Path(f"{name}.log"))
    model, _meta = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
    mu2 = sp.MutableDenseNDimArray(model.d2Inv_au.tolist())
    omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)
    phi3 = sp.MutableDenseNDimArray(cubic.reduced_cm.tolist())
    return mu1, mu2, phi3, omega


def _max_abs_tau(tau: dict[str, sp.Expr]) -> float:
    return max(abs(float(val)) for val in tau.values())


def test_h12h30_decomposition_reassembles_total():
    mu1, mu2, phi3, omega = _load_case("hfo")
    direct = channel_h12h30(mu1, mu2, phi3, omega, hbar=sp.Float(1.0))
    pieces = channel_h12h30_decomposed(mu1, mu2, phi3, omega, hbar=sp.Float(1.0))
    for key in direct:
        assert abs(float(direct[key] - pieces["total"][key])) < 1.0e-14
        assert abs(float(pieces["semidiagonal"][key] - pieces["semidiagonal_effective"][key])) < 1.0e-14


def test_h12h30_semidiagonal_pair_split_is_small_on_test_cases():
    for species in ("h2o", "h2s", "hfo"):
        mu1, mu2, phi3, omega = _load_case(species)
        pieces = channel_h12h30_decomposed(mu1, mu2, phi3, omega, hbar=sp.Float(1.0))
        split = _max_abs_tau(pieces["semidiagonal_split"])
        full = _max_abs_tau(pieces["semidiagonal"])
        assert split <= 0.25 * full


def test_h12h30_three_index_sector_vanishes_for_c2v_triatomics():
    for species in ("h2o", "h2s"):
        mu1, mu2, phi3, omega = _load_case(species)
        pieces = channel_h12h30_decomposed(mu1, mu2, phi3, omega, hbar=sp.Float(1.0))
        assert _max_abs_tau(pieces["three_index"]) < 1.0e-14
        assert _max_abs_tau(pieces["semidiagonal"]) > 1.0e-12


def test_h12h30_three_index_sector_is_active_for_hfo():
    mu1, mu2, phi3, omega = _load_case("hfo")
    pieces = channel_h12h30_decomposed(mu1, mu2, phi3, omega, hbar=sp.Float(1.0))
    assert _max_abs_tau(pieces["three_index"]) > 1.0e-12
    assert _max_abs_tau(pieces["semidiagonal"]) > 1.0e-12
    for key in pieces["three_index"]:
        assert abs(float(pieces["S3"][key] - pieces["three_index_effective"][key])) < 1.0e-14
        assert abs(float(pieces["three_index"][key] - pieces["S3"][key] - pieces["S3_addition"][key])) < 1.0e-14


def test_h12h30_three_index_split_exposes_cancellation_on_active_cases():
    for species in ("hfo", "h2co", "h2cs"):
        mu1, mu2, phi3, omega = _load_case(species)
        pieces = channel_h12h30_decomposed(mu1, mu2, phi3, omega, hbar=sp.Float(1.0))
        assert _max_abs_tau(pieces["three_index_split"]) > _max_abs_tau(pieces["three_index"])
