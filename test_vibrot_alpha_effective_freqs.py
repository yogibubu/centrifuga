#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from gaussian_vpt_parser import parse_gaussian_anharmonic_force_data, parse_gaussian_fchk_harmonic_data
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from rovib_distortion import ANGSTROM_TO_BOHR
from vibrot_alpha import alpha_matrix_from_harmonic_and_cubic_cm


def _build(name: str):
    fchk = parse_gaussian_fchk_harmonic_data(Path(f"{name}.fchk"))
    model, meta = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    return model, meta


def test_alpha_effective_frequencies_only_change_anharmonic_channel() -> None:
    model, _meta = _build("h2o")
    anh = parse_gaussian_anharmonic_force_data(Path("h2o.log"))
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)

    base = alpha_matrix_from_harmonic_and_cubic_cm(model, cubic.reduced_cm)
    eff = np.abs(np.asarray(model.vib_freq_cm, dtype=float)).copy()
    eff[0] *= 0.9
    shifted = alpha_matrix_from_harmonic_and_cubic_cm(
        model,
        cubic.reduced_cm,
        effective_frequencies_cm=eff,
    )

    assert np.allclose(base["alpha_coriolis_cm"], shifted["alpha_coriolis_cm"])
    assert np.allclose(base["alpha_inertia_cm"], shifted["alpha_inertia_cm"])
    assert not np.allclose(base["alpha_anharmonic_cm"], shifted["alpha_anharmonic_cm"])
    assert np.allclose(shifted["anharmonic_reference_frequencies_cm"], eff)
