#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from gaussian_vpt_parser import parse_gaussian_anharmonic_force_data, parse_gaussian_fchk_harmonic_data
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from rovib_distortion import ANGSTROM_TO_BOHR


def _build(name: str):
    fchk = parse_gaussian_fchk_harmonic_data(Path(f"{name}.fchk"))
    model, meta = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    return model, meta


def test_default_representation_policy() -> None:
    _, h2o = _build("h2o")
    _, h2s = _build("h2s")
    _, h2co = _build("h2co")
    _, h2cs = _build("h2cs")

    assert h2o["representation"] == "Ir"
    assert h2s["representation"] == "Ir"
    assert h2co["representation"] == "IIIr"
    assert h2cs["representation"] == "IIIr"


def test_cubic_alignment_is_frequency_only() -> None:
    model, _meta = _build("h2o")
    anh = parse_gaussian_anharmonic_force_data(Path("h2o.log"))
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)

    assert cubic.raw_au.shape[0] == model.vib_freq_cm.size
    assert cubic.reduced_cm.shape[0] == model.vib_freq_cm.size
    assert len(cubic.check.source_to_target) == model.vib_freq_cm.size
    assert len(cubic.check.target_to_source) == model.vib_freq_cm.size
    assert not cubic.check.is_identity
    assert np.isfinite(cubic.check.max_abs_freq_delta_cm)
    assert cubic.check.max_abs_freq_delta_cm <= 5.0
