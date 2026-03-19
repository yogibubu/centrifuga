"""Check normal-mode symmetry assignment from the harmonic model itself."""

from __future__ import annotations

from ceditt_gui import _build_harmonic_model_from_inputs
from symmetry_metadata import assign_normal_mode_irreps


def test_h2o_mode_symmetry_assignment_from_harmonic_model() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="h2o.fchk", xyz_path="", hessian_path="")
    labels = assign_normal_mode_irreps(model)
    assert labels == ["A1", "A1", "B1[x]"]


def test_h2co_mode_symmetry_assignment_from_harmonic_model() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="h2co.fchk", xyz_path="", hessian_path="")
    labels = assign_normal_mode_irreps(model)
    assert labels == ["B2[z]", "B1[y]", "A1", "A1", "A1", "B1[y]"]
