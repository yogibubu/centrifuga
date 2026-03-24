"""Check normal-mode symmetry assignment from the harmonic model itself."""

from __future__ import annotations

from ceditt_gui import _build_harmonic_model_from_inputs
from symmetry_metadata import assign_normal_mode_irreps, _nonabelian_character_table, _nonabelian_operation_keys


def test_h2o_mode_symmetry_assignment_from_harmonic_model() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="h2o.fchk", xyz_path="", hessian_path="")
    labels = assign_normal_mode_irreps(model)
    assert labels == ["A1", "A1", "B1[x]"]


def test_h2co_mode_symmetry_assignment_from_harmonic_model() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="h2co.fchk", xyz_path="", hessian_path="")
    labels = assign_normal_mode_irreps(model)
    assert labels == ["B2[z]", "B1[y]", "A1", "A1", "A1", "B1[y]"]


def test_nh3_mode_symmetry_assignment_from_harmonic_model() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="nh3.fchk", xyz_path="", hessian_path="")
    labels = assign_normal_mode_irreps(model)
    assert labels == ["A1", "E", "E", "A1", "E", "E"]


def test_nonabelian_character_support_for_d3_and_d3h() -> None:
    d3_labels = ["E", "C3z^1", "C3z^2", "C2_xy_3_0"]
    d3h_labels = ["E", "C3z^1", "C3z^2", "C2_xy_3_0", "sigma_xy", "S3", "sigma_v_3_0"]
    assert _nonabelian_operation_keys("D3", d3_labels) == ("E", "C3z^1", "C2_xy_3_0")
    assert _nonabelian_operation_keys("D3h", d3h_labels) == ("E", "C3z^1", "C2_xy_3_0", "sigma_xy", "S3", "sigma_v_3_0")
    assert _nonabelian_character_table("D3", ("E", "C3z^1", "C2_xy_3_0")) is not None
    assert _nonabelian_character_table("D3h", ("E", "C3z^1", "C2_xy_3_0", "sigma_xy", "S3", "sigma_v_3_0")) is not None


def test_nonabelian_character_support_for_c4v_d4_and_d4h() -> None:
    c4v_labels = ["E", "C4z^1", "sigma_xz"]
    d4_labels = ["E", "C4z^1", "C2x^1"]
    d4h_labels = ["E", "C4z^1", "C2x^1", "i", "S4", "sigma_xz"]
    assert _nonabelian_operation_keys("C4v", c4v_labels) == ("E", "C4z^1", "sigma_xz")
    assert _nonabelian_operation_keys("D4", d4_labels) == ("E", "C4z^1", "C2x^1")
    assert _nonabelian_operation_keys("D4h", d4h_labels) == ("E", "C4z^1", "C2x^1", "i", "S4", "sigma_xz")
    assert _nonabelian_character_table("C4v", ("E", "C4z^1", "sigma_xz")) is not None
    assert _nonabelian_character_table("D4", ("E", "C4z^1", "C2x^1")) is not None
    assert _nonabelian_character_table("D4h", ("E", "C4z^1", "C2x^1", "i", "S4", "sigma_xz")) is not None


def test_nonabelian_character_support_for_c6v_d6_and_d6h() -> None:
    c6v_labels = ["E", "C6z^1", "sigma_xz"]
    d6_labels = ["E", "C6z^1", "C2x^1"]
    d6h_labels = ["E", "C6z^1", "C2x^1", "i", "S6", "sigma_xz"]
    assert _nonabelian_operation_keys("C6v", c6v_labels) == ("E", "C6z^1", "sigma_xz")
    assert _nonabelian_operation_keys("D6", d6_labels) == ("E", "C6z^1", "C2x^1")
    assert _nonabelian_operation_keys("D6h", d6h_labels) == ("E", "C6z^1", "C2x^1", "i", "S6", "sigma_xz")
    assert _nonabelian_character_table("C6v", ("E", "C6z^1", "sigma_xz")) is not None
    assert _nonabelian_character_table("D6", ("E", "C6z^1", "C2x^1")) is not None
    assert _nonabelian_character_table("D6h", ("E", "C6z^1", "C2x^1", "i", "S6", "sigma_xz")) is not None


def test_nonabelian_character_support_for_c5v_d5_and_d5h() -> None:
    c5v_labels = ["E", "C5z^1", "sigma_xz"]
    d5_labels = ["E", "C5z^1", "C2x^1"]
    d5h_labels = ["E", "C5z^1", "C2x^1", "sigma_xy", "sigma_xz"]
    assert _nonabelian_operation_keys("C5v", c5v_labels) == ("E", "C5z^1", "sigma_xz")
    assert _nonabelian_operation_keys("D5", d5_labels) == ("E", "C5z^1", "C2x^1")
    assert _nonabelian_operation_keys("D5h", d5h_labels) == ("E", "C5z^1", "C2x^1", "sigma_xy", "sigma_xz")
    assert _nonabelian_character_table("C5v", ("E", "C5z^1", "sigma_xz")) is not None
    assert _nonabelian_character_table("D5", ("E", "C5z^1", "C2x^1")) is not None
    assert _nonabelian_character_table("D5h", ("E", "C5z^1", "C2x^1", "sigma_xy", "sigma_xz")) is not None


def test_c2h2_mode_symmetry_assignment_from_harmonic_model() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="c2h2.fchk", xyz_path="", hessian_path="")
    labels = assign_normal_mode_irreps(model)
    assert labels == ["Pi_g", "Pi_g", "Pi_u", "Pi_u", "Sigma_g+", "Sigma_u+", "Sigma_g+"]
