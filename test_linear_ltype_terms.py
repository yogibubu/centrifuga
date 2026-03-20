from types import SimpleNamespace

import numpy as np

from ceditt_gui import _build_harmonic_model_from_inputs
from distortion_workflow import compute_order2_quartic, linear_ltype_terms


def test_linear_ltype_terms_reports_degenerate_pair_diagnostics() -> None:
    model = SimpleNamespace(
        abc_mhz=np.array([float("inf"), 15000.0, 15000.0], dtype=float),
        moments_amu_a2=np.array([0.0, 10.0, 10.0], dtype=float),
        vib_freq_cm=np.array([700.0, 702.0, 1200.0], dtype=float),
        coriolis_zeta_pairs_xyz=np.array(
            [
                [[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        xyz_to_abc=("a", "b", "c"),
    )
    quartic_special = {"quartic_mhz": {"D": 0.012}}
    sextic_special = {"sextic_hz": {"H": 25.0}}
    out = linear_ltype_terms(model, quartic_special=quartic_special, sextic_special=sextic_special)
    assert out is not None
    assert abs(out["B_linear_cm"] - (15000.0 / 29979.2458)) < 1.0e-6
    assert len(out["pairs"]) == 1
    pair = out["pairs"][0]
    assert pair["modes"] == (0, 1)
    assert abs(pair["freq_cm"] - 701.0) < 1.0e-12
    assert abs(pair["zeta_parallel"] - 0.8) < 1.0e-12
    assert abs(pair["q_tJ_diagnostic_hz"] - 9600.0) < 1.0e-9
    assert abs(pair["q_tH_diagnostic_hz"] - 20.0) < 1.0e-9
    assert out["diagnostic_only"] is False
    conv = out["literature_convention"]
    assert conv["name"] == "circular_vibrational_angular_momentum_basis"
    assert conv["basis"] == ["I_l", "Z_l", "X_l", "Y_l"]
    assert conv["active_minimal_channel"] == "X_l"
    pure_branch = out["pure_rotational_branch"]
    pair_branch = out["pairwise_ltype_branch"]
    assert pure_branch["kind"] == "linear_pure_rotational"
    assert abs(pure_branch["scalars"]["D_mhz"] - 0.012) < 1.0e-12
    assert abs(pure_branch["scalars"]["H_hz"] - 25.0) < 1.0e-12
    assert pair_branch["kind"] == "linear_pairwise_ltype"
    assert pair_branch["pair_count"] == 1
    assert pair_branch["active_operator_channel"] == "X_l"
    assert abs(pair_branch["driving_scalars"]["D_mhz"] - 0.012) < 1.0e-12
    assert abs(pair_branch["driving_scalars"]["H_hz"] - 25.0) < 1.0e-12
    assert pair_branch["rotational_feeds"]["quartic_feed_operator"] == "J^2 X_l"
    assert pair_branch["rotational_feeds"]["sextic_feed_operator"] == "(J^2)^2 X_l"
    assert pair_branch["rotational_feeds"]["quartic_feed_present"] is True
    assert pair_branch["rotational_feeds"]["sextic_feed_present"] is True
    assert out["operator_basis"] == "circular_doublet"
    assert out["operator_basis_alt"] == "real_doublet"
    assert "J^2 X_l" in out["effective_model"]
    assert "J^2 O_t" in out["effective_model_alt"]
    assert [entry["operator"] for entry in out["full_operator_basis_real"]] == ["I_t", "O_t", "X_t", "Y_t"]
    assert [entry["operator"] for entry in out["full_operator_basis_circular"]] == ["I_l", "Z_l", "X_l", "Y_l"]
    assert [entry["operator"] for entry in out["full_operator_basis_alt"]] == ["I_t", "O_t", "X_t", "Y_t"]
    terms = {term["operator"]: term["coefficient_hz"] for term in pair["operator_terms_hz"]}
    terms_alt = {term["operator"]: term["coefficient_hz"] for term in pair["operator_terms_alt_hz"]}
    qconv = pair["conventional_constants_hz"]
    qlit = pair["literature_constants_hz"]
    qlit0 = pair["literature_harmonic_estimate_hz"]
    qlitw = pair["literature_watson_estimate_hz"]
    qspec = pair["spectroscopic_linear_constants_hz"]
    qeff = pair["effective_linear_model_hz"]
    conv_map = pair["conventional_pair_mapping"]
    coeffs_real = pair["pair_basis_coefficients_real_hz"]
    coeffs_alt = pair["pair_basis_coefficients_alt_hz"]
    solver_basis = pair["solver_facing_circular_basis"]
    solver_blocks = pair["solver_facing_blocks_hz"]
    assert abs(terms["O_t"] - pair["q_t_abs_hz"]) < 1.0e-12
    assert abs(terms["J^2 O_t"] - 9600.0) < 1.0e-9
    assert abs(terms["(J^2)^2 O_t"] - 20.0) < 1.0e-9
    assert abs(terms_alt["X_l"] - pair["q_t_abs_hz"]) < 1.0e-12
    assert abs(terms_alt["J^2 X_l"] - 9600.0) < 1.0e-9
    assert abs(terms_alt["(J^2)^2 X_l"] - 20.0) < 1.0e-9
    assert abs(qconv["q_t"] - pair["q_t_abs_hz"]) < 1.0e-12
    assert abs(qconv["q_tJ"] - 9600.0) < 1.0e-9
    assert abs(qconv["q_tH"] - 20.0) < 1.0e-9
    assert abs(qlit["q_l"] - pair["q_t_abs_hz"]) < 1.0e-12
    assert abs(qlit["q_lJ"] - 9600.0) < 1.0e-9
    assert abs(qlit["q_lH"] - 20.0) < 1.0e-9
    assert abs(qlit0["q_l_leading"] - (2.0 * (15000.0 / 29979.2458) * 2.99792458e10 / 701.0)) < 1.0
    assert qlit0["formula"] == "q_e^(0) = 2 B_linear / omega_t"
    assert qlitw["q_l_watson"] >= qlit0["q_l_leading"]
    assert "sum_s f_st^2" in qlitw["formula"]
    assert isinstance(qlitw["terms"], list)
    assert abs(qspec["q_e0"] - qlit0["q_l_leading"]) < 1.0e-12
    assert abs(qspec["q_eW"] - qlitw["q_l_watson"]) < 1.0e-12
    assert qspec["q_v"] is None
    assert qeff["primary_basis"] == ["I_l", "Z_l", "X_l", "Y_l"]
    assert abs(qeff["constants_hz"]["q_eW"] - qspec["q_eW"]) < 1.0e-12
    assert abs(qeff["constants_hz"]["q_J_pair"] - 9600.0) < 1.0e-9
    assert abs(qeff["constants_hz"]["q_H_pair"] - 20.0) < 1.0e-9
    assert conv_map["status"] == "minimal_pairwise_ready"
    assert conv_map["active_channel"] == "X_l"
    assert conv_map["inactive_channels"] == ["I_l", "Z_l", "Y_l"]
    assert abs(conv_map["constants_hz"]["q_l"] - pair["q_t_abs_hz"]) < 1.0e-12
    assert abs(conv_map["constants_hz"]["q_lJ"] - 9600.0) < 1.0e-9
    assert abs(conv_map["constants_hz"]["q_lH"] - 20.0) < 1.0e-9
    assert abs(conv_map["legacy_alias_hz"]["q_t"] - pair["q_t_abs_hz"]) < 1.0e-12
    assert coeffs_real == {"I_t": 0.0, "O_t": pair["q_t_abs_hz"], "X_t": 0.0, "Y_t": 0.0}
    assert coeffs_alt == {"I_l": 0.0, "Z_l": 0.0, "X_l": pair["q_t_abs_hz"], "Y_l": 0.0}
    assert pair["pair_hamiltonian_matrix_real_hz"] == [[pair["q_t_abs_hz"], 0.0], [0.0, -pair["q_t_abs_hz"]]]
    assert pair["pair_hamiltonian_matrix_alt_hz"] == [[0.0, pair["q_t_abs_hz"]], [pair["q_t_abs_hz"], 0.0]]
    assert pair["pair_basis_coefficients_real_J2_hz"] == {"I_t": 0.0, "O_t": 9600.0, "X_t": 0.0, "Y_t": 0.0}
    assert pair["pair_basis_coefficients_alt_J2_hz"] == {"I_l": 0.0, "Z_l": 0.0, "X_l": 9600.0, "Y_l": 0.0}
    assert pair["pair_basis_coefficients_real_J4_hz"] == {"I_t": 0.0, "O_t": 20.0, "X_t": 0.0, "Y_t": 0.0}
    assert pair["pair_basis_coefficients_alt_J4_hz"] == {"I_l": 0.0, "Z_l": 0.0, "X_l": 20.0, "Y_l": 0.0}
    assert solver_basis == ["I_l", "Z_l", "X_l", "Y_l"]
    assert solver_blocks["J0"]["vector"] == [0.0, 0.0, qspec["q_eW"], 0.0]
    assert solver_blocks["J2"]["vector"] == [0.0, 0.0, 9600.0, 0.0]
    assert solver_blocks["J4"]["vector"] == [0.0, 0.0, 20.0, 0.0]
    assert solver_blocks["J0"]["matrix"] == [[0.0, qspec["q_eW"]], [qspec["q_eW"], 0.0]]


def test_linear_ltype_terms_assign_pair_labels_for_real_c2h2() -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path="c2h2.fchk")
    out = compute_order2_quartic(model)
    pairs = out["linear_ltype_terms"]["pairs"]
    assert out["linear_ltype_terms"]["pure_rotational_branch"]["kind"] == "linear_pure_rotational"
    assert out["linear_ltype_terms"]["pairwise_ltype_branch"]["pair_count"] == 2
    assert out["linear_ltype_terms"]["pairwise_ltype_branch"]["rotational_feeds"]["quartic_feed_present"] is True
    assert out["linear_ltype_terms"]["pairwise_ltype_branch"]["rotational_feeds"]["sextic_feed_present"] is False
    assert [pair["pair_label"] for pair in pairs] == ["Pi_g(1)", "Pi_u(1)"]
    assert [pair["pair_irrep"] for pair in pairs] == ["Pi_g", "Pi_u"]
    assert [entry["operator"] for entry in pairs[0]["full_operator_basis_real"]] == ["I_t", "O_t", "X_t", "Y_t"]


def test_compute_order2_quartic_passes_linear_special_projection_into_ltype_layer() -> None:
    model = SimpleNamespace(
        abc_mhz=np.array([float("inf"), 15000.0, 15000.0], dtype=float),
        moments_amu_a2=np.array([0.0, 10.0, 10.0], dtype=float),
        vib_freq_cm=np.array([700.0, 702.0, 1200.0], dtype=float),
        coriolis_zeta_pairs_xyz=np.array(
            [
                [[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        xyz_to_abc=("a", "b", "c"),
        dInv_au=np.zeros((3, 3, 3), dtype=float),
    )
    # Build a fake quartic tensor producing D = 0.012 MHz in the linear limit.
    model.dInv_au[1, 1, 0] = 1.0
    model.dInv_au[2, 2, 0] = 1.0
    out = compute_order2_quartic(model)
    assert out["special_quartic_projection"] is not None
    pair = out["linear_ltype_terms"]["pairs"][0]
    assert pair["q_tJ_diagnostic_hz"] > 0.0
