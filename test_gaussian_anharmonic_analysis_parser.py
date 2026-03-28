from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_analysis,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
)
import numpy as np

from anharmonic_partition import (
    build_anharmonic_filter_plan,
    classify_three_classes_cartesian,
    direct_y_matrix_cartesian,
    effective_frequencies_from_three_class_report,
    filter_semidiagonal_phi3,
    z_matrix_from_x_y,
)


def test_parse_gaussian_anharmonic_analysis_h2o() -> None:
    out = parse_gaussian_anharmonic_analysis("h2o.log")
    assert out.pt2_model == "Deperturbed VPT2 (DVPT2)"
    assert out.active_resonance_counts["fermi_12"] == 1
    assert out.active_resonance_counts["darling_22"] == 0
    assert out.x_total_cm.shape == (3, 3)
    assert abs(float(out.x_total_cm[2, 1]) + 23.6020) < 1.0e-4
    assert len(out.variational_overlaps) == 2
    assert out.variational_overlaps[0].dvpt2_state == "|1(1)>"
    assert abs(out.variational_overlaps[0].overlap - 0.707) < 5.0e-4
    assert len(out.fundamental_bands) == 3
    assert out.fundamental_bands[0].mode_index == 1
    assert abs(out.fundamental_bands[0].anharmonic_cm - 3569.128) < 1.0e-3


def test_parse_gaussian_anharmonic_analysis_h2co() -> None:
    out = parse_gaussian_anharmonic_analysis("h2co.log")
    assert out.active_resonance_counts["fermi_12"] == 2
    assert out.active_resonance_counts["darling_22"] == 0
    assert out.active_resonance_counts["darling_11"] == 0
    assert len(out.resonances) == 2
    assert out.resonances[0].lhs_modes == (5,)
    assert out.resonances[0].rhs_modes == (2, 6)
    assert out.resonances[0].status == "Active"
    assert abs(out.resonances[0].freq_diff_cm + 161.6695) < 1.0e-4
    assert abs(float(out.x_total_cm[4, 0]) + 138.0) < 1.0e-6
    assert len(out.variational_energies) == 3
    assert out.variational_energies[0].dvpt2_state == "5(1)"
    assert abs(out.variational_energies[0].deperturbed_energy_cm - 2780.261) < 1.0e-3
    assert abs(out.variational_energies[0].after_diag_energy_cm - 2718.760) < 1.0e-3
    assert len(out.fundamental_bands) == 6
    assert out.fundamental_bands[4].mode_index == 5
    assert abs(out.fundamental_bands[4].anharmonic_cm - 2718.760) < 1.0e-3


def test_parse_gaussian_anharmonic_analysis_c2h2_state_definitions() -> None:
    out = parse_gaussian_anharmonic_analysis("c2h2.log")
    assert out.active_resonance_counts["fermi_12"] == 0
    assert out.active_resonance_counts["darling_22"] == 13
    assert out.active_resonance_counts["darling_11"] == 0
    assert len(out.variational_state_definitions) > 20
    block5 = [item for item in out.variational_state_definitions if item.variational_state_index == 5]
    assert len(block5) == 2
    assert abs(block5[0].coefficient - 0.707107) < 1.0e-6
    assert block5[0].dvpt2_state == "|6(1);4(1)>"


def test_z_matrix_from_x_y_is_simple_difference() -> None:
    out = parse_gaussian_anharmonic_analysis("h2o.log")
    z = z_matrix_from_x_y(out.x_total_cm, out.x_coriolis_cm)
    assert z.shape == out.x_total_cm.shape
    assert abs(float(z[2, 1]) + 47.2777) < 1.0e-3


def test_parse_gaussian_anharmonic_force_data_includes_quartics() -> None:
    out = parse_gaussian_anharmonic_force_data("h2o.log")
    assert out.phi4_reduced_cm.shape == (3, 3, 3, 3)
    assert abs(float(out.phi4_reduced_cm[0, 0, 0, 0]) - 770.8416) < 1.0e-4
    assert abs(float(out.phi4_reduced_cm[1, 1, 0, 0]) + 257.92864) < 1.0e-4


def test_parse_gaussian_fchk_harmonic_data_accepts_missing_e_exponent_tokens() -> None:
    out = parse_gaussian_fchk_harmonic_data("nno_DPCS3.fchk")
    assert out.n_modes == 4
    assert out.point_group == "Cinfv"


def test_direct_y_matrix_cartesian_h2o_cartesian_reduction() -> None:
    force = parse_gaussian_anharmonic_force_data("h2o.log")
    analysis = parse_gaussian_anharmonic_analysis("h2o.log")
    y = direct_y_matrix_cartesian(
        force.frequencies_cm,
        force.phi3_reduced_cm,
        force.phi4_reduced_cm,
        coriolis_cm=analysis.x_coriolis_cm,
    )
    assert y.shape == (3, 3)
    assert abs(float(y[0, 0]) + 40.2151492823) < 1.0e-6
    assert abs(float(y[0, 1]) + 36.4948411733) < 1.0e-6
    assert abs(float(y[0, 2]) + 21.8350796586) < 1.0e-6


def test_direct_y_matrix_cartesian_h2co_selected_entries() -> None:
    force = parse_gaussian_anharmonic_force_data("h2co.log")
    analysis = parse_gaussian_anharmonic_analysis("h2co.log")
    y = direct_y_matrix_cartesian(
        force.frequencies_cm,
        force.phi3_reduced_cm,
        force.phi4_reduced_cm,
        coriolis_cm=analysis.x_coriolis_cm,
    )
    assert abs(float(y[0, 0]) + 32.9678484939) < 1.0e-6
    assert abs(float(y[0, 2]) + 27.3711685043) < 1.0e-6


def test_classify_three_classes_cartesian_h2o_keeps_fermi_as_warning_only() -> None:
    force = parse_gaussian_anharmonic_force_data("h2o.log")
    analysis = parse_gaussian_anharmonic_analysis("h2o.log")
    report = classify_three_classes_cartesian(force, analysis)
    mode1 = report.diagnostics[0]
    mode2 = report.diagnostics[1]
    assert mode1.assigned_class == "I"
    assert mode1.active_resonance_count == 1
    assert mode1.low_overlap is not None and abs(mode1.low_overlap - 0.707) < 5.0e-4
    assert mode1.omega_dvpt2_cm is not None and abs(mode1.omega_dvpt2_cm - 3604.162) < 1.0e-3
    assert mode1.omega_gvpt2_cm is not None and abs(mode1.omega_gvpt2_cm - 3569.128) < 1.0e-3
    assert mode1.qp_shift_cm is not None and abs(mode1.qp_shift_cm + 35.034) < 1.0e-3
    assert mode1.resonance_partner_modes == (2,)
    assert "active_fundamental_resonance_flag" in mode1.reasons
    assert "variational_overlap_warning" in mode1.reasons
    assert "large_qp_shift_warning" in mode1.reasons
    assert mode2.assigned_class == "I"
    assert report.unreliable_semidiagonal_pairs == ()


def test_classify_three_classes_cartesian_h2co_keeps_single_polyad_as_warning_only() -> None:
    force = parse_gaussian_anharmonic_force_data("h2co.log")
    analysis = parse_gaussian_anharmonic_analysis("h2co.log")
    report = classify_three_classes_cartesian(force, analysis)
    mode5 = report.diagnostics[4]
    assert mode5.assigned_class == "I"
    assert mode5.active_resonance_count == 2
    assert mode5.low_overlap is not None and abs(mode5.low_overlap - 0.562) < 5.0e-4
    assert mode5.omega_dvpt2_cm is not None and abs(mode5.omega_dvpt2_cm - 2780.261) < 1.0e-3
    assert mode5.omega_gvpt2_cm is not None and abs(mode5.omega_gvpt2_cm - 2718.760) < 1.0e-3
    assert mode5.qp_shift_cm is not None and abs(mode5.qp_shift_cm + 61.501) < 1.0e-3
    assert mode5.resonance_partner_modes == (2, 3, 6)
    assert "active_fundamental_resonance_flag" in mode5.reasons
    assert "variational_overlap_warning" in mode5.reasons
    assert "large_qp_shift_warning" in mode5.reasons
    assert report.unreliable_semidiagonal_pairs == ()


def test_build_anharmonic_filter_plan_keeps_warnings_separate_from_actions() -> None:
    force = parse_gaussian_anharmonic_force_data("h2co.log")
    analysis = parse_gaussian_anharmonic_analysis("h2co.log")
    report = classify_three_classes_cartesian(force, analysis)
    plan = build_anharmonic_filter_plan(report)
    assert plan.warning_modes == (5,)
    assert plan.warning_pairs == ((2, 5), (3, 5), (5, 6))
    assert plan.class2_modes == ()
    assert plan.class3_modes == ()
    assert plan.disable_semidiagonal_pairs == ()


def test_filter_semidiagonal_phi3_zeros_only_requested_pair() -> None:
    force = parse_gaussian_anharmonic_force_data("h2co.log")
    original = force.phi3_reduced_cm
    filtered = filter_semidiagonal_phi3(original, [(5, 6)])
    assert abs(float(filtered[4, 4, 5])) < 1.0e-12
    assert abs(float(filtered[5, 5, 4])) < 1.0e-12
    assert abs(float(filtered[4, 5, 5])) < 1.0e-12
    assert abs(float(filtered[5, 4, 4])) < 1.0e-12
    assert np.array_equal(original[0:4, 0:4, 0:4], filtered[0:4, 0:4, 0:4])


def test_effective_frequencies_from_three_class_report_replaces_only_class2_modes() -> None:
    force = parse_gaussian_anharmonic_force_data("h2co.log")
    analysis = parse_gaussian_anharmonic_analysis("h2co.log")
    report = classify_three_classes_cartesian(
        force,
        analysis,
        class2_max_frequency_cm=4000.0,
    )
    eff = effective_frequencies_from_three_class_report(force.frequencies_cm, report)
    assert abs(float(eff[0]) - float(report.diagnostics[0].omega_gvpt2_cm)) < 1.0e-6
    assert abs(float(eff[1]) - float(report.diagnostics[1].omega_gvpt2_cm)) < 1.0e-6
    assert abs(float(eff[4]) - float(force.frequencies_cm[4])) < 1.0e-6
