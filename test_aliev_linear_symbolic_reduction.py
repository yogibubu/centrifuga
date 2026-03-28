import sys

import sympy as sp

sys.path.insert(0, "/Users/vincenzobarone/centrifugal")

from scripts.aliev_linear_symbolic_reduction import (
    analyze_xf_cross_degree_structure,
    compare_compact_uv_with_bare_h24_uv,
    derive_cylindrical_xy_quartic_constraints,
    degree_audit_eq106_commutators,
    degree_audit_h24_r_terms,
    derive_h04_xy_tensor_from_eq86,
    derive_i_s11r_h04_xy_structure,
    derive_i_s11r_h04_xy_tensor_form,
    derive_regular_symmetric_limit_of_s11r,
    derive_singular_scaling_requirement_for_u_channel,
    derive_s11r_xy_generator_from_h22_offdiag,
    derive_linear_pair_mn_from_x,
    identify_only_possible_source_for_delta,
    locate_u_channel_after_bare_h24_analysis,
    compare_visible_eq101_degrees_with_compact_u,
    compare_visible_eq101_zeta2_with_compact_u_kernel,
    derive_beta_perpendicular_uv_dependence_from_rotlin,
    derive_beta_parallel_uv_dependence_from_rotlin,
    derive_dimensionless_u_kernel_mismatch,
    derive_missing_u_kernel_relative_to_eq101,
    derive_minimal_paper1_u_candidate_and_residual,
    prove_regular_s11r_branch_contradicts_generic_compact_u,
    prove_beta_perpendicular_depends_essentially_on_u,
    prove_beta_parallel_depends_essentially_on_u,
    prove_eq101_crop_is_zeta2_complete,
    prove_s11r_h04_can_only_feed_u_channel,
    prove_bare_h24_uv_equals_v_only_channel,
    prove_iJz_h04_vanishes_under_cylindrical_symmetry,
    prove_mn_inversion,
    prove_prefactor_bookkeeping,
    prove_r_cancellation_in_uv_block,
    transcribe_eq101_visible_zeta2_sector,
    prove_uv_symmetries,
)


def test_mn_inversion_is_exact() -> None:
    out = prove_mn_inversion()
    assert out["back_Xd_minus_Xd"] == 0
    assert out["back_Xu_minus_Xu"] == 0


def test_uv_symmetries_are_exact() -> None:
    out = prove_uv_symmetries()
    assert out["U_swap_minus_U"] == 0
    assert out["V_swap_plus_V"] == 0


def test_linear_pair_mn_specialization_is_exact() -> None:
    out = derive_linear_pair_mn_from_x()
    assert str(out["M_expr"]) == "B*z*(w_n - w_t)/(4*sqrt(w_n)*sqrt(w_t)*(w_n + w_t))"
    assert str(out["N_expr"]) == "B*z*(w_n + w_t)/(4*sqrt(w_n)*sqrt(w_t)*(w_n - w_t))"


def test_prefactor_bookkeeping_is_exact() -> None:
    out = prove_prefactor_bookkeeping()
    assert str(out["M*M+N*N"]) == "(Xdn*Xdp + Xun*Xup)/8"
    assert str(out["M*M-N*N"]) == "-(Xdn*Xup + Xdp*Xun)/8"
    assert out["common_prefactor_mm_nn"] == sp.Rational(1, 8)


def test_uv_block_cancels_r_term_exactly() -> None:
    out = prove_r_cancellation_in_uv_block()
    assert out["r_coefficient"] == 0


def test_compact_uv_is_not_just_bare_h24_uv() -> None:
    out = compare_compact_uv_with_bare_h24_uv()
    assert out["difference"] != 0


def test_xf_cross_has_incompatible_degree_structure() -> None:
    out = analyze_xf_cross_degree_structure()
    assert out["monomials_in_(B,z1,z2,r)"] == [(4, 2, 2, 0), (2, 1, 1, 1)]


def test_only_r_type_terms_can_absorb_delta() -> None:
    out = identify_only_possible_source_for_delta()
    assert out["delta_degree_(B,z1,z2,r)"] == (2, 1, 1, 0)
    assert out["only_possible_source_inside_H24"] == ["R-type pieces"]


def test_no_r_type_term_matches_pure_delta_grade() -> None:
    out = degree_audit_h24_r_terms()
    assert out["pure_delta_candidates"] == []


def test_bare_h24_uv_equals_v_only_channel() -> None:
    out = prove_bare_h24_uv_equals_v_only_channel()
    assert out["bare_plus_4_times_v_channel"] == 0


def test_u_channel_is_not_in_explicit_bare_h24_kernel() -> None:
    out = locate_u_channel_after_bare_h24_analysis()
    assert out["u_channel_location"] == "block-diagonal correction only"


def test_only_constructive_eq106_source_for_pure_u_is_s11_h04() -> None:
    out = degree_audit_eq106_commutators()
    assert out["constructive_sources_for_pure_U_channel"] == ["i[S11^(R),H04]"]


def test_s11r_xy_generator_matches_h22_offdiag_exactly() -> None:
    out = derive_s11r_xy_generator_from_h22_offdiag()
    assert sp.simplify(out["s_xy"] - sp.Symbol("X_xy") / (4 * (-sp.Symbol("B_x") + sp.Symbol("B_y")))) == 0


def test_i_s11r_h04_xy_structure_is_odd_quartic_only() -> None:
    out = derive_i_s11r_h04_xy_structure()
    A, B, C = sp.symbols("A B C")
    Jx, Jy = sp.symbols("Jx J_y")
    expected = (-4 * A + 2 * C) * Jx**3 * Jy + (4 * B - 2 * C) * Jx * Jy**3
    assert sp.expand(out["dH"] - expected) == 0


def test_i_s11r_h04_xy_tensor_form_is_exact() -> None:
    out = derive_i_s11r_h04_xy_tensor_form()
    tau_xxxx, tau_yyyy, tau_xxyy = sp.symbols("tau_xxxx tau_yyyy tau_xxyy")
    assert out["coeff_Jx3Jy"] == -tau_xxxx + 3 * tau_xxyy
    assert out["coeff_JxJy3"] == tau_yyyy - 3 * tau_xxyy


def test_h04_xy_tensor_is_inserted_exactly_from_eq86() -> None:
    out = derive_h04_xy_tensor_from_eq86()
    wx, wy = sp.symbols("w_x w_y", positive=True)
    Cxxx, Cxyy, Cyxx, Cyyy = sp.symbols("C_x_xx C_x_yy C_y_xx C_y_yy")
    assert out["tau_xxxx"] == -2 * (wx * Cxxx**2 + wy * Cyxx**2)
    assert out["tau_xxyy"] == -2 * (wx * Cxxx * Cxyy + wy * Cyxx * Cyyy)
    assert out["tau_yyyy"] == -2 * (wx * Cxyy**2 + wy * Cyyy**2)


def test_s11r_h04_feeds_only_symmetric_channel() -> None:
    out = prove_s11r_h04_can_only_feed_u_channel()
    s_kl, s_lk = sp.symbols("s_kl s_lk")
    assert sp.simplify(out["symmetry_condition"].subs({s_lk: s_kl})) == 0
    assert out["channel"] == "U-only"


def test_s11r_has_regular_symmetric_limit() -> None:
    out = derive_regular_symmetric_limit_of_s11r()
    assert out["s_xy_regular_limit"] == sp.Symbol("Uhat_xy")


def test_linear_xy_quartic_block_is_cylindrically_constrained() -> None:
    out = derive_cylindrical_xy_quartic_constraints()
    lam = sp.Symbol("lam")
    assert out["tau_xxxx"] == 4 * lam
    assert out["tau_yyyy"] == 4 * lam
    assert out["tau_xxyy"] == sp.Rational(4, 3) * lam


def test_iJz_h04_vanishes_in_exact_linear_limit() -> None:
    out = prove_iJz_h04_vanishes_under_cylindrical_symmetry()
    assert out["coeff_Jx3Jy_linear_limit"] == 0
    assert out["coeff_JxJy3_linear_limit"] == 0


def test_u_channel_requires_singular_s11r_branch() -> None:
    out = derive_singular_scaling_requirement_for_u_channel()
    Delta, a, b, c, Xxy, Uhat = sp.symbols("Delta a b c X_xy Uhat_xy")
    assert sp.simplify(out["coeff_Jx3Jy_first_order"] - Delta * (-a + 3 * c)) == 0
    assert sp.simplify(out["coeff_JxJy3_first_order"] - Delta * (b - 3 * c)) == 0
    assert out["singular_limit_Jx3Jy"] == Xxy * (-a + 3 * c) / 4
    assert out["singular_limit_JxJy3"] == Xxy * (b - 3 * c) / 4
    assert out["regularized_limit_Jx3Jy"] == 0
    assert out["regularized_limit_JxJy3"] == 0


def test_regular_s11r_branch_cannot_generate_generic_compact_u() -> None:
    out = prove_regular_s11r_branch_contradicts_generic_compact_u()
    assert out["regularized_commutator_limit_Jx3Jy"] == 0
    assert out["regularized_commutator_limit_JxJy3"] == 0
    assert out["compact_u_is_identically_zero"] != 0


def test_visible_eq101_zeta2_sector_is_symmetric_bilinear() -> None:
    out = transcribe_eq101_visible_zeta2_sector()
    wk, wl, wm = sp.symbols("w_k w_l w_m", positive=True)
    swap = {wk: wl, wl: wk}
    assert sp.simplify(out["kernel_1"].subs(swap, simultaneous=True) - out["kernel_1"]) == 0
    assert sp.simplify(out["kernel_2"].subs(swap, simultaneous=True) - out["kernel_2"]) == 0


def test_eq101_crop_is_complete_for_zeta2_sector() -> None:
    out = prove_eq101_crop_is_zeta2_complete()
    assert out["zeta2_term_count_in_eq101"] == 2
    assert out["closing_bracket_visible"] is True
    assert out["equation_number_visible"] is True
    assert out["zeta2_sector_complete_in_crop"] is True


def test_visible_eq101_zeta2_kernel_is_not_compact_u_kernel() -> None:
    out = compare_visible_eq101_zeta2_with_compact_u_kernel()
    assert out["difference"] != 0


def test_visible_eq101_degree_split_matches_compact_u_split() -> None:
    out = compare_visible_eq101_degrees_with_compact_u()
    assert out["compact_U_degrees_(C,zeta,k4)"]["r_part"] == (0, 0, 1)
    assert out["compact_U_degrees_(C,zeta,k4)"]["zeta2_part"] == (0, 2, 0)
    assert out["visible_eq101_degrees_(C,zeta,k4)"]["k4"] == (0, 0, 1)
    assert out["visible_eq101_degrees_(C,zeta,k4)"]["zeta2"] == (0, 2, 0)


def test_missing_u_kernel_relative_to_eq101_is_nonzero() -> None:
    out = derive_missing_u_kernel_relative_to_eq101()
    assert out["delta_u_kernel"] != 0


def test_minimal_paper1_u_candidate_does_not_reach_aliev() -> None:
    out = derive_minimal_paper1_u_candidate_and_residual()
    assert out["residual_to_aliev"] != 0


def test_dimensionless_u_kernel_mismatch_has_simple_closed_form() -> None:
    out = derive_dimensionless_u_kernel_mismatch()
    x, y = sp.symbols("x y", positive=True)
    expected = -(x * y - 1) * (x**2 + y**2 + 2) / (
        4 * (x - 1) * (x + 1) * (y - 1) * (y + 1)
    )
    assert sp.simplify(out["mismatch_after_simple_1_over_4_rescaling"] - expected) == 0


def test_beta_parallel_uv_piece_from_rotlin_is_exact() -> None:
    out = derive_beta_parallel_uv_dependence_from_rotlin()
    wi, wj, Uij, Vij = sp.symbols("w_i w_j U_ij V_ij")
    assert out["term71"] == Uij**2 * (wi + wj)
    assert out["term72"] == Vij**2 * (wj - wi)
    assert out["term7_uv_piece"] == -16 * Uij**2 * wi - 16 * Uij**2 * wj + 16 * Vij**2 * wi - 16 * Vij**2 * wj


def test_beta_parallel_depends_essentially_on_u() -> None:
    out = prove_beta_parallel_depends_essentially_on_u()
    wi, wj, Uij, Vij = sp.symbols("w_i w_j U_ij V_ij")
    assert out["beta_parallel_uv_entry_point"] == "Term7 only"
    assert out["u_dependence_is_nonzero"] == -32 * Uij * (wi + wj)
    assert out["v_dependence_is_nonzero"] == 32 * Vij * (wi - wj)
    assert out["u_survives_quadratically"] is True


def test_beta_perpendicular_uv_piece_from_rotlin_is_exact() -> None:
    out = derive_beta_perpendicular_uv_dependence_from_rotlin()
    wt, wj, Utj, Vtj = sp.symbols("w_t w_j U_tj V_tj")
    assert out["term7_uv_piece"] == -16 * Utj**2 * wt - 16 * Utj**2 * wj + 16 * Vtj**2 * wt - 16 * Vtj**2 * wj


def test_beta_perpendicular_depends_essentially_on_u() -> None:
    out = prove_beta_perpendicular_depends_essentially_on_u()
    wt, wj, Utj, Vtj = sp.symbols("w_t w_j U_tj V_tj")
    assert out["beta_perpendicular_uv_entry_point"] == "Term7 only"
    assert sp.simplify(out["u_dependence_is_nonzero"] - (-32 * Utj * (wt + wj))) == 0
    assert sp.simplify(out["v_dependence_is_nonzero"] - (32 * Vtj * (wt - wj))) == 0
    assert out["u_survives_quadratically"] is True
