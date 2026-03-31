from __future__ import annotations

import numpy as np
import sympy as sp

from scripts.h30h30_diagonal_symbolic_probe import _build_diag_only_collapsed, _diag_scaffold_sum
from scripts.h30h30_diagonal_subspace_probe import _collapsed_diag_vector
from scripts.h30h30_diagonal_rotmix_probe import _collapsed_tau_vector, UNIVERSAL_DIAG
from scripts.h30h30_diag_benchmark_basis import build_one_mode_benchmark_basis
from scripts.h30h30_sector_symbolic_probe import _collapsed_sector_vector
from quartic_channels import (
    _H30H30_DIAG1_COEFFS,
    _H30H30_DIAG_BENCH_RESID11_COEFFS,
    _H30H30_DIAG_BENCH_RESID13_COEFFS,
    _H30H30_DIAG1_III_III_CORR_COEFFS,
    _H30H30_DIAG1_III_III_ROTMIX_COEFFS,
    _H30H30_MANUAL_IIJ_PIVOT_COEFFS,
    _H30H30_MANUAL_IIJ_ORDER2_COEFFS,
    _H30H30_MANUAL_IIJ_ORDER3_COEFFS,
    _H30H30_WATERLIKE_BASIS1_WEIGHTS,
    _H30H30_WATERLIKE_BASIS2_WEIGHTS,
    _H30H30_WATERLIKE_SCAFFOLD_BASIS1_WEIGHTS,
    _H30H30_WATERLIKE_SCAFFOLD_BASIS2_WEIGHTS,
    _H30H30_WATERLIKE_REDUCED_III_IIJ_WEIGHTS,
    _H30H30_WATERLIKE_REDUCED_IIJ_IIJ_WEIGHTS,
    channel_h30h30_decomposed,
)
import derive_watson_quartic_vanvleck as dv


def _assert_small(expr, tol: float = 1.0e-5) -> None:
    assert abs(float(sp.N(expr))) < tol


def test_h30h30_diag_scaffold_matches_one_mode_symbolic_baseline():
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, 1)
    phi3 = sp.MutableDenseNDimArray.zeros(1, 1, 1)
    omega = (sp.Symbol("omega0", positive=True),)
    hbar = sp.Symbol("hbar", positive=True)
    Axx, Ayy, Azz, phi = sp.symbols("Axx Ayy Azz phi", real=True)

    mu1[0, 0, 0] = Axx
    mu1[1, 1, 0] = Ayy
    mu1[2, 2, 0] = Azz
    phi3[0, 0, 0] = phi

    pieces = channel_h30h30_decomposed(mu1, phi3, omega, hbar)
    leading = pieces["diag_0_iii_iii"]
    corr = pieces["diag_1_iii_iii_correction_candidate"]

    assert sp.simplify(leading["tau_xxxx"] - (-hbar * phi**2 * Axx**2 / (1200 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_yyyy"] - (-hbar * phi**2 * Ayy**2 / (270000 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_zzzz"] - (-hbar * phi**2 * Azz**2 / (675 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_xxyy"] - (-hbar * phi**2 * Axx * Ayy / (9000 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_xxzz"] - (-hbar * phi**2 * Axx * Azz / (450 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_yyzz"] - (-hbar * phi**2 * Ayy * Azz / (6750 * omega[0] ** 7))) == 0
    assert all(sp.simplify(corr[key]) != 0 for key in corr)


def test_h30h30_total_is_currently_the_full_diagonal_core():
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, 2)
    phi3 = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    omega = (sp.Float("0.02"), sp.Float("0.03"))
    hbar = sp.Float("1.0")

    mu1[0, 0, 0] = sp.Float("2.0")
    mu1[1, 1, 0] = sp.Float("3.0")
    mu1[2, 2, 0] = sp.Float("5.0")
    mu1[0, 0, 1] = sp.Float("7.0")
    mu1[1, 1, 1] = sp.Float("11.0")
    mu1[2, 2, 1] = sp.Float("13.0")
    phi3[0, 0, 0] = sp.Float("1.2")
    phi3[1, 1, 1] = sp.Float("0.7")
    phi3[0, 1, 1] = sp.Float("0.4")

    pieces = channel_h30h30_decomposed(mu1, phi3, omega, hbar)
    pure = pieces["pure_diagonal_total"]
    total = pieces["total"]
    for key in pure:
        _assert_small(sp.simplify(pure[key] - pieces["diag_0_iii_iii"][key]))
    for key in total:
        _assert_small(sp.simplify(
            total[key]
            - pure[key]
            - pieces["diag_1_iii_iii_correction_candidate"][key]
        ))


def test_h30h30_extended_total_recombines_current_diagnostic_sum():
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, 2)
    phi3 = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    omega = (sp.Float("0.02"), sp.Float("0.03"))
    hbar = sp.Float("1.0")

    mu1[0, 0, 0] = sp.Float("2.0")
    mu1[1, 1, 0] = sp.Float("3.0")
    mu1[2, 2, 0] = sp.Float("5.0")
    mu1[0, 0, 1] = sp.Float("7.0")
    mu1[1, 1, 1] = sp.Float("11.0")
    mu1[2, 2, 1] = sp.Float("13.0")
    phi3[0, 0, 0] = sp.Float("1.2")
    phi3[1, 1, 1] = sp.Float("0.7")
    phi3[0, 1, 1] = sp.Float("0.4")

    pieces = channel_h30h30_decomposed(mu1, phi3, omega, hbar)
    extended = pieces["extended_total"]
    rebuilt = {
        key: sp.simplify(
            pieces["diag_0_iii_iii"][key]
            + pieces["diag_1_iii_iii_correction_candidate"][key]
            + pieces["diag_1_iii_iij_0"][key]
            + pieces["placeholder_residual"][key]
        )
        for key in extended
    }
    for key in extended:
        _assert_small(sp.simplify(extended[key] - rebuilt[key]))


def test_h30h30_semidiagonal_scaffold_is_zero_in_one_mode_and_active_in_two_modes():
    hbar = sp.Symbol("hbar", positive=True)

    mu1_1 = sp.MutableDenseNDimArray.zeros(3, 3, 1)
    phi3_1 = sp.MutableDenseNDimArray.zeros(1, 1, 1)
    omega_1 = (sp.Symbol("omega0", positive=True),)
    mu1_1[0, 0, 0] = sp.Integer(2)
    mu1_1[1, 1, 0] = sp.Integer(3)
    mu1_1[2, 2, 0] = sp.Integer(5)
    phi3_1[0, 0, 0] = sp.Symbol("phi", real=True)
    pieces_1 = channel_h30h30_decomposed(mu1_1, phi3_1, omega_1, hbar)
    for key in pieces_1["diag_1_iii_iij_0"]:
        assert sp.simplify(pieces_1["diag_1_iii_iij_0"][key]) == 0
    for key in pieces_1["diag_0_iii_iij_1_candidate"]:
        assert sp.simplify(pieces_1["diag_0_iii_iij_1_candidate"][key]) == 0
    for key in pieces_1["diag_0_iii_iij_2_resonance_candidate"]:
        assert sp.simplify(pieces_1["diag_0_iii_iij_2_resonance_candidate"][key]) == 0
    for key in pieces_1["diag_0_iii_iij_2_regularized_preview"]:
        assert sp.simplify(pieces_1["diag_0_iii_iij_2_regularized_preview"][key]) == 0
    for key in pieces_1["diag_0_iij_iij_1_candidate"]:
        assert sp.simplify(pieces_1["diag_0_iij_iij_1_candidate"][key]) == 0
    for key in pieces_1["diag_0_iij_iij_2_candidate"]:
        assert sp.simplify(pieces_1["diag_0_iij_iij_2_candidate"][key]) == 0

    mu1_2 = sp.MutableDenseNDimArray.zeros(3, 3, 2)
    phi3_2 = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    omega_2 = (sp.Symbol("omega0", positive=True), sp.Symbol("omega1", positive=True))
    mu1_2[0, 0, 0] = sp.Integer(2)
    mu1_2[0, 0, 1] = sp.Integer(7)
    mu1_2[1, 1, 0] = sp.Integer(3)
    mu1_2[1, 1, 1] = sp.Integer(11)
    mu1_2[2, 2, 0] = sp.Integer(5)
    mu1_2[2, 2, 1] = sp.Integer(13)
    phi3_2[0, 0, 0] = sp.Symbol("phi000", real=True)
    phi3_2[0, 0, 1] = sp.Symbol("phi001", real=True)
    phi3_2[1, 1, 0] = sp.Symbol("phi110", real=True)
    pieces_2 = channel_h30h30_decomposed(mu1_2, phi3_2, omega_2, hbar)
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_1_iii_iij_0"].values())
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_0_iii_iij_1_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_0_iii_iij_2_resonance_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_0_iii_iij_2_regularized_preview"].values())
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_0_iij_iij_1_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_0_iij_iij_2_candidate"].values())


def test_h30h30_candidate_scaffold_is_not_in_total_yet():
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, 2)
    phi3 = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    omega = (sp.Float("0.02"), sp.Float("0.03"))
    hbar = sp.Float("1.0")
    mu1[0, 0, 0] = sp.Float("2.0")
    mu1[0, 0, 1] = sp.Float("7.0")
    mu1[1, 1, 0] = sp.Float("3.0")
    mu1[1, 1, 1] = sp.Float("11.0")
    mu1[2, 2, 0] = sp.Float("5.0")
    mu1[2, 2, 1] = sp.Float("13.0")
    phi3[0, 0, 0] = sp.Float("1.2")
    phi3[0, 0, 1] = sp.Float("0.4")

    pieces = channel_h30h30_decomposed(mu1, phi3, omega, hbar)
    for key in pieces["pure_diagonal_total"]:
        _assert_small(sp.simplify(
            pieces["pure_diagonal_total"][key] - pieces["diag_0_iii_iii"][key]
        ))
    for key in pieces["total"]:
        _assert_small(sp.simplify(
            pieces["total"][key]
            - pieces["pure_diagonal_total"][key]
            - pieces["diag_1_iii_iii_correction_candidate"][key]
        ))
    rebuilt_extended = {
        key: sp.simplify(
            pieces["diag_0_iii_iii"][key]
            + pieces["diag_1_iii_iii_correction_candidate"][key]
            + pieces["diag_1_iii_iij_0"][key]
            + pieces["placeholder_residual"][key]
        )
        for key in pieces["extended_total"]
    }
    for key in pieces["extended_total"]:
        _assert_small(sp.simplify(rebuilt_extended[key] - pieces["extended_total"][key]))
    rebuilt_preview = {
        key: sp.simplify(
            pieces["diag_0_iii_iii"][key]
            + pieces["diag_1_iii_iii_correction_candidate"][key]
            + pieces["diag_0_iii_iij_2_regularized_preview"][key]
        )
        for key in pieces["resonance_preview_total"]
    }
    for key in pieces["resonance_preview_total"]:
        _assert_small(sp.simplify(rebuilt_preview[key] - pieces["resonance_preview_total"][key]))
    rebuilt_diagpair = {
        key: sp.simplify(
            pieces["diag_0_iii_iii"][key]
            + pieces["diag_1_iii_iii_correction_candidate"][key]
        )
        for key in pieces["diagonal_pair_preview_total"]
    }
    for key in pieces["diagonal_pair_preview_total"]:
        _assert_small(sp.simplify(rebuilt_diagpair[key] - pieces["diagonal_pair_preview_total"][key]))


def test_h30h30_d1_over_d0_coefficients_are_not_uniformly_small():
    ratios = {
        key: abs(float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key] / _H30H30_DIAG1_COEFFS[key])))
        for key in _H30H30_DIAG1_COEFFS
    }
    assert max(ratios.values()) > 10.0
    assert min(ratios.values()) < 1.0


def test_h30h30_d1_over_d0_coefficients_are_sign_indefinite():
    signed = [
        float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key] / _H30H30_DIAG1_COEFFS[key]))
        for key in _H30H30_DIAG1_COEFFS
    ]
    assert any(val > 0 for val in signed)
    assert any(val < 0 for val in signed)


def test_h30h30_d0_d1_coefficient_vectors_are_strongly_collinear():
    mon_keys = ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")
    v0 = np.array([float(sp.N(_H30H30_DIAG1_COEFFS[key])) for key in mon_keys], dtype=float)
    v1 = np.array([float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key])) for key in mon_keys], dtype=float)
    corr = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    assert abs(corr) > 0.9


def test_h30h30_diag_only_symbolic_probe_has_universal_ratios():
    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_V = 4
    dv.PRUNE_MAX_J = 4
    for n_modes in (1, 2):
        hrv1, v3, omega, hbar, mu1, phi3 = _build_diag_only_collapsed(n_modes, seed=7)
        h_input = dv.build_targeted_input("H30,H30", hrv1, {}, v3, {})
        k_full, _s_series = dv.build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=4)
        quartic = dv.extract_quartic_rot_ground(k_full[4])
        tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
        exprs = dv.decompose_tau_by_symbol_class(tau)["H30,H30"]
        diag_expected = {
            "tau_xxxx": -sp.Rational(1, 1728),
            "tau_yyyy": -sp.Rational(1, 1728),
            "tau_zzzz": -sp.Rational(1, 1728),
            "tau_xxyy": -sp.Rational(1, 864),
            "tau_xxzz": -sp.Rational(1, 864),
            "tau_yyzz": -sp.Rational(1, 864),
        }
        for component, expected in diag_expected.items():
            ratio = sp.simplify(exprs[component] / (_diag_scaffold_sum(mu1, phi3, omega, component) * hbar))
            assert sp.simplify(ratio - expected) == 0


def test_h30h30_manual_sector_recursion_matches_full_builder_for_small_case():
    rot_pairs = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}
    full = _collapsed_sector_vector(
        1,
        7,
        allowed_classes={"iii", "iij"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=False,
    )
    manual = _collapsed_sector_vector(
        1,
        7,
        allowed_classes={"iii", "iij"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=True,
    )
    assert np.allclose(full, manual, atol=1.0e-12, rtol=1.0e-12)


def test_h30h30_manual_iij_sector_dominates_iii_iij_followup_probe():
    rot_pairs = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}
    iii = _collapsed_sector_vector(
        2,
        7,
        allowed_classes={"iii"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=True,
    )
    iij = _collapsed_sector_vector(
        2,
        7,
        allowed_classes={"iij"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=True,
    )
    both = _collapsed_sector_vector(
        2,
        7,
        allowed_classes={"iii", "iij"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=True,
    )
    interference = both - iii - iij
    assert np.linalg.norm(interference) > 0.0
    assert np.linalg.norm(interference) / np.linalg.norm(iij) < 3.0e-2


def test_h30h30_manual_iij_pivot_is_not_closed_by_existing_followup_candidates():
    from quartic_channels import (
        _H30H30_DIAG1_III_IIJ0_COEFFS,
        _H30H30_DIAG0_III_IIJ1_COEFFS,
        _H30H30_DIAG0_III_IIJ2_COEFFS,
        _H30H30_IIJ_IIJ1_COEFFS,
        _H30H30_IIJ_IIJ2_COEFFS,
    )

    mon_keys = ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")
    rot_pairs = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}
    iij = _collapsed_sector_vector(
        2,
        7,
        allowed_classes={"iij"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=True,
    )
    basis = np.column_stack([
        np.array([float(sp.N(_H30H30_DIAG1_III_IIJ0_COEFFS[k])) for k in mon_keys], dtype=float),
        np.array([float(sp.N(_H30H30_DIAG0_III_IIJ1_COEFFS[k])) for k in mon_keys], dtype=float),
        np.array([float(sp.N(_H30H30_DIAG0_III_IIJ2_COEFFS[k])) for k in mon_keys], dtype=float),
        np.array([float(sp.N(_H30H30_IIJ_IIJ1_COEFFS[k])) for k in mon_keys], dtype=float),
        np.array([float(sp.N(_H30H30_IIJ_IIJ2_COEFFS[k])) for k in mon_keys], dtype=float),
    ])
    coeffs, *_ = np.linalg.lstsq(basis, iij, rcond=None)
    relres = np.linalg.norm(iij - basis @ coeffs) / np.linalg.norm(iij)
    assert relres > 0.4


def test_h30h30_manual_iij_coeff_vector_matches_manual_sector_probe():
    rot_pairs = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}
    iij = _collapsed_sector_vector(
        2,
        7,
        allowed_classes={"iij"},
        diag_rot_only=False,
        rot_pairs=rot_pairs,
        manual=True,
    )
    coeff_vec = np.array(
        [float(sp.N(_H30H30_MANUAL_IIJ_PIVOT_COEFFS[k])) for k in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")],
        dtype=float,
    )
    assert np.allclose(iij, coeff_vec, atol=1.0e-12, rtol=1.0e-12)


def test_h30h30_manual_iij_order_decomposition_matches_closed_bch_ratios():
    for key, value in _H30H30_MANUAL_IIJ_PIVOT_COEFFS.items():
        assert sp.simplify(_H30H30_MANUAL_IIJ_ORDER2_COEFFS[key] - 3 * value) == 0
        assert sp.simplify(_H30H30_MANUAL_IIJ_ORDER3_COEFFS[key] + 2 * value) == 0


def test_h30h30_manual_iij_bch_prefactor_is_closed_in_restricted_probe():
    import derive_watson_quartic_vanvleck as dv
    from scripts.h30h30_sector_symbolic_probe import _filter_v3

    dv.CHANNEL_AWARE = True
    dv.PRUNE_MAX_J = 4
    dv.PRUNE_MAX_V = 4
    _h, hrv1, hrv2, v3, v4, omega, hbar, class_syms = dv.build_hprime_collapsed(
        n_modes=2,
        seed=7,
        diag_rot_only=False,
        rot_pairs={(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)},
        symbolic_omega=False,
    )
    v3f = _filter_v3(v3, {"iij"})
    h1 = dv.build_targeted_input("H30,H30", hrv1, hrv2, v3f, v4)[1]
    h_series = {0: {}, 1: h1, 2: {}, 3: {}, 4: {}}
    partial1 = dv.bch_transform(h_series, {}, omega, hbar, max_order=1)
    _, off1 = dv.split_diag_offdiag(partial1[1], omega, hbar)
    s1 = dv.solve_s_order(off1, omega, hbar)
    partial2 = dv.bch_transform(h_series, {1: s1}, omega, hbar, max_order=2)
    _, off2 = dv.split_diag_offdiag(partial2[2], omega, hbar)
    s2 = dv.solve_s_order(off2, omega, hbar)
    a_sym, _b_sym, c_sym, _d_sym = class_syms
    scale = a_sym**2 * c_sym**2 * hbar

    def tau_pair(expr):
        exprn = dv.prune_expr(dv.normal_order_expr(expr))
        quartic = dv.extract_quartic_rot_ground(exprn)
        tau = dv.tau_constants_from_poly(dv.commuting_projection(quartic))
        piece = dv.decompose_tau_by_symbol_class(tau).get("H30,H30", {})
        if not piece:
            return None
        return sp.simplify(piece["tau_xxxx"] / scale), sp.simplify(piece["tau_xxyy"] / scale)

    order2_expected = (
        -sp.Rational(624163585, 449474199552),
        -sp.Rational(929643954971, 288947699712000),
    )
    order3_expected = (
        sp.Rational(624163585, 674211299328),
        sp.Rational(929643954971, 433421549568000),
    )
    assert tau_pair(dv.scale_expr(dv.comm_expr(s2, dv.comm_expr(s1, h1)), sp.Rational(1, 4))) == order2_expected
    h0_branch = dv.comm_with_h0(s1, omega, hbar)
    assert tau_pair(dv.scale_expr(dv.comm_expr(s2, dv.comm_expr(s1, h0_branch)), sp.Rational(1, 6))) == order3_expected


def test_h30h30_one_offdiag_rotational_block_deforms_universal_diag_core():
    rot_pairs = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}
    full = _collapsed_tau_vector(2, 7, rot_pairs=rot_pairs)
    assert any(sp.simplify(full[key] - UNIVERSAL_DIAG[key]) != 0 for key in UNIVERSAL_DIAG)


def test_h30h30_one_mode_full_rotation_deforms_diag_only_core():
    diag = _collapsed_diag_vector(1, 7, diag_rot_only=True)
    full = _collapsed_diag_vector(1, 7, diag_rot_only=False)
    assert np.allclose(diag[:3], full[:3])
    assert not np.allclose(diag[3:], full[3:])


def test_h30h30_one_mode_rotmix_delta_is_supported_on_mixed_components():
    diag = _collapsed_diag_vector(1, 7, diag_rot_only=True)
    full = _collapsed_diag_vector(1, 7, diag_rot_only=False)
    delta = full - diag
    assert np.allclose(delta[:3], 0.0)
    assert not np.allclose(delta[3:], 0.0)


def test_h30h30_current_diag1_placeholder_is_not_mixed_only():
    pure_entries = [
        float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key]))
        for key in ("tau_xxxx", "tau_yyyy", "tau_zzzz")
    ]
    assert any(abs(val) > 0.0 for val in pure_entries)


def test_h30h30_rotmix_diag1_candidate_is_mixed_only():
    assert _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_xxxx"] == 0
    assert _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_yyyy"] == 0
    assert _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_zzzz"] == 0
    assert _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_xxyy"] == -sp.Rational(529, 12441600)
    assert _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_xxzz"] == -sp.Rational(3, 5120000)
    assert _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_yyzz"] == -sp.Rational(121, 1382400)


def test_h30h30_one_mode_benchmark_derived_basis_has_rank_four():
    basis, labels = build_one_mode_benchmark_basis()
    assert labels == [
        "diag(seed=7)",
        "rotmix(seed=7)",
        "residual(seed=11)",
        "residual(seed=13)",
    ]
    assert np.linalg.matrix_rank(basis) == 4


def test_h30h30_one_mode_benchmark_derived_basis_reconstructs_sampled_seeds():
    basis, _labels = build_one_mode_benchmark_basis()
    for seed in (7, 11, 13):
        full = _collapsed_diag_vector(1, seed, diag_rot_only=False)
        coeffs, *_ = np.linalg.lstsq(basis, full, rcond=None)
        residual = full - basis @ coeffs
        assert np.linalg.norm(residual) < 1.0e-12


def test_h30h30_benchmark_residual_candidates_are_nontrivial():
    for coeffs in (_H30H30_DIAG_BENCH_RESID11_COEFFS, _H30H30_DIAG_BENCH_RESID13_COEFFS):
        assert any(abs(float(sp.N(val))) > 0.0 for val in coeffs.values())


def test_h30h30_waterlike_basis_weights_are_nontrivial():
    for weights in (
        _H30H30_WATERLIKE_BASIS1_WEIGHTS,
        _H30H30_WATERLIKE_BASIS2_WEIGHTS,
        _H30H30_WATERLIKE_SCAFFOLD_BASIS1_WEIGHTS,
        _H30H30_WATERLIKE_SCAFFOLD_BASIS2_WEIGHTS,
        _H30H30_WATERLIKE_REDUCED_III_IIJ_WEIGHTS,
        _H30H30_WATERLIKE_REDUCED_IIJ_IIJ_WEIGHTS,
    ):
        assert any(abs(float(sp.N(val))) > 0.0 for val in weights.values())


def test_h30h30_waterlike_preview_total_contains_named_candidates():
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, 2)
    phi3 = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    omega = (sp.Float("0.02"), sp.Float("0.03"))
    hbar = sp.Float("1.0")
    mu1[0, 0, 0] = sp.Float("2.0")
    mu1[0, 0, 1] = sp.Float("7.0")
    mu1[1, 1, 0] = sp.Float("3.0")
    mu1[1, 1, 1] = sp.Float("11.0")
    mu1[2, 2, 0] = sp.Float("5.0")
    mu1[2, 2, 1] = sp.Float("13.0")
    phi3[0, 0, 0] = sp.Float("1.2")
    phi3[0, 0, 1] = sp.Float("0.4")
    phi3[1, 1, 0] = sp.Float("0.6")

    pieces = channel_h30h30_decomposed(mu1, phi3, omega, hbar)
    assert any(sp.simplify(val) != 0 for val in pieces["waterlike_basis1_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["waterlike_basis2_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["waterlike_scaffold_basis1_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["waterlike_scaffold_basis2_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["waterlike_reduced_iii_iij_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["waterlike_reduced_iij_iij_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["manual_iij_order2_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["manual_iij_order3_candidate"].values())
    assert any(sp.simplify(val) != 0 for val in pieces["manual_iij_pivot_candidate"].values())
    for key in pieces["manual_iij_pivot_candidate"]:
        _assert_small(
            sp.simplify(
                pieces["manual_iij_order2_candidate"][key]
                + pieces["manual_iij_order3_candidate"][key]
                - pieces["manual_iij_pivot_candidate"][key]
            )
        )
    for key in pieces["trusted_total"]:
        _assert_small(
            sp.simplify(
                pieces["manual_iij_preview_total"][key]
                - pieces["trusted_total"][key]
                - pieces["manual_iij_pivot_candidate"][key]
            )
        )
        _assert_small(
            sp.simplify(
                pieces["manual_iij_bch_closed_preview_total"][key]
                - pieces["manual_iij_preview_total"][key]
            )
        )
        _assert_small(
            sp.simplify(
                pieces["manual_iij_order2_preview_total"][key]
                + pieces["manual_iij_order3_preview_total"][key]
                - pieces["trusted_total"][key]
                - pieces["manual_iij_bch_closed_preview_total"][key]
            )
        )
