#!/usr/bin/env python3
"""Focused tests for the perturbative scalar-``Dv`` module."""

from __future__ import annotations

import numpy as np

from dv_perturbative import (
    DvPerturbativeInputs,
    assemble_Dv,
    compute_Dv_for_state,
    compute_phi_iijk,
    linear_state_factors,
    split_beta_by_mode_kind,
)
from linear_dv_derivation import derive_reduced_linear_dv_equations
from linear_dv_aliev_terms import (
    build_beta_equation_skeletons,
    build_partially_readable_families,
    build_readable_auxiliary_families,
    build_reduced_auxiliary_bridge,
    build_structured_family_skeletons,
)
from linear_dv_operator_blocks import derive_reduced_linear_operator_projection
from quartic_observable_model import build_linear_dv_model


def _co2_like_reduced_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freq = np.array([667.0, 672.0, 1388.0, 2349.0], dtype=float)

    coriolis = np.array(
        [
            [0.0, 0.14, 0.05, 0.02],
            [0.14, 0.0, 0.04, 0.01],
            [0.05, 0.04, 0.0, 0.03],
            [0.02, 0.01, 0.03, 0.0],
        ],
        dtype=float,
    )

    phi3 = np.zeros((4, 4, 4), dtype=float)
    phi3[0, 0, 0] = 12.0
    phi3[0, 0, 1] = 2.4
    phi3[0, 0, 2] = 1.0
    phi3[1, 1, 1] = 11.0
    phi3[1, 1, 0] = 2.2
    phi3[1, 1, 2] = 0.8
    phi3[2, 2, 2] = 9.5
    phi3[2, 2, 0] = 0.7
    phi3[2, 2, 3] = 1.4
    phi3[3, 3, 3] = 8.0
    phi3[3, 3, 2] = 1.1

    phi_iijk = np.array(
        [
            [9.0, 1.6, 0.7, 0.2],
            [1.4, 8.5, 0.5, 0.1],
            [0.6, 0.4, 7.0, 1.2],
            [0.2, 0.1, 1.1, 6.0],
        ],
        dtype=float,
    )
    return freq, coriolis, phi3, phi_iijk


def test_compute_phi_iijk_from_single_mode_hessian_scans() -> None:
    freq, _coriolis, phi3, phi_iijk = _co2_like_reduced_data()
    h0 = np.diag(freq * freq)
    steps = np.array([0.03, 0.04, 0.02, 0.025], dtype=float)
    scans: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for i, step in enumerate(steps):
        h_minus = h0.copy()
        h_plus = h0.copy()
        for k in range(freq.size):
            linear = phi3[i, i, k] * step
            quad = 0.5 * phi_iijk[i, k] * step * step
            h_plus[i, k] += linear + quad
            h_minus[i, k] += -linear + quad
            h_plus[k, i] = h_plus[i, k]
            h_minus[k, i] = h_minus[i, k]
        scans[i] = (h_minus, h_plus)

    recovered = compute_phi_iijk(h0, scans, steps)
    assert np.allclose(recovered, phi_iijk, atol=1.0e-12)


def test_assemble_Dv_keeps_physical_branches_separate_for_co2_like_model() -> None:
    freq, coriolis, phi3, phi_iijk = _co2_like_reduced_data()
    mode_kinds = ("perpendicular", "perpendicular", "parallel", "parallel")
    inputs = DvPerturbativeInputs(
        frequencies_cm=freq,
        mode_kinds=mode_kinds,
        coriolis_matrix=coriolis,
        cubic_force_constants=phi3,
        phi_iijk=phi_iijk,
        dj_equilibrium=2.5e-4,
    )

    result = assemble_Dv(inputs, resonance_floor_cm=5.0, partner_weight=0.5)

    expected_cor = np.zeros_like(freq)
    for i in range(freq.size):
        for j in range(i + 1, freq.size):
            strength = coriolis[i, j] ** 2 / (freq[i] + freq[j])
            expected_cor[i] += 0.5 * strength / freq[i]
            expected_cor[j] += 0.5 * strength / freq[j]

    expected_anh = np.zeros_like(freq)
    expected_direct = np.zeros_like(freq)
    expected_cubic = np.zeros_like(freq)
    for i in range(freq.size):
        for k in range(freq.size):
            direct_strength = phi_iijk[i, k] / (8.0 * freq[i] * (freq[i] + freq[k]))
            cubic_strength = -(phi3[i, i, i] * phi3[i, i, k] / (2.0 * freq[i] + freq[k])) / (
                8.0 * freq[i] * (freq[i] + freq[k])
            )
            expected_anh[i] += direct_strength + cubic_strength
            expected_direct[i] += direct_strength
            expected_cubic[i] += cubic_strength
            if k != i:
                expected_anh[k] += 0.5 * (direct_strength + cubic_strength)
                expected_direct[k] += 0.5 * direct_strength
                expected_cubic[k] += 0.5 * cubic_strength

    assert np.allclose(result.coriolis.beta, expected_cor)
    assert np.allclose(result.anharmonic.beta, expected_anh)
    assert np.allclose(result.total.beta, expected_cor + expected_anh)
    assert np.allclose(result.reduced_terms.coriolis, expected_cor)
    assert np.allclose(result.reduced_terms.quartic_direct, expected_direct)
    assert np.allclose(result.reduced_terms.cubic_dressing, expected_cubic)
    assert abs(result.dj_equilibrium - 2.5e-4) <= 1.0e-15
    assert result.mode_kinds == mode_kinds


def test_compute_Dv_for_arbitrary_state_matches_linear_expansion() -> None:
    freq, coriolis, phi3, phi_iijk = _co2_like_reduced_data()
    quanta = np.array([1, 0, 2, 1], dtype=int)
    mode_kinds = ("perpendicular", "perpendicular", "parallel", "parallel")

    inputs = DvPerturbativeInputs(
        frequencies_cm=freq,
        mode_kinds=mode_kinds,
        coriolis_matrix=coriolis,
        cubic_force_constants=phi3,
        phi_iijk=phi_iijk,
        dj_equilibrium=1.0e-4,
    )
    result = compute_Dv_for_state(inputs, quanta, resonance_floor_cm=5.0, partner_weight=0.5)

    weights = linear_state_factors(quanta, mode_kinds)
    expected = float(result.dj_equilibrium - np.dot(result.total.beta, weights))
    coriolis_only = float(np.dot(result.coriolis.beta, weights))
    anh_only = float(np.dot(result.anharmonic.beta, weights))

    assert abs(result.value_for_state(quanta) - expected) <= 1.0e-15
    assert abs(result.value_for_state(quanta) - (inputs.dj_equilibrium - coriolis_only - anh_only)) <= 1.0e-15


def test_split_beta_by_mode_kind_separates_parallel_and_perpendicular_modes() -> None:
    beta = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    parts = split_beta_by_mode_kind(beta, ("perpendicular", "parallel", "parallel", "perpendicular"))
    assert np.allclose(parts["parallel"], np.array([0.0, 2.0, 3.0, 0.0]))
    assert np.allclose(parts["perpendicular"], np.array([1.0, 0.0, 0.0, 4.0]))


def test_general_quartic_observable_backend_projects_linear_dv() -> None:
    freq, coriolis, phi3, phi_iijk = _co2_like_reduced_data()
    mode_kinds = ("perpendicular", "perpendicular", "parallel", "parallel")
    model, components = build_linear_dv_model(
        freq,
        mode_kinds,
        coriolis,
        phi3,
        phi_iijk,
        dj_equilibrium=3.0e-4,
    )

    quanta = np.array([0, 1, 2, 0], dtype=int)
    weights = linear_state_factors(quanta, mode_kinds)
    expected = float(3.0e-4 + np.dot(components.total, weights))

    assert model.observable_labels == ("Dv",)
    assert abs(model.project_scalar("Dv", quanta) - expected) <= 1.0e-15
    contribs = model.contribution_values_for_state(quanta)
    recombined = (
        contribs["coriolis"]["Dv"]
        + contribs["quartic_direct"]["Dv"]
        + contribs["cubic_dressing"]["Dv"]
        + 3.0e-4
    )
    assert abs(model.project_scalar("Dv", quanta) - recombined) <= 1.0e-15


def test_symbolic_linear_dv_derivation_has_expected_weight_structure() -> None:
    eqs = derive_reduced_linear_dv_equations(3, ("parallel", "perpendicular", "parallel"))
    assert len(eqs.beta_by_mode) == 3
    assert str(eqs.weights_by_mode[0]) == "v0 + 1/2"
    assert str(eqs.weights_by_mode[1]) == "v1 + 1"
    assert str(eqs.weights_by_mode[2]) == "v2 + 1/2"
    assert "D_J" in str(eqs.dv_expression)


def test_operator_level_linear_dv_projection_uses_quartic_carrier() -> None:
    proj = derive_reduced_linear_operator_projection(2, ("parallel", "perpendicular"))
    carrier = str(proj.carrier.polynomial)
    assert "l**2" in carrier
    assert "J" in carrier
    assert str(proj.weights_by_mode[0]) == "v0 + 1/2"
    assert str(proj.weights_by_mode[1]) == "v1 + 1"
    assert "D_J" in str(proj.dv_expression)


def test_auxiliary_bridge_exposes_parallel_and_perpendicular_families() -> None:
    bridge = build_reduced_auxiliary_bridge(("parallel", "perpendicular", "parallel", "perpendicular"))
    part = bridge["partition"]
    aux = bridge["auxiliaries"]
    assert part.parallel == (0, 2)
    assert part.perpendicular == (1, 3)
    assert bridge["status"] == "symbolic_scaffold_only"
    assert (0, 0) in aux.x_nt
    assert (1, 1) in aux.u_tt
    assert 0 in aux.r_n


def test_readable_auxiliary_families_expose_x_and_r_formulas() -> None:
    readable = build_readable_auxiliary_families(("parallel", "perpendicular", "parallel"))
    assert (0, 0) in readable.x_nt_upper
    assert (1, 0) in readable.x_nt_lower
    assert 0 in readable.r_n
    assert (0, 1) in readable.f_nn
    assert "zeta_0_0" in str(readable.x_nt_upper[(0, 0)])
    assert "omega_n0**2 - omega_t0**2" in str(readable.x_nt_upper[(0, 0)])
    text = str(readable.r_n[0])
    assert "D_J" in text
    assert "B" in text
    assert "k_0_0_0" in text
    assert "zeta_0_0" in str(readable.f_nn[(0, 1)])
    assert "B**2" in str(readable.f_nn[(0, 1)])


def test_structured_family_skeletons_preserve_index_topology() -> None:
    skeleton = build_structured_family_skeletons(("parallel", "perpendicular", "parallel", "perpendicular"))
    assert (0, 0) in skeleton.f_nn
    assert (1, 1) in skeleton.f_nn
    assert (0, 0) in skeleton.f_nt
    assert (1, 1) in skeleton.u_tt
    assert (0, 1) in skeleton.v_tt
    text_fnn = str(skeleton.f_nn[(0, 0)])
    assert "K_Fnn_0_0_0" in text_fnn
    assert "zeta_0_0" in text_fnn
    assert "K_Fnt_0_0_0" in str(skeleton.f_nt[(0, 0)])
    assert "K_U_1_1_0" in str(skeleton.u_tt[(1, 1)])


def test_partially_readable_families_expose_rnt_rtt_and_fnt_outer_structure() -> None:
    partial = build_partially_readable_families(("parallel", "perpendicular", "parallel", "perpendicular"))
    assert (0, 0) in partial.r_nt
    assert (1, 1) in partial.r_tt
    assert (0, 1) in partial.f_nt
    text_rnt = str(partial.r_nt[(0, 0)])
    text_rtt = str(partial.r_tt[(1, 1)])
    text_fnt = str(partial.f_nt[(0, 1)])
    assert "Ct_0" in text_rnt
    assert "k_rtt_1_1_0" in text_rtt
    assert "K_Fnt_0_1_0" in text_fnt


def test_beta_equation_skeletons_split_parallel_and_perpendicular_families() -> None:
    beta = build_beta_equation_skeletons(("parallel", "perpendicular", "parallel", "perpendicular"))
    assert 0 in beta.beta_parallel
    assert 1 in beta.beta_parallel
    assert 0 in beta.beta_perpendicular
    assert 1 in beta.beta_perpendicular
    text_par = str(beta.beta_parallel[0])
    text_perp = str(beta.beta_perpendicular[0])
    assert "Qpar_0" in text_par
    assert "r_0" in text_par or "D_J" in text_par
    assert "Qperp_0" in text_perp
    assert "K_U_0_0_0" in text_perp
