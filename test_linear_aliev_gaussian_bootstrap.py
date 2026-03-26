#!/usr/bin/env python3
"""Smoke tests for the Gaussian -> reduced linear-Aliev payload bootstrap."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import (
    build_explicit_aliev_L_model,
    build_explicit_aliev_beta_breakdown,
    build_explicit_aliev_dv_model,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from gaussian_force_constant_units import raw_cubic_to_reduced_cm, raw_quartic_to_reduced_cm
from gaussian_vpt_parser import parse_gaussian_anharmonic_force_data
from scripts.build_linear_aliev_payload_from_gaussian import build_payload, _principal_coriolis_projection


def test_c2h2_gaussian_bootstrap_builds_reduced_payload() -> None:
    payload = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
    )
    assert payload["metadata"]["quartic_mode"] == "reduced"
    assert payload["metadata"]["parallel_mode_indices_0based"] == [4, 5, 6]
    assert payload["metadata"]["perpendicular_pairs_0based"] == [(0, 1), (2, 3)]
    assert len(payload["omega_parallel"]) == 3
    assert len(payload["omega_perpendicular"]) == 2
    assert len(payload["zeta_nt"]) == 3
    assert len(payload["zeta_nt"][0]) == 2
    assert len(payload["zeta_pair_blocks"]) == 3
    assert len(payload["zeta_pair_blocks"][0]) == 2
    assert np.asarray(payload["zeta_pair_blocks"], dtype=float).shape == (3, 2, 2, 2)
    assert len(payload["k4_reduced"]) == 3
    assert payload["metadata"]["coordinate_normalization"] == "aliev"
    assert payload["metadata"]["zeta_reduction"] == "principal_direction"


def test_c2h2_gaussian_bootstrap_feeds_linear_aliev_models() -> None:
    payload = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced")
    dv = build_explicit_aliev_dv_model(inputs)
    l_model = build_explicit_aliev_L_model(inputs)
    value = dv.value_for_state((0, 0, 0, 0, 0))
    assert len(dv.beta_parallel) == 3
    assert len(dv.beta_perpendicular) == 2
    assert value.is_real is not False
    assert l_model.value.is_real is not False


def test_c2h2_gaussian_bootstrap_supports_alternative_cn_convention() -> None:
    payload_default = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        cn_source="alpha_perp_with_Bxx_equals_minus_alpha_perp",
    )
    payload_half = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        cn_source="alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
    )
    assert payload_default["metadata"]["cn_source"] == "alpha_perp_with_Bxx_equals_minus_alpha_perp"
    assert payload_half["metadata"]["cn_source"] == "alpha_perp_with_Bxx_equals_minus_half_alpha_perp"
    assert payload_half["bxx_parallel"][0] == 0.5 * payload_default["bxx_parallel"][0]


def test_c2h2_gaussian_bootstrap_supports_didq_based_cn_convention() -> None:
    payload = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        cn_source="didq_linear_v_iscr",
    )
    assert payload["metadata"]["cn_source"] == "didq_linear_v_iscr"
    assert len(payload["bxx_parallel"]) == 3
    assert all(np.isfinite(np.asarray(payload["bxx_parallel"], dtype=float)))


def test_c2h2_gaussian_bootstrap_supports_alternative_zeta_reduction() -> None:
    payload_principal = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="principal_direction",
    )
    payload_norm = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="norm",
    )
    payload_max = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="maxabs",
    )
    assert payload_principal["metadata"]["zeta_reduction"] == "principal_direction"
    assert payload_norm["metadata"]["zeta_reduction"] == "norm"
    assert payload_max["metadata"]["zeta_reduction"] == "maxabs"
    assert np.max(np.abs(np.asarray(payload_norm["zeta_nt"]) - np.asarray(payload_max["zeta_nt"]))) > 0.1


def test_c2h2_gaussian_bootstrap_supports_component_tb_zeta_reduction() -> None:
    payload = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="component_tb",
    )
    zeta = np.asarray(payload["zeta_nt"], dtype=float)
    assert payload["metadata"]["zeta_reduction"] == "component_tb"
    assert zeta.shape == (3, 2)
    assert np.max(np.abs(zeta)) < 0.2


def test_c2h2_gaussian_bootstrap_supports_all_componentwise_zeta_variants() -> None:
    payload_ta = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="component_ta",
    )
    payload_ua = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="component_ua",
    )
    payload_ub = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="component_ub",
    )
    assert payload_ta["metadata"]["zeta_reduction"] == "component_ta"
    assert payload_ua["metadata"]["zeta_reduction"] == "component_ua"
    assert payload_ub["metadata"]["zeta_reduction"] == "component_ub"
    assert np.asarray(payload_ta["zeta_nt"]).shape == (3, 2)
    assert np.asarray(payload_ua["zeta_nt"]).shape == (3, 2)
    assert np.asarray(payload_ub["zeta_nt"]).shape == (3, 2)


def test_c2h2_gaussian_bootstrap_supports_pair_offdiag_reduction() -> None:
    payload_tb = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="component_tb",
    )
    payload_ua = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="component_ua",
    )
    payload_offdiag = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        zeta_reduction="pair_offdiag",
    )
    z_tb = np.asarray(payload_tb["zeta_nt"], dtype=float)
    z_ua = np.asarray(payload_ua["zeta_nt"], dtype=float)
    z_off = np.asarray(payload_offdiag["zeta_nt"], dtype=float)
    assert payload_offdiag["metadata"]["zeta_reduction"] == "pair_offdiag"
    assert np.max(np.abs(z_off - 0.5 * (z_tb + z_ua))) < 1.0e-15


def test_principal_coriolis_projection_is_rotation_invariant() -> None:
    block = np.array([[-0.9928555500435392, 0.11932249053611421], [0.11932249053611393, 0.9928555500435390]])
    ref = _principal_coriolis_projection(block)
    angle = 0.371
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = rot.T @ block @ rot
    got = _principal_coriolis_projection(rotated)
    assert abs(ref - got) < 1.0e-12


def test_pair_seed_block_invariant_is_rotation_invariant() -> None:
    block_a = np.array([[-0.21623585244207097, 0.0005710266677840482], [0.0005710266677840482, 0.216235852442071]], dtype=float)
    block_b = np.array([[-0.9928555500435392, 0.11932249053611421], [0.11932249053611393, 0.9928555500435390]], dtype=float)
    ref = float(np.sum(block_a * block_b))
    angle = 0.371
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_a = block_a @ rot
    rot_b = block_b @ rot
    got = float(np.sum(rot_a * rot_b))
    assert abs(ref - got) < 1.0e-12


def test_c2h2_gaussian_bootstrap_defaults_to_gaussian_qe_bridge_seed() -> None:
    payload = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
    )
    assert payload["metadata"]["pair_seed_source"] == "gaussian_qe_source"
    assert payload["metadata"]["pair_seed_status"] == "non_physical_bridge"
    assert len(payload["pair_seed_perpendicular"]) == 2
    assert all(np.isfinite(np.asarray(payload["pair_seed_perpendicular"], dtype=float)))


def test_c2h2_gaussian_bootstrap_supports_formula_pair_seed_path() -> None:
    payload = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        pair_seed_source="formula",
    )
    assert payload["metadata"]["pair_seed_source"] == "formula"
    assert payload["pair_seed_perpendicular"] is None


def test_gaussian_raw_force_constant_reconversions_match_reduced_printout() -> None:
    anh = parse_gaussian_anharmonic_force_data("/Users/vincenzobarone/centrifugal/gaussian/c2h2.log")
    phi3 = raw_cubic_to_reduced_cm(anh.phi3_raw_au, anh.frequencies_cm)
    phi4 = raw_quartic_to_reduced_cm(anh.phi4_raw_au, anh.frequencies_cm)
    assert np.max(np.abs(phi3 - anh.phi3_reduced_cm)) < 0.1
    assert np.max(np.abs(phi4 - anh.phi4_reduced_cm)) < 0.1


def test_c2h2_gaussian_bootstrap_can_use_raw_au_reconverted_force_constants() -> None:
    payload_reduced = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        force_constant_source="reduced",
    )
    payload_raw = build_payload(
        fchk_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk",
        log_path="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log",
        force_constant_source="raw_au_reconverted",
    )
    assert payload_reduced["metadata"]["force_constant_source"] == "reduced"
    assert payload_raw["metadata"]["force_constant_source"] == "raw_au_reconverted"
    assert np.max(np.abs(np.asarray(payload_reduced["k4_reduced"]) - np.asarray(payload_raw["k4_reduced"]))) < 0.1
