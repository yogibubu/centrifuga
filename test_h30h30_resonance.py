#!/usr/bin/env python3
"""Tests for H30H30 resonance diagnostics."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from h30h30_resonance import (
    current_h30h30_factor_diagnostics,
    enumerate_current_h30h30_terms,
    martin_ratio,
    regularized_two_level_term,
    smooth_switch,
)
from quartic_channels import channel_h30h30_decomposed


def test_martin_ratio_and_switch_are_well_behaved():
    assert martin_ratio(2.0, 4.0) == 0.5
    assert math.isinf(martin_ratio(1.0, 0.0))
    assert smooth_switch(0.0, center=1.0, width=0.1, method="tanh") < 1e-6
    assert smooth_switch(2.0, center=1.0, width=0.1, method="erf") > 0.999


def test_regularized_two_level_term_matches_shifted_limit_far_from_resonance():
    value = regularized_two_level_term(1.0, 100.0, metric_center=1.0, metric_width=0.05, level_shift_cm=1.0)
    expected = 1.0 / 101.0
    assert abs(value - expected) < 1e-4


def test_enumerate_current_h30h30_terms_tracks_diagonal_modal_terms():
    mu1 = np.zeros((3, 3, 2))
    mu1[0, 0, 0] = 2.0
    mu1[0, 0, 1] = 3.0
    mu1[1, 1, 0] = 5.0
    mu1[2, 2, 1] = 7.0
    phi3 = np.zeros((2, 2, 2))
    phi3[0, 0, 0] = 11.0
    phi3[1, 1, 1] = 13.0
    omega_cm = [1000.0, 1600.0]

    terms = enumerate_current_h30h30_terms(mu1, phi3, omega_cm, wilson_factor_cm_per_au=2.0)
    assert len(terms) == 12

    diag_terms = [term for term in terms if term.component == "tau_xxxx" and term.diagonal_modes]
    assert len(diag_terms) == 2
    assert diag_terms[0].family == "sum_plus_one_current"
    assert diag_terms[0].modes == (0, 0, 0)
    assert math.isclose(diag_terms[0].coupling_cm, 2.0 * 11.0 * 2.0 * 2.0)


def test_current_h30h30_factor_diagnostics_are_not_martin_ready_yet():
    diags = current_h30h30_factor_diagnostics([1000.0, 1600.0])
    assert diags
    assert any(not item.martin_ready for item in diags)
    names = {item.scaffold for item in diags}
    assert "diag_1_iii_iii" in names
    assert "diag_1_iii_iij_0" in names
    assert "diag_0_iii_iij_1_candidate" in names


def test_signed_iii_iij_candidate_is_the_first_martin_ready_family():
    diags = current_h30h30_factor_diagnostics([1000.0, 1600.0])
    signed = [item for item in diags if item.scaffold == "diag_0_iii_iij_2_resonance_candidate"]
    assert signed
    assert all(item.martin_ready for item in signed)
    assert any(abs(f - (2.0 * 1000.0 - 1600.0)) < 1e-9 for f in signed[0].factors_cm)


def test_regularized_preview_softens_signed_h30h30_candidate_near_resonance():
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, 2)
    phi3 = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    mu1[0, 0, 0] = sp.Float("2.0")
    mu1[0, 0, 1] = sp.Float("3.0")
    mu1[1, 1, 0] = sp.Float("1.5")
    mu1[1, 1, 1] = sp.Float("2.5")
    mu1[2, 2, 0] = sp.Float("1.0")
    mu1[2, 2, 1] = sp.Float("2.0")
    phi3[0, 0, 0] = sp.Float("0.9")
    phi3[0, 0, 1] = sp.Float("0.4")
    omega = (sp.Float("0.0200"), sp.Float("0.0395"))

    pieces = channel_h30h30_decomposed(mu1, phi3, omega, sp.Float("1.0"))
    pert = pieces["diag_0_iii_iij_2_resonance_candidate"]
    reg = pieces["diag_0_iii_iij_2_regularized_preview"]

    pert_norm = sum(abs(float(v)) for v in pert.values())
    reg_norm = sum(abs(float(v)) for v in reg.values())
    assert reg_norm < pert_norm
