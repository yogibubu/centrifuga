#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import derive_watson_quartic_vanvleck as old
from tools.vanvleck import derive_watson_octic_from_loworder_vanvleck as dlo


def test_loworder_octic_problem_reproduces_quartic_slice():
    old.PRUNE_MAX_J = 4
    old.PRUNE_MAX_V = 4
    old.CHANNEL_AWARE = False

    _hprime, h12, h22, h30, h40, omega, hbar = old.build_hprime(n_modes=1, diag_rot_only=True)
    old_input = {1: old.add_expr(h12, h30), 2: old.add_expr(h22, h40)}
    old_k, _old_s = old.build_effective_to_order(old_input, omega=omega, hbar=hbar, max_order=4)
    old_quartic = {}
    for order in range(1, 5):
        for jword, coeff in old.extract_quartic_rot_ground(old_k[order]).items():
            old_quartic[jword] = old_quartic.get(jword, 0) + coeff
    old_tau = old.tau_constants_from_poly(old.commuting_projection(old_quartic))

    out = dlo.derive_loworder_octic_problem(
        n_modes=1,
        diag_rot_only=True,
        max_order=4,
        max_j=8,
        max_v=8,
        keep_origin_fn=old.keep_origin,
        combine_origin_fn=old.combine_origin,
        generator_keep_fn=old.keep_for_generator,
    )
    new_quartic = out["analysis"]["quartic_total"]
    new_tau = old.tau_constants_from_poly(old.commuting_projection(new_quartic))

    for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
        assert sp.simplify(old_tau[name] - new_tau[name]) == 0


def test_loworder_octic_problem_minimal_case_has_no_octic_block_yet():
    out = dlo.derive_loworder_octic_problem(n_modes=1, diag_rot_only=True, max_order=4, max_j=8, max_v=8)
    assert out["analysis"]["quartic_total"]
    assert out["analysis"]["octic_total"] == {}
