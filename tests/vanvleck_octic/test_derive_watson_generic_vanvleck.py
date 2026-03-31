#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import derive_watson_quartic_vanvleck as old
from tools.vanvleck import derive_watson_generic_vanvleck as new


def test_generic_driver_reproduces_existing_quartic_engine_for_full_order2_input():
    old.PRUNE_MAX_J = 4
    old.PRUNE_MAX_V = 4
    old.CHANNEL_AWARE = True

    _hprime, h12, h22, h30, h40, omega, hbar = old.build_hprime(n_modes=1, diag_rot_only=True)
    old_input = {1: old.add_expr(h12, h30), 2: old.add_expr(h22, h40)}
    old_k, _old_s = old.build_effective_to_order(old_input, omega=omega, hbar=hbar, max_order=4)
    old_quartic = {}
    for order in range(1, 5):
        for jword, coeff in old.extract_quartic_rot_ground(old_k[order]).items():
            old_quartic[jword] = old_quartic.get(jword, 0) + coeff
    old_tau = old.tau_constants_from_poly(old.commuting_projection(old_quartic))

    out = new.build_generic_reference_problem(
        n_modes=1,
        potential_orders=(3, 4),
        rotder_orders=(1, 2),
        order_assignment={1: ("H12", "H30"), 2: ("H22", "H40")},
        max_order=4,
        max_j=4,
        max_v=4,
        diag_rot_only=True,
        keep_origin_fn=old.keep_origin,
        combine_origin_fn=old.combine_origin,
        generator_keep_fn=old.keep_for_generator,
    )
    new_quartic = out["analysis"]["quartic_total"]
    new_tau = old.tau_constants_from_poly(old.commuting_projection(new_quartic))

    for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
        assert sp.simplify(old_tau[name] - new_tau[name]) == 0


def test_generic_driver_exposes_sextic_and_octic_blocks():
    out = new.build_generic_reference_problem(
        n_modes=1,
        potential_orders=(3, 4, 5, 6),
        rotder_orders=(1, 2, 3),
        order_assignment={1: ("H12", "H30"), 2: ("H22", "H40"), 3: ("H32", "H50"), 4: ("H60",)},
        max_order=4,
        max_j=8,
        max_v=6,
        diag_rot_only=True,
    )
    assert "quartic_total" in out["analysis"]
    assert "sextic_total" in out["analysis"]
    assert "octic_total" in out["analysis"]
