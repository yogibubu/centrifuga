#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import vanvleck_bch_engine as vbe


def test_normal_order_word_contracts_same_mode():
    word = (("a", 0), ("ad", 0))
    terms = dict(vbe.normal_order_word(word))
    assert terms[(("ad", 0), ("a", 0))] == 1
    assert terms[tuple()] == 1


def test_delta_energy_matches_creation_annihilation_balance():
    hbar = sp.Symbol("hbar")
    omega0, omega1 = sp.symbols("omega0 omega1")
    vword = (("ad", 0), ("a", 1), ("a", 0))
    de = vbe.delta_energy(vword, hbar, (omega0, omega1))
    assert sp.simplify(de - hbar * (-omega0 + omega1 + omega0)) == 0


def test_build_effective_to_order_keeps_diagonal_block():
    hbar = sp.Symbol("hbar")
    omega0 = sp.Symbol("omega0", positive=True)
    h_input = {
        1: {
            (tuple(), (vbe.Jy, vbe.Jy), ("H12",)): sp.Integer(3),
            ((("a", 0),), (vbe.Jy, vbe.Jy), ("H12",)): sp.Integer(5),
        }
    }
    k, s = vbe.build_effective_to_order(
        h_input,
        omega=(omega0,),
        hbar=hbar,
        max_order=1,
        max_j=4,
        max_v=2,
    )
    assert k[1][(tuple(), (vbe.Jy, vbe.Jy), ("H12",))] == 3
    assert s[1][((("a", 0),), (vbe.Jy, vbe.Jy), ("H12",))] == -5 / (hbar * omega0)


def test_keep_origin_filter_is_respected():
    hbar = sp.Symbol("hbar")
    omega0 = sp.Symbol("omega0", positive=True)
    h_input = {
        1: {
            (tuple(), (vbe.Jy, vbe.Jy), ("KEEP",)): sp.Integer(1),
            (tuple(), (vbe.Jy, vbe.Jy), ("DROP",)): sp.Integer(2),
        }
    }
    k, _ = vbe.build_effective_to_order(
        h_input,
        omega=(omega0,),
        hbar=hbar,
        max_order=1,
        max_j=4,
        max_v=0,
        keep_origin_fn=lambda origin: origin == ("KEEP",),
    )
    assert k[1] == {(tuple(), (vbe.Jy, vbe.Jy), ("KEEP",)): 1}
