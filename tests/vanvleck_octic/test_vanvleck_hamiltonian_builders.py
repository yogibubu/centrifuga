#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import vanvleck_hamiltonian_builders as vhb
import vanvleck_bch_engine as vbe


def test_build_potential_block_cubic_has_expected_word_count_for_one_mode():
    hbar = sp.Symbol("hbar")
    omega0 = sp.Symbol("omega0", positive=True)
    phi3 = vhb.symbolic_tensor((1, 1, 1), "phi3")
    block = vhb.build_potential_block(3, phi=phi3, omega=(omega0,), hbar=hbar, origin=("H30",))
    assert len(block) == 8
    for (vword, jword, origin), coeff in block.items():
        assert len(vword) == 3
        assert jword == ()
        assert origin == ("H30",)
        assert coeff.has(phi3[(0, 0, 0)])


def test_build_rotder_block_quadratic_has_expected_rotational_carrier():
    hbar = sp.Symbol("hbar")
    omega0 = sp.Symbol("omega0", positive=True)
    mu2 = vhb.symbolic_tensor((3, 3, 1, 1), "mu2")
    block = vhb.build_rotder_block(
        2,
        mu=mu2,
        omega=(omega0,),
        hbar=hbar,
        origin=("H22",),
        diag_rot_only=True,
    )
    assert block
    for (vword, jword, origin), coeff in block.items():
        assert len(vword) == 2
        assert len(jword) == 2
        assert jword[0] == jword[1]
        assert origin == ("H22",)
        assert coeff.has(hbar)


def test_build_generic_blocks_supports_orders_beyond_quartic():
    out = vhb.build_generic_blocks(n_modes=1, potential_orders=(3, 4, 5, 6), rotder_orders=(1, 2, 3))
    assert "H30" in out["blocks"]
    assert "H40" in out["blocks"]
    assert "H50" in out["blocks"]
    assert "H60" in out["blocks"]
    assert "H12" in out["blocks"]
    assert "H22" in out["blocks"]
    assert "H32" in out["blocks"]
    assert out["hprime"]
