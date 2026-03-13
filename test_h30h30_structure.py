from __future__ import annotations

import sympy as sp

from quartic_channels import channel_h30h30_decomposed


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
    leading = pieces["diag_1_iii_iii"]

    assert sp.simplify(leading["tau_xxxx"] - (-hbar * phi**2 * Axx**2 / (1200 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_yyyy"] - (-hbar * phi**2 * Ayy**2 / (270000 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_zzzz"] - (-hbar * phi**2 * Azz**2 / (675 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_xxyy"] - (-hbar * phi**2 * Axx * Ayy / (9000 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_xxzz"] - (-hbar * phi**2 * Axx * Azz / (450 * omega[0] ** 7))) == 0
    assert sp.simplify(leading["tau_yyzz"] - (-hbar * phi**2 * Ayy * Azz / (6750 * omega[0] ** 7))) == 0


def test_h30h30_total_recombines_leading_and_residual():
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
    total = pieces["total"]
    rebuilt = {
        key: sp.simplify(
            pieces["diag_1_iii_iii"][key]
            + pieces["diag_1_iii_iij_0"][key]
            + pieces["placeholder_residual"][key]
        )
        for key in total
    }
    for key in total:
        assert sp.simplify(total[key] - rebuilt[key]) == 0


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
    pieces_2 = channel_h30h30_decomposed(mu1_2, phi3_2, omega_2, hbar)
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_1_iii_iij_0"].values())
    assert any(sp.simplify(val) != 0 for val in pieces_2["diag_0_iii_iij_1_candidate"].values())


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
    rebuilt = {
        key: sp.simplify(
            pieces["diag_1_iii_iii"][key]
            + pieces["diag_1_iii_iij_0"][key]
            + pieces["placeholder_residual"][key]
        )
        for key in pieces["total"]
    }
    for key in pieces["total"]:
        assert sp.simplify(rebuilt[key] - pieces["total"][key]) == 0
