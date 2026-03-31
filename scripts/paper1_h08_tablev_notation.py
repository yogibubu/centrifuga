#!/usr/bin/env python3
"""Exact transcription of the Table V notation block used by H08.

This records only what is fully legible in the notation section printed below
Table V.  In particular, it fixes the exact definitions of E and F in terms of
R, X, and H02.
"""

from __future__ import annotations

import sympy as sp


def paper1_tablev_notation_sums() -> dict[str, str]:
    return {
        "Sigma_tilde": "block-diagonal sum",
        "Sigma_star": "sum with resonant terms omitted",
    }


def paper1_tablev_notation_ef_blocks() -> dict[str, sp.Expr]:
    """Return the exact visible E/F definitions from the notation block."""
    H02 = sp.Symbol("H02", commutative=False)

    Rp_kl = sp.Symbol("R'_kl", commutative=False)
    Rt_k = sp.Symbol("R̃_k", commutative=False)
    Rt_l = sp.Symbol("R̃_l", commutative=False)
    Rt_m = sp.Symbol("R̃_m", commutative=False)
    Xkl = sp.Symbol("X_kl", commutative=False)
    Xukl = sp.Symbol("X^kl", commutative=False)
    i = sp.I

    Ekl = sp.Symbol("E_kl", commutative=False)
    Elk = sp.Symbol("E_lk", commutative=False)
    Eu_kl = sp.Symbol("E^kl", commutative=False)
    Eu_lk = sp.Symbol("E^lk", commutative=False)
    Fkl = sp.Symbol("F_kl", commutative=False)
    Flk = sp.Symbol("F_lk", commutative=False)
    Fu_kl = sp.Symbol("F^kl", commutative=False)
    Fu_lk = sp.Symbol("F^lk", commutative=False)

    sum_E = sp.Symbol(
        "Σ_m{X_km(4R'_l + R̃_m - R̃_l) + X_lm(4R'_k + R̃_m - R̃_k)}",
        commutative=False,
    )
    sum_Eu = sp.Symbol(
        "Σ_m{X^km(4R'_l + R̃_m - R̃_l) + X^lm(4R'_k + R̃_m - R̃_k)}",
        commutative=False,
    )
    sum_F = sp.Symbol(
        "Σ_m{X_km(6R'_l + R̃_m - R̃_l) + X_lm(6R'_k + R̃_m - R̃_k)}",
        commutative=False,
    )
    sum_Fu = sp.Symbol(
        "Σ_m{X^km(6R'_l + R̃_m - R̃_l) + X^lm(6R'_k + R̃_m - R̃_k)}",
        commutative=False,
    )

    return {
        "E_kl": sp.Eq(Ekl, Rp_kl - sp.Rational(1, 12) * sum_E + i * sp.Symbol("[X_kl,H02]", commutative=False) / 2),
        "E_lk": sp.Eq(Elk, Rp_kl - sp.Rational(1, 12) * sum_E + i * sp.Symbol("[X_kl,H02]", commutative=False) / 2),
        "E^kl": sp.Eq(Eu_kl, sp.Rational(1, 12) * sum_Eu + i * sp.Symbol("[X^kl,H02]", commutative=False) / 2),
        "E^lk": sp.Eq(Eu_lk, sp.Rational(1, 12) * sum_Eu + i * sp.Symbol("[X^kl,H02]", commutative=False) / 2),
        "F_kl": sp.Eq(Fkl, Rp_kl - sp.Rational(1, 24) * sum_F + i * sp.Symbol("[X_kl,H02]", commutative=False) / 3),
        "F_lk": sp.Eq(Flk, Rp_kl - sp.Rational(1, 24) * sum_F + i * sp.Symbol("[X_kl,H02]", commutative=False) / 3),
        "F^kl": sp.Eq(Fu_kl, sp.Rational(1, 24) * sum_Fu + i * sp.Symbol("[X^kl,H02]", commutative=False) / 3),
        "F^lk": sp.Eq(Fu_lk, sp.Rational(1, 24) * sum_Fu + i * sp.Symbol("[X^kl,H02]", commutative=False) / 3),
    }


def paper1_tablev_notation_minimal_dependency_map() -> dict[str, tuple[str, ...]]:
    return {
        "E": ("R_prime", "X", "H02"),
        "F": ("R_prime", "X", "H02"),
        "primitive_visible_objects": ("R_prime", "R_tilde", "X", "H02"),
    }


def main() -> None:
    for key, eq in paper1_tablev_notation_ef_blocks().items():
        print(key, ":", eq)


if __name__ == "__main__":
    main()
