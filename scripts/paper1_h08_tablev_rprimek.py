#!/usr/bin/env python3
"""Exact visible definition of the one-index prime object R'_k from Table V."""

from __future__ import annotations

import sympy as sp


def paper1_tablev_rprimek_definition() -> sp.Eq:
    wk, wl = sp.symbols("omega_k omega_l", positive=True)
    Rp_k = sp.Symbol("R'_k", commutative=False)
    Rl = sp.Symbol("R_l", commutative=False)
    Rp_kl = sp.Symbol("R'_{kl}", commutative=False)
    Rk = sp.Symbol("R_k", commutative=False)
    H02 = sp.Symbol("H02", commutative=False)
    comm = sp.Symbol("[R_k,H02]", commutative=False)
    sum_term = sp.Symbol("Σ_l R_l R'_{kl}/omega_l", commutative=False)
    return sp.Eq(Rp_k, -sum_term - sp.I * comm / wk)


def paper1_tablev_rprimek_support() -> dict[str, tuple[str, ...]]:
    return {
        "R'_k": ("R", "R'_kl", "H02"),
        "expanded": ("mu1", "phi4", "zeta", "omega", "B"),
    }


def main() -> None:
    print(paper1_tablev_rprimek_definition())
    print(paper1_tablev_rprimek_support())


if __name__ == "__main__":
    main()
