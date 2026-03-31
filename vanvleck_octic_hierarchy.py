#!/usr/bin/env python3
"""Exact octic hierarchy scaffolding for the generic Van Vleck engine.

This module does not guess closed formulas for H08. It records the exact
operator hierarchy printed in Watson--Aliev and expresses it as perturbative
input data that can be fed to the generic BCH engine block by block.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class OcticHierarchy:
    H02: sp.Symbol
    H04: sp.Symbol
    H06: sp.Symbol
    H08: sp.Symbol
    S03: sp.Symbol
    S05: sp.Symbol
    S07: sp.Symbol
    term0: sp.Expr
    term1: sp.Expr
    term2: sp.Expr
    term3: sp.Expr
    term4: sp.Expr
    term5: sp.Expr
    term6: sp.Expr
    tilde_H08: sp.Expr


def comm(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return sp.Symbol(f"[{sp.sstr(a)},{sp.sstr(b)}]", commutative=False)


def printed_octic_hierarchy() -> OcticHierarchy:
    H02 = sp.Symbol("H02", commutative=False)
    H04 = sp.Symbol("H04", commutative=False)
    H06 = sp.Symbol("H06", commutative=False)
    H08 = sp.Symbol("H08", commutative=False)
    S03 = sp.Symbol("S03", commutative=False)
    S05 = sp.Symbol("S05", commutative=False)
    S07 = sp.Symbol("S07", commutative=False)

    term0 = H08
    term1 = -sp.I / 6 * comm(S03, comm(S03, comm(S03, H02)))
    term2 = -sp.Rational(1, 2) * comm(S03, comm(S03, H04))
    term3 = sp.I * comm(S03, H06)
    term4 = sp.I * comm(S05, H04)
    term5 = -comm(S05, comm(S03, H02))
    term6 = sp.I * comm(S07, H02)
    tilde = term0 + term1 + term2 + term3 + term4 + term5 + term6
    return OcticHierarchy(H02, H04, H06, H08, S03, S05, S07, term0, term1, term2, term3, term4, term5, term6, tilde)


def hierarchy_coefficients() -> dict[str, sp.Expr]:
    return {
        "H08": sp.Integer(1),
        "S03S03S03H02": -sp.I / 6,
        "S03S03H04": -sp.Rational(1, 2),
        "S03H06": sp.I,
        "S05H04": sp.I,
        "S05S03H02": -sp.Integer(1),
        "S07H02": sp.I,
    }


def hierarchy_dependencies() -> dict[str, tuple[str, ...]]:
    return {
        "bare_block": ("H08",),
        "completion_terms": (
            "S03S03S03H02",
            "S03S03H04",
            "S03H06",
            "S05H04",
            "S05S03H02",
            "S07H02",
        ),
        "lower_order_inputs": ("H02", "H04", "H06", "S03", "S05", "S07"),
    }


__all__ = ["OcticHierarchy", "comm", "printed_octic_hierarchy", "hierarchy_coefficients", "hierarchy_dependencies"]
