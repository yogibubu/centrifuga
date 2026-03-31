#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import vanvleck_bch_engine as vbe
import vanvleck_perturbative_input as vpi


def _block(tag: str, coeff: int) -> dict[vbe.Key, sp.Expr]:
    return {(tuple(), (vbe.Jy, vbe.Jy), (tag,)): sp.Integer(coeff)}


def test_compose_input_series_combines_blocks_by_order():
    series = vpi.compose_input_series(
        {
            1: (_block("A", 1), _block("B", 2)),
            2: (_block("C", 3),),
        }
    )
    assert series[1][(tuple(), (vbe.Jy, vbe.Jy), ("A",))] == 1
    assert series[1][(tuple(), (vbe.Jy, vbe.Jy), ("B",))] == 2
    assert series[2][(tuple(), (vbe.Jy, vbe.Jy), ("C",))] == 3


def test_collapse_block_coefficients_replaces_values_only():
    A = sp.Symbol("A", real=True)
    block = {
        (tuple(), (vbe.Jy, vbe.Jy), ("H12",)): sp.Integer(7),
        (tuple(), (vbe.Jz, vbe.Jz), ("H12",)): sp.Integer(9),
    }
    collapsed = vpi.collapse_block_coefficients(block, class_symbol=A, weight=sp.Rational(3, 2))
    assert set(collapsed) == set(block)
    assert all(sp.simplify(val - sp.Rational(3, 2) * A) == 0 for val in collapsed.values())


def test_perturbative_origin_catalog_lists_unique_origins():
    series = {
        1: {
            (tuple(), (vbe.Jy, vbe.Jy), ("H12",)): 1,
            (tuple(), (vbe.Jy, vbe.Jy), ("H30",)): 2,
        },
        2: {
            (tuple(), (vbe.Jy, vbe.Jy), ("H22",)): 3,
        },
    }
    catalog = vpi.perturbative_origin_catalog(series)
    assert catalog[1] == (("H12",), ("H30",))
    assert catalog[2] == (("H22",),)
